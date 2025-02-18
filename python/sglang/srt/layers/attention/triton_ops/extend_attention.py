# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.utils import is_hip

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

is_hip_ = is_hip()


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

# <NT> extend kernel����chunked prefill
# ������ 1��prefill(prefill��chunk��ĵ�һ��chunk)
#       2��extend(prefill��chunk�󣬳��˵�һ�����������chunk���Լ������extend)��
#       3������˳����decode(���ڼ�϶���Ӵ�����prefill��decode�������һ��batch)
@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    B_req_idx,
    B_Seq_Len,
    B_Start_Loc_Extend,
    B_Seq_Len_Extend,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_req_to_tokens_b,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
): 
    cur_seq = tl.program_id(0)      # ��������0���Ը�batch�������������֣���һ��block��Ӧһ�����С�q��һ�л��߶��л��Ӧbacth���һ�����С�
    cur_head = tl.program_id(1)     # ������
    cur_block_m = tl.program_id(2)  # ������
    cur_kv_head = cur_head // kv_group_num

    cur_seq_len = tl.load(B_Seq_Len + cur_seq)  # B_Seq_Len ��һ���б���������������һ�£������±꼴���б�Ԫ���±꣬��Ǹ�batch��ÿ�����еĳ��ȡ�
    cur_seq_len_extend = tl.load(B_Seq_Len_Extend + cur_seq) # B_Seq_Len_Extend ����������ơ�
    cur_seq_len_prefix = cur_seq_len - cur_seq_len_extend    # ���г��� = prefixǰ׺���� + extend��չ����

    cur_seq_prefix_start_in_loc = 0
    cur_seq_extend_start_contiguous = tl.load(B_Start_Loc_Extend + cur_seq) # extend����ʼ�㣬����q_extend��k_extend��v_extend��
    cur_batch_req_idx = tl.load(B_req_idx + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL) # q��head_dim
    offs_dv = tl.arange(0, BLOCK_DV)    # v��head_dim
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    #��extend��������ʼ��+�̻߳��ֵ��з�Χ��* ������ + �з���head_id*head_dim + �̻߳��ֵ��з�Χ���õ�extend��q_tile
    offs_q = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    # ����LqΪ576��288, ��������¾�Ϊ0.
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # K(n,k) �����ϣ���n�����ϴ�0��ǰ׺���ȷ�Χ��ȡk_tile, �����q_tile��gemm��ȡk_tileʱ����ת�õ�����ȡ�����QxKT���㡣���н��зֿ�softmax��gemm v.
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        # <NT> cur_batch_req_idx����ǰreq�ڸ�batch�µ���ƫ������ stride_req_to_tokens_b: һ�е���������
        # Req_to_tokens�����req��λ��tokens��������start_n + offs_n: �߳�������ķ�Χ
        # �õ�offs_b_loc_prefix��prefix��ƫ�������������ڸ�ƫ������Req_to_tokens���ҵ���Ӧprefix��k_buffer�д�ŵ�λ�á�
        offs_b_loc_prefix = cur_batch_req_idx * stride_req_to_tokens_b + (
            cur_seq_prefix_start_in_loc + start_n + offs_n
        )
        offs_kv_loc = tl.load(Req_to_tokens + offs_b_loc_prefix, mask=mask_n, other=0)

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q.to(k.dtype), k)
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # <NT> ����������ǰ׺���֣�ȫ����������ʷ���ݣ������Ҫȫ���㡣
    # ��������Ҫ����KV����չ���֣��ⲿ��KV�л���Q�ĸ���token��ʱ������ǰ���ص���ϵ�������ģ���У���Ҫ����δ�������ݣ���ֻ��Ҫ�����������򼴿ɡ���
    # n�ķ�Χ��
    # stage 2: compute the trianlge part
    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_contiguous + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_contiguous + start_n + offs_n[None, :])
                * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        # <NT> ������ģ�ͣ������������������mask
        mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
            start_n + offs_n[None, :]
        )
        mask_causual &= mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_contiguous + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o, acc / deno[:, None], mask=mask_m[:, None] & mask_dv[None, :]
    )

# <NT> ����chuncked prefill�� prefill��decode��ϵ����ݡ�
# ��Ҫ��prefillΪ�����������涼Χ����gemm����tl.dot��gemm���С�
# ����������decode����cur_seq�������֣�һ��batch�ﲻ���ж��ͬһ���е�decode���ݡ����㷽ʽ��prefill�ķ�ʽ�����㣬ֻ��q�Ӿ����˻�����������
# �����: prefill�Ǽ����ܼ��ͣ�decode�Ƿô��ܼ��ͣ�prefill��q��decode��qС����prefill��kv���ܻ��decode��С��decode��ʷ���г���k�����N���ܻ���󡣺�ʱָ����˭����
#        �����������߳���Դ����һ���࣬������볬�����е�deocde������(kvcache�ܴ�)���Ƿ������Ӱ���kernel�Ľ���ʱ��(prefill����ȫ�����ˣ���decode��Զû����)��
def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    b_seq_len_extend,
    b_start_loc_extend,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    # <NT> ���һά����ÿ��ͷ���������� head_dim
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )
    # <NT> ��GQA�ȷ���attention���Ὣ��ѯͷ���飬ÿ�鹲��һ������ֵ���Լ��ټ��������ڴ�ʹ�á�
    # �����ַ�������£�GQA ��λ����Ϣ�Ĳ�׽������� MHA ���ܻ�������������Ϊ���鹲���ֵ�ķ�ʽ��һ���̶���������ÿ����ѯͷ��λ����Ϣ�Ķ���ѧϰ�Ͳ�׽��
    # ��ˣ����� PE(Position Encoding, λ�ñ���) ���԰��� GQA ���õ�������������д�Ԫ��λ�ù�ϵ���Ӷ���׼ȷ�ؼ���ע����Ȩ�ء�
    # ע�⣺MHAҲ���Լ�PE����һ���ڷ���attention�г�����
    #
    # ���Lq��576ά�ģ���ô����head_dim=512��pe_dim=64��Ҳ˵�������PE��ѵ�������ģ�ͨ��proj��õ�qkvʱ��˳�����������
    # ͬ��Lq 288 = head_dim 256 + pe_dim 32��
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    # batch_size�����е����������q�������Ҫ�٣�q��һ�л��߶��л��Ӧbacth���һ�����У���grid��batch_size���֣���ζ�����г������ж̵����ݶ��������ͬ�������̣߳�
    batch_size, head_num = b_seq_len.shape[0], q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel[grid](
        q_extend,      
        k_extend,       # extend���֣�ֻ������������
        v_extend,       # extend���֣�ֻ������������
        o_extend,
        k_buffer,       # q_extend��k_buffer��prefix����ȫ����
        v_buffer,       #         ��v_buffer��prefix����ȫ����
        req_to_tokens,  # 2) ����req_idxȥReq_to_tokens���ҵ�prefix���±꣬������K_Buffer��ȥ��prefix��Ӧ��k cache����q_extend����
        b_req_idx,      # 1��ͨ��B_req_idx�ҵ�req���±� req_idx
        b_seq_len,      # һ���б��������batch����������һ�£������±꼴���б�Ԫ�ص��±꣬����Ÿ�batch��ÿ�����еĳ��ȡ�
        b_start_loc_extend, # extend����ʼ�㣬����q_extend��k_extend��v_extend����ʼ��
        b_seq_len_extend,   # �� b_seq_len ���ƣ�Ҳ��һ���б�����Ӧ����extend�ĳ��ȣ�seq_len = prefix_len + seq_len_extend
        sm_scale,           # Q x KT������������ӷ�ĸ�����Ա��ַ����ȶ�����ΪQKT���Կ�����dk��Ԫ�س˻��������ŵĻ���dkԽ�󣬷���Խ�����ź�ɱ���softmax�������С�ں���Χ�ڡ�(����Խ�󣬱�������Խ��ɢ��softmax������׳��ּ������)
        kv_group_num,       # ������������GQA�У�q��head���kv��headҪ�࣬��������=q_head/kv_head.
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        req_to_tokens.stride(0),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend
