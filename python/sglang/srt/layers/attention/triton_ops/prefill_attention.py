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
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

# <NT> 
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    # tl.program_id()��ȡ��ǰ������x-0,y-1,z-2ά���ϵ�block_id. ��[grid]��Ӧ(batch, head, triton.cdiv(max_input_len, BLOCK))
    cur_batch = tl.program_id(0)  # ����q���ԣ���block������������������tile��ʼ�е�����ƫ�ƣ���λ����Ӧbatch���ڸ�batch�����£���һ������������������зֿ���㡣
    cur_head = tl.program_id(1)   # ����q���ԣ���block��������
    start_m = tl.program_id(2)    # ����q���ԣ���block��������

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)                 # ��ǰ�����batch�ĳ��ȣ������ǰ�m�����ŵġ�
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)   # ��ǰ�����batch�����������еĶ�Ӧ�кš�

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_m��һά�ģ�������BLOCK_M�������Ǵ�start_m * BLOCK_M �� start_m * BLOCK_M + BLOCK_M��
    # offs_m[:, None]��None�ڵڶ�ά����ʾ���任�ɶ�ά��С(BLOCK_M,1)����BLOCK_M��1�С�
    # cur_batch_in_all_start_index��һ�����֣���offs_m[:, None]���й㲥��Ӧλ��ӣ��������m�����ƫ�ơ�
    # ����stride_qbs����Q����һ��Ԫ�ظ�������offs_m[:, None]������˼�Ǵ��������任��һά���ݵ��е��±�ƫ������
    # ���еĻ����ϣ�����cur_head * stride_qh ���е�ƫ�ƣ�cur_head��ͷ�±꣬stride_qh��head_dim��С��ʹ��ת����Ӧhead�������ϡ���ʱά�ȴ�С��Ȼ��(BLOCK_M,1)
    # offs_dҲ��һά�ģ�������BLOCK_DMODEL��offs_d[None, :]��չά���ڵ�һά������չ��(1,BLOCK_DMODEL)��һ��BLOCK_DMODEL�С�
    # A(BLOCK_M,1)��B(1,BLOCK_DMODEL)������ӣ�A�����з�����BLOCK_DMODEL�Σ��õ�(BLOCK_M, BLOCK_DMODEL); ͬ��B�����з�����BLOCK_M�Ρ���Ԫ�ض�Ӧλһһ��ӡ�
    # ���յõ���off_qά����(BLOCK_M, BLOCK_DMODEL)�������Ԫ�ر�ʾ��ѡȡ��һ��tile���ݵĸ���Ԫ��ƫ������
    #
    # (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs �ҵ�block�������
    # cur_head * stride_qh �����ҵ���ǰhead��Ӧ�����׵�ַ��offs_d������ס�������е����ݣ��õ�һ��tile
    # ��������stride_qbs��ÿ��head����stride_qh
    # b0/h0   h1   h2   h3   ...
    # b1/h0   h1   h2
    # b2/h0   h1   h2
    # b3
    # ...
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    # ����Q���󣬼�gemm��A����ֿ��ѡȡ��Χ��m��d���еģ�d��Ӧgemm�е�Kά�ȡ�
    # K��V���󶼳䵱gemm�е�B����
    # ��ΪK��Ҫת�ã�����ȡoffs_n[None, :]ά����(1, BLOCK_N), �������ֵ�Ա�ʾ�У�ֻ��ʵ���Ų��ŵ����С�
    # ����������תΪһά��ֵ�����±꣬�ټ���ͷ����ƫ�ã���������������ɡ�
    # ��������offs_d[:, None]�ṩ(BLOCK_DMODEL,1)�����յ�����������off_k(BLOCK_DMODEL, BLOCK_N). 
    # ��Ϊ���������д�ŵ����е������������д�ŵ����е���������������ֱ�ӵ���tl.dot��gemm���õ��ľ���QxKT
    # QK��ά�Ƚ�����(BLOCK_M, BLOCK_DMODEL) x (BLOCK_DMODEL, BLOCK_N) = (BLOCK_M, BLOCK_N)
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    # off_v(BLOCK_N, BLOCK_DMODEL). V�䵱B������Ҫת�ã�����offs_n[:, None]+offs_d[None, :]���������У��������С�
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk # head_dim�����ϵ�mask����ΪBLOCK_DMODEL����next_power_of_2(LK)�õ��ģ�ʵ�ʳ�����Χ������Ҫ��������

    # ֱ�Ӵ��׵�ַQ���϶�ά����off_q����ȡ��Q�Ķ�Ӧtile��
    # offs_m[:, None] < cur_batch_seq_len����ʾ����������cur_batch_seq_len��blockȡ�����һ��M����tileʱ�����ܻᳬ����Χ����Ҫ����������seq_len���з����š�
    # mask_d[None, :]���ų����з����ϳ�����Χ�Ĳ��֡�
    # other=0.0 ��ʾ��mask�����������Ϊ0��gemmʱ����0��Ӱ�졣
    # ���������qΪ���ģ����������һ��q��tile����������k��v�Ķ��tile���м��㡣
    # ���ڹ̶�q_tile(m,k)�������, K�䵱B������Ҫ��q_tile(m,k)����Ŀ���k_tile(k,0~n)����n���顣
    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # block����ĸ�batch����ʼ�к� ���� ��batch�ĳ��ȣ��������ü��㡣block_start_loc < cur_batch_seq_lenʱ��1��else��0
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # ����k�����n�������Ҫ��q_tile[m,k]����Ŀ�k_tile[k,0~n]
    # ����Ƿ����ģ�ͣ���n���ǵ�ǰbatch�ĳ��ȴ�С��
    # ��������ģ�ͣ�������������C��˵Ӧ�����ϵ����»�һ���ߣ�ֻ�������µ������Σ���mҪ����n��
    # 1 0 0 0
    # 1 1 0 0
    # 1 1 1 0
    # 1 1 1 1
    # ��Ϊ���ģ�ͼ���ʱ�޷�������������ݣ�
    # ����k��˵��n����������ķ��򣬶���q����m����������ķ���q��m��ĳ����ʱ��ֻ�ܿ���k��n�ڶ�Ӧ���ǰ�沿�����ݡ�
    # end_n��һ����ֵ��start_m��block id�����κ��blockȡ��q_tile�޷��뱻���ε�k_tile���㡣
    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    # block_mask��0��1��Ϊ0ʱ������ֱ��������Ϊ1ʱ��end_n�ĳ���Ϊ׼��ÿ�μ���stepΪBLOCK_N��
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # tl.multiple_of()�ǽ�start_n����ΪBLOCK_N�ı�����������Ǳ����������Ƕ�һ��������
        # -- compute qk ----
        # cur_batch_in_all_start_index �ǵ�ǰbatch����ʼ�У���q����һ����ֵ��start_nҲ��ǰȡ��n����block�����кţ�stride_kbs��k������������
        # mask�����γ�����batch���ȵĲ��֡�Q x KT => Q(m,d) x K(n,d)^T = (m,n), off_k�ж���������ת�ã�����ֱ����dot����gemm��
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) # gemm
        qk *= sm_scale     # 1/sqrt(dk), dk��head_dim

        if IS_CAUSAL:
            # �������ģ�ͽ�һ�����Σ�n��Χ�ڸ�batch�����ڣ�mҪ����n��������������0��ӣ���ÿ�仯���������������븺������ӣ������Ϊ�����
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            # �����ģ��ֻ���Ʋ�����batch���ȷ�Χ����
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # flash attention�ķֿ�softmax���㲿��
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)            # ��ά��1�������ֵ�����ҳ�ÿ�е����ֵ (һ����һ��token������ֵ)
        p = tl.exp(qk - m_ij[:, None])  # �����ȥ���ֵ��ȡexpָ��
        l_ij = tl.sum(p, 1)             # Ȼ��ÿ����ֵ�ۼ�����
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)   # m_ij �� ��ʷ��m_i �Ҹ����ߣ���Ϊ�µ����ֵ
        alpha = tl.exp(m_i - m_i_new)     # �õ�����ϵ��
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]    # ��ǰ��Ľ��������ϵ������

        # �������softmax�ļ��㣬����v��gemm�����µ�acc��
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype) # �� p ����������תΪ�� v ��ͬ
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # ��forѭ���ļ�����acc����ŵ���������Ӧtileλ����
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )

# <NT> 
def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True
):
    if is_cuda_available and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    # ����QKV�Ĳ���ά����(total_token_num, head, head_dim)����Ӧ��ͷע�������ơ�
    # ��ͷע����������QKV��ά�ȶ���LxD, L�����г��ȣ�D�Ǵ�Ƕ��ά�ȡ�
    # ����ͷע�������ƣ������ÿ��ͷ����ά����Lx(D/h)��head_dimΪD/h��h��ͷ���������Lq�����ӦD/h��
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq**0.5) # ���Ź�һ�����ӣ�Attention���㹫ʽ��softmax((Q x KT) / sm_sclae) x V
    batch, head = b_seq_len.shape[0], q.shape[1]
    # �� GQA �����ѯע������Grouped Query Attention���У�Q��head���KV��headҪ�࣬���ܱ�����
    # GQA ����ѯͷQ_H�ֳ�G�飬ÿ�鹲��һ��K��V�������˼���ֵ�������������˼��������ڴ�����
    # �ڱ�׼MHA, QKV��ͷ������һ���ġ�
    kv_group_num = q.shape[1] // k.shape[1]  

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK)) # triton.cdiv������ȡ���ĳ���, BLOCK��һ��block�ĸ�������ݴ�С, ��ʾz������Ҫ���ٸ�block��������Щ���ݡ�
    num_warps = 4 if Lk <= 64 else 8

    # BLOCK_M �� BLOCK_N ���� BLOCK = 128.
    # 
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,               # ����GQA��Q��KV��ͷ����һ���⣬������attention��Ϊ1.
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk), # ��һ�� 2 ���ݴη�, ��LK=7���򷵻�8��2^3
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,  # ��������ģ�ͣ���Ҫ��mask��δ�����ֵ����ݲ�������㡣
        num_warps=num_warps,  # һ��block���ж��ٸ�warp, Ϊ4��8����һ��CTA��128��256���߳�
        num_stages=1,         # �����ˮ��stages
        Lk=Lk,                # K�����head_dim��С��������
    )
