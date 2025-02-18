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

# <NT> extend kernel用于chunked prefill
# 即包含 1）prefill(prefill分chunk后的第一个chunk)
#       2）extend(prefill分chunk后，除了第一个以外的其他chunk；以及常规的extend)，
#       3）穿插顺带的decode(会在间隙中捎带，是prefill与decode可以组成一个batch)
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
    cur_seq = tl.program_id(0)      # 行索引，0号以该batch里序列数量划分，即一个block对应一个序列。q的一行或者多行会对应bacth里的一个序列。
    cur_head = tl.program_id(1)     # 列索引
    cur_block_m = tl.program_id(2)  # 行索引
    cur_kv_head = cur_head // kv_group_num

    cur_seq_len = tl.load(B_Seq_Len + cur_seq)  # B_Seq_Len 是一个列表，长度与序列数量一致，序列下标即是列表元素下标，标记该batch下每个序列的长度。
    cur_seq_len_extend = tl.load(B_Seq_Len_Extend + cur_seq) # B_Seq_Len_Extend 与上面的类似。
    cur_seq_len_prefix = cur_seq_len - cur_seq_len_extend    # 序列长度 = prefix前缀长度 + extend扩展长度

    cur_seq_prefix_start_in_loc = 0
    cur_seq_extend_start_contiguous = tl.load(B_Start_Loc_Extend + cur_seq) # extend的起始点，包括q_extend，k_extend和v_extend。
    cur_batch_req_idx = tl.load(B_req_idx + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL) # q的head_dim
    offs_dv = tl.arange(0, BLOCK_DV)    # v的head_dim
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    #（extend的数据起始点+线程划分的行范围）* 总列数 + 列方向head_id*head_dim + 线程划分的列范围，得到extend的q_tile
    offs_q = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    # 除了Lq为576或288, 其他情况下均为0.
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

    # K(n,k) 矩阵上，在n方向上从0到前缀长度范围，取k_tile, 逐个与q_tile做gemm。取k_tile时，按转置的索引取，完成QxKT计算。进行进行分块softmax和gemm v.
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        # <NT> cur_batch_req_idx：当前req在该batch下的行偏移量； stride_req_to_tokens_b: 一行的总列数？
        # Req_to_tokens存放由req定位到tokens的索引，start_n + offs_n: 线程所负责的范围
        # 得到offs_b_loc_prefix是prefix的偏移量。进而基于该偏移量在Req_to_tokens中找到对应prefix在k_buffer中存放的位置。
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

    # <NT> 上面计算的是前缀部分，全部都属于历史数据，因此需要全计算。
    # 而这里需要计算KV的扩展部分，这部分KV中会与Q的各个token在时序上有前后重叠关系。而因果模型中，需要屏蔽未来的数据，则只需要计算三角区域即可。、
    # n的范围从
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

        # <NT> 针对因果模型，用于屏蔽三角区域的mask
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

# <NT> 负责chuncked prefill或 prefill与decode混合的数据。
# 主要以prefill为主，所以里面都围绕着gemm进行tl.dot的gemm进行。
# 混合在里面的decode会以cur_seq进行区分，一个batch里不会有多个同一序列的decode数据。计算方式按prefill的方式来计算，只是q从矩阵退化成了向量。
# 问题点: prefill是计算密集型，decode是访存密集型，prefill的q大，decode的q小，但prefill的kv可能会比decode的小，decode历史序列长，k矩阵的N可能会更大。耗时指不定谁长。
#        且里面分配的线程资源都是一样多，如果混入超长序列的deocde在里面(kvcache很大)，是否会严重影响该kernel的结束时间(prefill部分全算完了，但decode还远没结束)？
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
    # <NT> 最后一维都是每个头的特征长度 head_dim
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )
    # <NT> 如GQA等分组attention，会将查询头分组，每组共享一个键和值，以减少计算量和内存使用。
    # 在这种分组机制下，GQA 对位置信息的捕捉能力相对 MHA 可能会有所减弱，因为分组共享键值的方式在一定程度上限制了每个查询头对位置信息的独立学习和捕捉。
    # 因此，引入 PE(Position Encoding, 位置编码) 可以帮助 GQA 更好地理解输入序列中词元的位置关系，从而更准确地计算注意力权重。
    # 注意：MHA也可以加PE，但一般在分组attention中常见。
    #
    # 如果Lq是576维的，那么就有head_dim=512和pe_dim=64。也说明这里的PE是训练出来的，通过proj层得到qkv时，顺带计算出来。
    # 同理Lq 288 = head_dim 256 + pe_dim 32。
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
    # batch_size是序列的数量，会比q矩阵的行要少，q的一行或者多行会对应bacth里的一个序列，而grid按batch_size划分，意味着序列长和序列短的数据都会分配相同数量的线程？
    batch_size, head_num = b_seq_len.shape[0], q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel[grid](
        q_extend,      
        k_extend,       # extend部分，只计算三角区域
        v_extend,       # extend部分，只计算三角区域
        o_extend,
        k_buffer,       # q_extend与k_buffer里prefix部分全计算
        v_buffer,       #         与v_buffer里prefix部分全计算
        req_to_tokens,  # 2) 拿着req_idx去Req_to_tokens里找到prefix的下标，进而在K_Buffer中去除prefix对应的k cache，与q_extend计算
        b_req_idx,      # 1）通过B_req_idx找到req的下标 req_idx
        b_seq_len,      # 一个列表，长度与该batch的序列数量一致，序列下标即是列表元素的下标，标记着该batch下每个序列的长度。
        b_start_loc_extend, # extend的起始点，包括q_extend，k_extend和v_extend的起始点
        b_seq_len_extend,   # 与 b_seq_len 类似，也是一个列表，但对应的是extend的长度，seq_len = prefix_len + seq_len_extend
        sm_scale,           # Q x KT下面的缩放因子分母，可以保持方差稳定，因为QKT可以看作是dk个元素乘积，不缩放的话，dk越大，方差越大。缩放后可保持softmax的输入大小在合理范围内。(方差越大，表明数据越离散，softmax输出容易出现极端情况)
        kv_group_num,       # 分组数量，如GQA中，q的head会比kv的head要多，分组数量=q_head/kv_head.
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
