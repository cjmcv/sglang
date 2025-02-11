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
    # tl.program_id()获取当前程序在x-0,y-1,z-2维度上的block_id. 与[grid]对应(batch, head, triton.cdiv(max_input_len, BLOCK))
    cur_batch = tl.program_id(0)  # 对于q而言，是block的行索引，决定所属tile起始行的整体偏移，定位到对应batch。在该batch基础下，进一步按下面的行列索进行分块计算。
    cur_head = tl.program_id(1)   # 对于q而言，是block的列索引
    start_m = tl.program_id(2)    # 对于q而言，是block的行索引

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)                 # 当前计算的batch的长度，长度是按m方向排的。
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)   # 当前计算的batch在整个矩阵中的对应行号。

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_m是一维的，长度是BLOCK_M，数据是从start_m * BLOCK_M 到 start_m * BLOCK_M + BLOCK_M。
    # offs_m[:, None]中None在第二维，表示将其换成二维大小(BLOCK_M,1)，即BLOCK_M行1列。
    # cur_batch_in_all_start_index是一个数字，与offs_m[:, None]进行广播对应位相加，整体进行m方向的偏移。
    # 乘以stride_qbs，即Q矩阵一行元素个数，则offs_m[:, None]表达的意思是从行数，变换成一维数据的行的下标偏移量。
    # 在行的基础上，加上cur_head * stride_qh 做列的偏移，cur_head是头下标，stride_qh是head_dim大小，使跳转到对应head的数据上。此时维度大小仍然是(BLOCK_M,1)
    # offs_d也是一维的，长度是BLOCK_DMODEL，offs_d[None, :]扩展维度在第一维，即扩展成(1,BLOCK_DMODEL)，一行BLOCK_DMODEL列。
    # A(BLOCK_M,1)和B(1,BLOCK_DMODEL)进行相加，A会沿列方向复制BLOCK_DMODEL次，得到(BLOCK_M, BLOCK_DMODEL); 同理B会沿行方向复制BLOCK_M次。逐元素对应位一一相加。
    # 最终得到的off_q维度是(BLOCK_M, BLOCK_DMODEL)，里面的元素表示所选取的一个tile数据的各个元素偏移量。
    #
    # (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs 找到block负责的行
    # cur_head * stride_qh 进而找到当前head对应的列首地址，offs_d进而框住所负责列的数据，得到一个tile
    # 总列数：stride_qbs，每个head列数stride_qh
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
    # 上面Q矩阵，即gemm的A矩阵分块的选取是围绕m和d进行的，d对应gemm中的K维度。
    # K和V矩阵都充当gemm中的B矩阵。
    # 因为K需要转置，所以取offs_n[None, :]维度是(1, BLOCK_N), 里面的数值仍表示行，只是实际排布放到了列。
    # 乘以列数，转为一维数值的行下标，再加上头的行偏置，完成行索引的生成。
    # 列索引由offs_d[:, None]提供(BLOCK_DMODEL,1)。最终的索引矩阵是off_k(BLOCK_DMODEL, BLOCK_N). 
    # 因为其索引中行存放的是列的数据索引，列存放的是行的数据索引，所以直接调用tl.dot做gemm，得到的就是QxKT
    # QK的维度将会是(BLOCK_M, BLOCK_DMODEL) x (BLOCK_DMODEL, BLOCK_N) = (BLOCK_M, BLOCK_N)
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    # off_v(BLOCK_N, BLOCK_DMODEL). V充当B矩阵不需要转置，则用offs_n[:, None]+offs_d[None, :]，行仍是行，列仍是列。
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk # head_dim方向上的mask，因为BLOCK_DMODEL是由next_power_of_2(LK)得到的，实际超出范围部分需要舍弃掉。

    # 直接从首地址Q加上二维索引off_q，获取到Q的对应tile。
    # offs_m[:, None] < cur_batch_seq_len，表示行数不超过cur_batch_seq_len，block取到最后一个M方向tile时，可能会超出范围，需要将其舍弃。seq_len按行方向排。
    # mask_d[None, :]，排除掉列方向上超出范围的部分。
    # other=0.0 表示被mask掉的数据填充为0，gemm时计算0无影响。
    # 整体计算以q为中心，在这里加载一次q的tile，在下面与k和v的多个tile进行计算。
    # 则在固定q_tile(m,k)的情况下, K充当B矩阵需要跟q_tile(m,k)计算的块有k_tile(k,0~n)，共n个块。
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

    # block负责的该batch的起始行号 超过 该batch的长度，则丢弃不用计算。block_start_loc < cur_batch_seq_len时得1，else得0
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # 遍历k矩阵的n方向的需要跟q_tile[m,k]计算的块k_tile[k,0~n]
    # 如果是非因果模型，则n就是当前batch的长度大小。
    # 如果是因果模型，则对于输出矩阵C来说应从左上到右下画一条线，只计算左下的三角形，即m要大于n。
    # 1 0 0 0
    # 1 1 0 0
    # 1 1 1 0
    # 1 1 1 1
    # 因为因果模型计算时无法看到后面的数据，
    # 对于k来说，n是序列延申的方向，对于q，则m是序列延申的方向。q的m到某个点时，只能看到k的n在对应点的前面部分数据。
    # end_n是一个数值，start_m是block id，屏蔽后该block取的q_tile无法与被屏蔽的k_tile计算。
    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    # block_mask是0或1，为0时，整体直接跳过，为1时以end_n的长度为准。每次计算step为BLOCK_N。
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # tl.multiple_of()是将start_n调整为BLOCK_N的倍数。本身就是倍数，可能是多一个保护？
        # -- compute qk ----
        # cur_batch_in_all_start_index 是当前batch的起始行，与q共用一个数值。start_n也当前取的n方向block的首行号，stride_kbs是k矩阵总列数。
        # mask是屏蔽超过该batch长度的部分。Q x KT => Q(m,d) x K(n,d)^T = (m,n), off_k中对索引做了转置，可以直接用dot计算gemm。
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) # gemm
        qk *= sm_scale     # 1/sqrt(dk), dk是head_dim

        if IS_CAUSAL:
            # 根据因果模型进一步屏蔽，n范围在该batch长度内，m要大于n。符合条件的与0相加，即每变化。不符合条件的与负无穷相加，即结果为负无穷。
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            # 非因果模型只限制不超过batch长度范围即可
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # flash attention的分块softmax计算部分
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)            # 在维度1上找最大值，即找出每行的最大值 (一行是一个token的特征值)
        p = tl.exp(qk - m_ij[:, None])  # 自身减去最大值后，取exp指数
        l_ij = tl.sum(p, 1)             # 然后每行数值累加起来
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)   # m_ij 与 历史的m_i 找更大者，作为新的最大值
        alpha = tl.exp(m_i - m_i_new)     # 得到更新系数
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]    # 与前面的结果按缩放系数叠加

        # 以上完成softmax的计算，随后跟v做gemm，更新到acc中
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype) # 将 p 的数据类型转为跟 v 相同
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # 将for循环的计算结果acc，存放到输出矩阵对应tile位置上
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

    # 这里QKV的参数维度是(total_token_num, head, head_dim)，对应多头注意力机制。
    # 单头注意力机制中QKV的维度都是LxD, L是序列长度，D是词嵌入维度。
    # 而多头注意力机制，则对于每个头，其维度是Lx(D/h)，head_dim为D/h，h是头数。下面的Lq等则对应D/h。
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq**0.5) # 缩放归一化因子，Attention计算公式是softmax((Q x KT) / sm_sclae) x V
    batch, head = b_seq_len.shape[0], q.shape[1]
    # 在 GQA 分组查询注意力（Grouped Query Attention）中，Q的head会比KV的head要多，且能被整除
    # GQA 将查询头Q_H分成G组，每组共享一个K和V。减少了键和值的数量，降低了计算量和内存需求。
    # 在标准MHA, QKV的头数都是一样的。
    kv_group_num = q.shape[1] // k.shape[1]  

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK)) # triton.cdiv是向上取整的除法, BLOCK是一个block的负责的数据大小, 表示z方向需要多少个block来处理这些数据。
    num_warps = 4 if Lk <= 64 else 8

    # BLOCK_M 和 BLOCK_N 都是 BLOCK = 128.
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
        kv_group_num=kv_group_num,               # 除了GQA中Q和KV的头数不一致外，其他的attention均为1.
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk), # 下一个 2 的幂次方, 如LK=7，则返回8即2^3
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,  # 如果是因果模型，需要加mask，未来出现的数据不参与计算。
        num_warps=num_warps,  # 一个block里有多少个warp, 为4或8，即一个CTA有128或256个线程
        num_stages=1,         # 软件流水的stages
        Lk=Lk,                # K矩阵的head_dim大小，总列数
    )
