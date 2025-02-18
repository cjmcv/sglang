import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# <NT> 计算一个分割指针数组 seg_indptr，用于将数据按照专家（expert）进行分割, 得到每个专家对应token段的起始位置
# 通过二分查找的方式在排序后的数组 reorder_topk_ids 中找到特定专家 expert 应该插入的位置，
# 然后将该位置的索引值存储到 seg_indptr 中相应的专家位置加 1 的地方。
@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)

# <NT> 根据 reorder_ids 数组创建一个从源索引（src_id）到目标索引（dst_id）的映射 src2dst.
# 也就是说，对于每个源索引，找到它在重排序后的目标索引，并记录下来。
@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)

# <NT> 预处理MoE专家并行的topk ID,生成重排序信息
# 例子: https://mp.weixin.qq.com/s/hRVpCFynybW37jogW9_BXA
# 原始的token到专家的分配 (topk_ids)
#   token_idx:     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#   expert_ids:    [1, 3, 2, 1, 0, 2, 3, 1, 2, 0]
#
# 执行run_moe_ep_preproess后:
#   排序后的专家ID (reorder_topk_ids): [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
#   每个专家负责的token段位置 (seg_indptr):
#      expert_id:     [0,    1,    2,    3,    4]
#      seg_indptr:    [0,    2,    5,    8,    10]
#      含义：
#        - expert 0 处理索引 0-1 的token
#        - expert 1 处理索引 2-4 的token
#        - expert 2 处理索引 5-7 的token
#        - expert 3 处理索引 8-9 的token
#   原始位置到重排序后位置的映射 (src2dst):
#      原始位置:      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#      重排序后位置:   [4, 9, 2, 3, 7, 5, 8, 6, 0, 1]
#   这样重排序后，相同专家要处理的token就被组织在了一起
#      重排序后位置:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#      专家ID:        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):
    # 对专家id, 从小到大排序，reorder_topk_ids是排序后的id好，reorder_ids是调整后对应的原始索引
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    # 初始化分段指针和源目标映射数组. 
    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    # 计算每个专家对应token段的起始位置. numel()是得到元素总数
    compute_seg_indptr_triton_kernel[(num_experts,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    # 计算源索引到目标索引的映射
    compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE
    )
    return reorder_topk_ids, src2dst, seg_indptr


# <NT> 预重排序kernel, 将输入数据重新排列并应用缩放    
# 该kernel将输入数据按照专家分配重新排列, 并对分配到当前rank的专家数据进行缩放处理。
# 对于每个输入token, 遍历其选择的topk个专家, 如果专家属于当前rank, 则将该token的数据拷贝到对应位置并应用缩放因子。
# input_ptr输入张量 排序得到 gateup_input_ptr的实际输入张量
@triton.jit
def pre_reorder_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0) # 拿到当前处理的输入token索引
    # 进而计算得到当前token的src2dst和topk_ids指针位置, 以及输入数据的对应索引
    src2dst_ptr = src2dst_ptr + src_idx * topk  
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size
    # 遍历当前token选择的topk个专家
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        # 确保专家属于当前rank
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            # 计算缩放因子
            if a1_scales_ptr is not None:
                scale = 1.0 / tl.load(a1_scales_ptr + expert_id - start_expert_id)
            else:
                scale = 1.0
            # 获取目标位置索引和指针
            dst_idx = tl.load(src2dst_ptr + idx)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            # 按块处理hidden_size维度的数据
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + tl.arange(0, BLOCK_SIZE)
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
                out_data = (in_data * scale).to(OutDtype)
                tl.store(dst_ptr + offset, out_data, mask=mask)


@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)

# <NT> 该函数将专家输出重新排序并加权求和,生成最终输出。
# 主要步骤:
#   1. 获取输入数据类型和程序ID
#   2. 计算各指针偏移量
#   3. 对每个block:
#      - 创建零向量用于累加
#      - 对每个topk专家:
#        * 如果专家ID在范围内,加载并累加其输出
#   4. 如果没有计算过的专家,输出全零向量
@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    # 以源索引, 计算各指针的实际位置
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    # 标记是否有专家参与计算
    computed = False
    # 计算存储位置
    store_ptr = output_ptr + src_idx * hidden_size
    # 按block大小遍历hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        # 创建零向量用于累加
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        # 遍历topk个专家
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            # 检查专家ID是否在有效范围内
            if expert_id >= start_expert_id and expert_id <= end_expert_id:
                computed = True
                # 加载目标索引和权重
                dst_idx = tl.load(src2dst_ptr + idx)
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                # 计算加载位置并加载数据
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                # 加权累加
                sum_vec += in_data * weigh_scale
        # 存储累加结果
        tl.store(store_ptr + offset, sum_vec, mask=mask)

    # 如果没有专家参与计算,输出全零
    if computed == False:
        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            tl.store(
                store_ptr + offset, tl.zeros([BLOCK_SIZE], dtype=InDtype), mask=mask
            )


@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    idx = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            idx = bs

    idx_start = tl.load(m_num_tiles_indptr + idx)

    m_range_start = tl.load(seg_indptr + idx) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + idx + 1), m_range_start + BLOCK_SIZE_M)
    expert_id = tl.load(weight_indices + idx)
    return m_range_start, m_range_end, expert_id


@triton.jit
def grouped_gemm_triton_kernel(
    a,
    b,
    c,
    batch_size,
    N,
    K,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    use_fp8_w8a8,
    scale_a,
    scale_b,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, expert_id = compute_m_range(
        pid_m, batch_size, seg_indptr, weight_indices, m_num_tiles_indptr, BLOCK_SIZE_M
    )
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a + (m_range_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    b_ptr = b + (
        (expert_id * b_stride_0)
        + (n_range_start + offs_bn[:, None]) * b_stride_1
        + offs_k[None, :]
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(
            a_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        b_tile = tl.load(
            b_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        accumulator = tl.dot(a_tile, b_tile.T, accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    if use_fp8_w8a8:
        scale_a_value = tl.load(scale_a + expert_id)
        scale_b_value = tl.load(scale_b + expert_id)
        accumulator *= scale_a_value * scale_b_value
    c_tile = accumulator.to(c_dtype)

    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptr, c_tile, mask=c_mask)


@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
):
    # <NT> 仅支持权重列优先，nn.Linear权重里(out_features, in_features)对应。行优先的gemm，b矩阵应该是(in_features, out_features)
    assert weight_column_major == True  # TODO: more
    if use_fp8_w8a8:
        assert scale_a is not None and scale_b is not None

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
    }

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](
        m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"]
    )

    grid = lambda META: (
        triton.cdiv(a.size(0), META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(b.size(1), META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_triton_kernel[grid](
        a,
        b,
        c,
        batch_size,
        b.size(1),
        b.size(2),
        seg_indptr,
        weight_indices,
        m_num_tiles_indptr,
        use_fp8_w8a8,
        scale_a,
        scale_b,
        a.stride(0),
        b.stride(0),
        b.stride(1),
        **config,
    )
    return c
