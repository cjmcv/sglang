import logging
from contextlib import contextmanager
from typing import Tuple

import torch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = get_bool_env_var("SGLANG_DEEPGEMM_SANITY_CHECK")


# <NT> deepseek的moe计算过程: 
#    1）计算前的数据：如有100个token，假设每个token特征长度为128，则原始数据维度是[100,128]。每层的路由专家权重部分共有256个专家，权重维度是[n,k]，则共有[256,n,k]的权重数据。
#    2）在deepseek的moe计算中，topk被设为8，即表示每个token会选择8个专家网络进行计算。
#    3）一个批次数据进行计算时并不会激活所有专家，如该批次的所有token共激活了20个专家（每个token选8个，所有token都选完后，只命中了这20个专家），
#       即会对应有20个独立的gemm，则对应grouped_gemm中分组将会是20。B矩阵将会是这256个专家中被选出的20个专家的权重组合，每个专家对应一个组，即B矩阵维度是[20,n,k]
#    4）A矩阵是输入数据，一行对应一个token，一个token选topk个专家，即一个token会参与这20个组中的8个组的计算, 即会重复出现8次，总数据量会是[100*topk, 128].
#       应要与B矩阵的分组对应，m=100*topk里会有分组操作，与权重的分组一一对应。
#
# <NT> deepgemm的分组gemm计算kernel分了mask和continous两个，计算示意图：https://zhuanlan.zhihu.com/p/27867600492
# mask api: A矩阵的维度会是[groups, max_m, k]，groups表示当前轮次被激活的专家数，max_m表示当前轮次计算，一个专家被分配最多的token数量，其他分组以该最大token数进行凑整。
#           使每个组的m维度都被凑成max_m，但实际每个组的token数会小于等于max_m，所以需要有一个掩码来表示该group里m维度上哪些数据不需要被计算。
#           masked_m参数：是一个长度为groups的tensor，里面存在的数据是对应每次group中，A矩阵不需要参与计算的部分。
#           且A和B矩阵的分组顺序是一一对应摆放号的，A的0号group的数据对应B的0号group的数据。
#      假设专家负载不均衡，某一个专家对应的token数很多，其他专家的很多，则max_m会比较大，稀疏性会很高。
# continous api: A矩阵的每个token选topk个专家会参与topk个组的计算，即会重复topk次，维度是[m*topk，k]，也可以看成是[groups, (m*topk/groups), k]（实际是按[m*topk，k]排布的）。
#                B矩阵同mask_api，而A矩阵是全部数据都参与计算的，没有填充没用数据，不需要masked_m去屏蔽数据。
#                但因为需要知道A矩阵每行应该要对应哪个组，多设置了m_indices作为索引，长度为[m*topk]，里面一个元素表示A矩阵该行数据应对应哪个分组。
#
# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_deep_gemm_config(gpu_id, server_args)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
