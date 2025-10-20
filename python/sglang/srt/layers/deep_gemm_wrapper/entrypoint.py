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


# <NT> deepseek��moe�������: 
#    1������ǰ�����ݣ�����100��token������ÿ��token��������Ϊ128����ԭʼ����ά����[100,128]��ÿ���·��ר��Ȩ�ز��ֹ���256��ר�ң�Ȩ��ά����[n,k]������[256,n,k]��Ȩ�����ݡ�
#    2����deepseek��moe�����У�topk����Ϊ8������ʾÿ��token��ѡ��8��ר��������м��㡣
#    3��һ���������ݽ��м���ʱ�����ἤ������ר�ң�������ε�����token��������20��ר�ң�ÿ��tokenѡ8��������token��ѡ���ֻ��������20��ר�ң���
#       �����Ӧ��20��������gemm�����Ӧgrouped_gemm�з��齫����20��B���󽫻�����256��ר���б�ѡ����20��ר�ҵ�Ȩ����ϣ�ÿ��ר�Ҷ�Ӧһ���飬��B����ά����[20,n,k]
#    4��A�������������ݣ�һ�ж�Ӧһ��token��һ��tokenѡtopk��ר�ң���һ��token�������20�����е�8����ļ���, �����ظ�����8�Σ�������������[100*topk, 128].
#       ӦҪ��B����ķ����Ӧ��m=100*topk����з����������Ȩ�صķ���һһ��Ӧ��
#
# <NT> deepgemm�ķ���gemm����kernel����mask��continous����������ʾ��ͼ��https://zhuanlan.zhihu.com/p/27867600492
# mask api: A�����ά�Ȼ���[groups, max_m, k]��groups��ʾ��ǰ�ִα������ר������max_m��ʾ��ǰ�ִμ��㣬һ��ר�ұ���������token���������������Ը����token�����д�����
#           ʹÿ�����mά�ȶ����ճ�max_m����ʵ��ÿ�����token����С�ڵ���max_m��������Ҫ��һ����������ʾ��group��mά������Щ���ݲ���Ҫ�����㡣
#           masked_m��������һ������Ϊgroups��tensor��������ڵ������Ƕ�Ӧÿ��group�У�A������Ҫ�������Ĳ��֡�
#           ��A��B����ķ���˳����һһ��Ӧ�ڷźŵģ�A��0��group�����ݶ�ӦB��0��group�����ݡ�
#      ����ר�Ҹ��ز����⣬ĳһ��ר�Ҷ�Ӧ��token���ܶ࣬����ר�ҵĺܶ࣬��max_m��Ƚϴ�ϡ���Ի�ܸߡ�
# continous api: A�����ÿ��tokenѡtopk��ר�һ����topk����ļ��㣬�����ظ�topk�Σ�ά����[m*topk��k]��Ҳ���Կ�����[groups, (m*topk/groups), k]��ʵ���ǰ�[m*topk��k]�Ų��ģ���
#                B����ͬmask_api����A������ȫ�����ݶ��������ģ�û�����û�����ݣ�����Ҫmasked_mȥ�������ݡ�
#                ����Ϊ��Ҫ֪��A����ÿ��Ӧ��Ҫ��Ӧ�ĸ��飬��������m_indices��Ϊ����������Ϊ[m*topk]������һ��Ԫ�ر�ʾA�����������Ӧ��Ӧ�ĸ����顣
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
