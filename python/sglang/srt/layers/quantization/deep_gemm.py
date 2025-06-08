import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm.contrib.concurrent import thread_map

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var, get_device_sm, get_int_env_var, is_cuda

logger = logging.getLogger(__name__)
_ENABLE_JIT_DEEPGEMM = False

try:
    import deep_gemm
    from deep_gemm import get_num_sms
    from deep_gemm.jit import build
    from deep_gemm.jit.compiler import get_nvcc_compiler
    from deep_gemm.jit_kernels.gemm import get_best_configs
    from deep_gemm.jit_kernels.runtime import FP8GemmRuntime, GemmType

    sm_version = get_device_sm()
    if sm_version == 90:
        if get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true"):
            _ENABLE_JIT_DEEPGEMM = True
except ImportError:
    logger.warning("Failed to import deepgemm, disable _ENABLE_JIT_DEEPGEMM.")


def get_enable_jit_deepgemm():
    return _ENABLE_JIT_DEEPGEMM


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = get_bool_env_var(
    "SGL_JIT_DEEPGEMM_PRECOMPILE", "true"
)
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = get_bool_env_var("SGL_IS_FIRST_RANK_ON_NODE", "true")
_COMPILE_WORKERS = get_int_env_var("SGL_JIT_DEEPGEMM_COMPILE_WORKERS", 4)
_IN_PRECOMPILE_STAGE = get_bool_env_var("SGL_IN_DEEPGEMM_PRECOMPILE_STAGE", "false")

# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGL_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
_USE_NVRTC_DEFAULT = "0"
if _ENABLE_JIT_DEEPGEMM:
    try:
        get_nvcc_compiler()
    except:
        logger.warning(
            "NVCC Compiler not found, use NVRTC for DeepGEMM JIT "
            "and may have performance loss with some cases."
        )
        _USE_NVRTC_DEFAULT = "1"
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", _USE_NVRTC_DEFAULT)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    # Generate m_max
    m_max = 1024 * 16
    if server_args.chunked_prefill_size < 1:
        m_max = 1024 * 64
    elif server_args.chunked_prefill_size > 8192:
        m_max = server_args.chunked_prefill_size * 2
    m_max = min(1024 * 128, m_max)
    _BUILTIN_M_LIST = list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = ServerArgs.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE or not _IN_PRECOMPILE_STAGE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()


@dataclass
class DeepGemmKernelHelper:
    name: str
    compile_func: Callable[
        [
            int,
            int,
            int,
            Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
        ],
        None,
    ]
    configure_func: Callable[
        [int, int, int, int, int],
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
    ]


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


def _compile_warning_1():
    if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
        logger.warning(
            "Entering DeepGEMM JIT Pre-Compile session. "
            "And it may takes a long time(Typically 10-20 mins) "
            "if you have not run `sglang.compile_deep_gemm`. "
            "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
            " for pre-compilation to reduce the overhead if you have not run it before. "
            "For example: "
            "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
        )


def _compile_warning_2():
    logger.warning(
        "Entering DeepGEMM JIT Single Kernel Compile session. "
        "And it will makes inference throughput becomes flaky. "
        "Please run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
        " for pre-compilation to solve this issue. "
        "For example: "
        "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
    )


def _compile_grouped_gemm_nt_f8f8bf16_masked_one(
    n: int,
    k: int,
    num_groups: int,
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    kwargs = {
        "GEMM_TYPE": GemmType.GroupedMasked,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("m_grouped_gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


def _compile_grouped_gemm_nt_f8f8bf16_contig_one(
    n: int,
    k: int,
    num_groups: int,
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128
    kwargs = {
        "GEMM_TYPE": GemmType.GroupedContiguous,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("m_grouped_gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


def _compile_gemm_nt_f8f8bf16_one(
    n: int,
    k: int,
    _: int,  # _ is a dummy parameter to align with other interfaces
    config: Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]],
) -> None:
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = config
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128
    kwargs = {
        "GEMM_TYPE": GemmType.Normal,
        "NUM_TMA_THREADS": num_tma_threads,
        "NUM_MATH_THREADS_PER_GROUP": num_math_threads_per_group,
        "N": n,
        "K": k,
        "NUM_GROUPS": 1,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SWIZZLE_D_MODE": smem_config[1],
        "BLOCK_N_PADDING": smem_config[2],
        "NUM_STAGES": num_stages,
        "NUM_TMA_MULTICAST": tma_multicast_config[0],
        "IS_TMA_MULTICAST_ON_A": tma_multicast_config[1],
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config[0],
    }

    code = FP8GemmRuntime.generate(kwargs)
    _ = build("gemm_fp8_fp8_bf16_nt", code, FP8GemmRuntime, kwargs)


_KERNEL_HELPER_DICT: Dict[DeepGemmKernelType, DeepGemmKernelHelper] = {
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: DeepGemmKernelHelper(
        name="m_grouped_gemm_fp8_fp8_bf16_nt_masked",
        compile_func=_compile_grouped_gemm_nt_f8f8bf16_masked_one,
        configure_func=lambda m, n, k, num_groups, num_sms: get_best_configs(
            m, n, k, num_groups, num_sms, is_grouped_masked=True
        ),
    ),
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: DeepGemmKernelHelper(
        name="m_grouped_gemm_fp8_fp8_bf16_nt_contiguous",
        compile_func=_compile_grouped_gemm_nt_f8f8bf16_contig_one,
        configure_func=lambda m, n, k, _, num_sms: get_best_configs(
            m, n, k, 1, num_sms, is_grouped_contiguous=True
        ),
    ),
    DeepGemmKernelType.GEMM_NT_F8F8BF16: DeepGemmKernelHelper(
        name="gemm_fp8_fp8_bf16_nt",
        compile_func=_compile_gemm_nt_f8f8bf16_one,
        configure_func=lambda m, n, k, _, num_sms: get_best_configs(
            m, n, k, 1, num_sms
        ),
    ),
}


def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: Optional[List[int]] = None,
) -> None:

    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        kernel_helper = _KERNEL_HELPER_DICT[kernel_type]
        _compile_warning_1()
        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_helper.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        # NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
        num_sms = get_num_sms()
        collected_configs = set()
        for m in m_list if m_list is not None else _BUILTIN_M_LIST:
            # Put config into set to get unique configs and reduce cases to be compiled
            collected_configs.add(
                kernel_helper.configure_func(m, n, k, num_groups, num_sms)
            )
        compile_func = lambda config: kernel_helper.compile_func(
            n, k, num_groups, config
        )
        thread_map(compile_func, collected_configs, max_workers=_COMPILE_WORKERS)

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
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)

    with _log_jit_build(expected_m, n, k, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            lhs, rhs, out, masked_m, expected_m
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)

    with _log_jit_build(m, n, k, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GEMM_NT_F8F8BF16
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, 1)

    with _log_jit_build(m, n, k, kernel_type):
        deep_gemm.gemm_fp8_fp8_bf16_nt(lhs, rhs, out)


@contextmanager
def _log_jit_build(M: int, N: int, K: int, kernel_type: DeepGemmKernelType):
    if _IN_PRECOMPILE_STAGE:
        yield
        return

    from deep_gemm.jit.runtime import RuntimeCache

    origin_func = RuntimeCache.get

    def __patched_func(self, *args, **kwargs):
        ret = origin_func(self, *args, **kwargs)
        if ret is None:
            kernel_helper = _KERNEL_HELPER_DICT[kernel_type]
            _compile_warning_2()
            logger.warning(
                f"DeepGEMM JIT Compiling for <{kernel_helper.name}> M={M}, N={N}, K={K}. Please wait."
            )
        return ret

    RuntimeCache.get = __patched_func
    yield
    RuntimeCache.get = origin_func


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
