"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

# <NT> ReqToTokenPool / TokenToKVPoolAllocator / KVCache (MHA/MLA/DoubleSparse)
# kvcache的管理主要由这三个类组成，
# ReqToTokenPool管理seq的槽位，负责从seq到token位置的映射，里面存放的元素是seq所属的token在 TokenToKVPoolAllocator 的槽位的索引。
#               因为token id数值范围很广，不能直接用于内存索引，所以需要动态地将token id转为实际索引，kv_indices。
# TokenToKVPoolAllocator管理 token 的槽位，负责从 token 到实际 kvcache 内存索引的映射，根据里面存放的数据可以找到任意token对应的 KVCache 的内存块。
# KVCache管理实际的kvcache内存，通过TokenToKVPoolAllocator拿到的token对应的内存索引，可以取出其对应的kvcache数据。
# 
# 除了这三类，还有一类叫HostKVCache(MHA/MLA/DoubleSparse)，管理host端的kvcache内存，除此之外与普通的KVCache不同点还有它兼顾了TokenToKVPoolAllocator的功能，也管理了token的槽位！！关系有点乱。。
import abc
import logging
import threading
from enum import IntEnum
from functools import wraps
from typing import List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import triton
import triton.language as tl

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import (
    debug_timing,
    get_compiler_backend,
    is_cuda,
    next_power_of_2,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()


# <NT> 将一个请求 req 映射到其对应的 token 位置.
# 创建脉络：Scheduler.__init__: self.req_to_token_pool, self.token_to_kv_pool = self.tp_worker.get_memory_pool() , 从tp_worker中get出。Scheduler在构建ScheduleBatch时会传入。
#          TpModelWorker.get_memory_pool: return (self.model_runner.req_to_token_pool, self.model_runner.token_to_kv_pool) , 
#                                         即从model_runner中拿到。ModelRunner是在TpModelWorker.init里创建的。
#          ModelRunner.init_memory_pool: 进行req_to_token_pool的实际创建操作。
#                                       self.req_to_token_pool = ReqToTokenPool(
#                                             size=max_num_reqs + 1,
#                                             max_context_len=self.model_config.context_len + 4,
#                                             device=self.device,
#                                             enable_memory_saver=self.server_args.enable_memory_saver,
#                                        )
#          即实际创建和管理是在 ModelRunner 里进行的，但 Scheduler和ScheduleBatch 会对其进行相关操作。关键操作在ScheduleBatch里。
#
# 初始化函数：size：填充为max_num_reqs + 1，就是最大请求数量+1。
#            max_context_len：是模型最大能处理的token数量，除非是用户手动设置，否则填的是模型文件配套的config文件的数据。
#            enable_memory_saver：一个节省内存的python包。
#            self.req_to_token: 实际的内存池，内存分配的是(size, max_context_len)的二维tensor，类型是int32。
#            self.free_slots：按size大小创建一个free_slots列表，里面的值是从0到size-1，表示这些编号对应位置的请求req是空的，可以往里面插入新请求。
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region():
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))

    # <NT> 在ScheduleBatch在执行prepare_for_extend时调用。
    # 遍历该batch所有req，如对应req的pre_len不为0，即该req之前有计算过，有前缀。
    # 则会把这些前缀的下标都写入到self.req_to_token：self.req_to_token_pool.write( (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices)
    # indices将会是一个二维元组，行是从alloc获得的req空位的下标，列是从0到pre_len的位置上分别写入req.prefix_indices的所有token下标。
    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    # <NT> alloc在ScheduleBatch.prepare_for_extend->alloc_req_slots里调用，会根据该batch有多少个req，从而调用alloc拿到相应数量的slot空位下标，
    # 后面会往self.req_to_token的对应下标里写数据。其中need_size是req数量，如need_size为5，会将self.free_slots的前5个数据给出去并做更新。
    # 如初始状态下，前5个数据是0-4，表示这5个数据对应的下标位置是空的，可以插入新的请求。
    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter


# <NT> 将一个token映射到其kvcache数据存放的位置
# 创建过程与ReqToTokenPool一致，但是TokenToKVPool会按架构区分：
# MLATokenToKVPool / DoubleSparseTokenToKVPool / MHATokenToKVPool (除了MLA和DoubleSparse, 剩下的都是MHA(GQA的kvcache与MHA的一致))
# 命名为kvcache，size是max_total_num_tokens由剩余显存计算得出，dtype只区分auto和fp8
# 
# 与ReqToTokenPool的关系，ReqToTokenPool可由req拿到所属token在kvcache上的存放位置，即kvcache ids.
# 通过这个与token一一对应的kvcache id去BaseTokenToKVPool里定位到该token具体的kvcache存放位置。
#
# 这里的TokenToKVPoolAllocator是对kvcache做了进一步的封装，负责token槽位的管理，实际内存由KVCache管理。
class TokenToKVPoolAllocator:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        # <NT> 将free_slots初始化为从1-size+1的数据，用来表示token的空闲槽位的下标。
        self.clear()

        self._kvcache = kvcache

    def available_size(self):
        return len(self.free_slots)

    def get_kvcache(self):
        return self._kvcache

    # <NT> 在ScheduleBatch.prepare_for_extend中调用 out_cache_loc = self.alloc_token_slots(extend_num_tokens)，
    #      need_size=extend_num_tokens，对应的是新构建的batch中当前当前需要做extend计算的token数,
    #      按token数申请空槽位，返回的是充当下标的tensor，在set_kv_buffer函数中作为loc参数输入，
    #      里面每个元素表示的是token的kvcache存放的下标，对应该轮ScheduleBatch计算的输出结果。
    #      普通mha的kvcache维度是[layer_id][token][head_num*head_dim]，这里返回的是空闲的kvcache id集合，与token一一对应。
    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def backup_state(self):
        return self.free_slots

    def restore_state(self, free_slots):
        self.free_slots = free_slots

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []

# <NT> MHATokenToKVPool 
# 创建例子，在ModelRunner中：
# self.token_to_kv_pool = MHATokenToKVPool(
#     self.max_total_num_tokens,    # 等于 self.profile_max_num_token(total_gpu_memory)，从(load model后)剩余gpu内存中按架构进行估算，主要分mla和非mla。
#     dtype=self.kv_cache_dtype,    # 分auto / fp8_e5m2 和 fp8_e4m3
#     head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),  # 总的kv_head数量 除以 张量并行节点数
#     head_dim=self.model_config.head_dim,                                   # head_dim, kv_head和q_head的head dim都是一样的
#     layer_num=self.model_config.num_hidden_layers,   # mha层数量，每个层的kvcache是独立的。进一步以qwen2为例，搜class Qwen2Model(nn.Module)阅读
#     device=self.device,
#     enable_memory_saver=self.server_args.enable_memory_saver,
# )
class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self._create_buffers()

        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    # <NT-TODO> 每个层一份，每份按最大token数分配，每个token是head_num*head_dim的矩阵，k和v独立。
    # 即cache在底层是一大块连续空间，使用radix cache是对这一大块空间构建额外的索引组成树状。
    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr() for i in range(self.layer_num)
        ] + [self.get_value_buffer(i).data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes for i in range(self.layer_num)
        ] + [self.get_value_buffer(i).nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        self.k_buffer[layer_id - self.start_layer][indices] = k_data
        self.v_buffer[layer_id - self.start_layer][indices] = v_data

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    # <NT> loc是从 TokenToKVPoolAllocator.alloc 中申请得到的与token一一对应的kvcache id集合，对应一个batch计算的输出token。
    # 布局：[layer_id][token][head_num*head_dim]
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v


@torch.compile
def fused_downcast(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dtype: torch.dtype,
    store_dtype: torch.dtype,
    max_fp8: float,
    min_fp8: float,
):
    cache_k = cache_k / k_scale
    cache_k = torch.clamp(cache_k, min_fp8, max_fp8)
    cache_v = cache_v / v_scale
    cache_v = torch.clamp(cache_v, min_fp8, max_fp8)
    cache_k = cache_k.to(dtype)
    cache_v = cache_v.to(dtype)
    cache_k = cache_k.view(store_dtype)
    cache_v = cache_v.view(store_dtype)
    return cache_k, cache_v


# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True, backend=get_compiler_backend())
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        with self.memory_saver_adapter.region():
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # <NT> k和v共存，布局同样也是[layer_id][token][head_num*head_dim]，
            # 只是head_num=1, head_dim=kv_lora_rank + qk_rope_head_dim
            # 与mha的kv分开管理不同，这里kv在同一个kv_buffer上。
            # 其中
            self.kv_buffer = [
                torch.zeros(
                    (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]

        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # <NT> shape是(size + 1, 1, kv_lora_rank + qk_rope_head_dim)
    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer]

    # <NT-TODO> [..., : self.kv_lora_rank]是取最后一个维度的前kv_lora_rank的数据，
    # 即shape是(size + 1, 1, self.kv_lora_rank)， 为什么是这个维度？
    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    # <NT> 在deepseek v2中cache_k对应的是latent_cache
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
        )

    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        for i in range(self.layer_num):
            self.kv_buffer[i][indices] = flat_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        self.kv_buffer[layer_id - self.start_layer][indices] = flat_data


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label

    def get_flat_data(self, indices):
        pass

    def transfer(self, indices, flat_data):
        pass

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pass

# <NT-TODO> PROTECTED: 
#      SYNCED: 从host到device已经同步好
#      BACKUP：从device到host端已经备份好
class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


def synchronized(debug_only=False):
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (not debug_only) or self.debug:
                return func(self, *args, **kwargs)
                with self.lock:
                    return func(self, *args, **kwargs)
            else:
                return True

        return wrapper

    return _decorator


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool,
        device: str,
        page_size: int,
    ):
        self.device_pool = device_pool
        self.dtype = device_pool.store_dtype
        self.pin_memory = pin_memory
        self.device = device
        self.page_size = page_size
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the host memory pool size to the page size
        self.size = self.size - (self.size % self.page_size)
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        if requested_bytes > host_mem.available - ten_gb:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{host_mem.available / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data_by_layer(self, indices, layer_id):
        raise NotImplementedError()

    @abc.abstractmethod
    def assign_flat_data(self, indices, flat_data):
        raise NotImplementedError()

    @synchronized()
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        if self.debug:
            self.mem_state[select_index] = MemoryStateInt.RESERVED

        return select_index

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        if self.debug:
            self.mem_state[indices] = MemoryStateInt.IDLE
        return len(indices)

    @synchronized(debug_only=True)
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized(debug_only=True)
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized(debug_only=True)
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_backup(self, indices: torch.Tensor):
        if not self.is_synced(indices):
            raise ValueError(
                f"The host memory slots should be in SYNCED state before turning into BACKUP. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def protect_write(self, indices: torch.Tensor):
        if not self.is_reserved(indices):
            raise ValueError(
                f"The host memory slots should be RESERVED before write operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def protect_load(self, indices: torch.Tensor):
        if not self.is_backup(indices):
            raise ValueError(
                f"The host memory slots should be in BACKUP state before load operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def complete_io(self, indices: torch.Tensor):
        if not self.is_protected(indices):
            raise ValueError(
                f"The host memory slots should be PROTECTED during I/O operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.SYNCED

# <NT> Host版本，需要输入device_pool，根据host_to_device_ratio来申请host端内存。
# 内存大小为device_pool.size * host_to_device_ratio。
class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool, host_to_device_ratio, host_size, pin_memory, device, page_size
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        return torch.empty(
            (2, self.layer_num, self.size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, :, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, :, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[:, layer_id - self.start_layer, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, :, indices] = flat_data

    def write_page_all_layers(self, host_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[0, j, h_index : h_index + self.page_size].copy_(
                    device_pool.k_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )
                self.kv_buffer[1, j, h_index : h_index + self.page_size].copy_(
                    device_pool.v_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, host_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.k_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    0, layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )
            device_pool.v_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    1, layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool, host_to_device_ratio, host_size, pin_memory, device, page_size
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )

    def init_kv_buffer(self):
        return torch.empty(
            (
                self.layer_num,
                self.size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[layer_id - self.start_layer, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, indices] = flat_data

    def write_page_all_layers(self, host_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[j, h_index : h_index + self.page_size].copy_(
                    device_pool.kv_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, host_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.kv_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )
