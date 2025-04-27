from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid: str, value: torch.Tensor):
        self.rid = rid
        self.value = value

# <NT> ChunkCache 基于ReqToTokenPool和TokenToKVPoolAllocator进行构建，用于管理KVCache。
class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def reset(self):
        pass

	# <NT> 不具备跨seq共享前缀的作用
    def match_prefix(self, **unused_kwargs) -> Tuple[List[int], int]:
        return [], None
        
    # <NT> req结束时调用，入参通常只填req，则token_id_len将会是prompts+decode的所有token的长度。
    # 对应直接按req取出其对应的kv_indices，对二级cache都执行内存释放操作。
    # 问题: 为什么token_to_kv_pool也要释放？如果其他seq有相同的token，则无法共享。
    # 答：如果seq结束时不释放其所有token的kvcache，需要约定什么时候才能释放，长期不释放，内存会爆。
    #     而约定什么时候释放的问题，可以由radix cache来解决，即构建基数树，用LRU（最近最少使用，基于计时器）释放。
    def cache_finished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any):
        return 0

    def pretty_print(self):
        return ""
