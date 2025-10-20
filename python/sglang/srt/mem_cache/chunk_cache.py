from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


# <NT> ChunkCache 基于ReqToTokenPool和TokenToKVPoolAllocator进行构建，用于管理KVCache。
class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
        )

    # <NT> req结束时调用，入参通常只填req，则token_id_len将会是prompts+decode的所有token的长度。
    # 对应直接按req取出其对应的kv_indices，对二级cache都执行内存释放操作。
    # 问题: 为什么token_to_kv_pool也要释放？如果其他seq有相同的token，则无法共享。
    # 答：如果seq结束时不释放其所有token的kvcache，需要约定什么时候才能释放，长期不释放，内存会爆。
    #     而约定什么时候释放的问题，可以由radix cache来解决，即构建基数树，用LRU（最近最少使用，基于计时器）释放。
    def cache_finished_req(self, req: Req, is_insert: bool = True):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        page_size: int,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
