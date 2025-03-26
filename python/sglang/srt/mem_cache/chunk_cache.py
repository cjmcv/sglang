from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid, value):
        self.rid = rid
        self.value = value

# <NT> ChunkCache 基于ReqToTokenPool和BaseTokenToKVPool进行构建，用于管理前缀。
class ChunkCache(BasePrefixCache):
    def __init__(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        self.disable = True
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.reset()

    def reset(self):
        self.entries = {}

    # <NT> rid是req_id, key是最长前缀token id集 (围绕prompts+decode到当前阶段为止，已有的token id合集)
    # entries是围绕req_id进行的，一个rid对应一个entry，entry.value维系着一个req的所用token的kvcache映射id号，从req_to_token中取出。
    # 注1: key是token id是针对分词器的，每个token（由token id标记）在使用了token_to_kv_pool的空槽后，会对应有一个kvcache id号，表示该token在kvcache的存放位置。
    # 而entry.value对应的就是这个kvcache的位置，通过req_to_token_pool得到。
    # 注2：chunk cache中以rid为单位进行数据维护，可能并不需要输入key，只是为了与radix cache保持接口一致。
    #      radix cache中需要围绕的key去做索引，可以进一步找到kvcache id。
    def match_prefix(self, rid: int, key: List[int]) -> Tuple[List[int], int]:
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    # <NT> req结束时调用，入参通常只填req，则token_id_len将会是prompts+decode的所有token的长度。
    # 对应直接按req取出其对应的kv_indices，对二级cache都执行内存释放操作。
    # 问题: 为什么token_to_kv_pool也要释放？如果其他seq有相同的token，则无法共享。
    # 答：如果seq结束时不释放其所有token的kvcache，需要约定什么时候才能释放，长期不释放，内存会爆。
    #     而约定什么时候释放的问题，可以由radix cache来解决，即构建基数树，用LRU（最近最少使用，基于计时器）释放。
    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool.free(kv_indices)

        if req.rid in self.entries:
            del self.entries[req.rid]

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.fill_ids)
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = kv_indices
        req.last_node = entry

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    def inc_lock_ref(self, node):
        return 0

    def dec_lock_ref(self, node):
        return 0

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0
