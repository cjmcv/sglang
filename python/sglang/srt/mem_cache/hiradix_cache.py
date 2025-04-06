import heapq
import logging
import threading
import time
from typing import List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    MLATokenToKVPool,
    MLATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import _key_match_page_size1 as _key_match

logger = logging.getLogger(__name__)

# <NT> HiRadixCache 分层级offload的RadixCache
# 1. 取出的self.kv_cache默认会是显存的，根据其类型分配与之对应的host端的kvcache，内存大小为device_pool.size * host_to_device_ratio。
# 2. 新增一个HiCacheController，同时输入gpu端和host端的token_to_kv_pool（kvcache），用于二者的数据交互。
class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
    ):
        if page_size != 1:
            raise ValueError(
                "Page size larger than 1 is not yet supported in HiRadixCache."
            )
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache, hicache_ratio
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache, hicache_ratio
            )
        else:
            raise ValueError(f"Only MHA and MLA supports swap kv_cache to host.")

        self.tp_group = tp_cache_group
        self.page_size = page_size

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            load_cache_event=self.load_cache_event,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = 1
        self.load_back_threshold = 10
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, self.page_size, disable=False
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()

    # <NT> 获取该子节点的高度，从子节点出发往根节点方向走，每走一步高度加1.
    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    # <NT> 将要写的数据通过cache_controller提交到队列中，返回host内存的索引，并标记到node.host_value中。
    # 如果没有返回索引，则表示host的内存池里没有空槽位，需要先清除掉一些数据挪腾位置出来，再进行上面的操作。
    # 因为是异步拷贝，所以使用ongoing_write_through去记录正在拷贝的节点，拷贝结束再清除。
    def write_backup(self, node: TreeNode):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            self.inc_lock_ref(node)
        else:
            return None

        return len(host_indices)

    # <NT> 在_insert_helper中调用，自增该节点的命中次数，当达到阈值并且host_value为空时，将数据从显存拷贝到内存。
    def inc_hit_count(self, node: TreeNode):
        if self.cache_controller.write_policy != "write_through_selective":
            return
        node.hit_count += 1
        if node.host_value is None and node.hit_count > self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    # <NT> 检查清理 write_backup 中启动的已完成的写回操作。
    # 1）基于ack_write_queue的qsize构建一个tensor (queue_size)，其元素只有一个值，表示队列大小。
    # 2）调用all_reduce，对queue_size执行 ReduceOp.MIN 的 allreduce操作，
    #    取所有TP节点中队列大小的最小值作为统一的队列大小。这样可以保证在更新基数缓存（radix cache）时，所有进程的操作是同步的。
    # 3）按统一的队列大小去逐个 dec_lock_ref 和 del self.ongoing_write_through。
    def writing_check(self):
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.dec_lock_ref(self.ongoing_write_through[ack_id])
            del self.ongoing_write_through[ack_id]

    # <NT> 对已完成的加载操作进行清理，包括复位node的.loading变量，删除 ongoing_load_back 元素。
    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                start_node, end_node = self.ongoing_load_back[ack_id]
                self.dec_lock_ref(end_node)
                while end_node != start_node:
                    assert end_node.loading
                    end_node.loading = False
                    end_node = end_node.parent
                # clear the reference
                del self.ongoing_load_back[ack_id]
            except Exception:
                break

    def evictable_size(self):
        return self.evictable_size_

    # <NT> 大体与 RadixCache.evict 一样，先收集树的叶子节点构建小顶堆，并从小到大逐个pop出，进行检索。
    # 主要差别在 if x.host_value 部分，这里的 write_policy 目前版本默认是 "write_through_selective"，
    # 如果 x 的数据没有被写入到 host 端，则直接释放显存并删除节点。
    # 如果 x 的数据有被写入到 host 端，则删除device数据，保留host端备份。
    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        pending_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if x.host_value is None:
                if self.cache_controller.write_policy == "write_back":
                    num_evicted += self.write_backup(x)
                elif self.cache_controller.write_policy == "write_through_selective":
                    num_evicted += self._evict_write_through_selective(x)
                else:
                    assert (
                        self.cache_controller.write_policy != "write_through"
                    ), "write_through should be inclusive"
                    raise NotImplementedError
            else:
                num_evicted += self._evict_write_through(x)

            for child in x.parent.children.values():
                if child in pending_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                self.writing_check()
                time.sleep(0.1)

    # <NT> 释放有host端数据的节点，调用 evict_device 对节点的 device侧数据做释放。
    # 释放后 node.value = None，节点仍存在于树里，后续访问到这种节点，会得到 node.evicted 为 True。
    def _evict_write_through(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    # <NT> 释放没有host端数据的节点
    def _evict_write_through_selective(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    # <NT> 在 write_backup 中调用，释放host端的空槽位。只释放evicted的节点。
    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue
            assert x.lock_ref == 0 and x.host_value is not None

            assert self.cache_controller.evict_host(x.host_value) > 0
            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    # <NT> write_backup相对应，从host端加载数据到device端。
    # write_backup是单节点写，load_back是以last_node为起点，往parent方向的整条前缀路径的所有节点的加载。
    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # <NT> nodes_to_load是所有需要加载的节点，得到的host_indices则包含有多有节点的indices.
        # 分层加载是基于 host_indices 切片后进行的。
        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    # <NT> 从host端把数据加载回device端。开头使用if last_node.evicted过滤，
    # 因为该功能只适用于host端有备份而device数据已删除的情况
    def init_load_back(
        self,
        last_node: TreeNode,
        prefix_indices: torch.Tensor,
        mem_quota: Optional[int] = None,
    ):
        assert (
            len(prefix_indices) == 0 or prefix_indices.is_cuda
        ), "indices of device kV caches should be on GPU"
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                prefix_indices = ( 
                    loading_values
                    if len(prefix_indices) == 0
                    else torch.cat([prefix_indices, loading_values])
                )
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )

            while last_node.evicted:
                last_node = last_node.parent

        return last_node, prefix_indices

    def read_to_load_cache(self):
        self.load_cache_event.set()

    def match_prefix(self, key: List[int], include_evicted=False, **kwargs):
        if self.disable:
            return [], self.root_node

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.tensor([], dtype=torch.int64)

        last_node_global = last_node
        while last_node.evicted:
            last_node = last_node.parent

        if include_evicted:
            return value, last_node, last_node_global
        else:
            return value, last_node

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.time()
        value = []
        while len(key) > 0 and key[0] in node.children.keys():
            child = node.children[key[0]]
            child.last_access_time = time.time()
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]
        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.host_value is not None:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    # <NT> 节点插入的关键函数，可以先看RadixCache的_insert_helper注释。
    # 先查看待插入的token能否找到直接对应上的节点(pagesize为1，节点以key[0]作为词典的key)
    # 如果有:
    #   看前缀长度是否等于要插入的token。
    #   如果等于，即是能全匹配该节点：
    #     查看节点是否为evicted，
    #     如果是，表示该节点的device数据已经被同步到了host端。需要重新填充device数据，并更新host对应位置的状态。
    #     否则执行inc_hit_count表示该节点被命中次数+1，达到阈值后会被备份到host。
    #     并继续把待插入token剩下部分递归执行插入操作。
    #   否则是部分匹配，则需要分裂节点
    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if child.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    child.value = value[:prefix_len]
                    self.evictable_size_ += len(value[:prefix_len])
                    self.token_to_kv_pool_host.update_synced(child.host_value)
                    return self._insert_helper(
                        child, key[prefix_len:], value[prefix_len:]
                    )
                else:
                    self.inc_hit_count(child)
                    return prefix_len + self._insert_helper(
                        child, key[prefix_len:], value[prefix_len:]
                    )

            # partial match, split the node
            new_node = self._split_node(child.key, child, prefix_len)
            if new_node.evicted:
                new_node.value = value[:prefix_len]
                self.token_to_kv_pool_host.update_synced(new_node.host_value)
                self.evictable_size_ += len(new_node.value)
                return self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )
            else:
                self.inc_hit_count(new_node)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)

            if self.cache_controller.write_policy == "write_through":
                self.write_backup(new_node)
        return 0

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list
