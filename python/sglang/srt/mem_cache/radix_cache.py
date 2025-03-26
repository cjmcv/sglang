from __future__ import annotations

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

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None                     # token id
        self.value = None                   # token在kvcache的存放位置，kv_indices。
        self.lock_ref = 0                   # 引用计数，表示有多少个正在计算的seq用着它。
        self.last_access_time = time.time() # 时间标记，用于LRU淘汰策略

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None
	
	# <NT> 魔术方法/双下方法, 表示小于。
    # 当使用 < 运算符比较两个对象时，Python 会自动调用对象的__lt__方法来确定它们的大小关系。
    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time

# <NT> 找出 key0 和 key1 两个列表中从 开头开始 连续相同元素的数量。
def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

# <NT> RadixCache 基于ReqToTokenPool和BaseTokenToKVPool额外构建的索引
class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    # <NT> 外部调用主接口，输入要查找的key(token id), 返回能在cache中找到的kv_indices(token在kvcache的存放位置)。
    # last_node是匹配前缀命中的最后一个节点。给出去主要是提供一个快速检索的入口，当要再次检索节点时，不需要重新匹配。
    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return value, last_node[0]

    # <NT> 用于插入key的主入口，实际使用时value将会是从req_to_token_pool中取出的kv_indices, 
    # 表示token在kvcache中的存放位置。
    def insert(self, key: List, value=None):
        if self.disable:
            return 0
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    # <NT> 当seq结束时调用。
    # if self.disable，即禁用RadixCache，里面的内容与ChunkCache的cache_finished_req函数内容基本一致，直接释放两个pool的相关内容。
    # 如未被disable，则正常使用RadixCache：
    #     一般调用token_ids会为空，所以算出token_ids会是prompts和decode所有结果的合集。
    #     按token_ids的长度从req_to_token_pool里取出该seq所有token的kvcache索引。同时执行插入操作，将该seq的所有token_ids插入到
    # 基数树中，即把新计算得到的未被插入树中的部分添加进去。new_prefix_len一般会是token_ids的长度。与ChunkCache相比，
    # 主要少了token_to_kv_pool的整体释放，而req_to_token_pool还是要正常释放的。
    # 疑问点：为什么要执行这部分的释放，self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])
    #        在cache_unfinished_req中也有同样的执行情况。
    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if self.disable:
            if token_ids is None:
                token_ids_len = len(req.origin_input_ids) + len(req.output_ids) - 1
            else:
                token_ids_len = len(token_ids)

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :token_ids_len
            ]
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids)
        assert len(new_indices) == len(token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    # <NT> 驱逐：根据num_tokens的数量，从树中驱逐相应数量的token（一个节点会有一个或多个token），为准备加入的新token腾位置。
    # 首先收集树的叶子节点(没有children的节点)，构建小顶堆，因为leaves是TreeNode集合，所以通过其方法__lt__确定大小。
    # 看__lt__的定义可知，堆顶的是last_access_time最小者，即最长时间未被访问的。并从小到大逐个pop出，进行检索。
    # 如果pop出的叶子节点目前未被使用(x.lock_ref > 0), 则对其做释放处理，并累计释放的节点中，包含的token数量(num_evicted).
    # 使驱逐的token数量达到num_tokens。
    #
    # heapq.heapify 函数能够把一个列表转化为小顶堆结构。
    # heapq.heappop 返回堆顶的最小值，并维持小顶堆性质。
    # heapq.heappush 插入一个元素到堆里，并维持小顶堆性质。
    # evict_callback 会调用self.token_to_kv_pool.free，完成kvcache的内存释放。 
    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    # <NT> 增加计数，表示正在被使用。
    # 基于子节点req.last_node，往根节点方向遍历，即会遍历所有前缀节点，所有前缀节点的计数lock_ref都会加1。
    # 如果遍历到某个节点，其本身被标记为lock_ref==0，表示在这之前未被seq所使用，处于可驱逐状态。
    # 而现在需要将其从可驱逐状态转为保护状态，所以evictable_size_会减少，对应protected_size_增加。
    # delta是evictable_size_的差额。
    # 注：req.last_node一般从match_prefix中得到。
    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    # <NT> 与inc_lock_ref想对应。
    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    # <NT> 树里可被驱逐淘汰的token有多少
    def evictable_size(self):
        return self.evictable_size_

    # <NT> 树里不可被驱逐淘汰的token有多少，即正在被使用着的部分
    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    ##### Internal Helper Functions #####
    # <NT> 递归匹配前缀，返回匹配上前缀的部分token的kv_indices，同时返回被匹配上的最后一个节点。
    # 每个节点被访问时，都会更新时间，太长时间未被访问的会淘汰掉。
    # 每个节点的key是一个列表，包含有一个token id集合，并以集合的首个token id作为词典检索的key。
    # 先查找待匹配key的首个token是否存在于该节点为基础的子节点中，如不存在可直接结束。
    # 如存在，进一步看该子节点的key与待匹配的key，从头数，有多长是一致的，即能匹配上的。
    #     如果能匹配上的部分比子节点的key要少，
    #         说明该子节点需要按匹配上的部分进行分裂，把子节点未匹配的部分key分离出来。
    #         new_node.key存放的是匹配上的部分。new_node充当last_node，后面的都不需要匹配了，都属于不命中的部分。
    #     否则：
    #         说明该子节点能全命中，需要进一步检索该子节点的子节点，直到待匹配的key全部匹配完，或出现到某个子节点只能匹配部分的情况（就是上面的情况）。
    # （如有疑惑，可结合_insert_helper一起看）
    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    # <NT> 调用 new_node = self._split_node(child.key, child, prefix_len)
    # 如child节点有key：abcd，需要匹配的key为abef，有前缀ab，未完全覆盖节点上的key，此时需要对节点进行分裂。
    # 函数参数：key为abcd，child是节点本身，split_len将会是前缀2
    # new_node.key: ab, 只留存前缀匹配部分。value也一样只留存前缀匹配的部分。key是token id，而value是token在kvcache的存放位置，二者一一对应，所以也可以用相同的长度进行截取。
    # new_node.children: 子节点是原分裂前的节点child，其词典key被设为c，即非匹配部分的首个key值，child.key自然就是cd了。
    # child[abcd]  分裂 ==>> 父 new_node[ab] -> 子 chile[cd]  返回父节点
    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    # <NT> 外层通过insert函数进入，输入self.root_node，key是token ids，value与key的值一致。
    # 随后该函数会递归调用自身，直到完成插入。
    # 首先判断该节点的众多子节点中，是否有包含key[0]
    #    如有：
    #         取出key[0]所在的子节点child，以child为起始点，通过_key_match计算出于key相重叠的部分有多长 prefix_len
    #         如果prefix_len长度等于child节点上的key的总长度：
    #             表示二者的前缀涵盖了子节点child上所有的key。
    #             如果prefix_len长度等于正匹配的key的总长度：
    #                 表示二者的前缀同样也涵盖了插入的key，即要插入的key已完全被加入缓存了，不再需要任何其他操作
    #             否则：
    #                 表示要插入的key有一部分没有被缓存。所以从key中剔除掉已被缓存的头prefix_len个token，
    #                 取出未被缓存的部分key和value，基于child节点为基础上进入插入操作。
    #         否则：
    #             表示二者的前缀也未涵盖子节点child上的所有key，则需要对child子节点做分裂操作。
    #             如child节点有key为abcd，需要匹配的key为abef，有前缀ab，未完全覆盖节点上的key，此时需要对节点进行分裂。
    #             即有 child[abcd] <=_split_node=> 父 new_node[ab] -> 子 chile[cd]
    #             随后在 new_node[ab] 下继续往下递归插入未匹配部分的ef。
    #    (上面"如有"内部各分支都有return，不会继续往下走)
    #    如无：
    #         新增一节点存放需要匹配的key，并充当当前节点的子节点。
    #         key和value都是list，节点的parent只会有一个，所以直接指向node；
    #         而node的children因为会有很多个，按key列表中的首个token充当代表，作为children的索引。
    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)
                
            new_node = self._split_node(child.key, child, prefix_len)
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
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        if node.evicted:
            return 0
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    # <NT> 收集叶子节点，即没有children的节点。
    # children类型是defaultdict(TreeNode)，values就是TreeNode合集。
    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
