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
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:

    counter = 0

    # <NT> TreeNode.key  : token id
    #      TreeNode.value: token��kvcache�Ĵ��λ�ã�kv_indices
    #      TreeNode.lock_ref        : ���ü�������ʾ�ж��ٸ����ڼ����req��������
    #      TreeNode.last_access_time: ʱ���ǣ�����LRU��̭����
    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

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
	
	# <NT> ħ������/˫�·���, ��ʾС�ڡ�
    # ��ʹ�� < ������Ƚ���������ʱ��Python ���Զ����ö����__lt__������ȷ�����ǵĴ�С��ϵ��
    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time

# <NT> ���page_sizeΪ1��������ϰ汾ֻ֧��page_sizeΪ1.
#      �ҳ� key0 �� key1 �����б��д� ��ͷ��ʼ ������ͬԪ�ص�������
def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

# <NT> ���page_size��Ϊ1���������_key_match_page_size1���ƣ�
#      ��page_size��tokenΪһ��page������page��Ƚϣ�����Ҫpage����ÿ��token��һ�²���������page���.
#      ��page_size=2�����б�abcd��abce����ƥ���ϵ�token��ab�����᷵��2.
#      ��page_size=1, ƥ���ϵ�token����abc�����᷵��3.
def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


# <NT> RadixCache ����ReqToTokenPool��BaseTokenToKVPool���⹹��������
# page_size: ��Ϊ2����ʾ����token����һ����Ϊһ��page��ƥ��ǰ׺ʱ��pageΪ��λ����ƥ�䡣page��������token��һ�£�������page��һ�¡�
class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    # <NT> �ⲿ�������ӿڣ�����Ҫ���ҵ�key(token id), ��������cache���ҵ���kv_indices(token��kvcache�Ĵ��λ��)��
    # last_node��ƥ��ǰ׺���е����һ���ڵ㡣����ȥ��Ҫ���ṩһ�����ټ�������ڣ���Ҫ�ٴμ����ڵ�ʱ������Ҫ����ƥ�䡣
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
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, last_node

    # <NT> ���ڲ���key������ڣ�ʵ��ʹ��ʱvalue�����Ǵ�req_to_token_pool��ȡ����kv_indices, 
    # ��ʾtoken��kvcache�еĴ��λ�á�
    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    # <NT> ��req����ʱ���á�
    # if self.disable��������RadixCache�������������ChunkCache��cache_finished_req�������ݻ���һ�£�ֱ���ͷ�����pool��������ݡ�
    # ��δ��disable��������ʹ��RadixCache��
    #     token_ids����prompts��decode���н���ĺϼ�����token_ids�ĳ��ȴ�req_to_token_pool��ȡ����seq����token��kvcache������
    # ͬʱִ�в������������req���¼���õ���δ���������еĲ�����ӽ�ȥ��
    # new_prefix_lenһ�����token_ids�ĳ��ȡ���ChunkCache��ȣ���Ҫ����token_to_kv_pool�������ͷţ���req_to_token_pool����Ҫ�����ͷŵġ�
    # ���ʵ㣺ΪʲôҪִ���ⲿ�ֵ��ͷţ�self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])
    #        ��cache_unfinished_req��Ҳ��ͬ����ִ�������
    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len], page_aligned_kv_indices
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    # <NT> ��reqδ����ʱ���á�
    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(page_aligned_token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    # <NT> ���𣺸���num_tokens��������������������Ӧ������token��һ���ڵ����һ������token����Ϊ׼���������token��λ�á�
    # �����ռ�����Ҷ�ӽڵ�(û��children�Ľڵ�)������С���ѣ���Ϊleaves��TreeNode���ϣ�����ͨ���䷽��__lt__ȷ����С��
    # ��__lt__�Ķ����֪���Ѷ�����last_access_time��С�ߣ����ʱ��δ�����ʵġ�����С�������pop�������м�����
    # ���pop����Ҷ�ӽڵ�Ŀǰδ��ʹ��(x.lock_ref > 0), ��������ͷŴ������ۼ��ͷŵĽڵ��У�������token����(num_evicted).
    # ʹ�����token�����ﵽnum_tokens��
    #
    # heapq.heapify �����ܹ���һ���б�ת��ΪС���ѽṹ��
    # heapq.heappop ���ضѶ�����Сֵ����ά��С�������ʡ�
    # heapq.heappush ����һ��Ԫ�ص������ά��С�������ʡ�
    def evict(self, num_tokens: int):
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

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    # <NT> ���Ӽ���������1��ʾ���ڱ�ʹ�á�
    # �����ӽڵ�req.last_node�������ڵ㷽������������������ǰ׺�ڵ㣬����ǰ׺�ڵ�ļ���lock_ref�����1��
    # ���������ĳ���ڵ㣬�䱾�����Ϊlock_ref==0����ʾ����֮ǰδ��seq��ʹ�ã����ڿ�����״̬��
    # ��������Ҫ����ӿ�����״̬תΪ����״̬������evictable_size_����٣���Ӧprotected_size_���ӡ�
    # delta��evictable_size_�Ĳ�
    # ע��req.last_nodeһ���match_prefix�еõ���
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

    # <NT> ��inc_lock_ref���Ӧ��
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

    # <NT> ����ɱ�������̭��token�ж���
    def evictable_size(self):
        return self.evictable_size_

    # <NT> ���ﲻ�ɱ�������̭��token�ж��٣������ڱ�ʹ���ŵĲ���
    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    # <NT> �ݹ�ƥ��ǰ׺������ƥ����ǰ׺�Ĳ���token��kv_indices��ͬʱ���ر�ƥ���ϵ����һ���ڵ㡣
    # ÿ���ڵ㱻����ʱ���������ʱ�䣬̫��ʱ��δ�����ʵĻ���̭����
    # ÿ���ڵ��key��һ���б�������һ��token id���ϣ����Լ��ϵ��׸�token id��Ϊ�ʵ������key����page_sizeΪ1ʱ����
    # �Ȳ��Ҵ�ƥ��key�Ƿ�����ڸýڵ�Ϊ�������ӽڵ��У��粻���ڿ�ֱ�ӽ�����
    # ����ڣ���һ�������ӽڵ��key���ƥ���key����ͷ�����ж೤��һ�µģ�����ƥ���ϵġ�
    #     �����ƥ���ϵĲ��ֱ��ӽڵ��keyҪ�٣�
    #         ˵�����ӽڵ���Ҫ��ƥ���ϵĲ��ֽ��з��ѣ����ӽڵ�δƥ��Ĳ���key���������
    #         new_node.key��ŵ���ƥ���ϵĲ��֡�new_node�䵱last_node������Ķ�����Ҫƥ���ˣ������ڲ����еĲ��֡�
    #     ����
    #         ˵�����ӽڵ���ȫ���У���Ҫ��һ���������ӽڵ���ӽڵ㣬ֱ����ƥ���keyȫ��ƥ���꣬����ֵ�ĳ���ӽڵ�ֻ��ƥ�䲿�ֵ����������������������
    # �������ɻ󣬿ɽ��_insert_helperһ�𿴣�
    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    # <NT> ���� new_node = self._split_node(child.key, child, prefix_len)
    # ��child�ڵ���key��abcd����Ҫƥ���keyΪabef����ǰ׺ab��δ��ȫ���ǽڵ��ϵ�key����ʱ��Ҫ�Խڵ���з��ѡ�
    # ����������keyΪabcd��child�ǽڵ㱾��split_len������ǰ׺2
    # new_node.key: ab, ֻ����ǰ׺ƥ�䲿�֡�valueҲһ��ֻ����ǰ׺ƥ��Ĳ��֡�key��token id����value��token��kvcache�Ĵ��λ�ã�����һһ��Ӧ������Ҳ��������ͬ�ĳ��Ƚ��н�ȡ��
    # new_node.children: �ӽڵ���ԭ����ǰ�Ľڵ�child����ʵ�key����Ϊc������ƥ�䲿�ֵ��׸�keyֵ��child.key��Ȼ����cd�ˡ�
    # child[abcd]  ���� ==>> �� new_node[ab] -> �� chile[cd]  ���ظ��ڵ�
    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        self._record_remove_event(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)

        return new_node

    # <NT> ���ͨ��insert�������롣���ú�����ݹ��������ֱ����ɲ��롣
    # �����жϸýڵ���ڶ��ӽڵ��У��Ƿ��а���key[0](��page_size>1, ���һ��key�ĳɵ�һ��page)
    #    ���У�
    #         ȡ��key[0]���ڵ��ӽڵ�child����childΪ��ʼ�㣬ͨ��_key_match�������key���ص��Ĳ����ж೤ prefix_len
    #         ���prefix_len���ȵ���child�ڵ��ϵ�key���ܳ��ȣ�
    #             ��ʾ���ߵ�ǰ׺�������ӽڵ�child�����е�key��
    #             ���prefix_len���ȵ�����ƥ���key���ܳ��ȣ�
    #                 ��ʾ���ߵ�ǰ׺ͬ��Ҳ�����˲����key����Ҫ�����key����ȫ�����뻺���ˣ�������Ҫ�κ���������
    #             ����
    #                 ��ʾҪ�����key��һ����û�б����档���Դ�key���޳����ѱ������ͷprefix_len��token��
    #                 ȡ��δ������Ĳ���key��value������child�ڵ�Ϊ�����Ͻ�����������
    #         ����
    #             ��ʾ���ߵ�ǰ׺Ҳδ�����ӽڵ�child�ϵ�����key������Ҫ��child�ӽڵ������Ѳ�����
    #             ��child�ڵ���keyΪabcd����Ҫƥ���keyΪabef����ǰ׺ab��δ��ȫ���ǽڵ��ϵ�key����ʱ��Ҫ�Խڵ���з��ѡ�
    #             ���� child[abcd] <=_split_node=> �� new_node[ab] -> �� chile[cd]
    #             ����� new_node[ab] �¼������µݹ����δƥ�䲿�ֵ�ef��
    #    (����"����"�ڲ�����֧����return���������������)
    #    ���ޣ�
    #         ����һ�ڵ�����Ҫƥ���key�����䵱��ǰ�ڵ���ӽڵ㡣
    #         key��value����list���ڵ��parentֻ����һ��������ֱ��ָ��node��
    #         ��node��children��Ϊ���кܶ������key�б��е��׸�token�䵱������Ϊchildren��������
    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    # <NT> �ռ�Ҷ�ӽڵ㣬��û��children�Ľڵ㡣
    # children������defaultdict(TreeNode)��values����TreeNode�ϼ���
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

    def _record_store_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            parent_block_hash = hash(tuple(node.parent.key))
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=node.key,
                    block_size=len(node.key),
                    lora_id=None,
                )
            )

    def _record_remove_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache(None, None, page_size=1, disable=False)

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
