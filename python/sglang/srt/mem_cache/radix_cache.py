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
        self.value = None                   # token��kvcache�Ĵ��λ�ã�kv_indices��
        self.lock_ref = 0                   # ���ü�������ʾ�ж��ٸ����ڼ����seq��������
        self.last_access_time = time.time() # ʱ���ǣ�����LRU��̭����

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

# <NT> �ҳ� key0 �� key1 �����б��д� ��ͷ��ʼ ������ͬԪ�ص�������
def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

# <NT> RadixCache ����ReqToTokenPool��BaseTokenToKVPool���⹹��������
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

    # <NT> ���ڲ���key������ڣ�ʵ��ʹ��ʱvalue�����Ǵ�req_to_token_pool��ȡ����kv_indices, 
    # ��ʾtoken��kvcache�еĴ��λ�á�
    def insert(self, key: List, value=None):
        if self.disable:
            return 0
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    # <NT> ��seq����ʱ���á�
    # if self.disable��������RadixCache�������������ChunkCache��cache_finished_req�������ݻ���һ�£�ֱ���ͷ�����pool��������ݡ�
    # ��δ��disable��������ʹ��RadixCache��
    #     һ�����token_ids��Ϊ�գ��������token_ids����prompts��decode���н���ĺϼ���
    #     ��token_ids�ĳ��ȴ�req_to_token_pool��ȡ����seq����token��kvcache������ͬʱִ�в������������seq������token_ids���뵽
    # �������У������¼���õ���δ���������еĲ�����ӽ�ȥ��new_prefix_lenһ�����token_ids�ĳ��ȡ���ChunkCache��ȣ�
    # ��Ҫ����token_to_kv_pool�������ͷţ���req_to_token_pool����Ҫ�����ͷŵġ�
    # ���ʵ㣺ΪʲôҪִ���ⲿ�ֵ��ͷţ�self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])
    #        ��cache_unfinished_req��Ҳ��ͬ����ִ�������
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

    # <NT> ���𣺸���num_tokens��������������������Ӧ������token��һ���ڵ����һ������token����Ϊ׼���������token��λ�á�
    # �����ռ�����Ҷ�ӽڵ�(û��children�Ľڵ�)������С���ѣ���Ϊleaves��TreeNode���ϣ�����ͨ���䷽��__lt__ȷ����С��
    # ��__lt__�Ķ����֪���Ѷ�����last_access_time��С�ߣ����ʱ��δ�����ʵġ�����С�������pop�������м�����
    # ���pop����Ҷ�ӽڵ�Ŀǰδ��ʹ��(x.lock_ref > 0), ��������ͷŴ������ۼ��ͷŵĽڵ��У�������token����(num_evicted).
    # ʹ�����token�����ﵽnum_tokens��
    #
    # heapq.heapify �����ܹ���һ���б�ת��ΪС���ѽṹ��
    # heapq.heappop ���ضѶ�����Сֵ����ά��С�������ʡ�
    # heapq.heappush ����һ��Ԫ�ص������ά��С�������ʡ�
    # evict_callback �����self.token_to_kv_pool.free�����kvcache���ڴ��ͷš� 
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

    # <NT> ���Ӽ�������ʾ���ڱ�ʹ�á�
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

    ##### Internal Helper Functions #####
    # <NT> �ݹ�ƥ��ǰ׺������ƥ����ǰ׺�Ĳ���token��kv_indices��ͬʱ���ر�ƥ���ϵ����һ���ڵ㡣
    # ÿ���ڵ㱻����ʱ���������ʱ�䣬̫��ʱ��δ�����ʵĻ���̭����
    # ÿ���ڵ��key��һ���б�������һ��token id���ϣ����Լ��ϵ��׸�token id��Ϊ�ʵ������key��
    # �Ȳ��Ҵ�ƥ��key���׸�token�Ƿ�����ڸýڵ�Ϊ�������ӽڵ��У��粻���ڿ�ֱ�ӽ�����
    # ����ڣ���һ�������ӽڵ��key���ƥ���key����ͷ�����ж೤��һ�µģ�����ƥ���ϵġ�
    #     �����ƥ���ϵĲ��ֱ��ӽڵ��keyҪ�٣�
    #         ˵�����ӽڵ���Ҫ��ƥ���ϵĲ��ֽ��з��ѣ����ӽڵ�δƥ��Ĳ���key���������
    #         new_node.key��ŵ���ƥ���ϵĲ��֡�new_node�䵱last_node������Ķ�����Ҫƥ���ˣ������ڲ����еĲ��֡�
    #     ����
    #         ˵�����ӽڵ���ȫ���У���Ҫ��һ���������ӽڵ���ӽڵ㣬ֱ����ƥ���keyȫ��ƥ���꣬����ֵ�ĳ���ӽڵ�ֻ��ƥ�䲿�ֵ����������������������
    # �������ɻ󣬿ɽ��_insert_helperһ�𿴣�
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

    # <NT> ���� new_node = self._split_node(child.key, child, prefix_len)
    # ��child�ڵ���key��abcd����Ҫƥ���keyΪabef����ǰ׺ab��δ��ȫ���ǽڵ��ϵ�key����ʱ��Ҫ�Խڵ���з��ѡ�
    # ����������keyΪabcd��child�ǽڵ㱾��split_len������ǰ׺2
    # new_node.key: ab, ֻ����ǰ׺ƥ�䲿�֡�valueҲһ��ֻ����ǰ׺ƥ��Ĳ��֡�key��token id����value��token��kvcache�Ĵ��λ�ã�����һһ��Ӧ������Ҳ��������ͬ�ĳ��Ƚ��н�ȡ��
    # new_node.children: �ӽڵ���ԭ����ǰ�Ľڵ�child����ʵ�key����Ϊc������ƥ�䲿�ֵ��׸�keyֵ��child.key��Ȼ����cd�ˡ�
    # child[abcd]  ���� ==>> �� new_node[ab] -> �� chile[cd]  ���ظ��ڵ�
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

    # <NT> ���ͨ��insert�������룬����self.root_node��key��token ids��value��key��ֵһ�¡�
    # ���ú�����ݹ��������ֱ����ɲ��롣
    # �����жϸýڵ���ڶ��ӽڵ��У��Ƿ��а���key[0]
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
