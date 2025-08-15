from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# <NT> 使用torch原生attention api: torch.nn.functional.scaled_dot_product_attention
# 参数: query: Tensor [batch_size, num_heads, seq_len_q, head_dim]
#       key:   Tensor [batch_size, num_heads_kv, seq_len_kv, head_dim]
#       value: Tensor [batch_size, num_heads_kv, seq_len_kv, head_dim]  注意不支持head_dim_v，value的head_dim需与query的一致。
#       attn_mask: Optional[Tensor]，注意力掩码，[batch_size, seq_len, seq_len], 掩码值非零，则对应位置注意力分数会被设置为 -inf，从而在计算 softmax 时被忽略。
#       dropout_p: float = 0.0。在训练阶段，可以设置一个非零值来随机丢弃一些注意力权重，以防止过拟合。
#       is_causal: bool = False, 如果设置为 True，则会自动创建一个因果掩码，使得每个位置只能关注到它之前的位置，常用于自回归解码任务。
#       scale: Optional[float] = None, 如未指定，则会自动计算为 1 / sqrt(head_dim)，即每个注意力头的嵌入维度的倒数的平方根。
#       enable_gqa: bool = False，如果设置为 True，则会根据 query 的头数对 key 和 value 进行重复扩展，以支持分组查询。（这里只要q的头数不等于k的头数就为True，搜use_gqa）
# 返回值: 输出 Tensor [batch_size, num_heads, seq_len_q, head_dim]
class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # <NT> api的维度需求是HDN[batch_size, num_heads, seq_len_q, head_dim]，而sglang的是NHD，所以需要调转一下维度顺序。
        # 这里的batch_size都为1，在进api之前再插入batch_size维度。进到这里的qkv都没有batch_size维度(使用fa3等backend也一样)，为什么？
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        # <NT> seq_lens是该ForwardBatch中包含的seq的数量的长度集合，元素总数等于batch_size, 每个元素表示对应seq的长度。
        # 循环每个seq，每个seq单独处理，相当于循环内每次计算的batch_size都为1.
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            # <NT> extend_prefix_lens: 每个请求的前缀长度，即已经预先计算好 KV 缓存的部分序列长度
            #      extend_seq_lens: 每个请求需要处理的序列长度，即当前批次中需要计算的部分序列长度
            #      extend_prefix_len + extend_seq_len = seq_len (已打印确认)
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            # <NT> query维度调转后，中间的是num_tokens，q都是需要处理的内容，对应extend_seq_lens，
            # num_tokens是包含了该batch的所有extend_seq_len之和，需要切分出该seq所属的tokens。
            per_req_query = query[:, start_q:end_q, :]
            # <NT> seq_len_kv是该seq包含已计算的前缀和当前需要计算的部分的总长度。
            # 将当前需要计算的per_req_query填充到per_req_query_redudant中prefill_seq_len_q往后的位置上。
            # 则per_req_query_redudant的0到prefill_seq_len_q范围内，即前缀的内容目前是空的。
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # <NT> 围绕seq_idx获取kvcache，首先从 req_pool_indices 中找到该seq对应的req_pool的索引。
            # 基于该索引可以直接找到该req的token在req_to_token中的位置。 req_to_token中每行对应一个req，
            # 这里取出对应req行的前seq_len_kv列的数据，赋给per_req_tokens，每个元素表示一个token的索引值。
            # 如qwen2.5中k_cache[49518, 2, 128]，per_req_tokens[8240]，通过per_req_tokens的8240个索引值，
            # 找出k_cache中对应的第一维的8240行，组合成一个新张量[8240,2,128]=>[num_token, nheads, head_dim].
            # 最后query.dim()-2=1，即.movedim(0, 1)将0号维度挪到1号维度上，即调整成[nheads, num_token, head_dim]，
            # 所以该例子中per_req_key[2, 8240, 128]。
            # k_cache和v_cache共用一份索引，所以per_req_value[2, 8240, 128]。
            # 内存连续性问题: per_req_key 是 k_cache 的一个视图（view），它共享 k_cache 的内存, 所以如果k_cache里内存连续，
            # 但索引值per_req_tokens并不是连续的，如取0,3,5号token, 则内存也不是连续的！
            #
            # req_pool_indices是在prepare_for_extend中，组batch时调用 req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
            # 为每个req申请空槽位，即每个forwardBatch都会维护一个，负责该batch的所有req在req_to_token中的索引位置。
            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                # <NT> sage attention 的接入方式
                # sageattn(
                #     per_req_query_redudant.unsqueeze(0),
                #     per_req_key.unsqueeze(0),
                #     per_req_value.unsqueeze(0),
                #     sm_scale=scaling,
                #     tensor_layout="HND",
                #     is_causal=causal,
                # )
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False
