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

# <NT> ʹ��torchԭ��attention api: torch.nn.functional.scaled_dot_product_attention
# ����: query: Tensor [batch_size, num_heads, seq_len_q, head_dim]
#       key:   Tensor [batch_size, num_heads_kv, seq_len_kv, head_dim]
#       value: Tensor [batch_size, num_heads_kv, seq_len_kv, head_dim]  ע�ⲻ֧��head_dim_v��value��head_dim����query��һ�¡�
#       attn_mask: Optional[Tensor]��ע�������룬[batch_size, seq_len, seq_len], ����ֵ���㣬���Ӧλ��ע���������ᱻ����Ϊ -inf���Ӷ��ڼ��� softmax ʱ�����ԡ�
#       dropout_p: float = 0.0����ѵ���׶Σ���������һ������ֵ���������һЩע����Ȩ�أ��Է�ֹ����ϡ�
#       is_causal: bool = False, �������Ϊ True������Զ�����һ��������룬ʹ��ÿ��λ��ֻ�ܹ�ע����֮ǰ��λ�ã��������Իع��������
#       scale: Optional[float] = None, ��δָ��������Զ�����Ϊ 1 / sqrt(head_dim)����ÿ��ע����ͷ��Ƕ��ά�ȵĵ�����ƽ������
#       enable_gqa: bool = False���������Ϊ True�������� query ��ͷ���� key �� value �����ظ���չ����֧�ַ����ѯ��������ֻҪq��ͷ��������k��ͷ����ΪTrue����use_gqa��
# ����ֵ: ��� Tensor [batch_size, num_heads, seq_len_q, head_dim]
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

        # <NT> api��ά��������HDN[batch_size, num_heads, seq_len_q, head_dim]����sglang����NHD��������Ҫ��תһ��ά��˳��
        # �����batch_size��Ϊ1���ڽ�api֮ǰ�ٲ���batch_sizeά�ȡ����������qkv��û��batch_sizeά��(ʹ��fa3��backendҲһ��)��Ϊʲô��
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        # <NT> seq_lens�Ǹ�ForwardBatch�а�����seq�������ĳ��ȼ��ϣ�Ԫ����������batch_size, ÿ��Ԫ�ر�ʾ��Ӧseq�ĳ��ȡ�
        # ѭ��ÿ��seq��ÿ��seq���������൱��ѭ����ÿ�μ����batch_size��Ϊ1.
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            # <NT> extend_prefix_lens: ÿ�������ǰ׺���ȣ����Ѿ�Ԥ�ȼ���� KV ����Ĳ������г���
            #      extend_seq_lens: ÿ��������Ҫ��������г��ȣ�����ǰ��������Ҫ����Ĳ������г���
            #      extend_prefix_len + extend_seq_len = seq_len (�Ѵ�ӡȷ��)
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            # <NT> queryά�ȵ�ת���м����num_tokens��q������Ҫ��������ݣ���Ӧextend_seq_lens��
            # num_tokens�ǰ����˸�batch������extend_seq_len֮�ͣ���Ҫ�зֳ���seq������tokens��
            per_req_query = query[:, start_q:end_q, :]
            # <NT> seq_len_kv�Ǹ�seq�����Ѽ����ǰ׺�͵�ǰ��Ҫ����Ĳ��ֵ��ܳ��ȡ�
            # ����ǰ��Ҫ�����per_req_query��䵽per_req_query_redudant��prefill_seq_len_q�����λ���ϡ�
            # ��per_req_query_redudant��0��prefill_seq_len_q��Χ�ڣ���ǰ׺������Ŀǰ�ǿյġ�
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # <NT> Χ��seq_idx��ȡkvcache�����ȴ� req_pool_indices ���ҵ���seq��Ӧ��req_pool��������
            # ���ڸ���������ֱ���ҵ���req��token��req_to_token�е�λ�á� req_to_token��ÿ�ж�Ӧһ��req��
            # ����ȡ����Ӧreq�е�ǰseq_len_kv�е����ݣ�����per_req_tokens��ÿ��Ԫ�ر�ʾһ��token������ֵ��
            # ��qwen2.5��k_cache[49518, 2, 128]��per_req_tokens[8240]��ͨ��per_req_tokens��8240������ֵ��
            # �ҳ�k_cache�ж�Ӧ�ĵ�һά��8240�У���ϳ�һ��������[8240,2,128]=>[num_token, nheads, head_dim].
            # ���query.dim()-2=1����.movedim(0, 1)��0��ά��Ų��1��ά���ϣ���������[nheads, num_token, head_dim]��
            # ���Ը�������per_req_key[2, 8240, 128]��
            # k_cache��v_cache����һ������������per_req_value[2, 8240, 128]��
            # �ڴ�����������: per_req_key �� k_cache ��һ����ͼ��view���������� k_cache ���ڴ�, �������k_cache���ڴ�������
            # ������ֵper_req_tokens�����������ģ���ȡ0,3,5��token, ���ڴ�Ҳ���������ģ�
            #
            # req_pool_indices����prepare_for_extend�У���batchʱ���� req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
            # Ϊÿ��req����ղ�λ����ÿ��forwardBatch����ά��һ���������batch������req��req_to_token�е�����λ�á�
            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                # <NT> sage attention �Ľ��뷽ʽ
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
