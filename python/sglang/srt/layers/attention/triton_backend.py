from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )

        self.num_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        if forward_batch.forward_mode.is_decode():
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_head,
                    self.num_kv_splits,
                    self.v_head_dim + 1,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            max_extend_len = None
        else:
            attn_logits = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        self.forward_metadata = attn_logits, max_extend_len

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len

        self.cuda_graph_start_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_attn_logits = torch.empty(
            (max_bs, self.num_head, self.num_kv_splits, self.v_head_dim + 1),
            dtype=torch.float32,
            device="cuda",
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        assert encoder_lens is None, "Not supported"
        assert forward_mode.is_decode(), "Not supported"
        assert spec_info is None, "Not supported"

        self.forward_metadata = (
            self.cuda_graph_attn_logits,
            None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        self.cuda_graph_start_loc.zero_()
        self.cuda_graph_start_loc[1:bs] = torch.cumsum(seq_lens[: bs - 1], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


    # <NT> 该函数会在(python/sglang/srt/models/qwen2.py#157 -> self.attn)里调用。
    # self.attn是RadixAttention对象，在forward时，里面会根据forward_batch.forward_mode.is_decode()来确定调用forward_extend还是forward_decode。
    # 输入参数的qkv由attention层之前的线性层计算输出的。其中的k和v的进入forwad时就会被添加到该ForwardBatch维护的kvcache buffer中 (set_kv_buffer)，每次调用都会被叠加。
    # 
    # forward_extend里，kernel输入新的k和v，同时还要将kvcache里的数据拿出来计算，同时输入进行计算。
    #    extend的bacth里，数据同属于一个序列，对应的是同一份kvcache，从历史kvcache里拿出来的数据属于prefix，所有数据都是历史数据，需要单独先计算。
    #                    然后还要跟新的kv计算，新的kv可能会包含未来数据，在因果模型里只需要计算三角区域即可。所以需要区分开两部分来计算。
    # forward_decode里，k和v就只用从kvcache buffer里读出来的。
    #    decode的batch里，数据来自多个不同的序列，每个序列会有各自的kvcache，里面需要将batch里的每一行(一个token)与其对应的kv矩阵(该序列所有历史kvcache)做gemv。
    # （参考 https://flashinfer.ai/2024/02/02/introduce-flashinfer.html 中的图1，Append对应Extend）
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        attn_logits, _ = self.forward_metadata

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            attn_logits,
            self.num_kv_splits,
            layer.scaling,
            layer.logit_cap,
        )
        return o
