# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm import _custom_ops as ops

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_to_tensor_quant,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda_available, is_hip

is_hip_ = is_hip()

if is_cuda_available():
    from sgl_kernel import bmm_fp8


# <NT> MLP层由gate_up_proj -> act_fn -> down_proj共两个线性层和一个激活层组成。
# 其中第一个线性层gate_up_proj采用MergedColumnParallelLinear做TP，第二次线性层down_proj采用RowParallelLinear做TP。
# gate_up_proj的ColumnParallelLinear是权重按列切分，负责的元素能完整计算完，后面还有其他层需要计算时，可以先不汇聚结果(all gather)。
#     其输出维度是[intermediate_size] * 2，intermediate_size一般比hidden_size要大不少，在这输出维度大的情况下, 将输出结果汇总的代价较高，所以按列切分不汇总会更合适。
# down_proj的RowParallelLinear是权重按行切分，负责的元素都没算完，需要对结果做allreduce, 以完成整个MLP的计算。因输出维度是hidden_size，较小，通信压力也小。
#     其次，RowParallelLinear有参数input_is_parallel，即表示数据已经切分好了，可以直接在ColumnParallelLinear的结果上计算。
#     反之，如果仍采用ColumnParallelLinear，它要求每个设备接收 完整的输入数据 进行计算，而前面的结果是已经将数据切分好了，要整合的话得多一次all gather通信才行。
#     另外，输入维度是[intermediate_size] * 2，较大，占用的显存也会比较多。
# 例子看srt/layers/linear.py, 层定义处。
class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        # <NT> SiluAndMul会将输入[intermediate_size] * 2做silu后对位相乘合并，输出得到intermediate_size
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

# <NT> 用于管理和控制专家混合（MoE）层中的专家路由。
# 基本作用：
#     专家选择：根据输入数据的特征，MoEGate决定将数据分配给哪些专家处理，确保每个输入由最合适的专家处理。
#     负载均衡：MoEGate通过动态调整专家分配，防止某些专家过载或闲置，提升计算资源利用率。
#     稀疏性控制：MoEGate通过限制每个输入激活的专家数量，保持模型的计算效率，避免激活过多专家导致计算成本上升。
#     可扩展性：MoEGate支持动态增减专家数量，使模型能够灵活扩展，适应不同任务需求。
# 其中的参数 config.n_routed_experts 指每个输入样本（或token）实际被路由到的专家数量，用于控制每个输入样本激活的专家数量。
#           config.num_experts_per_tok 是指每个输入样本（或token）在路由过程中候选的专家数量。
#           都涉及到了上面的负载均衡/稀疏性控制/可扩展性。
# 本质上的实现就是一个线性层。
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config)

        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        # <NT> 
        # n_routed_experts: 每个token实际被路由到的专家数量，无论是否采用EP，都需要用到。
        # num_experts_per_tok: 每个token在路由过程中候选的专家数量
        # moe_intermediate_size: 每个专家网络中中间层（Intermediate Layer）的隐藏单元数量，每个专家通常是一个独立的FFN。而这个参数就是这个FFN中间层的维度。
        # n_group: 专家分组数量，对于每个输入样本，门控网络会为所有专家或专家组计算得分。选择得分较高的专家或专家组，通常选取topk（对应下面的topk_group）。
        #          一旦确定了每个样本要使用的专家或专家组，就将样本输入到相应的专家网络进行计算。
        #          1) 分组内专家并行计算：在每个选中的专家组内，各个专家可以并行地对分配给它们的样本进行处理，这样可以充分利用计算资源，提高计算效率。
        #          2) 跨组计算协调：如果涉及多个专家组，需要对跨组的计算进行协调，确保每个样本都能得到正确的处理。
        #          3) 根据门控分数，对各个专家的输出进行加权求和.
        # e_score_correction_bias: 用于对专家选择的得分进行修正. 如由于数据分布不均，导致某些专家可能会被门控网络频繁选中，则可以用它来调整；
        self.experts = MoEImpl(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
        )

        # <NT> shared_experts 是指在MoE层中，所有输入样本共享的专家数量。
        # 共享专家 与其他的普通专家不同之处在于，它会在gate门控网络前面执行，输入的是所有的隐层状态，即针对所有样本，通常用于学习输入数据中的通用特征。
        # 而普通专家是在gate后，选择部分普通专家进行后续处理。
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = (
            self.experts(hidden_states=hidden_states, router_logits=router_logits)
            * self.routed_scaling_factor
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        # <NT> 这里tp_size > 1, 不仅只针对TP，对于EP而言tp_size也是大于1，启用EP时，ep_size=tp_size，且TP不使用。这里的MoE的TP和EP只能二选一。
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            # FIXME: quick fix for skip quantization
            prefix=f"self_attn.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # TODO, support head_size 192
        self.attn = RadixAttention(
            self.num_local_heads,
            256,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        q = torch.nn.functional.pad(q, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        k = torch.nn.functional.pad(k, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        v = torch.nn.functional.pad(v, [0, 256 - self.v_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, 256)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        use_dp=False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim               # 不使用旋转位置编码部分的维度
        self.qk_rope_head_dim = qk_rope_head_dim               # 使用旋转位置编码部分的维度
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim # 二者相加就是总的维度
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank    # lora针对Q的低秩矩阵的秩
        self.kv_lora_rank = kv_lora_rank  # lora针对K和V的低秩矩阵的秩
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads if use_dp else num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # <NT> 使用dp时，这些线性层都是普通的线性层，而不使用DP时才会有Parallel的线性层。
        # 即这里的实现中，DP和TP无法同时用于加速MLA。
        if use_dp:
            # For data parallel attention
            if self.q_lora_rank is not None:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ReplicatedLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            else:
                self.q_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            self.kv_b_proj = ReplicatedLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
            )
            # O projection.
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
            )
        else:
            # For tensor parallel attention
            if self.q_lora_rank is not None:
                # lora模块计算：Y=X*W0 + X*A*B?
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ColumnParallelLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
            )
            # O projection.
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            # FIXME: quick fix for skip quantization
            prefix=f"self_attn.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        # <NT> 在权重吸收的模式下，kv的上采样被挪到了attention kernel后面，则进入到attention kernel的是压缩后的kv，
        # 压缩后的kv只有一份，可以看作是所有q共享一个kv head，可采用mqa的kernel，即num_kv_heads=1。
        # kv_lora_rank 是 LoRA 中低秩矩阵的秩（rank），决定了低秩矩阵的大小。
        # head_dim 为 self.kv_lora_rank + self.qk_rope_head_dim, 即压缩后的矩阵维度加上应用rope后的矩阵维度。(deepseek中压缩的部分不应用rope，位置编码和非位置编码会区分开)
        # 详情看: https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf
        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
        )
        # <NT> 普通模式下，进入到attention kernel前，kv会被上采样还原到压缩前的状态，
        # num_kv_heads等于num_local_heads，则计算与mha一致，即有q和kv的head数量一致，一一对应，而不再是所有q对应一份kv。
        # self.num_local_heads = num_heads // tp_size，在DeepSeek V3中num_heads是32。 
        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        def no_absorb() -> bool:
            if global_server_args_dict["enable_flashinfer_mla"]:
                # Flashinfer MLA: Only do not use absorb when prefilling/extending without radix cache
                return (
                    global_server_args_dict["disable_radix_cache"]
                    and forward_batch.forward_mode.is_extend()
                )
            else:
                # Triton: Use normal computation for prefill and use weight absorption for extend/decode
                # <NT> 在chunked prefill下，prefill阶段也可能是is_extend，需要额外判断extend_prefix_lens是否为0，为0才算是prefill的开头第一个chunk。
                # MQ: 为什么prefix为0时不能使用？
                return (
                    forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                    and forward_batch.extend_prefix_lens.sum() == 0
                )

        if no_absorb():
            return self.forward_normal(positions, hidden_states, forward_batch)
        else:
            if is_hip_:
                if (
                    os.getenv("SGLANG_ROCM_FUSED_DECODE_MLA") == "1"
                    and forward_batch.forward_mode.is_decode()
                ):
                    return self.forward_absorb_fused_mla_rope(
                        positions, hidden_states, forward_batch
                    )
                else:
                    return self.forward_absorb(positions, hidden_states, forward_batch)
            else:
                return self.forward_absorb(positions, hidden_states, forward_batch)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        # <NT> 单独将q的pe(位置编码)部分拿出来，需要计算rotary_emb 旋转位置编码，对应q的apply rope部分，算完后合并回去q，充当mha的q的输入。
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        # <NT> 调用mha
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        # <NT> 取出q_pe，需要进而计算位置编码 rotary_emb，对应weight_absorption中的q的apply rope部分
        # 另外q_nope，需要跟先乘以w_kc，对应weight_absorption中的Wuc Linear1
        # q_nope经过线性层得到的q_nope_out会跟q_pe经过rope计算结果合并在一起，充当mqa的q输入。
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        # [seq_len, head_dim, latent_dim+rope_dim]
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input
        k_pe = k_input[..., self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_input[..., self.kv_lora_rank :] = q_pe
        k_input[..., self.kv_lora_rank :] = k_pe

        # <NT> 调用mqa
        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

		# <NT> 需要多乘以w_vc，将维度升回去。对应weight_absorption中的Wuc Linear2
        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output


def all_gather(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch, rank, world_size, group
):
    if world_size == 1:
        return input_tensor

    all_lens = forward_batch.global_num_tokens
    max_len = max(forward_batch.global_num_tokens)

    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    group.all_gather_into_tensor(forward_batch.gathered_buffer, padded_tensor)

    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len : i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = (
            not global_server_args_dict["disable_mla"]
            and global_server_args_dict["enable_dp_attention"]
        )
        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        if not global_server_args_dict["disable_mla"]:
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                use_dp=self.enable_dp_attention,
            )
        else:
            self.self_attn = DeepseekV2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
            )

        # <NT> 
        # config.n_routed_experts is not None 表示必须是MoE架构
        # config.first_k_dense_replace 表示在模型从前 k 个全连接层开始，是否用某种特定的结构或操作替换原有的全连接层。这里的某种特定结构即MoE。
        #                              如为0，表示所有mlp都是DeepseekV2MoE。为2，表示前面两层是DeepseekV2MLP，后面都是DeepseekV2MoE。
        # config.moe_layer_freq 参数通常用于控制 MoE 层在模型中的分布频率，
        #                       如为2，表示每隔 2 层插入一个 MoE 层。
        if is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(config=config, quant_config=quant_config)
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention, MLA
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected, MoE+MLP 组合，组合方式取决于first_k_dense_replace和moe_layer_freq两个参数。
        if self.enable_dp_attention:
            # <NT> 使用dp attention时，在MLA后需要先进行all_gather, 同步各节点的数据并行计算情况。再进MoE或MLP。
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
            hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states[start_idx:end_idx]
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

# <NT> DeepseekV2Model定义
# -> VocabParallelEmbedding 
# -> for 多个相同的DeepseekV2DecoderLayer (数量是num_hidden_layers)
#    -> RMSNorm
#    -> DeepseekV2AttentionMLA - self attention
#       -> forward_normal
#       -> forward_absorb 权重吸收，
#    -> RMSNorm
#    -> DeepseekV2MoE + DeepseekV2MLP组合 (由first_k_dense_replace和moe_layer_freq两个参数控制组合情况,在V3和R1里都是分别为3和1) 
#          <DeepseekV2MoE>
#       -> shared_experts (DeepseekV2MLP类型，共享部分，每个样本都会进入计算)
#       -> MoEGate        (门控网络，选择专家)
#       -> experts, 有 EPMoE    (专家并行，一个节点存放部分专家) 
#                   或 FusedMoE（所有专家混合在一起，然后用TP拆分每个专家网络的权重，打包做gemm）
#       -> all reduce 专家结果 （无论是EP还是TP）
#          <DeepseekV2MLP>
#       ->  gate_up_proj - MergedColumnParallelLinear
#       ->  act_fn       - SiluAndMul
#       ->  down_proj    - RowParallelLinear - all reduce
# -> RMSNorm
class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

# <NT> 最外层类
class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(config, quant_config)
        if global_server_args_dict["enable_dp_attention"]:
            # 采用dp attention时，用普通线性层即可？
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            # <NT> 如果不是采用dp attention (dp+tp方案)，则默认为采用TP (TP也可以为1，即单节点)
            # 使用 ParallelLMHead(VocabParallelEmbedding). 
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        # <NT> DeepseekV2ForCausalLM本身就是一个大的torch.nn.Module.
        # named_parameters 是 torch.nn.Module 类的一个方法，其主要作用是返回一个生成器，该生成器会迭代产生模型中所有可训练参数的名称以及对应的参数本身。
        # 即会返回上面__init__(self)中所定义的可训练参数的名称和对应的参数张量信息。
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                # <NT> 如在MoE中create_weights时，会将权重加载函数通过set_weight_attrs，未为属性注册到nn.Module中。由这里取出每个层对应的加载函数，进行实际的加载操作。
                # python/sglang/srt/layers/moe/ep_moe/layer.py#161
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        if not global_server_args_dict["disable_mla"]:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    w = ops.awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
                else:
                    w = self_attn.kv_b_proj.weight
                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
                # This may affect the accuracy of fp8 model.
                if hasattr(self.quant_config, "weight_block_size") and w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if is_hip_:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                if (
                    hasattr(self.quant_config, "weight_block_size")
                    and w.dtype == torch.int8
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                # <NT> 注意到 w = self_attn.kv_b_proj.weight，即w_kc/w_vc是从升维恢复成多头的线性层的权重中拆分出来的。
                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                    if is_hip_:
                        self_attn.w_scale *= 2.0

# <NT> DeepSeek v3与v2在模型结构上完全一致，均使用了MoE和MLA，仅在参数上有所区别。
#
# V3 采用 FP8 训练，并开源了原生 FP8 权重，推理部署也更倾向于使用FP8。
# R1 是基于 DeepSeek V3 模型进一步进行训练得到的，V3是通用型的大语言模型，R1 则是推理优先的模型，侧重于处理复杂的推理任务。
# V3 总参数 6710 亿，每 token 激活 370 亿参数；R1 有不同规模的蒸馏版本，参数范围在 15 亿到 700 亿之间。
#
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
