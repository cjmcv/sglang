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

# <NT> huggingface的模型会由configuration_deepseek.py和modeling_deepseek.py两个文件组成，
# configuration_deepseek.py：包含各种超参数和设置信息，这些信息决定了模型的架构、规模和行为。通过配置类，用户可以方便地对模型进行定制和调整；
#                            https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/configuration_deepseek.py
# modeling_deepseek.py：实现 DeepSeek 模型的具体架构和前向传播逻辑。它定义了模型的各个组件，如嵌入层、注意力层、前馈网络等，并将这些组件组合成完整的模型；
#                       https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# 如在ktransformer的实现，是在huggingface模型定义的基础上，将其中某些算子更换成自己的，从而达到优化的效果。
# 当前的sglang模型文件主要对应上面modeling_deepseek.py。

import os
from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import ModelConfigForExpertLocation
from sglang.srt.managers.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    LazyValue,
    add_prefix,
    bind_or_assign,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2

    from sglang.srt.layers.quantization.deep_gemm import (
        grouped_gemm_nt_f8f8bf16_masked as deep_gemm_grouped_gemm_nt_f8f8bf16_masked,
    )
else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )

if _use_aiter:
    from aiter.rotary_embedding import get_rope

logger = logging.getLogger(__name__)


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()


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
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        # <NT> SiluAndMul会将输入[intermediate_size] * 2做silu后对位相乘合并，输出得到intermediate_size
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch=None):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

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
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
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
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if global_server_args_dict["disable_shared_experts_fusion"]
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id

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

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        # <NT> deepseek开源的deepep方案 / 原生的EP方案 / 原生的FusedMoE的TP方案
        # n_routed_experts: 每个token实际被路由到的专家数量，无论是否采用EP，都需要用到。
        # num_experts_per_tok: 每个token在路由过程中候选的专家数量
        # moe_intermediate_size: 每个专家网络中中间层（Intermediate Layer）的隐藏单元数量，每个专家通常是一个独立的FFN。而这个参数就是这个FFN中间层的维度。
        # n_group: 专家分组数量，对于每个输入样本，门控网络会为所有专家或专家组计算得分。选择得分较高的专家或专家组，通常选取topk（对应下面的topk_group）。
        #          一旦确定了每个样本要使用的专家或专家组，就将样本输入到相应的专家网络进行计算。
        #          1) 分组内专家并行计算：在每个选中的专家组内，各个专家可以并行地对分配给它们的样本进行处理，这样可以充分利用计算资源，提高计算效率。
        #          2) 跨组计算协调：如果涉及多个专家组，需要对跨组的计算进行协调，确保每个样本都能得到正确的处理。
        #          3) 根据门控分数，对各个专家的输出进行加权求和.
        # e_score_correction_bias: 用于对专家选择的得分进行修正. 如由于数据分布不均，导致某些专家可能会被门控网络频繁选中，则可以用它来调整；
        self.experts = get_moe_impl_class()(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        # <NT> shared_experts 是指在MoE层中，所有输入样本共享的专家数量。
        # 共享专家 与其他的普通专家不同之处在于，它会在gate门控网络前面执行，输入的是所有的隐层状态，即针对所有样本，通常用于学习输入数据中的通用特征。
        # 而普通专家是在gate后，选择部分普通专家进行后续处理。
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if global_server_args_dict["enable_deepep_moe"]
                    else {}
                ),
            )

        self.top_k = config.num_experts_per_tok

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,
                return_recv_hook=True,
            )

        self._enable_deepep_moe = global_server_args_dict["enable_deepep_moe"]

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        if not self._enable_deepep_moe:
            return self.forward_normal(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        # <NT> 这里tp_size > 1, 不仅只针对TP，对于EP而言tp_size也是大于1，启用EP时，ep_size=tp_size，且TP不使用。这里的MoE的TP和EP只能二选一。
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    # <NT> 1) hidden_states 输入通过gate网络得到路由分数 router_logits，同时也经过共享专家得到共享专家的输出 shared_output。
    #      2) 根据gate网络输出去选择专家，topk_idx和topk_weights都是[m,topk]，一行对应一个token，一列对应一个专家id及其在该轮计算的比重。
    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        forward_mode = forward_batch.forward_mode
        shared_output = None
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_mode=forward_mode,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                hidden_states=final_hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states):
        if self.num_fused_shared_experts == 0:
            return self.shared_experts(hidden_states)
        else:
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            state.topk_weights_local, state.topk_idx_local = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                num_token_non_padded=state.forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            self.deepep_dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                (
                    state.hidden_states_experts_input,
                    state.topk_idx_dispatched,
                    state.topk_weights_dispatched,
                    state.reorder_topk_ids,
                    state.num_recv_tokens_per_expert,
                    state.seg_indptr,
                    state.masked_m,
                    state.expected_m,
                ) = self.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.hidden_states_experts_output = self.experts(
            hidden_states=state.pop("hidden_states_experts_input"),
            topk_idx=state.topk_idx_dispatched,
            topk_weights=state.topk_weights_dispatched,
            reorder_topk_ids=state.pop("reorder_topk_ids"),
            seg_indptr=state.pop("seg_indptr"),
            masked_m=state.pop("masked_m"),
            expected_m=state.pop("expected_m"),
            num_recv_tokens_per_expert=state.pop("num_recv_tokens_per_expert"),
            forward_mode=state.forward_batch.forward_mode,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.pop("topk_idx_dispatched"),
                topk_weights=state.pop("topk_weights_dispatched"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.deepep_dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

# <NT> 模型中MLA的参数值: https://zhuanlan.zhihu.com/p/25449691772
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
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
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
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
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
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.flashinfer_mla_disable_ragged = global_server_args_dict[
            "flashinfer_mla_disable_ragged"
        ]
        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.rocm_fused_decode_mla = get_bool_env_var(
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = get_int_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", 8192
        )

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        def _dispatch_mla_subtype():
            if _is_hip:
                if (
                    self.rocm_fused_decode_mla
                    and forward_batch.forward_mode.is_decode()
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE
                else:
                    return AttnForwardMethod.MLA
            else:
                return AttnForwardMethod.MLA

        # <NT> flashinfer和FA3里实现的都是最底层的attention kernel，对应RadixAttention类型的 attn_mha / attn_mqa 的forward实现kernel。
        # 实际mla是更外层的概念，包含了 低秩压缩、动态稀疏、位置编码适配 三大核心创新，这里MLA层会包含这些内容，在经过各种处理后，才调用 attn_mha 或 attn_mqa 的计算kernel。
        # 1）将原始高维KV矩阵投影至低秩潜在空间，生成联合压缩的隐向量 cKV（隐层状态经过线性层就得到cKV#搜 kv_a_proj_with_mqa ），
        #    仅需存储压缩后的隐向量，extend阶段通过线性变换重建原始KV，使用mha计算。decode阶段采用权重吸收优化，全程采用cKV。
        # 2）轻量级网络实时预测各注意力头对当前输入的贡献度，仅保留贡献度最高的头部，关闭低效计算单元。（代码中未体现？）
        # 3）将Query/Key的RoPE维度与非RoPE维度解耦，RoPE仅作用于部分维度，确保压缩后潜在向量仍携带位置感知能力。
        #
        # <NT> MLA在prefill/extend模式下会进入普通模式采用MHA实现，在decode阶段进入MLA的权重吸收模式采用MQA实现.
        # 原因: https://zhuanlan.zhihu.com/p/1897225385751585767
        #    1) 在prefill阶段，矩阵吸收后，MQA的HeadDim变为576+512，MHA使用pad后的vheadDim192，所以有（576 + 512）/（192+192）=2.833 ，
        #                     如果MHA的V不使用pad，则MQA实现相比MHA实现：（576+512) / (192+128)=3.4。所以MQA相比MHA的实现方式，计算量更大。
        #                     所以MQA相比MHA的实现方式，计算量更大。
        #    2) 在decode阶段，q为1，MQA的计算量会小于MHA，序列越长收益越大。
        if self.attention_backend == "flashinfer":
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            if (
                not self.flashinfer_mla_disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()
        elif self.attention_backend == "fa3":
            # Flash Attention: Use MHA with chunked KV cache when prefilling on long sequences.
            if forward_batch.extend_prefix_lens_cpu is not None:
                sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
            if (
                forward_batch.forward_mode.is_extend()
                and not self.disable_chunked_prefix_cache
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and (
                    sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
                    or sum_extend_prefix_lens == 0
                )
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif self.attention_backend == "aiter":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        return self.forward_core(s)

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj

        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, None, forward_batch, None

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            inner_state = self.forward_absorb_fused_mla_rope_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        else:
            raise NotImplementedError

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        # <NT> 单独将q的pe(位置编码)部分拿出来，需要计算rotary_emb 旋转位置编码，对应q的apply rope部分，算完后合并回去q，充当mha的q的输入。
        # qk_nope_head_dim：不参与位置编码的维度=>低秩投影路径； qk_rope_head_dim：参与RoPE旋转的维度=>位置敏感路径
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        # <NT> 从 低秩 到 原始维度的重建，用于后面的mha
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

        return q, k, v, forward_batch

    def forward_normal_core(self, q, k, v, forward_batch):
        # <NT> 调用mha
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                q = self.q_a_layernorm(q)
                k_nope = self.kv_a_layernorm(k_nope)

            k_nope = k_nope.unsqueeze(1)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        # <NT> 取出q_pe，需要进而计算位置编码 rotary_emb，对应weight_absorption中的q的apply rope部分
        # 另外q_nope，需要跟先乘以w_kc，对应weight_absorption中的Wuc Linear1
        # q_nope经过线性层得到的q_nope_out会跟q_pe经过rope计算结果合并在一起，充当mqa的q输入。
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        return q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator

    def forward_absorb_core(
        self, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator
    ):
        if self.attention_backend == "fa3" or self.attention_backend == "flashinfer":
            attn_output = self.attn_mqa(
                q_nope_out, k_nope, k_nope, forward_batch, q_rope=q_pe, k_rope=k_pe
            )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = attn_bmm_output[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
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

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
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

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
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
        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
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

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            lse = torch.transpose(lse, 0, 1).contiguous()
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
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

        return q, k, v, forward_batch

    def forward_normal_chunked_kv_core(self, q, k, v, forward_batch):
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        lse = torch.transpose(lse, 0, 1).contiguous()

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if any(forward_batch.extend_prefix_lens_cpu):
            # Only initialize the info once
            if forward_batch.num_prefix_chunks is None:
                forward_batch.prepare_chunked_prefix_cache_info(q.device)

            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
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
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        # <NT> 
        # config.n_routed_experts is not None 表示必须是MoE架构
        # config.first_k_dense_replace 表示在模型从前 k 个全连接层开始，是否用某种特定的结构或操作替换原有的全连接层。这里的某种特定结构即MoE。
        #                              如为0，表示所有mlp都是DeepseekV2MoE。为2，表示前面两层是DeepseekV2MLP，后面都是DeepseekV2MoE。
        # config.moe_layer_freq 参数通常用于控制 MoE 层在模型中的分布频率，
        #                       如为2，表示每隔 2 层插入一个 MoE 层。
        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch.forward_mode
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


# <NT> DeepseekV2Model定义
# -> VocabParallelEmbedding 
# -> for 多个相同的DeepseekV2DecoderLayer (数量是num_hidden_layers)
#    -> RMSNorm
#    -> DeepseekV2AttentionMLA - self attention
#       -> forward_normal
#       -> forward_absorb 权重吸收，
#    -> RMSNorm
#    -> DeepseekV2MoE + DeepseekV2MLP组合 (由first_k_dense_replace和moe_layer_freq两个参数控制组合情况,在V3和R1里都是分别为3和1) 
#       <DeepseekV2MoE>
#       -> shared_experts (DeepseekV2MLP类型，共享部分，每个样本都会进入计算)
#       -> MoEGate        (门控网络，选择专家)
#       -> experts, 有 EPMoE    (专家并行，一个节点存放部分专家) 
#                   或 FusedMoE（所有专家混合在一起，然后用TP拆分每个专家网络的权重，打包做gemm）
#       -> all reduce 专家结果 （无论是EP还是TP）
#       <DeepseekV2MLP>
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
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dp_size = get_local_attention_dp_size()

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        total_num_layers = len(self.layers)
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None

        normal_num_layers = (
            self.first_k_dense_replace
            if forward_batch.can_run_tbo
            else total_num_layers
        )
        for i in range(normal_num_layers):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, zero_allocator
                )

        if normal_num_layers != total_num_layers:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_num_layers:],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_num_layers - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)
        self.dp_size = get_local_attention_dp_size()

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = (
            0
            if global_server_args_dict["disable_shared_experts_fusion"]
            else self.config.n_shared_experts
        )
        if self.num_fused_shared_experts > 0:
            # Only Deepseek V3/R1 can use shared experts fusion optimization now.
            if (
                not _is_cuda
                or self.config.architectures[0] != architecture
                or self.config.n_routed_experts != 256
            ):
                self.num_fused_shared_experts = 0
                global_server_args_dict["disable_shared_experts_fusion"] = True
                log_info_on_rank0(
                    logger,
                    "Only Deepseek V3/R1 on NV-platform can use shared experts fusion optimization. Shared experts fusion optimization is disabled.",
                )
            elif (
                global_server_args_dict["enable_deepep_moe"]
                or global_server_args_dict["enable_ep_moe"]
            ):
                self.num_fused_shared_experts = 0
                global_server_args_dict["disable_shared_experts_fusion"] = True
                log_info_on_rank0(
                    logger,
                    "Deepseek V3/R1 can not use shared experts fusion optimization when in deepep_moe or ep_moe mode. Shared experts fusion optimization is disabled.",
                )
        elif self.num_fused_shared_experts == 0:
            if (
                _is_cuda
                and torch.cuda.get_device_capability("cuda") >= (9, 0)
                and self.config.architectures[0] == architecture
                and self.config.n_routed_experts == 256
                and (
                    not (
                        global_server_args_dict["enable_deepep_moe"]
                        or global_server_args_dict["enable_ep_moe"]
                    )
                )
            ):
                self.num_fused_shared_experts = self.config.n_shared_experts
                global_server_args_dict["disable_shared_experts_fusion"] = False
                log_info_on_rank0(
                    logger,
                    "Deepseek V3/R1 with fp8/fp4 can use shared experts fusion optimization when SM version >=90. Shared experts fusion optimization is enabled.",
                )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def post_load_weights(self, is_nextn=False, weight_names=None):

        # Perform post-processing after loading weights
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.config.num_hidden_layers)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        # filter the nextn layer.
                        if layer_id != self.config.num_hidden_layers:
                            layer_ids.add(layer_id)

        for layer_id in layer_ids:
            self_attn = (
                self.model.layers[layer_id].self_attn
                if not is_nextn
                else self.model.decoder.self_attn
            )
            if hasattr(self_attn.kv_b_proj, "qweight"):
                # AWQ compatible
                if _is_cuda:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = awq_dequantize(
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
            # Fix deepseek v3 blockwise bmm by using deep_gemm
            use_deep_gemm_bmm = False
            model_dtype = torch.get_default_dtype()

            if w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                if (
                    hasattr(self.quant_config, "weight_block_size")
                    and self.quant_config.weight_block_size is not None
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv

                    if (
                        _is_cuda
                        and weight_block_size[0] == 128
                        and weight_block_size[1] == 128
                        and model_dtype == torch.bfloat16
                    ):
                        if _ENABLE_JIT_DEEPGEMM and get_bool_env_var(
                            "SGL_USE_DEEPGEMM_BMM", "false"
                        ):
                            block_scale = weight_scale
                            use_deep_gemm_bmm = True
                        else:
                            w = block_quant_dequant(
                                weight,
                                weight_scale,
                                weight_block_size,
                                model_dtype,
                            )
                    else:
                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                else:
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale

                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                    self_attn.w_scale = scale

            if w.dtype == torch.int8:
                if hasattr(self.quant_config, "weight_block_size"):
                    # block-wise int8 need it
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                else:
                    # channel-wise int8 need it
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                        torch.bfloat16
                    )

            # <NT> 注意到 w = self_attn.kv_b_proj.weight，即w_kc/w_vc是从升维恢复成多头的线性层的权重中拆分出来的。
            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            if not use_deep_gemm_bmm:
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                self_attn.w_vc = bind_or_assign(
                    self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                )
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = bind_or_assign(
                        self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                    )
                    if _is_hip:
                        self_attn.w_scale *= 2.0
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign(
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign(
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
                self_attn.use_deep_gemm_bmm = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is not None:
                if self.quant_config.get_name() == "w8a8_int8":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif (
                    self.quant_config.get_name() == "fp8"
                    or self.quant_config.get_name() == "blockwise_int8"
                ):
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale_inv",
                        "gate_proj.weight",
                        "gate_proj.weight_scale_inv",
                        "up_proj.weight",
                        "up_proj.weight_scale_inv",
                    ]
                elif self.quant_config.get_name() == "awq":
                    suffix_list = [
                        "down_proj.qweight",
                        "down_proj.qzeros",
                        "down_proj.scales",
                        "gate_proj.qweight",
                        "gate_proj.qzeros",
                        "gate_proj.scales",
                        "up_proj.qweight",
                        "up_proj.qzeros",
                        "up_proj.scales",
                    ]
                elif self.quant_config.get_name() == "modelopt_fp4":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "down_proj.weight_scale_2",
                        "down_proj.input_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "gate_proj.weight_scale_2",
                        "gate_proj.input_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                        "up_proj.weight_scale_2",
                        "up_proj.input_scale",
                    ]
                else:
                    raise ValueError(
                        f"Unsupported shared expert fusion for quantization: {self.quant_config.get_name()}."
                    )
            else:
                suffix_list = [
                    "down_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                ]
            names_to_remove = []

            moe_layers = (
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                )
                if not is_nextn
                else [nextn_layer_id]
            )

            for moe_layer in tqdm(
                moe_layers,
                desc=f"Cloning {self.num_fused_shared_experts} "
                "shared expert into MoE",
            ):
                for suffix in suffix_list:
                    shared_expert_weight_name = (
                        f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    )
                    weights_list.append(
                        (
                            f"model.layers.{moe_layer}."
                            f"mlp.experts."
                            f"{self.config.n_routed_experts + 0}"
                            f".{suffix}",
                            weights_dict[shared_expert_weight_name],
                        )
                    )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        # <NT> DeepseekV2ForCausalLM本身就是一个大的torch.nn.Module.
        # named_parameters 是 torch.nn.Module 类的一个方法，其主要作用是返回一个生成器，该生成器会迭代产生模型中所有可训练参数的名称以及对应的参数本身。
        # 即会返回上面__init__(self)中所定义的可训练参数的名称和对应的参数张量信息。
        params_dict = dict(self.named_parameters())
        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

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
                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                                )
                            )
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if (
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            if any(scale in name for scale in ["k_scale", "v_scale"]):
                                name = name.replace("_proj", "attn_mqa")
                            else:
                                logger.warning(
                                    f"Unknown scale found in checkpoint: {name}"
                                )
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


# <NT> DeepSeek v3与v2在模型结构上完全一致，均使用了MoE和MLA，仅在参数上有所区别。
#
# V3 采用 FP8 训练，并开源了原生 FP8 权重，推理部署也更倾向于使用FP8。
# R1 是基于 DeepSeek V3 模型进一步进行训练得到的，V3是通用型的大语言模型，R1 则是推理优先的模型，侧重于处理复杂的推理任务。
# V3 总参数 6710 亿，每 token 激活 370 亿参数；R1 有不同规模的蒸馏版本，参数范围在 15 亿到 700 亿之间。
#
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
