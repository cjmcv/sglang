# Copyright 2024 SGLang Team
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

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from sglang.srt.utils import get_compiler_backend, is_cuda

_is_cuda = is_cuda()

from sglang.srt.managers.utils import ExpertDistributionRecorder

expert_distribution_recorder = ExpertDistributionRecorder()


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    if _is_cuda:
        from sgl_kernel import topk_softmax
    else:
        from vllm import _custom_ops as ops

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if _is_cuda:
        topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),
        )
    else:
        ops.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),
        )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek V2/V3/R1 series models
# <NT> 对门控网络输出计算softmax，然后按分组计算每组的分数，以每组的最高值来取topk的组。
# 并构建掩码筛选出得分较高的部分的权重及其下标。
# torch.topk返回的是topk的数值和下标，dim=-1表示沿着最后一个维度，如果是二维向量，则dim=-1是列，即从列方向找出每一行的topk个元素及其下标，维度是[n, k], 行数n不变，列是topk的k。
@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    # <NT> view是按分组num_expert_group，重新调整分数矩阵的维度，-1即最后一个维度的连续排布的分数是一组的
    # max(dim=-1).values，即在最后一个维度上取最大值。
    # group_scores的维度将会是[num_token, num_expert_group], 里面每个元素表示一组内所有专家的分数的最高值。
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

# <NT> 根据门控网络gate的输出，选择使用哪些专家
# 大体有grouped_topk，biased_grouped_topk和fused_topk
def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
):
    # DeekSeekv2 uses grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
    elif torch_native and custom_routing_function is None:
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    expert_distribution_recorder.record_new_token(topk_ids)

    return topk_weights, topk_ids
