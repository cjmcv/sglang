"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

from typing import Callable, Optional

import torch
from torch.nn import functional as F

from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
from sglang.srt.layers.moe.topk import select_experts


def fused_moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    inplace: bool = True,
    no_combine: bool = False,
) -> torch.Tensor:
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        correction_bias=correction_bias,
        torch_native=True,
    )

    w13_weights = layer.w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = layer.w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if activation == "silu":
        x1 = F.silu(x1)
    elif activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))

# <NT> moe_forward_native 详解
def moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:

    # <NT> 根据门控网络gate的输出router_logits，选择使用哪些专家
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        correction_bias=correction_bias,
        torch_native=True,
    )

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    len_experts = layer.num_experts

    # <NT> 待细究?
    # topk_ids.shape[0]是num_token, topk_ids.shape[1]是topk中的k，参考grouped_topk笔记
    # 用scatter_沿着维度1(第一个参数)，即列方向，在topk_ids对应位置填1(最后一个参数)，构建一个计数矩阵。
    # cnts，矩阵的每一行对应一个样本，每一列对应一个专家，矩阵元素表示该样本是否选择了对应的专家
    # cnts.sum，沿着第0维累加，行是每个token，列对应专家。0维累加得到每个专家的token数。某一token(行)选中的某个专家(列)，对应元素是1，没选中的是0.
    # view(-1)展开成一维，根据元素大小，得到元素从小到大的排序索引。
    # idxs // topk_ids.shape[1]， idxs的元素个数是topk_ids.shape[0] * topk_ids.shape[1]，除以topk_ids.shape[1]就剩下了行数。
    # 则这一句就是挑出排好序的token，一个token一行。
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    if activation == "silu":
        act = SiluAndMul()
    elif activation == "gelu":
        act = GeluAndMul()
    else:
        raise ValueError(f"Unsupported activation: {activation=}")

    outputs = []
    start_idx = 0
    # <NT> tokens_per_expert是一维列表，i是列表下标，也对应专家id；num_tokens是列表下标i对应的元素，表示该专家要负责的tokens数量。
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # <NT> 选出对应专家负责的tokens和对应专家的权重
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = layer.w13_weight[i]
        layer_w2_weight = layer.w2_weight[i]

        # <NT> w1和w3是充当一个线性层，共同用于gate_up_proj。因为是两块合并计算，用TP时应选用MergedColumnParallel。具体看MergedColumnParallel定义解析。
        gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)
        gate_up = act(gate_up)
        expert_out = F.linear(gate_up, layer_w2_weight)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out
