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

# <NT> moe_forward_native ���
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

    # <NT> �����ſ�����gate�����router_logits��ѡ��ʹ����Щר��
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

    # <NT> ��ϸ��?
    # topk_ids.shape[0]��num_token, topk_ids.shape[1]��topk�е�k���ο�grouped_topk�ʼ�
    # ��scatter_����ά��1(��һ������)�����з�����topk_ids��Ӧλ����1(���һ������)������һ����������
    # cnts�������ÿһ�ж�Ӧһ��������ÿһ�ж�Ӧһ��ר�ң�����Ԫ�ر�ʾ�������Ƿ�ѡ���˶�Ӧ��ר��
    # cnts.sum�����ŵ�0ά�ۼӣ�����ÿ��token���ж�Ӧר�ҡ�0ά�ۼӵõ�ÿ��ר�ҵ�token����ĳһtoken(��)ѡ�е�ĳ��ר��(��)����ӦԪ����1��ûѡ�е���0.
    # view(-1)չ����һά������Ԫ�ش�С���õ�Ԫ�ش�С���������������
    # idxs // topk_ids.shape[1]�� idxs��Ԫ�ظ�����topk_ids.shape[0] * topk_ids.shape[1]������topk_ids.shape[1]��ʣ����������
    # ����һ����������ź����token��һ��tokenһ�С�
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
    # <NT> tokens_per_expert��һά�б�i���б��±꣬Ҳ��Ӧר��id��num_tokens���б��±�i��Ӧ��Ԫ�أ���ʾ��ר��Ҫ�����tokens������
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # <NT> ѡ����Ӧר�Ҹ����tokens�Ͷ�Ӧר�ҵ�Ȩ��
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = layer.w13_weight[i]
        layer_w2_weight = layer.w2_weight[i]

        # <NT> w1��w3�ǳ䵱һ�����Բ㣬��ͬ����gate_up_proj����Ϊ������ϲ����㣬��TPʱӦѡ��MergedColumnParallel�����忴MergedColumnParallel���������
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
