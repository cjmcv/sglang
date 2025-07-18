# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from sglang.srt.model_executor.model_runner import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1


@dataclass
class EmbeddingPoolerOutput:
    embeddings: torch.Tensor

# <NT> 一个从隐藏状态中汇集特定信息的层。
# （在qwen2模型中有使用，跟LogitsProcessor放在同一级别位置。
#   python/sglang/srt/models/qwen2.py#357, 如果需要返回embedding数据，则调用它，否则调用LogitsProcessor）
# 该层执行以下操作：
#     根据pool方法提取特定标记或聚合数据。
#     若有指定，则对输出进行归一化处理。
#     将结构化结果以PoolerOutput的形式返回。
class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:
        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        elif self.pooling_type == PoolingType.CLS:
            prompt_lens = forward_batch.extend_seq_lens
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)
