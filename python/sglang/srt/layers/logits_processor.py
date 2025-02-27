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
"""Logits processing."""

import dataclasses
import logging
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[torch.Tensor] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # The logprobs of the next tokens.                              shape: [#seq]
    next_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: torch.Tensor = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: List = None
    input_top_logprobs_idx: List = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_extend() and forward_batch.return_logprob:
            extend_return_logprob = True
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_logprob_pruned_lens_cpu = [
                extend_len - start_len
                for extend_len, start_len in zip(
                    forward_batch.extend_seq_lens_cpu,
                    forward_batch.extend_logprob_start_lens_cpu,
                )
            ]
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_logprob_pruned_lens_cpu
            ) = False

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
        )

# <NT> 对模型输出的对数（logits）进行后处理，
# 对数（logits）是模型在输出层经过线性变换后得到的原始分数，尚未经过 Softmax 函数转换为概率分布。
# LogitsProcessor 的作用就是在将这些 logits 转换为概率之前，对其进行修改和调整，以实现特定的文本生成控制目标。
# 主要功能，如：
#   约束生成结果：通过对 logits 进行处理，可以限制模型生成的文本范围，例如避免生成一些不合法、不恰当或者不符合任务要求的词汇。
#   控制生成风格：可以根据不同的需求调整生成文本的风格，如生成更保守、多样化或者更具创造性的文本。
#   提高生成质量：通过过滤掉不合理的词汇或者调整词汇的得分，提高生成文本的质量和连贯性。
# 常见类型及实现方式：
#   * 重复词惩罚处理器（RepetitionPenaltyLogitsProcessor）
#     对已经出现过的词汇的 logits 进行惩罚，通常是将其得分乘以一个小于 1 的惩罚因子。避免生成重复的内容，使生成的文本更加多样化。
#   * 最大长度处理器（MaxLengthLogitsProcessor）
#     检查当前生成的文本长度，如果达到最大长度，修改 logits 的值。
#   * 词汇过滤处理器（ForcedBOSTokenLogitsProcessor、ForcedEOSTokenLogitsProcessor 等）
#     将指定词汇的 logits 设置为一个非常大的值，同时将其他词汇的 logits 设置为一个相对较小的值，使得指定词汇在经过 Softmax 后具有极高的概率被选中
class LogitsProcessor(nn.Module):
    def __init__(
        self, config, skip_all_gather: bool = False, logit_scale: Optional[float] = None
    ):
        super().__init__()
        self.config = config
        self.logit_scale = logit_scale
        self.do_tensor_parallel_all_gather = (
            not skip_all_gather and get_tensor_model_parallel_world_size() > 1
        )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
    ) -> LogitsProcessorOutput:
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

        # Get the last hidden states and last logits for the next token prediction
        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
        ):
            pruned_states = hidden_states
            sample_indices = None
        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
            sample_indices = None
        else:
            # Slice the requested tokens to compute logprob
            sample_index_pt = -1
            sample_indices = []
            pt, pruned_states, pruned_input_ids = 0, [], []
            for start_len, extend_len in zip(
                logits_metadata.extend_logprob_start_lens_cpu,
                logits_metadata.extend_seq_lens_cpu,
            ):
                pruned_states.append(hidden_states[pt + start_len : pt + extend_len])
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                pruned_input_ids.append(input_ids[pt + start_len : pt + extend_len])
                pt += extend_len

            pruned_states = torch.cat(pruned_states)

        # Compute logits for both input and sampled tokens.
        logits = self._get_logits(pruned_states, lm_head, logits_metadata)
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )

        if (
            not logits_metadata.extend_return_logprob
            or logits_metadata.capture_hidden_mode.need_capture()
        ):
            # Decode mode or extend mode without return_logprob.
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=(
                    hidden_states
                    if logits_metadata.capture_hidden_mode.is_full()
                    else (
                        pruned_states
                        if logits_metadata.capture_hidden_mode.is_last()
                        else None
                    )
                ),
            )
        else:
            input_logprobs = logits
            del hidden_states, logits

            # Normalize the logprob w/o temperature, top-p
            input_logprobs = self.compute_temp_top_p_normalized_logprobs(
                input_logprobs, logits_metadata
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                (
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                ) = self.get_top_logprobs(input_logprobs, logits_metadata)
            else:
                input_top_logprobs_val = input_top_logprobs_idx = None

            input_token_logprobs = input_logprobs[
                torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
                torch.cat(
                    [
                        torch.cat(pruned_input_ids)[1:],
                        torch.tensor([0], device=input_logprobs.device),
                    ]
                ),
            ]

            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
            )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states."""

        if hasattr(lm_head, "weight"):
            logits = torch.matmul(hidden_states, lm_head.weight.T)
        else:
            # GGUF models
            logits = lm_head.linear_method.apply(lm_head, hidden_states, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self.do_tensor_parallel_all_gather:
            logits = tensor_model_parallel_all_gather(logits)

        logits = logits[:, : self.config.vocab_size].float()

        if self.final_logit_softcapping:
            fused_softcap(logits, self.final_logit_softcapping)

        return logits

    @staticmethod
    def get_top_logprobs(all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()

        input_top_logprobs_val, input_top_logprobs_idx = [], []

        pt = 0
        for k, pruned_len in zip(
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            input_top_logprobs_val.append(
                [values[pt + j][:k] for j in range(pruned_len - 1)]
            )
            input_top_logprobs_idx.append(
                [indices[pt + j][:k] for j in range(pruned_len - 1)]
            )
            pt += pruned_len

        return input_top_logprobs_val, input_top_logprobs_idx

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        # TODO: Implement the temp and top-p normalization
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value

    # Manual tanh implementation using exp
    exp2x = tl.exp(2 * x)
    x = (exp2x - 1) / (exp2x + 1)

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits
