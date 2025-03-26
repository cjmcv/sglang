from __future__ import annotations

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
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

# <NT> ScheduleBatch: 由Scheduler管理，包含高层级的调度数据，这些数据主要在cpu上。
#      ModelWorkerBatch: 由TpModelWorker管理，是ScheduleBatch的子集，仅包含模型在gpu上推理相关的数据。
#      ForwardBatch: 由ModelRunner管理，包含推理实际推理使用的张量数据。

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.speculative.spec_info import SpecInfo, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Put some global args for easy access
global_server_args_dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "disable_mla": ServerArgs.disable_mla,
    "torchao_config": ServerArgs.torchao_config,
    "enable_nan_detection": ServerArgs.enable_nan_detection,
    "enable_dp_attention": ServerArgs.enable_dp_attention,
    "enable_ep_moe": ServerArgs.enable_ep_moe,
    "device": ServerArgs.device,
    "enable_flashinfer_mla": ServerArgs.enable_flashinfer_mla,
    "disable_radix_cache": ServerArgs.disable_radix_cache,
}

logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message="Unknown error", status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


@dataclasses.dataclass
class ImageInputs:
    """The image related inputs."""

    pixel_values: Union[torch.Tensor, np.array]
    image_hashes: Optional[list] = None
    image_sizes: Optional[list] = None
    image_offsets: Optional[list] = None
    image_pad_len: Optional[list] = None
    pad_values: Optional[list] = None
    modalities: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # Llava related
    aspect_ratio_ids: Optional[List[torch.Tensor]] = None
    aspect_ratio_mask: Optional[List[torch.Tensor]] = None

    # QWen2-VL related
    image_grid_thws: List[Tuple[int, int, int]] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # MiniCPMV related
    # All the images in the batch should share the same special image
    # bound token ids.
    im_start_id: Optional[torch.Tensor] = None
    im_end_id: Optional[torch.Tensor] = None
    slice_start_id: Optional[torch.Tensor] = None
    slice_end_id: Optional[torch.Tensor] = None
    tgt_sizes: Optional[list] = None

    @staticmethod
    def from_dict(obj: dict):
        ret = ImageInputs(
            pixel_values=obj["pixel_values"],
            image_hashes=obj["image_hashes"],
        )

        # Use image hash as fake token_ids. We use this as the key for prefix matching in the radix cache.
        # Please note that if the `input_ids` is later used in the model forward,
        # you also need to clamp the values within the range of [0, vocab_size) to avoid out-of-bound
        # errors in cuda kernels. See also llava.py for example.
        ret.pad_values = [x % (1 << 30) for x in ret.image_hashes]

        optional_args = [
            "image_sizes",
            "modalities",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
            "im_start_id",
            "im_end_id",
            "slice_start_id",
            "slice_end_id",
            "tgt_sizes",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret

    def merge(self, other):
        assert self.pixel_values.shape[1:] == other.pixel_values.shape[1:]
        self.pixel_values = np.concatenate([self.pixel_values, other.pixel_values])

        # Use image hash as fake token_ids. We use this as the key for prefix matching in the radix cache.
        # Please note that if the `input_ids` is later used in the model forward,
        # you also need to clamp the values within the range of [0, vocab_size) to avoid out-of-bound
        # errors in cuda kernels. See also llava.py for example.
        self.image_hashes += other.image_hashes
        self.pad_values = [x % (1 << 30) for x in self.image_hashes]

        optional_args = [
            "image_sizes",
            "image_offsets",
            "image_pad_len",
            # "modalities", # modalities should be ["multi-images"] (one entry) even for multiple images
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
        ]
        for arg in optional_args:
            if getattr(self, arg, None) is not None:
                setattr(self, arg, getattr(self, arg) + getattr(other, arg))


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: Tuple[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_path: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        session_id: Optional[str] = None,
        custom_logit_processor: Optional[str] = None,
        eos_token_ids: Optional[Set[int]] = None,
    ):
        # Input and output info
        self.rid = rid   # <NT> Req id，单数据整型
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        # <NT> promts的token ids
        self.origin_input_ids = origin_input_ids
        # Each decode stage's output ids
        # <NT> 每个decode阶段生成的输出token id，在prefill/extend阶段为空，每次处理batch计算结果时，
        #      会将当前的next_token_id append进去，即会保留所有decode结果。
        #      注：ScheduleBatch里的output_ids仅保存当前batch的输出结果，即也就是下一batch要计算的next_token_id。
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        # <NT> 被填充了的token_id, 包括原始prompts的输入token(self.origin_input_ids)和后面decode已生成的token(self.output_ids)，
        #      最后一段会是当前batch计算的输出，也是下一个batch要计算的输入next_token_id
        self.fill_ids = None
        self.session_id = session_id
        self.input_embeds = input_embeds

        # Sampling info
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor

        # Memory pool info
        self.req_pool_idx = None

        # Check finish
        self.tokenizer = None
        self.finished_reason = None
        self.to_abort = False
        self.stream = stream
        self.eos_token_ids = eos_token_ids

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.vid = 0  # version id to sync decode status with in detokenizer_manager
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.image_inputs: Optional[ImageInputs] = None

        # Prefix info
        self.prefix_indices = []
        # Tokens to run prefill. input_tokens - shared_prefix_tokens.
        # Updated if chunked.
        self.extend_input_len = 0
        self.last_node = None

        # Chunked prefill
        # <NT> 表示该req被chunked的数量，当其小于等于0时，表示该req的prefill/extend阶段已完成。
        #      prefile/extend阶段，每次batch推理会完成一次的chunk计算，is_being_chunked会减1.
        self.is_being_chunked = 0

        # For retraction
        self.is_retracted = False

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num

        # Logprobs (return values)
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = None
        self.hidden_states = []

        # Logprobs (internal values)
        # The tokens is prefilled but need to be considered as decode tokens
        # and should be updated for the decode logprobs
        self.last_update_decode_tokens = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar: Optional[BaseGrammarObject] = None

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0
        self.lora_path = lora_path

    def extend_image_inputs(self, image_inputs):
        if self.image_inputs is None:
            self.image_inputs = image_inputs
        else:
            self.image_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    # <NT> 只在Schedule.get_new_batch_prefill中调用, 会对self.prefix_indices赋值，表示前缀token在kvcache_pool中的位置。
    # self.rid是req的id号，在chunk cache中充当索引找到对应的seq的前缀。
    # 而在radix cache中只需要用到key。
    #
    # origin_input_ids：一个req的prompts的token ids，对于每个req自创建后，一直不会被修改
    #
    # output_ids：有两个同名变量，Req.output_ids和ScheduleBatch.output_ids. 
    # ScheduleBatch.output_ids 对应的是上一轮计算输出的next_token_ids，会在准备新一轮batch计算时充当输入。
    #                          在Scheduler.run_batch中，一个batch计算完后输出next_token_ids, 
    #                          会执行batch.output_ids = next_token_ids，传给ScheduleBatch，用于充当下一轮计算的输入。
    # Req.output_ids在Scheduler.process_batch_result_prefill和Scheduler.process_batch_result_decode中被调用，每次batch推理后会处理batch_result,
    #               将输出的next_token_id都append到output_ids里，所以req的output_ids会存放该req的到当前阶段之前所有decode输出，
    #               以及当前阶段的decode输出，即下一轮推理要用到的next_token_id。
    #               Req.output_ids仅针对decode阶段，prefill阶段一般都为0.
    #
    # 实际调试数据：
    # fill_ids=  49, output_ids=0, prefix_indices=1, extend_input_len=48
    # ModelRunner.forward extend
    # ModelRunner.cuda_graph_runner.replay
    # fill_ids=8926, output_ids=0, prefix_indices=48, extend_input_len=8878
    # ModelRunner.forward extend
    # fill_ids=8926, output_ids=0, prefix_indices=2096, extend_input_len=6830
    # fill_ids=57, output_ids=0, prefix_indices=48, extend_input_len=9
    # ModelRunner.forward extend
    # fill_ids=8926, output_ids=0, prefix_indices=4144, extend_input_len=4782
    # fill_ids=57, output_ids=0, prefix_indices=48, extend_input_len=9
    # ModelRunner.forward extend
    # fill_ids=8926, output_ids=0, prefix_indices=6192, extend_input_len=2734
    # fill_ids=57, output_ids=0, prefix_indices=48, extend_input_len=9
    # ModelRunner.forward extend
    # fill_ids=8926, output_ids=0, prefix_indices=8240, extend_input_len=686
    # fill_ids=57, output_ids=0, prefix_indices=48, extend_input_len=9
    # ModelRunner.forward extend
    # ModelRunner.cuda_graph_runner.replay 。。。
    # 
    # 观察上面数据可以看到，当出现超长文本时，每次进入这里origin_input_ids都是不会变的，就是prompts的总长度，
    # 因为在这里的都是处理prompt部分，不包含解码部分而output_ids一直为零（这函数为什么要使用output_ids，原因不明确，可能跟全局使用fill_ids的风格有关）。
    #
    # self.prefix_indices是匹配到的前缀token在kvcache的位置，除了这里调用了tree_cache.match_prefix之外。
    # 主要的更新地方是：Scheduler.get_new_batch_prefill -> SchedulePolicy.calc_priority -> _compute_prefix_matches -> self.tree_cache.match_prefix.
    #
    def init_next_round_input(self, tree_cache: Optional[BasePrefixCache] = None):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            # tree cache is None if the prefix is not computed with tree cache.
            self.prefix_indices, self.last_node = tree_cache.match_prefix(
                rid=self.rid, key=self.adjust_max_prefix_ids()
            )
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    # <NT> 最长前缀的token id集
    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def get_next_inc_detokenization(self):
        if self.tokenizer is None:
            return False, ""
        read_ids, read_offset = self.init_incremental_detokenize()
        surr_ids = read_ids[:read_offset]

        surr_text = self.tokenizer.decode(
            surr_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )
        new_text = self.tokenizer.decode(
            read_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )

        if len(new_text) > len(surr_text) and not new_text.endswith("�"):
            return True, new_text[len(surr_text) :]

        return False, ""

    def check_finished(self):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT()
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= last_token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= last_token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= (
                        last_token_id in self.tokenizer.additional_stop_token_ids
                    )
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def jump_forward_and_retokenize(self, jump_forward_str, next_state):
        if self.origin_input_text is None:
            # Recovering text can only use unpadded ids
            self.origin_input_text = self.tokenizer.decode(
                self.origin_input_ids_unpadded
            )

        all_text = self.origin_input_text + self.decoded_text + jump_forward_str
        all_ids = self.tokenizer.encode(all_text)
        if not all_ids:
            logger.warning("Encoded all_text resulted in empty all_ids")
            return False

        prompt_tokens = len(self.origin_input_ids_unpadded)
        if prompt_tokens > len(all_ids):
            logger.warning("prompt_tokens is larger than encoded all_ids")
            return False

        if all_ids[prompt_tokens - 1] != self.origin_input_ids_unpadded[-1]:
            # TODO(lsyin): fix token fusion
            logger.warning(
                "Token fusion between input and output, try to avoid this by removing the space at the end of the input."
            )
            return False

        old_output_ids = self.output_ids
        self.output_ids = all_ids[prompt_tokens:]
        self.decoded_text = self.decoded_text + jump_forward_str
        self.surr_offset = prompt_tokens
        self.read_offset = len(all_ids)

        # NOTE: A trick to reduce the surrouding tokens decoding overhead
        for i in range(0, INIT_INCREMENTAL_DETOKENIZATION_OFFSET):
            surr_text_ = self.tokenizer.decode(
                all_ids[self.read_offset - i : self.read_offset]
            )
            if not surr_text_.endswith("�"):
                self.surr_offset = self.read_offset - i
                break

        # update the inner state of the grammar
        self.grammar.jump_and_retokenize(old_output_ids, self.output_ids, next_state)

        if self.return_logprob:
            # For fast-forward part's logprobs
            k = 0
            for i, old_id in enumerate(old_output_ids):
                if old_id == self.output_ids[i]:
                    k = k + 1
                else:
                    break
            self.output_token_logprobs_val = self.output_token_logprobs_val[:k]
            self.output_token_logprobs_idx = self.output_token_logprobs_idx[:k]
            self.output_top_logprobs_val = self.output_top_logprobs_val[:k]
            self.output_top_logprobs_idx = self.output_top_logprobs_idx[:k]
            self.logprob_start_len = prompt_tokens + k
            self.last_update_decode_tokens = len(self.output_ids) - k

        return True

    def reset_for_retract(self):
        self.prefix_indices = []
        self.last_node = None
        self.extend_input_len = 0
        self.is_retracted = True

        # For incremental logprobs
        # TODO: Fix the `logprob_start_len`
        self.last_update_decode_tokens = 0
        self.logprob_start_len = 10**9

    def __repr__(self):
        return (
            f"rid(n={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}"
        )


bid = 0


@dataclasses.dataclass
class ScheduleBatch:
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int32
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    req_pool_indices: torch.Tensor = None  # shape: [b], int32
    seq_lens: torch.Tensor = None  # shape: [b], int64
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int32
    output_ids: torch.Tensor = None  # shape: [b], int32

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    can_run_dp_cuda_graph: bool = False

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: int = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None

    # For encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[SpecInfo] = None

    # Enable custom logit processor
    enable_custom_logit_processor: bool = False

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: ReqToTokenPool,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=any(req.return_logprob for req in reqs),
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            enable_custom_logit_processor=enable_custom_logit_processor,
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    # <NT> 直接调用ReqToTokenPool.alloc，为每个req申请空槽位
    def alloc_req_slots(self, num_reqs: int):
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "Out of memory. "
                "Please set a smaller number for `--max-running-requests`."
            )
        return req_pool_indices

    # <NT> 主要调用BaseTokenToKVPool.alloc, 为需要extend的tokens，申请空的槽位。
    # 槽位总数量是kvcache最大可容纳的token数(根据load model后剩余显存计算得出)
    # 如果alloc返回None，表示空闲槽位不足，需要结合radix cache或chunk cache去执行淘汰策略。
    # 腾出足够位置给新的token。
    # 问题：为什么进入到该函数的token都需要分配新槽位，而不会是以前就存在的token。
    # 答：因为prepare_for_extend中extend_num_tokens是去掉了r.prefix_indices
    #     (通过radix/chunk cache检索得出，radix/chunk cache是token_to_kv_pool内存额外构建用于管理前缀的索引)，
    #     即表示这部分在kvcache中并没有找到。所以都属于新的token，需要新的槽位。
    def alloc_token_slots(self, num_tokens: int):
        out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)

        if out_cache_loc is None:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens, self.token_to_kv_pool.free)
                out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)

            if out_cache_loc is None:
                phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
                logger.error(
                    f"{phase_str} out of memory. Try to lower your batch size.\n"
                    f"Try to allocate {num_tokens} tokens.\n"
                    f"Avaliable tokens: {self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()}\n"
                )
                if self.tree_cache is not None:
                    self.tree_cache.pretty_print()
                exit(1)

        return out_cache_loc

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.image_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(self.encoder_lens_cpu, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len

        # Reassign
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int32).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int32).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert len(self.out_cache_loc) == self.extend_num_tokens

    # <NT-TODO> ScheduleBatch.prepare_for_extend，在Scheduler.get_new_batch_prefill中被调用, 
    #           该函数是准备一个全新的prefill batch，所有里面的req全都是新进入的。
    # 功能包括：1. 为每个req填充req_to_token_pool，建立req与其token_id的映射位置。
    #          2. 根据涵盖的所有req的信息，填充self的变量信息，
    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        bs = len(self.reqs)
        reqs = self.reqs
        # <NT> len(r.prefix_indices)即该请求r的前缀数量，总数fill_ids去掉前缀部分，剩下的就是extend的(输入)大小，
        # 因为fill_ids里包含有output_ids, 里面有包含上一batch计算得到的next_token_ids, 用于充当当前batch的输入。
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        # <NT> 将所有req的extend大小都累加起来，就是本次处理的extend总token数，可用于内存分配。
        extend_num_tokens = sum(len(ids) for ids in input_ids) 
        seq_lens = []
        pre_lens = []

        # Allocate memory
        # <NT> req_pool_indices：因为这个函数是准备一个全新的prefill batch，所以里面的req都要在req_to_token_pool中申请新的槽位, 用于从req映射到token
        #      out_cache_loc: 按上面计算得到的当前轮次需要extend的token数量，在token_to_kv_pool中申请的空间位置.
        req_pool_indices = self.alloc_req_slots(bs)
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)

        input_embeds = []

        pt = 0
        for i, req in enumerate(reqs):
            req.req_pool_idx = req_pool_indices[i]
            # <NT> prefix_indices是前缀token在token_to_kv_pool中的索引；
            # fill_ids是包含prompts和decode到当前阶段的所有decode输出token_id集合，如果是prefill/extend阶段，那seq_len就是prompts的token_id.
            # seq_len - pre_len 就是 prompts部分减去被缓存了的部分，剩下的就是还需要做extend的部分。
            # <NT-TODO>: 如果超长，会分段处理，分段处理的位置在哪里？
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids) 
            seq_lens.append(seq_len)                                      # seq_lens的每个元素表示该请求所对应的序列的长度。
            assert seq_len - pre_len == req.extend_input_len              # 二次核验: 序列长度 - 前缀长度 = extend长度

            # <NT> 如果该req的prefix_indices有内容，则将其按申请到的空位req_pool_idx（行）写入到req_to_token_pool的0-pre_len列中。
            #      因为是新req，从0开始写即可。
            if pre_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # If req.input_embeds is already a list, append its content directly
                input_embeds.extend(req.input_embeds)  # Use extend to avoid nesting

            if req.return_logprob:
                # Compute the relative logprob_start_len in an extend batch
                if req.logprob_start_len >= pre_len:
                    extend_logprob_start_len = min(
                        req.logprob_start_len - pre_len, req.extend_input_len - 1
                    )
                else:
                    raise RuntimeError(
                        f"This should never happen. {req.logprob_start_len=}, {pre_len=}"
                    )
                req.extend_logprob_start_len = extend_logprob_start_len

            # <NT-TODO> 预先更新，实际上还没有开始算？？？？？？？
            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)

        # Set fields
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.input_embeds = (
            torch.tensor(input_embeds).to(self.device, non_blocking=True)
            if input_embeds
            else None
        )

        self.out_cache_loc = out_cache_loc

        self.seq_lens_sum = sum(seq_lens)
        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]

        # Write to req_to_token_pool
        pre_lens = torch.tensor(pre_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        extend_lens = torch.tensor(self.extend_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        if global_server_args_dict["attention_backend"] != "torch_native":
            write_req_to_token_pool_triton[(bs,)](
                self.req_to_token_pool.req_to_token,
                self.req_pool_indices,
                pre_lens,
                self.seq_lens,
                extend_lens,
                self.out_cache_loc,
                self.req_to_token_pool.req_to_token.shape[1],
            )
        else:
            pt = 0
            for i in range(bs):
                self.req_to_token_pool.write(
                    (self.req_pool_indices[i], slice(pre_lens[i], self.seq_lens[i])),
                    self.out_cache_loc[pt : pt + self.extend_lens[i]],
                )
                pt += self.extend_lens[i]
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
            enable_overlap_schedule=self.enable_overlap,
        )

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)

    def check_decode_mem(self, buf_multiplier=1):
        bs = len(self.reqs) * buf_multiplier
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        self.tree_cache.evict(bs, self.token_to_kv_pool.free)

        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    # <NT> 当内存不足时，将正在decode的请求收回去，也把对应的cache都清除掉
    def retract_decode(self):
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = [i for i in range(len(self.reqs))]

        # TODO(lsyin): improve retraction policy for radix cache
        # output_ids和origin_input_ids进行排序，output_ids长度从大到小降序排列，如相同时，按origin_input_ids从小到大排。
        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),
                -len(self.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        first_iter = True
        while (
            self.token_to_kv_pool.available_size()
            < len(sorted_indices) * global_config.retract_decode_steps
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.token_to_kv_pool.available_size() > 0
                ), "No space left for only one request"
                break

            first_iter = False
            idx = sorted_indices.pop() # 因是降序排列，pop出来的是列表末尾的 output_ids 长度最小的。
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if isinstance(self.tree_cache, ChunkCache):
                # <NT> ChunkCache没有淘汰机制, req结束时直接释放相关token的kvcache，不会再提供给其他seq进行复用。
                # 详情见ChunkCache.cache_finished_req函数笔记。
                # ChunkCache does not have eviction
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
                del self.tree_cache.entries[req.rid]
            else:
                # <NT> RadixCache有淘汰机制, req结束后，其相关token的kvcache不会马上释放，因此可以提供给其他seq复用，
                # 淘汰采用LRU，节点访问时计时，长时间不访问导致超时时才释放。
                # TODO: apply more fine-grained retraction
                last_uncached_pos = len(req.prefix_indices)
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
                ]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)

                # release the last node
                self.tree_cache.dec_lock_ref(req.last_node)

                # NOTE(lsyin): we should use the newly evictable memory instantly.
                residual_size = (
                    len(sorted_indices) * global_config.retract_decode_steps
                    - self.token_to_kv_pool.available_size()
                )
                residual_size = max(0, residual_size)
                self.tree_cache.evict(residual_size, self.token_to_kv_pool.free)
            req.reset_for_retract()

        # 剔除batch中被sorted_indices.pop出来的部分
        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    # <NT> 
    def check_for_jump_forward(self, pad_input_ids_func):
        jump_forward_reqs = []
        keep_indices = set(i for i in range(len(self.reqs))) # 当前请求数的标号

        # 遍历该batch内的所有seq
        for i, req in enumerate(self.reqs):
            if req.grammar is not None:
                # 查找是否有可跳转的部分
                jump_helper = req.grammar.try_jump_forward(req.tokenizer)
                if jump_helper:
                    suffix_ids, _ = jump_helper  # suffix_ids 即为可跳转的部分，对于outlines是可跳转decode id，对于xgrammar则suffix_ids为空。

                    # Current ids, for cache and revert
                    cur_all_ids = tuple(req.origin_input_ids + req.output_ids)[:-1]
                    cur_output_ids = req.output_ids

                    req.output_ids.extend(suffix_ids) # req.output_ids 是每个decode阶段的输出id【xgrammar忽略】
                    decode_res, new_text = req.get_next_inc_detokenization() # 根据新增跳转部分id后的req.output_ids，去获取new_text【xgrammar忽略】
                    if not decode_res:
                        req.output_ids = cur_output_ids # 获取失败则复原id【xgrammar忽略】
                        continue

                    (
                        jump_forward_str,
                        next_state,
                    ) = req.grammar.jump_forward_str_state(jump_helper)

                    # Make the incrementally decoded text part of jump_forward_str
                    # so that the UTF-8 will not corrupt
                    # 根据jump_str和garmmar状态, 去更新req相关标记数据。
                    jump_forward_str = new_text + jump_forward_str
                    if not req.jump_forward_and_retokenize(
                        jump_forward_str, next_state
                    ):
                        req.output_ids = cur_output_ids
                        continue

                    # The decode status has diverged from detokenizer_manager
                    req.vid += 1

                    # insert the old request into tree_cache
                    self.tree_cache.cache_finished_req(req, cur_all_ids)

                    # re-applying image padding
                    if req.image_inputs is not None:
                        req.origin_input_ids = pad_input_ids_func(
                            req.origin_input_ids_unpadded, req.image_inputs
                        )

                    jump_forward_reqs.append(req)
                    keep_indices.remove(i)
                    
        # 根据keep_indices里没有的下标，清理self.reqs
        self.filter_batch(keep_indices=list(keep_indices))

        return jump_forward_reqs

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int32, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int32, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
            enable_overlap_schedule=self.enable_overlap,
        )

    # <NT> output_ids在Scheduler.run_batch中，会有batch.output_ids = next_token_ids，
    # 拿到上一轮数据的输出token ids，用于充当下一轮计算的输入。
    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        if self.spec_algorithm.is_eagle():
            return

        self.input_ids = self.output_ids
        self.output_ids = None
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(self.input_ids)

        # Alloc mem
        bs = len(self.reqs)
        self.out_cache_loc = self.alloc_token_slots(bs)

        if self.model_config.is_encoder_decoder:
            locs = self.encoder_lens + self.seq_lens
            self.prepare_encoder_info_decode()
        else:
            locs = self.seq_lens

        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.req_to_token_pool.write(
                (self.req_pool_indices, locs), self.out_cache_loc
            )
            self.seq_lens = self.seq_lens + 1
        else:
            # A faster in-place version
            self.req_to_token_pool.write(
                (self.req_pool_indices, locs), self.out_cache_loc
            )
            self.seq_lens.add_(1)
        self.seq_lens_sum += bs

    # <NT> 将self.reqs中keep_indices对应下标元素保留，去掉其他元素；
    # 如果keep_indices未指定，则将self.reqs中已完成或已经被chunked的去掉。
    def filter_batch(
        self,
        being_chunked_req: Optional[Req] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished() and self.reqs[i] is not being_chunked_req
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        new_indices = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.seq_lens = self.seq_lens[new_indices]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[new_indices]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, new_indices)
        if self.spec_info:
            self.spec_info.filter_batch(new_indices)

    def merge_batch(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.concat([self.output_ids, other.output_ids])
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
        self.reqs.extend(other.reqs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    # <NT> 区分batch里是否包含extend数据，如果是，需要提供extend相关信息。
    # ScheduleBatch里面包含的内容太多，不便全部带走，所以单独生成ModelWorkerBatch以供内层调用。
    # ModelWorkerBatch是个很纯粹的类，只有数据没有函数。
    def get_model_worker_batch(self):
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        global bid
        bid += 1
        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            global_num_tokens=self.global_num_tokens,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            image_inputs=[r.image_inputs for r in self.reqs],
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_paths=[req.lora_path for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.sampling_info.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
        )

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            enable_custom_logit_processor=self.enable_custom_logit_processor,
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]

    # For DP attention
    global_num_tokens: Optional[List[int]]
    can_run_dp_cuda_graph: bool

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]

    # For multimodal
    image_inputs: Optional[List[ImageInputs]]

    # For encoder-decoder
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # For LoRA
    lora_paths: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # The input Embeds
    input_embeds: Optional[torch.tensor] = None

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[SpecInfo] = None
    capture_hidden_mode: CaptureHiddenMode = None


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)

    # TODO: optimize this?
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )
