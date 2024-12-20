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
"""Constrained decoding with xgrammar backend."""

# Jump forward逻辑 - https://lmsys.org/blog/2024-02-05-compressed-fsm/
# v0.3采用的是压缩FSM，搭配outlines使用，优化关键点在于压缩FSM。
# 压缩前，运行过程中fsm的每个状态都可以计算出允许的转换并确定可接受的下一个token，限制decode的输出范围。缺点在于每次转换一个token，对应一次解码。
# 压缩，是在计算过程中，会把当前点FSM里的奇异转移边(即非常规部分，如喊了ab后面肯定接c没有其他可能)都检测出来，
#        并把相连续的部分直接打包成一个奇异路经，直接整体extend进去。而不需要再一个个decode（extend kernel的效率会比decode高很多，也省去了很多琐碎操作）

# v0.4中集成的陈天奇团队24年11月新推的xgrammar，里面有find_jump_forward_string的api，实现类似的功能，与该优化可无缝衔接。
    
# 具体调用逻辑：
# 1）在更新用于解码的batch数据(update_running_batch)时，遍历该batch内所有seq，由grammar去查找可jump的部分（check_for_jump_forward）。
# 2）如果某个seq能找到可jump的部分，将该seq当作已完成的seq，留存对应cache到基数树上，并将其从计算batch中移除出去，重新放回到waiting_queue。
# 3）在下一轮读取时，从wait_queue中重新把这个seq取出来放回到计算batch中，按新的extend请求来处理。

import logging
from typing import List, Tuple

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)

logger = logging.getLogger(__name__)


MAX_ROLLBACK_TOKENS = 200


class XGrammarGrammar(BaseGrammarObject):

    def __init__(
        self, matcher: GrammarMatcher, vocab_size: int, ctx: CompiledGrammar
    ) -> None:
        self.matcher = matcher # 将LLM的输出与指定的语法进行匹配，然后为下一个词元token生成掩码
        self.vocab_size = vocab_size
        self.ctx = ctx # CompiledGrammar可用于GrammarMatcher，以便高效地生成token掩码

    def accept_token(self, token: int):
        assert self.matcher.accept_token(token) # 接收一个token，并更新matcher的状态

    def try_jump_forward(self, tokenizer) -> Tuple[List[int], str]:
        # 查找用于jump-forward decoding的jump-forward符串。
        # 这是从当前matcher状态来看，肯定符合当前语法的最长字符串。
        # 该字符串可以成为LLM的输出，而无需大型语言模型进行解码。
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        # 将matcher回滚到之前的某个状态，回滚的幅度为(len(old_output_ids) - k)个token
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    # 为下一token预测分配bitmask。该bitmask是位于CPU上的一个shape为(batch_size, ceil(vocab_size / 32)) 的 32 位整型张量。
    # 有自行管理 CUDA 内存需求的用户可以自行使用 get_bitmask_shape 和 bitmask_dtype 来构建该张量。
    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    # 为下一token预测填充bitmask。输入的bitmask可通过 allocate_token_bitmask 函数生成，并且必须位于CPU上。
    # bitmask[index] 将用下一token的bitmask进行填充。
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    # 就地inplace将位掩码bitmask应用于对数几率logits
    # bitmask是一个按位压缩的 01 张量，其中 0 表示相应词元被屏蔽，1 表示相应词元未被屏蔽。
    # 它可以通过 allocate_token_bitmask 函数生成，并通过 fill_next_token_bitmask 函数进行填充。
    # 应用了bitmask后，被屏蔽的logits将被设置为-inf
    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if vocab_mask.device.type != logits.device.type:
            # vocab_mask must then be on the same device as logits
            # when applying the token bitmask, so we check and move if needed
            vocab_mask = vocab_mask.to(logits.device)

        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        matcher = GrammarMatcher(self.ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, self.ctx)


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        tokenizer_info = TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size

    # 有json和regex正则表达式两个模式，但xgrammar只支持json，outlines可支持json和regex
    def init_value_impl(self, key: Tuple[str, str]) -> XGrammarGrammar:

        key_type, key_string = key
        if key_type == "json":
            try:
                ctx = self.grammar_compiler.compile_json_schema(schema=key_string)
            except RuntimeError as e:
                logging.warning(
                    f"Skip invalid json_schema: json_schema={key_string}, {e=}"
                )
                return None
        elif key_type == "regex":
            logger.warning(
                "regex hasn't been supported by xgrammar yet. This is skipped."
            )
            return None
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS) # 通过 CompiledGrammar 创建 matcher
        return XGrammarGrammar(matcher, self.vocab_size, ctx)

    # 清除所有缓存的已编译语法, 会使用到场合通常有以下几种:
    # 1. 在模型切换或重置时
    # 2. 内存资源紧张时。缓存的语法较多，且很多短期可能用不上，需要时再重新编译对应所需的语法
    # 3. 出现语法相关错误或异常时.
    def reset(self):
        if self.grammar_compiler:
            self.grammar_compiler.clear_cache()
