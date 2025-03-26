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

# <NT> Jump forward�߼� - https://lmsys.org/blog/2024-02-05-compressed-fsm/
# v0.3���õ���ѹ��FSM������outlinesʹ�ã��Ż��ؼ�������ѹ��FSM��
# ѹ��ǰ�����й�����fsm��ÿ��״̬�����Լ���������ת����ȷ���ɽ��ܵ���һ��token������decode�������Χ��ȱ������ÿ��ת��һ��token����Ӧһ�ν��롣
# ѹ�������ڼ�������У���ѵ�ǰ��FSM�������ת�Ʊ�(���ǳ��沿�֣��纰��ab����϶���cû����������)����������
#        �����������Ĳ���ֱ�Ӵ����һ������·����ֱ������extend��ȥ��������Ҫ��һ����decode��extend kernel��Ч�ʻ��decode�ߺܶ࣬Ҳʡȥ�˺ܶ����������

# v0.4�м��ɵĳ������Ŷ�24��11�����Ƶ�xgrammar��������find_jump_forward_string��api��ʵ�����ƵĹ��ܣ�����Ż����޷��νӡ�
    
# ��������߼���
# 1���ڸ������ڽ����batch����(update_running_batch)ʱ��������batch������seq����grammarȥ���ҿ�jump�Ĳ��֣�check_for_jump_forward����
# 2�����ĳ��seq���ҵ���jump�Ĳ��֣�����seq��������ɵ�seq�������Ӧcache���������ϣ�������Ӽ���batch���Ƴ���ȥ�����·Żص�waiting_queue��
# 3������һ�ֶ�ȡʱ����wait_queue�����°����seqȡ�����Żص�����batch�У����µ�extend����������

# ע�� grammar���(outlines/xgrammar)�ǳ�ʼ����ʱ��ָ���ģ���grammarʵ������seqһһ�󶨣������Ҫ������seqʱָ��ģ��Ż���Ч����jump-forward����grammar��ģ����Ҳֻ����û��ṩ��ģ�塣
# grammarʹ�÷�ʽ��client.chat.completions.create ��ָ�� response_format��

import json
import logging
from typing import List, Optional, Tuple, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTagItem,
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
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.finished = False

    # <NT> ����һ��token��������matcher��״̬�����յ�token��prefill��decode��ȷ����token��
    # ��grammar��decode��Լ������BaseGrammarBackend.get_future_value�У���׼�����������batchʱ�����ݵ�ǰ״̬��ӽ�ȥ�ģ�����TODO
    def accept_token(self, token: int):
        assert self.matcher.accept_token(token)

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        # <NT> ��������jump-forward decoding��jump-forward������
        # ���Ǵӵ�ǰmatcher״̬�������϶����ϵ�ǰ�﷨����ַ�����
        # ���ַ������Գ�ΪLLM��������������������ģ�ͽ��н��롣
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
        # ��matcher�ع���֮ǰ��ĳ��״̬���ع��ķ���Ϊ(len(old_output_ids) - k)��token
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    # <NT> Ϊ��һtokenԤ�����bitmask����bitmask��λ��CPU�ϵ�һ��shapeΪ(batch_size, ceil(vocab_size / 32)) �� 32 λ����������
    # �����й��� CUDA �ڴ�������û���������ʹ�� get_bitmask_shape �� bitmask_dtype ��������������
    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    # <NT> Ϊ��һtokenԤ�����bitmask�������bitmask��ͨ�� allocate_token_bitmask �������ɣ����ұ���λ��CPU�ϡ�
    # bitmask[index] ������һtoken��bitmask������䡣
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    # <NT> �͵�inplace��λ����bitmaskӦ���ڶ�������logits
    # bitmask��һ����λѹ���� 01 ���������� 0 ��ʾ��Ӧ��Ԫ�����Σ�1 ��ʾ��Ӧ��Ԫδ�����Ρ�
    # ������ͨ�� allocate_token_bitmask �������ɣ���ͨ�� fill_next_token_bitmask ����������䡣
    # Ӧ����bitmask�󣬱����ε�logits��������Ϊ-inf
    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher, self.vocab_size, self.ctx, self.override_stop_tokens
        )


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
        override_stop_tokens = None

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens

    def _from_context(self, ctx: CompiledGrammar) -> XGrammarGrammar:
        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, ctx, self.override_stop_tokens)

    def dispatch_json(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            if key_string == "$$ANY$$":
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(schema=key_string)
        except RuntimeError as e:
            logging.warning(f"Skip invalid json_schema: json_schema={key_string}, {e=}")
            return None
        return self._from_context(ctx)

    def dispatch_ebnf(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_grammar(key_string)
        except RuntimeError as e:
            logging.warning(f"Skip invalid ebnf: ebnf={key_string}, {e=}")
            return None
        return self._from_context(ctx)

    def dispatch_regex(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            ctx = self.grammar_compiler.compile_regex(key_string)
        except RuntimeError as e:
            logging.warning(f"Skip invalid regex: regex={key_string}, {e=}")
            return None
        return self._from_context(ctx)

    def dispatch_structural_tag(self, key_string: str) -> Optional[XGrammarGrammar]:
        try:
            structural_tag = json.loads(key_string)
            tags = [
                StructuralTagItem(
                    begin=structure["begin"],
                    schema=json.dumps(structure["schema"]),
                    end=structure["end"],
                )
                for structure in structural_tag["structures"]
            ]
            ctx = self.grammar_compiler.compile_structural_tag(
                tags, structural_tag["triggers"]
            )
        except RuntimeError as e:
            logging.warning(f"Skip invalid regex: regex={key_string}, {e=}")
            return None
        return self._from_context(ctx)

    # <NT> ������л�����ѱ����﷨, ��ʹ�õ�����ͨ�������¼���:
    # 1. ��ģ���л�������ʱ
    # 2. �ڴ���Դ����ʱ��������﷨�϶࣬�Һܶ���ڿ����ò��ϣ���Ҫʱ�����±����Ӧ������﷨
    # 3. �����﷨��ش�����쳣ʱ.
    def reset(self):
        if self.grammar_compiler:
            self.grammar_compiler.clear_cache()
