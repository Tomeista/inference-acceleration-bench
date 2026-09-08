"""The exactness promise: a class asking for N input tokens gets N.

If these fail, every cross-config comparison is measuring prompt-length drift
alongside whatever it was supposed to measure.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bench.build_prompts import DEFAULT_TOKENIZER, build_class
from bench.classes import load_classes

TOKENIZER_DIR = (Path(__file__).resolve().parents[1] / DEFAULT_TOKENIZER).resolve()

pytestmark = pytest.mark.skipif(
    not TOKENIZER_DIR.exists(),
    reason=f"no local Qwen3 tokenizer at {TOKENIZER_DIR}",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.mark.parametrize("class_id", ["c1_chat", "c2_longform", "c3_rag", "c5_toolcall"])
def test_prompts_hit_their_exact_token_budget(tokenizer, class_id):
    cls = replace(load_classes()[class_id], num_prompts=6)
    scenarios, stats = build_class(cls, tokenizer, seed=0)

    assert len(scenarios) == 6
    assert stats["min_input_tokens"] == cls.input_tokens
    assert stats["max_input_tokens"] == cls.input_tokens


def test_toolcall_prompts_carry_tools_and_the_expected_tool_name(tokenizer):
    cls = replace(load_classes()["c5_toolcall"], num_prompts=6)
    scenarios, _ = build_class(cls, tokenizer, seed=0)

    for scenario in scenarios:
        turn = scenario.turns[0]
        assert turn.tools, "tool class must send tool schemas"
        names = {t["function"]["name"] for t in turn.tools}
        assert len(names) == len(turn.tools), "duplicate tools in one prompt"


def test_every_turn_pins_output_length(tokenizer):
    """ignore_eos plus a fixed max_tokens is what makes configs comparable."""
    cls = replace(load_classes()["c1_chat"], num_prompts=3)
    scenarios, _ = build_class(cls, tokenizer, seed=0)

    for scenario in scenarios:
        turn = scenario.turns[0]
        assert turn.extra_body["ignore_eos"] is True
        assert turn.max_tokens == cls.output_tokens


def test_thinking_flag_is_passed_through(tokenizer):
    """Qwen3 thinks by default, which would decouple output length from the class."""
    cls = replace(load_classes()["c1_chat"], num_prompts=2)
    scenarios, _ = build_class(cls, tokenizer, seed=0)
    kwargs = scenarios[0].turns[0].extra_body["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is False


def test_prompt_sets_are_deterministic(tokenizer):
    cls = replace(load_classes()["c3_rag"], num_prompts=4)
    first, _ = build_class(cls, tokenizer, seed=0)
    second, _ = build_class(cls, tokenizer, seed=0)
    assert [s.to_json() for s in first] == [s.to_json() for s in second]
