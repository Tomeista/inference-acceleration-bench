"""Build the frozen prompt sets.

Run once, commit the result. Every later benchmark run reads the same JSONL, so
a measurement taken in October is comparable to one taken in December, and the
Phase 2 quality pass scores the exact prompts whose latency was measured here.

The point of this module is exactness. A class declares `input_tokens`, and the
scenario it produces is that many tokens *after* the Qwen3 chat template has
been applied, tool schemas included. Anything less and a difference between two
server configs could be prompt-length drift rather than the thing under test.

Adding a class: write a builder, register it in BUILDERS under the `source` name
used in config/classes.yaml.

Usage:
    python -m bench.build_prompts --tokenizer ../quantization-cpu/Qwen3-0.6B-W8A16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bench import corpus
from bench.classes import PROMPTS_DIR, PromptClass, load_classes
from bench.scenarios import Scenario, Turn, write_scenarios

# Qwen3-0.6B and Qwen3-8B ship the same tokenizer and the same chat template, so
# the locally quantized 0.6B checkpoint is a valid stand-in for measuring 8B
# prompt lengths without downloading 16 GB of weights.
DEFAULT_TOKENIZER = "../quantization-cpu/Qwen3-0.6B-W8A16"

Assemble = Callable[[str], "tuple[list[dict[str, Any]], list[dict[str, Any]] | None]"]


@dataclass
class Draft:
    """A prompt under construction.

    `assemble` takes a filler string and returns the finished messages. The
    fitting loop below resizes that filler until the templated prompt lands on
    the class's exact token budget, which keeps every builder free of token
    arithmetic.
    """

    assemble: Assemble
    filler_pool: list[str]
    meta: dict[str, Any]


# --------------------------------------------------------------------------
# token fitting
# --------------------------------------------------------------------------


def _token_ids(out: Any) -> list[int]:
    """Normalize apply_chat_template output to a flat list of ids.

    transformers 4.x returns a list of ids; 5.x returns a BatchEncoding whose
    input_ids may be batched. Taking len() of the wrong one silently reports a
    prompt length of 2, which is exactly the kind of quiet wrongness the exact
    token budgets exist to prevent.
    """
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


class Measurer:
    """Counts tokens the way the server will see them."""

    def __init__(self, tokenizer, thinking: bool) -> None:
        self.tok = tokenizer
        self.thinking = thinking

    def count(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> int:
        out = self.tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            # Qwen3 injects an empty think block into the generation prompt when
            # thinking is disabled, which changes the prompt length. vLLM is
            # sent the matching chat_template_kwargs so the two agree.
            enable_thinking=self.thinking,
        )
        return len(_token_ids(out))

    def filler(self, pool: list[str], budget: int, rng: random.Random) -> str:
        """A filler string of roughly `budget` tokens drawn from `pool`."""
        if budget <= 0:
            return ""
        ids: list[int] = []
        guard = 0
        while len(ids) < budget and guard < 2000:
            piece = pool[rng.randrange(len(pool))]
            ids.extend(self.tok.encode(" " + piece, add_special_tokens=False))
            guard += 1
        return self.tok.decode(ids[:budget], skip_special_tokens=True)


def fit(
    draft: Draft, target: int, measurer: Measurer, rng: random.Random, max_iters: int = 24
) -> tuple[list[dict[str, Any]], list[dict] | None, int]:
    """Resize the filler until the templated prompt is exactly `target` tokens.

    Truncating a token sequence and decoding it back is not perfectly stable
    (a truncated word can retokenize into a different number of pieces), so this
    corrects iteratively rather than computing the budget in one shot. It keeps
    the closest attempt seen, which bounds the error even if it never lands
    exactly.
    """
    budget = target
    best: tuple[list[dict], list[dict] | None, int] | None = None

    for _ in range(max_iters):
        filler = measurer.filler(draft.filler_pool, budget, rng)
        messages, tools = draft.assemble(filler)
        n = measurer.count(messages, tools)

        if best is None or abs(n - target) < abs(best[2] - target):
            best = (messages, tools, n)
        if n == target:
            return messages, tools, n

        budget += target - n
        if budget < 0:
            budget = 0

    assert best is not None
    return best


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def build_chat(cls: PromptClass, idx: int, rng: random.Random) -> Draft:
    question = corpus.SHORT_QUESTIONS[idx % len(corpus.SHORT_QUESTIONS)]

    def assemble(filler: str):
        content = question if not filler else f"{question} For context: {filler}"
        return [{"role": "user", "content": content}], None

    return Draft(assemble, corpus.DOC_PARAGRAPHS, {"seed_question": question})


def build_longform(cls: PromptClass, idx: int, rng: random.Random) -> Draft:
    instruction = corpus.LONGFORM_INSTRUCTIONS[idx % len(corpus.LONGFORM_INSTRUCTIONS)]

    def assemble(filler: str):
        content = instruction if not filler else f"{instruction} Background notes: {filler}"
        return [{"role": "user", "content": content}], None

    return Draft(assemble, corpus.DOC_PARAGRAPHS, {"seed_instruction": instruction})


def build_rag(cls: PromptClass, idx: int, rng: random.Random) -> Draft:
    question = corpus.DOC_QUESTIONS[idx % len(corpus.DOC_QUESTIONS)]
    system = "Answer the question using only the document below. Be concise."

    def assemble(filler: str):
        # The filler is the document, so it absorbs the resizing while the
        # question stays intact at the end where the model will actually read it.
        user = f"Document:\n{filler}\n\nQuestion: {question}"
        return (
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            None,
        )

    return Draft(assemble, corpus.DOC_PARAGRAPHS, {"seed_question": question})


def build_toolcall(cls: PromptClass, idx: int, rng: random.Random) -> Draft:
    query, expected_tool = corpus.TOOL_QUERIES[idx % len(corpus.TOOL_QUERIES)]
    target_tool = next(
        t for t in corpus.TOOLS if t["function"]["name"] == expected_tool
    )
    distractors = [t for t in corpus.TOOLS if t["function"]["name"] != expected_tool]
    rng.shuffle(distractors)

    def assemble_with(n_distractors: int) -> Assemble:
        tools = [target_tool] + distractors[:n_distractors]
        rng_local = random.Random(idx)
        rng_local.shuffle(tools)

        def assemble(filler: str):
            content = query if not filler else f"{query} Additional context: {filler}"
            return (
                [
                    {"role": "system", "content": corpus.TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tools,
            )

        return assemble

    # Tool schemas are large and the template renders them into the prompt, so
    # the number of distractors has to adapt to the token budget rather than
    # being fixed. build_class holds the Measurer, so it does the shrinking via
    # the `_assemble_with` handle; three distractors is the starting point.
    return Draft(
        assemble_with(3),
        corpus.DOC_PARAGRAPHS,
        {
            "expected_tool": expected_tool,
            "seed_query": query,
            "_assemble_with": assemble_with,
        },
    )


BUILDERS: dict[str, Callable[[PromptClass, int, random.Random], Draft]] = {
    "chat": build_chat,
    "longform": build_longform,
    "rag": build_rag,
    "toolcall": build_toolcall,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_class(
    cls: PromptClass, tokenizer, seed: int
) -> tuple[list[Scenario], dict[str, Any]]:
    if cls.source not in BUILDERS:
        raise KeyError(
            f"class {cls.id!r} declares source {cls.source!r} but no builder is "
            f"registered; known: {sorted(BUILDERS)}"
        )
    builder = BUILDERS[cls.source]
    measurer = Measurer(tokenizer, cls.thinking)
    rng = random.Random(seed)

    scenarios: list[Scenario] = []
    actual_lengths: list[int] = []

    for idx in range(cls.num_prompts):
        draft = builder(cls, idx, rng)

        # A tool-bearing draft may not fit its budget with a full distractor
        # list. Shrink the list until the unpadded prompt leaves headroom.
        assemble_with = draft.meta.pop("_assemble_with", None)
        if assemble_with is not None:
            for n_distractors in (3, 2, 1, 0):
                candidate = assemble_with(n_distractors)
                messages, tools = candidate("")
                if measurer.count(messages, tools) <= cls.input_tokens - 8:
                    draft = Draft(candidate, draft.filler_pool, draft.meta)
                    draft.meta["n_distractor_tools"] = n_distractors
                    break
            else:
                draft = Draft(assemble_with(0), draft.filler_pool, draft.meta)
                draft.meta["n_distractor_tools"] = 0

        messages, tools, n_tokens = fit(draft, cls.input_tokens, measurer, rng)
        actual_lengths.append(n_tokens)

        turn = Turn(
            messages=messages,
            max_tokens=cls.output_tokens,
            temperature=cls.temperature,
            tools=tools,
            extra_body={
                # Without ignore_eos a config that happens to stop early looks
                # faster than it is. Every speed run generates exactly
                # max_tokens; the quality pass runs with natural stopping.
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": cls.thinking},
            },
        )
        scenarios.append(
            Scenario(
                scenario_id=f"{cls.id}-{idx:04d}",
                class_id=cls.id,
                turns=[turn],
                prompt_tokens_expected=n_tokens,
            )
        )

    exact = sum(1 for n in actual_lengths if n == cls.input_tokens)
    stats = {
        "class_id": cls.id,
        "source": cls.source,
        "num_prompts": len(scenarios),
        "target_input_tokens": cls.input_tokens,
        "output_tokens": cls.output_tokens,
        "exact_fit": exact,
        "min_input_tokens": min(actual_lengths),
        "max_input_tokens": max(actual_lengths),
        "mean_input_tokens": round(statistics.fmean(actual_lengths), 2),
    }
    return scenarios, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Local directory or HuggingFace id. Must match the served model.",
    )
    parser.add_argument("--classes", default="", help="Comma separated ids; default all enabled")
    parser.add_argument("--out-dir", default=str(PROMPTS_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    all_classes = load_classes()
    requested = [c.strip() for c in args.classes.split(",") if c.strip()]
    if requested:
        selected = [all_classes[c] for c in requested]
    else:
        selected = [c for c in all_classes.values() if c.enabled]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "classes": {},
    }

    for cls in selected:
        scenarios, stats = build_class(cls, tokenizer, args.seed)
        path = out_dir / f"{cls.id}.jsonl"
        write_scenarios(path, scenarios)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        stats["sha256_16"] = digest
        manifest["classes"][cls.id] = stats

        flag = "" if stats["exact_fit"] == stats["num_prompts"] else "  <-- inexact"
        print(
            f"{cls.id:16s} {stats['num_prompts']:4d} prompts  "
            f"target={cls.input_tokens:5d}  "
            f"exact={stats['exact_fit']}/{stats['num_prompts']}  "
            f"range=[{stats['min_input_tokens']}, {stats['max_input_tokens']}]  "
            f"{digest}{flag}"
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(selected)} prompt set(s) and manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
