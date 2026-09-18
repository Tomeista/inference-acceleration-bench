"""Turning generated text into a score, and saying how much to trust it.

Pure functions: strings in, numbers out. No server, no config, no MLflow. That
separation is deliberate rather than tidy-minded -- extraction is where this
feature can most easily manufacture its own finding. A regex that quietly stops
matching once a config's prose degrades would produce a beautiful accuracy cliff
that is entirely an artifact of the scorer, and it would be indistinguishable
from the result the study is looking for. So the extractors are tested here,
against the output damaged checkpoints actually emit, before anything reaches a
GPU.

Three conventions carry that caution into the numbers:

  * An extractor returns None rather than guessing. "No answer found" is a
    measurement -- `unparseable_rate` -- and the whole point is that
    instruction-following usually breaks before accuracy does.
  * Truncation is counted, not scored as wrong. A reply cut off at max_tokens is
    unmeasured; folding it into the wrong pile would credit the scorer with
    knowing something it does not.
  * Accuracy is reported twice: over every item, and over only the items that
    were actually scoreable. When those two diverge, the config is failing to
    answer rather than answering wrongly, and that is a different finding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

# How much generated text to keep per item in the artifact. Enough to see why an
# extraction failed -- the whole reason the raw text is kept at all -- without
# turning a 250-item run into a megabyte of JSONL.
TEXT_KEPT = 600


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

# The instructed format is "Answer: C". This also catches the ways a model
# drifts off it while still committing to a letter: "**Answer:** C",
# "Answer: (C)", "the answer is C", "Answer: The answer is C". findall plus
# last-match is what handles the last of those -- the first "answer" is followed
# by "The", which fails the letter group, and the engine moves on to the second.
_MC_LABELLED = re.compile(
    r"answer\s*(?:is\s*)?[:\-]?\s*\**\s*\(?\s*([A-Da-d])\s*[\)\.\:,]?",
    re.IGNORECASE,
)

# Fallback: a bare letter standing on its own. Deliberately uppercase-only,
# because a lowercase `a` is the English article and would match in almost any
# sentence -- the single most likely way this fallback could invent answers.
_MC_BARE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")

# MMLU-Pro runs to ten options, and the A-D extractor above cannot simply be
# widened to A-J. Two of the added letters are ordinary English words in
# uppercase -- "I" is the first-person pronoun, and a bare "A" opens sentences --
# so the loose fallback that is safe over four letters starts inventing answers
# over ten. That matters here more than it looks: 9% of MMLU-Pro's gold answers
# ARE "I", so a config whose prose degrades into "I I I ..." would score those
# items correct, and the resulting bump would be indistinguishable from a real
# result.
#
# Two changes, both narrowing:
#
#   the labelled form  a letter that is merely the first character of the next
#                      word no longer counts. "the answer is a bit unclear" and
#                      "the answer is I think so" match nothing rather than
#                      yielding A and I.
#   the bare fallback  drops "I" entirely. A reply of exactly "I" is scored
#                      unparseable rather than as a commitment to option I --
#                      the conservative direction, since `unparseable_rate` is a
#                      measurement and a wrong answer is not. "Answer: I" is
#                      unaffected; only the unlabelled form loses that letter.
_MC10_LABELLED = re.compile(
    r"answer\s*(?:is\s*)?[:\-]?\s*\**\s*\(?\s*([A-Ja-j])(?!\s*[A-Za-z])\s*[\)\.\:,]?",
    re.IGNORECASE,
)

_MC10_BARE = re.compile(r"(?<![A-Za-z])([A-HJ])(?![A-Za-z])")

# GSM8K's own convention, and what the prompt asks for.
_GSM_HASH = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")

# Fallback: the last number anywhere in the reply. Reasonable for arithmetic --
# a worked solution ends on its result -- and wrong often enough that
# `unparseable_rate` and the kept raw text both matter.
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_mc(text: str) -> str | None:
    """The letter this reply committed to, or None if it committed to none."""
    if not text:
        return None
    matches = _MC_LABELLED.findall(text)
    if matches:
        return matches[-1].upper()
    bare = _MC_BARE.findall(text)
    if bare:
        return bare[-1]
    return None


def extract_mc10(text: str) -> str | None:
    """The letter this reply committed to, over A-J, or None.

    Same shape as `extract_mc`, deliberately not the same function: the two
    suites have different letter ranges and therefore different false-positive
    surfaces, and sharing one extractor would mean widening MMLU's to A-J and
    changing what the four-option suite scores.
    """
    if not text:
        return None
    matches = _MC10_LABELLED.findall(text)
    if matches:
        return matches[-1].upper()
    bare = _MC10_BARE.findall(text)
    if bare:
        return bare[-1]
    return None


def normalize_number(raw: str) -> str:
    """Canonical form, so "$1,000.00" and "1000" are the same answer.

    String rather than float on purpose: the key is compared for equality, and
    round-tripping through a float would make 0.1 + 0.2 a scoring question.
    """
    cleaned = raw.strip().strip(".,").replace(",", "").replace("$", "").replace(" ", "")
    if not cleaned or cleaned in ("-", "."):
        return ""
    # Trailing zeros after a decimal point are formatting, not precision: the
    # gold answers are integers, and "18.00" is the same answer as "18".
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    if cleaned in ("", "-"):
        return "0"
    # "-0" and "0" are the same number and would otherwise score as a miss.
    return "0" if cleaned in ("-0", "0") else cleaned


def extract_numeric(text: str) -> str | None:
    """The number this reply committed to, normalized, or None."""
    if not text:
        return None
    hashed = _GSM_HASH.findall(text)
    if hashed:
        value = normalize_number(hashed[-1])
        return value or None
    numbers = _NUMBER.findall(text)
    if numbers:
        value = normalize_number(numbers[-1])
        return value or None
    return None


EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "mc": extract_mc,
    "mc10": extract_mc10,
    "numeric": extract_numeric,
}


# --------------------------------------------------------------------------
# the scorer contract
# --------------------------------------------------------------------------
#
# The three extractors above share a shape that the first three suites happened
# to have: read one string out of the reply, and be correct if it equals one
# gold string. Most of what the capability suite adds does not fit it.
#
#   ifeval      an item carries a list of instruction specs, and correctness is
#               per-instruction as well as per-prompt
#   bfcl_ast    the key is a set of acceptable calls, matched with type
#               coercion and optional-parameter rules
#   ruler       multi-needle items take partial credit
#   evalplus    correctness is "the test suite passed"
#   xstest      correctness is a refusal classification, not an answer
#
# So a scorer is given the whole key row and returns a verdict rather than a
# string. `extracted` survives as a field because it is what `agreement` joins
# on, and the paired agreement metric is the one this study leads with: for
# bfcl it is the canonicalized call, for ruler the extracted needle, for xstest
# the refusal label.
#
# `extra` is where a suite puts the metrics only it has. Two aggregation rules,
# by naming convention, because per-item means are wrong for some of them:
#
#   plain name        averaged over items. A 0/1 value therefore aggregates to
#                     a rate, which is what structure_failure and the rest want.
#   `<n>_num`/`<n>_den`  summed separately and divided, giving `<n>`. IFEval's
#                     instruction-level accuracy is a ratio of sums -- items
#                     carry different numbers of instructions, and a mean of
#                     per-item ratios would weight a one-instruction item as
#                     heavily as a five-instruction one.


@dataclass(frozen=True)
class ScoreResult:
    """What one scorer made of one reply."""

    extracted: str | None
    correct: bool
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Reply:
    """What the model sent back, as much of it as a scorer can need.

    A dataclass rather than a bare string because of native function calling.
    Under `--enable-auto-tool-choice` a tool call does not arrive as content at
    all: vLLM's parser decodes it and it comes back in `tool_calls`, leaving
    `text` empty. A scorer handed only the text would see nothing and score
    every successful tool call as a failure to answer.

    The pair is also exactly what separates BFCL's two failure modes. Calls
    decoded means the model produced well-formed JSON and the question is
    whether it was the *right* call; no calls but text present means the parser
    could not decode what it emitted, which is a structure failure. One field
    cannot express that difference.
    """

    text: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# reply, key row -> verdict. The key row is the whole line from
# `<suite>.key.jsonl`, so a scorer reaches its structured gold data through
# `row["meta"]` while `row["answer"]` stays the display string every suite has.
Scorer = Callable[[Reply, Mapping[str, Any]], ScoreResult]


def _from_extractor(extract: Callable[[str], str | None]) -> Scorer:
    """Lift one of the original string extractors into the general contract.

    Keeps `mc`, `mc10` and `numeric` scoring exactly what they scored before:
    the equality test below is the one `score_records` used to inline, so the
    committed eval sets produce byte-identical results across this change.
    """

    def score(reply: Reply, key: Mapping[str, Any]) -> ScoreResult:
        extracted = extract(reply.text)
        expected = key["answer"]
        return ScoreResult(
            extracted=extracted,
            correct=extracted is not None and extracted == expected,
        )

    return score


SCORERS: dict[str, Scorer] = {name: _from_extractor(fn) for name, fn in EXTRACTORS.items()}


def register_scorer(name: str, scorer: Scorer) -> None:
    """Add a scorer. Refuses to shadow an existing one.

    A silently replaced scorer would rescore a suite under the same name, and
    the MLflow store would hold two incomparable columns both called accuracy.
    """
    if name in SCORERS:
        raise KeyError(f"scorer {name!r} is already registered")
    SCORERS[name] = scorer


# --------------------------------------------------------------------------
# degeneracy and uncertainty
# --------------------------------------------------------------------------


def repetition_ratio(text: str, n: int = 4) -> float:
    """How much of this reply is the same n-gram over again.

    0.0 is text that never repeats itself; values approaching 1.0 are a model
    stuck in a loop, which is how a badly damaged checkpoint fails. Reported as
    its own metric because "produced 400 tokens of the same clause" and
    "answered incorrectly" are different failures that a single accuracy number
    merges.
    """
    words = text.split()
    if len(words) < n * 2:
        # Too short for repetition to mean anything; a two-word reply is not a
        # loop, and scoring it as one would flag every correct MC answer.
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for k successes in n trials.

    Wilson rather than the normal approximation, because a damaged config can
    sit near 0 and the BF16 reference near its ceiling, and the normal
    interval misbehaves at both ends -- it happily reports a lower bound below
    zero, which would be printed next to a real number as though it were one.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# scoring one suite
# --------------------------------------------------------------------------


@dataclass
class ItemResult:
    """One benchmark item, scored. Written per-item to the run's artifact."""

    scenario_id: str
    expected: str
    extracted: str | None
    correct: bool
    finish_reason: str | None
    truncated: bool
    repetition: float
    text: str
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "scenario_id": self.scenario_id,
            "expected": self.expected,
            "extracted": self.extracted,
            "correct": self.correct,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "repetition": round(self.repetition, 4),
            "text": self.text[:TEXT_KEPT],
        }
        # Omitted when empty, so the items files the original three suites
        # write -- and which later runs read back as the reference -- stay
        # byte-identical across this change.
        if self.extra:
            out["extra"] = {k: round(v, 6) for k, v in sorted(self.extra.items())}
        return out


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(thinking, answer, unterminated) for a reply that may carry a think block.

    Three cases, and the third is the one that matters.

      no block            ("", text, False). Every non-thinking suite.
      closed block        the reasoning, then the answer after `</think>`.
      block never closed  (everything, "", True) -- the model ran out of budget
                          mid-thought. The answer is *empty*, not "whatever the
                          reasoning happened to say last".

    That last case is why this returns a flag rather than just stripping. A
    truncated thinking trace often ends mid-derivation on a number or a letter,
    and scoring the reasoning text would read that as the model's answer: a
    config that thinks itself out of budget would score at chance instead of
    showing up as unparseable, which is the measurement the thinking arm is for.

    Only the *first* block is treated as reasoning. Qwen3 emits one; a second
    `<think>` inside the answer is content, and consuming it would silently
    delete part of a reply.
    """
    start = text.find(THINK_OPEN)
    if start == -1:
        # A reply that closes a block it never opened: vLLM's chat template can
        # emit the opening tag itself, so the model's text begins mid-thought.
        close = text.find(THINK_CLOSE)
        if close != -1:
            return text[:close], text[close + len(THINK_CLOSE) :], False
        return "", text, False

    close = text.find(THINK_CLOSE, start)
    if close == -1:
        return text, "", True
    return text[start + len(THINK_OPEN) : close], text[close + len(THINK_CLOSE) :], False


def thinking_metrics(thinking: str, answer: str, unterminated: bool) -> dict[str, float]:
    """Per-item cost of reasoning, in characters.

    Characters rather than tokens, deliberately. The server reports
    `completion_tokens` for the whole reply and does not split it at
    `</think>`, so a per-part token count would have to be apportioned -- an
    estimate presented in the same column as exact numbers. Characters are
    exact, need no tokenizer, and answer the question the arm exists to ask:
    whether a quantized config reasons for longer to reach the same answer.
    Report them beside the exact `completion_tokens` mean the cell already logs.
    """
    if not thinking and not unterminated:
        return {}
    total = len(thinking) + len(answer)
    return {
        "thinking_chars": float(len(thinking)),
        "answer_chars": float(len(answer)),
        "thinking_share": len(thinking) / total if total else 0.0,
        "thinking_unterminated": 1.0 if unterminated else 0.0,
    }


def _as_row(value: Any) -> Mapping[str, Any]:
    """A key entry as a row.

    `read_key` yields whole rows, but a bare gold string is the degenerate case
    of the same thing and several tests write keys that way. Normalizing here
    rather than at every call site keeps the scorer contract single-shaped.
    """
    return value if isinstance(value, Mapping) else {"answer": value}


def score_records(
    records: Iterable[Any], key: Mapping[str, Any], scorer: str
) -> list[ItemResult]:
    """Score the successful records of one suite against its answer key.

    `records` are `bench.client.RequestRecord`s, which is why this takes an
    iterable of anything with the right attributes rather than importing the
    class: it keeps this module free of the load client and therefore
    testable with two lines of fake.

    `key` maps scenario_id to the row from `<suite>.key.jsonl`. The scorer sees
    the whole row, because a structured suite's gold data does not fit in the
    single `answer` string.
    """
    if scorer not in SCORERS:
        raise KeyError(f"unknown scorer {scorer!r}; known: {sorted(SCORERS)}")
    score = SCORERS[scorer]

    results: list[ItemResult] = []
    for record in records:
        if not record.success:
            continue
        entry = key.get(record.scenario_id)
        if entry is None:
            raise KeyError(
                f"{record.scenario_id} was measured but is not in the answer key; "
                f"the prompt file and the key file disagree"
            )
        row = _as_row(entry)
        text = record.output_text or ""

        # Applied to every suite, not only the thinking arm. On a non-thinking
        # suite there is no block, so this is the identity -- and if a config
        # starts emitting one anyway, that is a finding rather than something to
        # be quietly scored as prose.
        thinking, answer, unterminated = split_thinking(text)
        reply = Reply(
            text=answer,
            # Structured calls, when native function calling decoded any. The
            # tuple is empty on every suite that does not use tools, so a
            # scorer that ignores it sees exactly what it saw before.
            tool_calls=tuple(getattr(record, "tool_calls", ()) or ()),
            finish_reason=record.finish_reason,
        )
        verdict = score(reply, row)
        extra = dict(verdict.extra)
        extra.update(thinking_metrics(thinking, answer, unterminated))

        results.append(
            ItemResult(
                scenario_id=record.scenario_id,
                expected=str(row["answer"]),
                extracted=verdict.extracted,
                correct=verdict.correct,
                finish_reason=record.finish_reason,
                truncated=record.finish_reason == "length",
                # Over the whole reply, thinking included: a config that loops
                # inside its reasoning is stuck, and scoring only the answer
                # would hide the single clearest sign of a damaged checkpoint.
                repetition=repetition_ratio(text),
                text=text,
                extra=extra,
            )
        )
    return results


def aggregate(results: list[ItemResult]) -> dict[str, float]:
    """The metric set for one (config, suite) cell.

    Two accuracies, on purpose. `accuracy` divides by every item and is the
    headline -- a config that cannot produce a parseable answer has failed the
    task, and hiding that behind a filter would flatter the most aggressive
    compression. `accuracy_parsed` divides by the items that were actually
    scoreable, so the gap between the two says whether a config is answering
    wrongly or has stopped answering at all.
    """
    n = len(results)
    if not n:
        return {}

    correct = sum(1 for r in results if r.correct)
    truncated = sum(1 for r in results if r.truncated)
    unparseable = sum(1 for r in results if r.extracted is None)
    scoreable = [r for r in results if r.extracted is not None and not r.truncated]

    lo, hi = wilson_interval(correct, n)
    out = {
        "n_items": float(n),
        "accuracy": correct / n,
        "accuracy_ci_lo": lo,
        "accuracy_ci_hi": hi,
        "unparseable_rate": unparseable / n,
        "truncated_rate": truncated / n,
        "repetition_ratio": sum(r.repetition for r in results) / n,
    }
    if scoreable:
        out["accuracy_parsed"] = sum(1 for r in scoreable if r.correct) / len(scoreable)
        out["n_scoreable"] = float(len(scoreable))
    out.update(aggregate_extra(results))
    return out


def aggregate_extra(results: list[ItemResult]) -> dict[str, float]:
    """Roll up the per-suite metrics in `ItemResult.extra`.

    Two rules, chosen by name (see the scorer contract above):

      `<n>_num` with `<n>_den`  summed separately, then divided. A ratio of
                                sums, for metrics whose denominator varies per
                                item -- IFEval items carry different numbers of
                                instructions, and averaging per-item ratios
                                would weight a one-instruction item as heavily
                                as a five-instruction one.
      anything else             averaged over the items that reported it.

    Averaged over the items that *reported* it, not over every item: a metric
    only some items can have (a needle depth that only multi-needle items
    carry) must not be diluted by the items it does not apply to.
    """
    if not results:
        return {}

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for name, value in result.extra.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    out: dict[str, float] = {}
    for name in sorted(totals):
        if name.endswith("_den"):
            continue
        if name.endswith("_num"):
            stem = name[: -len("_num")]
            den = totals.get(f"{stem}_den", 0.0)
            # A zero denominator means no item carried the metric at all.
            # Absent rather than zero, the same convention `agreement` follows:
            # 0.0 would read as "measured, and it was nothing".
            if den:
                out[stem] = totals[name] / den
            continue
        out[name] = totals[name] / counts[name]
    return out


def agreement(
    results: list[ItemResult], reference: dict[str, str | None]
) -> dict[str, float]:
    """How often this config gave the same answer as the reference config.

    The sensitive metric, and the reason the reference run is required to go
    first. Accuracy on 250 items has a +/-6pp interval, so it can sit flat while
    a config quietly changes a third of its answers; this catches that, because it
    is paired per item rather than compared in aggregate.

    Two unparseable replies count as agreeing. They are the same *behaviour*,
    and calling that a disagreement would make a config's agreement score depend
    on how the reference happened to fail.

    Returns {} when there is no reference, rather than 0.0. Same convention as
    `metrics.spec_decode_metrics` and `metrics.preemption_delta`: a metric that
    was never computed must not read as one that came out at zero.
    """
    shared = [r for r in results if r.scenario_id in reference]
    if not shared:
        return {}
    same = sum(1 for r in shared if r.extracted == reference[r.scenario_id])
    return {
        "agreement_with_reference": same / len(shared),
        "agreement_n": float(len(shared)),
    }


# --------------------------------------------------------------------------
# suite scorers
# --------------------------------------------------------------------------
#
# Registered here rather than in their own modules' import side effects, so
# that the set of scorers a run can use is visible in one place and a suite
# naming one that does not exist fails at load rather than mid-pass.


def score_ifeval(reply: Reply, key: Mapping[str, Any]) -> ScoreResult:
    """IFEval: four numbers per item, at two levels and two strictnesses.

    `correct` is prompt-level strict -- every instruction on the item obeyed,
    exactly as the reply came back. That is the headline and the harshest
    reading; the other three travel in `extra`.

    `extracted` is the per-instruction verdict pattern ("101"), not a pass/fail.
    Two configs can fail the same item for opposite reasons, and pairing them on
    a single bit would call that agreement. The pattern is what the agreement
    join compares, so it is also what makes IFEval a sensitive paired metric
    rather than a coarse one.
    """
    from bench.ifeval import evaluate, verdict_string

    meta = key["meta"]
    result = evaluate(meta["instruction_id_list"], meta["kwargs"], reply.text)
    return ScoreResult(
        extracted=verdict_string(result["strict_flags"]),
        correct=bool(result["strict"]),
        extra={
            "prompt_loose": 1.0 if result["loose"] else 0.0,
            "instruction_strict_num": float(result["n_strict_followed"]),
            "instruction_strict_den": float(result["n_instructions"]),
            "instruction_loose_num": float(result["n_loose_followed"]),
            "instruction_loose_den": float(result["n_instructions"]),
        },
    )


register_scorer("ifeval", score_ifeval)


def score_bfcl_ast(reply: Reply, key: Mapping[str, Any]) -> ScoreResult:
    """BFCL single-turn: the right call, with the right arguments.

    The two failure rates travel in `extra` and are the point of the suite.
    `structure_failure` is the model no longer producing a decodable call at
    all; `semantic_failure` is a well-formed call that is wrong. They break the
    integration in different ways and a merged accuracy hides which is which.

    `extracted` is the canonical call signature, so the agreement join pairs two
    configs on *what they called* rather than on whether each happened to be
    right. Two configs can both be wrong and wrong differently.
    """
    from bench.bfcl import score_item

    result = score_item(reply.tool_calls, reply.text, key["meta"].get("ground_truth"))
    return ScoreResult(
        extracted=result["extracted"],
        correct=bool(result["correct"]),
        extra={
            "structure_failure": 1.0 if result["structure_failure"] else 0.0,
            "semantic_failure": 1.0 if result["semantic_failure"] else 0.0,
        },
    )


register_scorer("bfcl_ast", score_bfcl_ast)
