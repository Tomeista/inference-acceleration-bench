"""BFCL's AST matcher: was that the right call, with the right arguments?

Tool calling is the capability the wider study is actually about -- an agent
emitting a call to an internal ticketing or CRM API -- and it is the one that
moves under quantization, because unlike multiple choice it is generation-based
and structured. A 4-point drop here is more malformed or wrong API calls, which
is more failed workflows falling back to a human.

Deterministic, with no judge anywhere: an item's key is the set of argument
values that would be acceptable, and matching is a comparison. That is what
makes BFCL reproducible enough to put in a thesis, and it is why the matcher is
reimplemented here rather than the harness imported.

Two failure modes, counted separately
-------------------------------------
This is the reason BFCL earns its place over a generic JSON benchmark, and the
split only exists because the model is asked in *native* function-calling mode:

  structure failure  nothing decodable came back. vLLM's hermes parser could not
                     turn the output into a call -- malformed JSON, missing
                     tags, or plain prose where a call was required. The model
                     has stopped producing the format.
  semantic failure   a well-formed call, but the wrong one: wrong function,
                     missing or invented parameters, wrong values or types.

They degrade differently and they cost differently in production -- one breaks
the integration, the other silently does the wrong thing -- so an aggregate
"accuracy" that merges them hides the finding.

Deliberately not implemented
----------------------------
The published matcher also covers Java and JavaScript type semantics. Only the
Python categories are drawn here, so those coercion rules would be dead code
pretending to be coverage.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# A gold parameter whose acceptable-value list contains the empty string is
# optional: the model may leave it out. BFCL encodes "this has a default" that
# way rather than with a flag, so it has to be read out of the values.
OPTIONAL = ""


def normalize_scalar(value: Any) -> Any:
    """Canonical form for comparing one argument value.

    Three equivalences, each of which would otherwise fail a call that is
    correct:

      10 and 10.0     the schema says integer, the model emitted a float, and
                      JSON does not distinguish them on the wire anyway
      "Paris" / "paris"  argument values are matched case-insensitively, and
                      surrounding whitespace is not a semantic difference
      True / "true"   models emit JSON booleans and the string form roughly
                      interchangeably

    Booleans are checked before numbers on purpose: `bool` is a subclass of
    `int` in Python, so `True` would otherwise normalize to `1.0` and match an
    argument whose gold value is the number 1.
    """
    if isinstance(value, bool):
        # Tagged, not returned bare. `bool` subclasses `int`, and `True == 1.0`
        # is true in Python, so returning the bool would let a call passing
        # `true` satisfy a parameter whose gold value is the number 1.
        return ("bool", value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return ("bool", lowered == "true")
        return lowered
    return value


def values_match(actual: Any, expected: Any) -> bool:
    """Whether one emitted argument value matches one acceptable value."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return False
        return all(values_match(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(values_match(a, e) for a, e in zip(actual, expected))
    return normalize_scalar(actual) == normalize_scalar(expected)


def argument_matches(actual: Any, acceptable: Sequence[Any]) -> bool:
    """Whether a value is one of the acceptable ones for its parameter."""
    return any(values_match(actual, candidate) for candidate in acceptable)


def call_matches(call: Mapping[str, Any], gold: Mapping[str, Any]) -> tuple[bool, str]:
    """Whether one decoded call satisfies one gold entry, and why not if it does not.

    `gold` is BFCL's shape: {function_name: {parameter: [acceptable, values]}}.
    The reason string is kept for the per-item artifact -- when a config starts
    failing, "wrong_value:unit" and "missing_required:height" point at different
    problems, and an aggregate accuracy points at neither.
    """
    (gold_name, gold_params), = gold.items()
    if call.get("name") != gold_name:
        return False, f"wrong_function:{call.get('name')!r}"

    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_an_object"

    for name, value in arguments.items():
        if name not in gold_params:
            # Not merely unhelpful: an argument the function does not take is
            # rejected by a real API, so this is a broken call.
            return False, f"unexpected_parameter:{name}"
        if not argument_matches(value, gold_params[name]):
            return False, f"wrong_value:{name}"

    for name, acceptable in gold_params.items():
        if name in arguments:
            continue
        if OPTIONAL not in acceptable:
            return False, f"missing_required:{name}"

    return True, ""


def decode_calls(tool_calls: Sequence[Mapping[str, Any]]) -> tuple[list[dict], str]:
    """Turn the client's raw tool-call fragments into calls, or say why not.

    Arguments arrive as the JSON *string* the model produced, deliberately kept
    unparsed by the client so that malformed JSON reaches here as data rather
    than as an exception somewhere upstream. Failing to parse it is a structure
    failure and is what this reports.
    """
    calls: list[dict] = []
    for raw in tool_calls:
        text = raw.get("arguments") or ""
        try:
            # An empty argument string is a call with no arguments, which is
            # legitimate for a zero-parameter function.
            arguments = json.loads(text) if text.strip() else {}
        except (ValueError, TypeError):
            return [], "undecodable_arguments"
        if not isinstance(arguments, dict):
            return [], "arguments_not_an_object"
        calls.append({"name": raw.get("name") or "", "arguments": arguments})
    return calls, ""


def score_item(
    tool_calls: Sequence[Mapping[str, Any]],
    text: str,
    ground_truth: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Score one BFCL item.

    `ground_truth` is None for the irrelevance category, where the right
    behaviour is to call nothing at all: the offered functions cannot answer the
    question, and a model that invents a plausible call anyway is the failure
    being measured. That category is the reason a tool-calling score cannot just
    be "did it call the right thing" -- knowing when not to call is half of what
    makes an agent usable.
    """
    irrelevance = ground_truth is None
    calls, decode_error = decode_calls(tool_calls)

    if irrelevance:
        # No structure failure is possible here: producing prose IS the correct
        # behaviour, so text-without-a-call is the right answer rather than a
        # parser failure.
        correct = not tool_calls
        return {
            "correct": correct,
            "extracted": "none" if correct else _signature(calls) or "undecodable",
            "structure_failure": False,
            "semantic_failure": not correct,
            "reason": "" if correct else "called_when_irrelevant",
        }

    if not tool_calls:
        # Nothing decoded. Either the model answered in prose where a call was
        # required, or the parser could not read what it emitted. Both are the
        # model failing to produce the format.
        return {
            "correct": False,
            "extracted": None,
            "structure_failure": True,
            "semantic_failure": False,
            "reason": "no_call_emitted" if text.strip() else "empty_reply",
        }

    if decode_error:
        return {
            "correct": False,
            "extracted": None,
            "structure_failure": True,
            "semantic_failure": False,
            "reason": decode_error,
        }

    if len(calls) != len(ground_truth):
        return {
            "correct": False,
            "extracted": _signature(calls),
            "structure_failure": False,
            "semantic_failure": True,
            "reason": f"wrong_call_count:{len(calls)}!={len(ground_truth)}",
        }

    # Parallel items expect several calls. BFCL matches them as a set rather
    # than in sequence: the model is not told what order to emit independent
    # calls in, and penalising an order it was never given would measure
    # something the task does not ask for. Greedy pairing is exact here because
    # an item's expected calls are distinct.
    remaining = list(ground_truth)
    reasons: list[str] = []
    for call in calls:
        for i, gold in enumerate(remaining):
            ok, _ = call_matches(call, gold)
            if ok:
                remaining.pop(i)
                break
        else:
            _, reason = call_matches(call, remaining[0])
            reasons.append(reason)

    correct = not remaining
    return {
        "correct": correct,
        "extracted": _signature(calls),
        "structure_failure": False,
        "semantic_failure": not correct,
        "reason": "" if correct else ";".join(reasons) or "unmatched_call",
    }


def _signature(calls: Sequence[Mapping[str, Any]]) -> str:
    """A compact, canonical rendering of what the model called.

    This is what the agreement join pairs two configs on, so it has to be
    stable under argument ordering -- otherwise two configs that made the same
    call would read as disagreeing because one serialized its keys differently.
    """
    parts = []
    for call in calls:
        arguments = call.get("arguments") or {}
        rendered = ",".join(
            f"{k}={json.dumps(arguments[k], sort_keys=True, ensure_ascii=False)}"
            for k in sorted(arguments)
        )
        parts.append(f"{call.get('name')}({rendered})")
    return "|".join(parts)
