"""The BFCL AST matcher, and the two failure modes it has to keep apart.

The hazard here is the mirror of the one in `test_ifeval`. A matcher that is too
lenient scores wrong calls as right and reports that quantization costs nothing;
a matcher that is too strict rejects correct calls and manufactures a collapse.
`test_every_frozen_items_own_ground_truth_is_accepted` is the guard against the
second -- it replays each item's published answer as though the model had
produced it -- and the per-rule cases below are the guard against the first.
"""

from __future__ import annotations

import json

import pytest

from bench.bfcl import _signature, decode_calls, normalize_scalar, score_item

GOLD = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def call(name: str, /, **arguments) -> dict:
    """One tool call in the shape the client hands the scorer.

    Arguments stay a JSON *string*, as the client leaves them, so that malformed
    JSON reaches the matcher as data rather than raising somewhere upstream.

    `name` is positional-only on purpose: real BFCL functions take a parameter
    called `name`, and a plain keyword would collide with this helper's own.
    """
    return {"name": name, "arguments": json.dumps(arguments)}


# --------------------------------------------------------------------------
# what counts as the right call
# --------------------------------------------------------------------------


def test_the_expected_call_is_accepted():
    assert score_item([call("calculate_triangle_area", base=10, height=5)], "", GOLD)["correct"]


def test_an_optional_parameter_may_be_supplied_or_omitted():
    """An acceptable-value list containing "" is how BFCL says "has a default".

    Both readings have to pass. Requiring it would fail every correct call that
    relied on the default; ignoring the marker would accept a missing *required*
    parameter, which is a broken call to a real API.
    """
    with_it = score_item(
        [call("calculate_triangle_area", base=10, height=5, unit="units")], "", GOLD
    )
    without = score_item([call("calculate_triangle_area", base=10, height=5)], "", GOLD)
    assert with_it["correct"] and without["correct"]


def test_a_missing_required_parameter_is_not_forgiven():
    result = score_item([call("calculate_triangle_area", base=10)], "", GOLD)
    assert not result["correct"]
    assert result["reason"] == "missing_required:height"
    assert result["semantic_failure"]


def test_an_invented_parameter_is_a_broken_call():
    """A real API rejects it, so accepting it here would overstate the model."""
    result = score_item(
        [call("calculate_triangle_area", base=10, height=5, colour="red")], "", GOLD
    )
    assert not result["correct"]
    assert result["reason"] == "unexpected_parameter:colour"


def test_the_wrong_function_is_named_as_such():
    result = score_item([call("compute_area", base=10, height=5)], "", GOLD)
    assert result["reason"].startswith("wrong_function")


@pytest.mark.parametrize("value", [10, 10.0], ids=["int", "float"])
def test_a_number_matches_however_it_was_serialized(value):
    """JSON does not distinguish 10 from 10.0, and models emit both. Failing a
    call over that would measure serialization, not capability."""
    assert score_item([call("calculate_triangle_area", base=value, height=5)], "", GOLD)[
        "correct"
    ]


def test_a_number_sent_as_a_string_is_still_wrong():
    """The line the coercion stops at, and it stops there deliberately.

    `10` and `10.0` are the same value written two ways. `"10"` is a different
    *type*, and an API with an integer parameter rejects it. Coercing across
    that boundary would let a model that had lost its typing score as correct,
    which is exactly the degradation the suite is meant to catch.
    """
    assert not score_item(
        [call("calculate_triangle_area", base="10", height=5)], "", GOLD
    )["correct"]


def test_argument_strings_are_matched_case_insensitively():
    assert score_item(
        [call("calculate_triangle_area", base=10, height=5, unit="UNITS")], "", GOLD
    )["correct"]


def test_a_boolean_never_matches_the_number_one():
    """`bool` subclasses `int` in Python, and `True == 1.0` is true there.

    So a numeric normalization that simply returns the value lets a call passing
    `true` satisfy a parameter whose gold value is the number 1 -- a type error
    scored as correct. The normalizer has to tag booleans to keep them apart,
    and comparing the raw values is not enough to prove it does.
    """
    assert normalize_scalar(True) != normalize_scalar(1)
    assert normalize_scalar(False) != normalize_scalar(0)
    assert normalize_scalar(True) == normalize_scalar("true")


def test_a_wrong_value_is_semantic_not_structural():
    """Well-formed JSON, wrong content. It must not land in the structure bucket:
    the two describe different production failures."""
    result = score_item([call("calculate_triangle_area", base=7, height=5)], "", GOLD)
    assert result["semantic_failure"] and not result["structure_failure"]
    assert result["reason"] == "wrong_value:base"


# --------------------------------------------------------------------------
# parallel calls
# --------------------------------------------------------------------------

PARALLEL = [
    {"get_weather": {"city": ["Paris"]}},
    {"get_weather": {"city": ["Berlin"]}},
]


def test_parallel_calls_are_matched_as_a_set_not_a_sequence():
    """The model is never told which order to emit independent calls in.

    Penalising the order would measure something the task does not ask for, and
    would make the score depend on an arbitrary choice the prompt left open.
    """
    forwards = score_item([call("get_weather", city="Paris"), call("get_weather", city="Berlin")], "", PARALLEL)
    backwards = score_item([call("get_weather", city="Berlin"), call("get_weather", city="Paris")], "", PARALLEL)
    assert forwards["correct"] and backwards["correct"]


def test_too_few_calls_is_a_semantic_failure():
    result = score_item([call("get_weather", city="Paris")], "", PARALLEL)
    assert not result["correct"]
    assert "wrong_call_count" in result["reason"]


def test_a_duplicated_call_does_not_satisfy_two_expectations():
    """Greedy pairing must consume each expected call at most once, or emitting
    the same call twice would pass an item that wanted two different ones."""
    result = score_item(
        [call("get_weather", city="Paris"), call("get_weather", city="Paris")], "", PARALLEL
    )
    assert not result["correct"]


# --------------------------------------------------------------------------
# structure vs semantics
# --------------------------------------------------------------------------


def test_prose_where_a_call_was_required_is_a_structure_failure():
    """The model has stopped producing the format. Under the hermes parser this
    is what arrives: no decodable call, and the text falls through as content."""
    result = score_item([], "Sure! The area of that triangle is 25 square units.", GOLD)
    assert result["structure_failure"] and not result["semantic_failure"]
    assert result["reason"] == "no_call_emitted"
    assert result["extracted"] is None


def test_malformed_argument_json_is_a_structure_failure():
    result = score_item([{"name": "calculate_triangle_area", "arguments": "{base: 10,"}], "", GOLD)
    assert result["structure_failure"]
    assert result["reason"] == "undecodable_arguments"


def test_a_zero_argument_call_is_not_malformed():
    """An empty argument string is a call with no arguments, which is legitimate
    for a zero-parameter function -- not a JSON parse failure."""
    calls, error = decode_calls([{"name": "ping", "arguments": ""}])
    assert not error and calls == [{"name": "ping", "arguments": {}}]


# --------------------------------------------------------------------------
# irrelevance: knowing when not to call
# --------------------------------------------------------------------------


def test_declining_to_call_is_correct_when_nothing_fits():
    result = score_item([], "None of these functions can answer that.", None)
    assert result["correct"]
    assert result["extracted"] == "none"


def test_inventing_a_call_when_nothing_fits_is_a_semantic_failure():
    result = score_item([call("determine_body_mass_index", weight=70, height=1.8)], "", None)
    assert not result["correct"]
    assert result["semantic_failure"] and not result["structure_failure"]
    assert result["reason"] == "called_when_irrelevant"


def test_prose_on_an_irrelevance_item_is_never_a_structure_failure():
    """Producing prose IS the correct behaviour here. Counting it as a parser
    failure would make the structure-failure rate rise with correctness."""
    assert not score_item([], "I cannot help with that.", None)["structure_failure"]


# --------------------------------------------------------------------------
# the agreement join
# --------------------------------------------------------------------------


def test_the_signature_is_stable_under_argument_order():
    """It is what the agreement join pairs two configs on. Were it order
    sensitive, two configs that made the identical call would read as
    disagreeing because one serialized its keys differently."""
    a = _signature([{"name": "f", "arguments": {"x": 1, "y": 2}}])
    b = _signature([{"name": "f", "arguments": {"y": 2, "x": 1}}])
    assert a == b


def test_two_different_wrong_calls_do_not_look_like_agreement():
    """Both configs are wrong; pairing them on a pass/fail bit would call that
    agreement and hide that they degraded in different directions."""
    a = _signature([{"name": "f", "arguments": {"x": 1}}])
    b = _signature([{"name": "f", "arguments": {"x": 2}}])
    assert a != b


# --------------------------------------------------------------------------
# against the frozen set
# --------------------------------------------------------------------------


def test_every_frozen_items_own_ground_truth_is_accepted(bfcl_key):
    """Replay each item's published answer as though the model had emitted it.

    This is the check that the matcher is not quietly too strict. A coercion
    rule that is slightly wrong -- a nested object compared by identity, a float
    that fails to match its integer -- would reject correct calls on some
    fraction of items, and every config would lose the same points. That reads
    as "tool calling degraded under compression" and is entirely the scorer's
    doing, which is the class of bug this whole file exists to prevent.
    """
    rejected = []
    for scenario_id, row in bfcl_key.items():
        ground_truth = row["meta"]["ground_truth"]
        if ground_truth is None:
            result = score_item([], "No suitable function is available.", None)
        else:
            calls = []
            for entry in ground_truth:
                (name, params), = entry.items()
                # The first non-optional acceptable value for each parameter is
                # a reply the published key says is correct.
                arguments = {
                    p: next(v for v in acceptable if v != "")
                    for p, acceptable in params.items()
                    if any(v != "" for v in acceptable)
                }
                calls.append(call(name, **arguments))
            result = score_item(calls, "", ground_truth)
        if not result["correct"]:
            rejected.append((scenario_id, row["meta"]["category"], result["reason"]))

    assert not rejected, f"{len(rejected)} gold answers rejected: {rejected[:5]}"


def test_the_frozen_set_covers_all_four_categories(bfcl_key):
    """Losing irrelevance in particular would turn this into a call-formatting
    score while still reporting a tool-calling number."""
    categories = {row["meta"]["category"] for row in bfcl_key.values()}
    assert categories == {"simple_python", "multiple", "parallel", "irrelevance"}


def test_every_frozen_item_asks_for_native_function_calling(bfcl_scenarios):
    """`tool_choice: auto` is what puts the deployed parser under test.

    `Turn.to_payload` pins tool_choice to "none" for the speed sweep, where tool
    schemas must render into the prompt without the parser buffering the stream.
    If that pin reached this suite, every item would come back as content, every
    item would score as a structure failure, and the cell would read as total
    collapse of tool calling.
    """
    for scenario in bfcl_scenarios:
        turn = scenario.turns[0]
        assert turn.tools, f"{scenario.scenario_id} offers no tools"
        assert turn.extra_body.get("tool_choice") == "auto", scenario.scenario_id
        assert turn.to_payload("m")["tool_choice"] == "auto", (
            f"{scenario.scenario_id}: extra_body did not override the speed-sweep pin"
        )
