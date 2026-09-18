"""Every IFEval verifier, with a case it must pass and a case it must fail.

The reason for the pairs: a verifier that always returns True is invisible in an
aggregate score. It would raise every config's IFEval number by the same amount,
leave the ranking intact, and look exactly like a benchmark on which
quantization does no damage -- which is the finding the study is trying to
measure. A verifier that always returns False is the same failure mirrored.

So nothing here is a formality. `test_no_verifier_is_a_constant` is the one that
would catch the whole class of bug at once; the per-instruction pairs are what
say which one broke.
"""

from __future__ import annotations

import json

import pytest

from bench.ifeval import (
    VERIFIERS,
    count_sentences,
    count_words,
    evaluate,
    loose_variants,
    verify,
    verify_loose,
)

# instruction id -> (kwargs, a reply that obeys it, a reply that does not)
CASES: dict[str, tuple[dict, str, str]] = {
    "punctuation:no_comma": ({}, "No commas here at all", "One, two, three"),
    "length_constraints:number_words": (
        {"relation": "at least", "num_words": 5},
        "one two three four five six",
        "too short",
    ),
    "length_constraints:number_sentences": (
        {"relation": "less than", "num_sentences": 3},
        "One sentence. Two sentences.",
        "One. Two. Three. Four.",
    ),
    "length_constraints:number_paragraphs": (
        {"num_paragraphs": 2},
        "First paragraph text.\n***\nSecond paragraph text.",
        "First paragraph text.\n***\nSecond.\n***\nThird.",
    ),
    "length_constraints:nth_paragraph_first_word": (
        {"first_word": "weekend", "num_paragraphs": 2, "nth_paragraph": 1},
        "Weekend plans are good.\n\nSecond paragraph here.",
        "Monday plans are good.\n\nSecond paragraph here.",
    ),
    "keywords:forbidden_words": (
        {"forbidden_words": ["banana", "apple"]},
        "I like oranges and pears",
        "I like bananas and this banana",
    ),
    "keywords:existence": (
        {"keywords": ["quantization", "latency"]},
        "Quantization affects latency",
        "Quantization affects speed",
    ),
    "keywords:frequency": (
        {"keyword": "sneaker", "relation": "at least", "frequency": 2},
        "sneaker and another sneaker",
        "just one sneaker",
    ),
    "keywords:letter_frequency": (
        {"letter": "#", "let_relation": "at least", "let_frequency": 3},
        "#one #two #three",
        "#one #two",
    ),
    "detectable_format:number_highlighted_sections": (
        {"num_highlights": 2},
        "*first highlight* and *second highlight*",
        "*only one highlight*",
    ),
    "detectable_format:number_bullet_lists": (
        {"num_bullets": 2},
        "* first point\n* second point",
        "* first point\n* second point\n* third point",
    ),
    "detectable_format:title": (
        {},
        "<<A Real Title>>\n\nBody text.",
        "A Real Title\n\nBody text.",
    ),
    "detectable_format:multiple_sections": (
        {"section_spliter": "PARAGRAPH", "num_sections": 2},
        "PARAGRAPH 1\ntext\nPARAGRAPH 2\nmore text",
        "PARAGRAPH 1\ntext only",
    ),
    "detectable_format:constrained_response": (
        {},
        "My answer is yes.",
        "Yes, definitely.",
    ),
    "detectable_format:json_format": (
        {},
        '{"answer": 42}',
        "Here is the answer: 42",
    ),
    "detectable_content:number_placeholders": (
        {"num_placeholders": 2},
        "Dear [name], your order [number] shipped.",
        "Dear [name], your order shipped.",
    ),
    "detectable_content:postscript": (
        {"postscript_marker": "P.S."},
        "Main body text.\nP.S. one more thing",
        "Main body text with no postscript.",
    ),
    "combination:two_responses": (
        {},
        "First response\n******\nSecond response",
        "Only one response",
    ),
    "combination:repeat_prompt": (
        {"prompt_to_repeat": "Write a poem about rain."},
        "Write a poem about rain. Rain falls softly.",
        "Rain falls softly.",
    ),
    "startend:quotation": (
        {},
        '"The whole reply is quoted"',
        "The whole reply is not quoted",
    ),
    "startend:end_checker": (
        {"end_phrase": "Any other questions?"},
        "Here is the answer. Any other questions?",
        "Here is the answer. Goodbye.",
    ),
    "change_case:english_lowercase": (
        {},
        "all lowercase text here",
        "Some Capitals Here",
    ),
    "change_case:english_capital": (
        {},
        "ALL CAPITALS HERE",
        "Not All Capitals Here",
    ),
    "change_case:capital_word_frequency": (
        {"capital_relation": "less than", "capital_frequency": 2},
        "only ONE capitalised word here",
        "TWO CAPITALISED words here",
    ),
    "language:response_language": (
        {"language": "de"},
        "Dies ist ein vollständiger deutscher Satz über Maschinen und Sprache.",
        "This is a complete English sentence about machines and language.",
    ),
}


def test_every_instruction_type_in_the_dataset_has_a_case():
    """The registry and this table must not drift apart.

    A verifier with no case here is untested; a case here with no verifier is a
    `verify` that raises mid-pass, on the GPU box, after the generation is spent.
    """
    assert set(CASES) == set(VERIFIERS)


@pytest.mark.parametrize("instruction_id", sorted(CASES))
def test_a_verifier_accepts_what_obeys_it(instruction_id):
    kwargs, good, _ = CASES[instruction_id]
    assert verify(instruction_id, kwargs, good) is True


@pytest.mark.parametrize("instruction_id", sorted(CASES))
def test_a_verifier_rejects_what_does_not(instruction_id):
    kwargs, _, bad = CASES[instruction_id]
    assert verify(instruction_id, kwargs, bad) is False


def test_no_verifier_is_a_constant():
    """The bug that would be invisible in the aggregate.

    A verifier stuck at True lifts every config's score equally, preserves the
    ranking, and reads as "quantization did no damage on IFEval" -- the study's
    own hypothesis, manufactured by the scorer. Stuck at False reads as total
    collapse. Either way the per-instruction pairs above are what catch it, and
    this is the assertion that says so in one place.
    """
    for instruction_id, (kwargs, good, bad) in CASES.items():
        assert verify(instruction_id, kwargs, good) != verify(
            instruction_id, kwargs, bad
        ), f"{instruction_id} returns the same verdict for an obeying and a disobeying reply"


def test_an_unknown_instruction_is_refused_rather_than_passed():
    """Silently passing an unimplemented constraint inflates the score."""
    with pytest.raises(KeyError, match="no verifier"):
        verify("detectable_format:interpretive_dance", {}, "anything")


# --------------------------------------------------------------------------
# strict vs loose
# --------------------------------------------------------------------------


def test_loose_forgives_a_preamble_that_strict_does_not():
    """The whole reason both are reported.

    The constraint is "the reply is all lowercase". The model obeys it and then
    adds a chatty first line. Strict says no, which is true but not about the
    capability; loose says yes, which is the more useful reading of what the
    model can do. The gap between the two scores is a format signal.
    """
    reply = "Sure! Here you go:\nthe rest of this reply is entirely lowercase"
    assert verify("change_case:english_lowercase", {}, reply) is False
    assert verify_loose("change_case:english_lowercase", {}, reply) is True


def test_loose_does_not_forgive_an_actual_violation():
    """Otherwise it would be a constant-True verifier wearing a disguise."""
    reply = "This Reply Has Capitals Throughout And No Wrapper To Strip"
    assert verify_loose("change_case:english_lowercase", {}, reply) is False


def test_the_strict_text_is_the_first_variant_tried():
    variants = loose_variants("first line\nsecond line")
    assert variants[0] == "first line\nsecond line"


def test_a_derived_variant_is_never_empty():
    """Several verifiers accept the empty string -- "no commas" is true of it.
    A one-line reply must not pass by deleting its only line."""
    assert "" not in loose_variants("one line only")[1:]


@pytest.mark.parametrize("instruction_id", sorted(CASES))
def test_loose_is_never_stricter_than_strict(instruction_id):
    """Loose is strict plus extra chances, so it can only ever be more lenient.

    This caught a real bug. The empty-variant filter was dropping the *original*
    text when it was empty, leaving nothing to check -- and several verifiers
    are vacuously true of "" ("no commas", "fewer than ten words"), so an empty
    reply passed strict and failed loose. On the frozen set that read as strict
    0.132 against loose 0.000: a config that had stopped answering entirely
    would have scored better on the harsher metric.
    """
    kwargs, good, bad = CASES[instruction_id]
    for reply in (good, bad, "", "   "):
        if verify(instruction_id, kwargs, reply):
            assert verify_loose(instruction_id, kwargs, reply), (
                f"{instruction_id}: strict passes {reply!r} but loose does not"
            )


def test_an_empty_reply_is_not_rescued_or_punished_by_looseness():
    """The specific regression: both levels must agree about an empty reply."""
    assert verify("punctuation:no_comma", {}, "") is True
    assert verify_loose("punctuation:no_comma", {}, "") is True


# --------------------------------------------------------------------------
# per-item aggregation
# --------------------------------------------------------------------------


def test_an_item_is_correct_only_if_every_instruction_holds():
    """Prompt-level accuracy is all-or-nothing, as the original defines it."""
    result = evaluate(
        ["punctuation:no_comma", "change_case:english_lowercase"],
        [{}, {}],
        "no commas and all lowercase",
    )
    assert result["strict"] is True

    partial = evaluate(
        ["punctuation:no_comma", "change_case:english_lowercase"],
        [{}, {}],
        "no commas but WITH capitals",
    )
    assert partial["strict"] is False
    # Still credited at the instruction level: one of the two was obeyed, and
    # that is the difference between "ignored the format" and "missed one rule".
    assert partial["n_strict_followed"] == 1
    assert partial["n_instructions"] == 2


def test_mismatched_instruction_and_kwargs_lists_are_refused():
    """They are parallel arrays in the source data; a zip would silently drop."""
    with pytest.raises(ValueError, match="kwargs entries"):
        evaluate(["punctuation:no_comma", "startend:quotation"], [{}], "text")


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def test_contractions_and_hyphenates_are_one_word_each():
    assert count_words("don't well-known state-of-the-art") == 3


def test_sentence_counting_ignores_trailing_whitespace_and_empty_pieces():
    assert count_sentences("One. Two. Three.") == 3
    assert count_sentences("  ") == 0


def test_a_relation_that_is_not_one_of_the_two_is_refused():
    """A typo in a kwargs field would otherwise silently score under the wrong
    comparison, and every item carrying it would be wrong in the same direction."""
    with pytest.raises(ValueError, match="unknown relation"):
        verify(
            "length_constraints:number_words",
            {"relation": "more than", "num_words": 5},
            "one two three four five six",
        )


# --------------------------------------------------------------------------
# against the real dataset
# --------------------------------------------------------------------------


def test_every_real_kwargs_entry_is_accepted_by_its_verifier(ifeval_source):
    """Sweep every instruction in the frozen set, not just the cases above.

    The hand-written kwargs are what I *think* the schema is. This is what it
    actually is, across every item the study will score: a verifier that reads
    `num_words` where the data says `num_word`, or assumes a key absent on some
    items, raises here rather than half way through a pass on the GPU box.

    Only crash-freedom is asserted, not the verdict -- these replies are not
    compliant answers to the items, so what each returns is meaningless. That it
    returns at all is the point.
    """
    expected = sum(len(row["instruction_id_list"]) for row in ifeval_source)
    checked = 0
    for row in ifeval_source:
        for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            for reply in ("A short reply.", '{"a": 1}', "*x*\n* y\n"):
                assert isinstance(verify(instruction_id, kwargs, reply), bool)
            checked += 1
    assert checked == expected
    assert checked > 300, f"only {checked} instructions in the frozen set; too few to sweep"


def test_the_frozen_draw_covers_every_instruction_type(ifeval_source):
    """The reason the draw is stratified.

    A type missing from the 250 is a constraint this study never measures, and
    the rarer types are exactly the ones a degraded config drops first. If this
    fails, the draw has stopped covering the benchmark and `n_items` or the
    stratification key needs revisiting -- not the verifier list.
    """
    drawn = {i for row in ifeval_source for i in row["instruction_id_list"]}
    assert drawn == set(VERIFIERS), f"not drawn: {sorted(set(VERIFIERS) - drawn)}"


def test_the_published_items_use_only_instructions_we_implement(ifeval_source):
    """A missing verifier must surface here, not as a KeyError mid-pass."""
    used = {i for row in ifeval_source for i in row["instruction_id_list"]}
    assert used <= set(VERIFIERS), f"no verifier for: {sorted(used - set(VERIFIERS))}"
