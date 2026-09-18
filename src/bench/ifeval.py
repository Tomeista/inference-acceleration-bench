"""IFEval's verifiable instructions, reimplemented.

IFEval asks the model to obey constraints a program can check -- "at least 300
words", "no commas", "wrap the whole reply in double quotes" -- so the score
needs no judge and no answer key beyond the constraint itself. That is why it is
in this suite: it is generation-based, so it moves under quantization the way
multiple choice does not, and it is deterministic, so two configs measured a
month apart are comparable.

Reimplemented here rather than imported, for the reason the rest of `evals/` is
frozen: the run path takes no dataset dependency and no harness dependency, and
every number the study reports comes out of code in this repository.

What that costs, stated plainly
-------------------------------
These scores will not match published IFEval numbers exactly. Two deliberate
deviations, both of which would otherwise drag a tokenizer model into the run
path:

  * words are counted with a regex, where the original uses `nltk.word_tokenize`
  * sentences are split with a regex, where the original uses `nltk.sent_tokenize`

Both differ on edge cases -- hyphenation, abbreviations, decimals -- so an item
sitting within a word or two of its threshold can be judged differently here.
The comparison this study makes is each config against its own BF16 reference on
byte-identical items, and a scorer that is consistently slightly different is
sound for that; a scorer that is *inconsistently* different is not, which is why
these are pure functions with no model, no randomness, and a test per verifier.

The one exception is `language:response_language`, which genuinely needs a
language identifier. `langdetect` is seeded to 0 at import, as the original
does: left unseeded it samples, and the same reply would verify differently on
two runs.

Strict and loose
----------------
Reported both ways, as the original does. Loose re-checks each instruction
against a handful of lightly edited variants of the reply -- markdown stars
removed, first and/or last line dropped -- and passes if any variant satisfies
it. It exists because models often wrap a compliant answer in "Sure, here you
go:", which fails a strict check for a reason that is not the constraint. The
gap between strict and loose is itself informative: a config whose loose score
holds while its strict score falls has kept the capability and lost the format.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

# --------------------------------------------------------------------------
# counting helpers
# --------------------------------------------------------------------------

# Words are runs of letters/digits, optionally carrying internal apostrophes or
# hyphens, so "don't" and "well-known" each count once.
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

# A sentence ends at . ! or ? followed by whitespace or the end of the text.
# Deliberately simple, and it over-splits on abbreviations ("Dr. Who" is two).
# The original's nltk splitter does not; see the module docstring.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")

# A highlighted section is *text* or **text** with something inside it.
_HIGHLIGHT = re.compile(r"\*{1,2}([^\*\n]+?)\*{1,2}")

# A title is <<like this>>.
_TITLE = re.compile(r"<<([^>]+)>>")

# A placeholder is [like this].
_PLACEHOLDER = re.compile(r"\[[^\[\]]*\]")

# The three answers `detectable_format:constrained_response` allows.
CONSTRAINED_RESPONSES = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

RELATIONS = ("at least", "less than")


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def count_sentences(text: str) -> int:
    """Non-empty chunks between sentence-ending punctuation."""
    stripped = text.strip()
    if not stripped:
        return 0
    pieces = [p for p in _SENTENCE_END.split(stripped) if p.strip()]
    return len(pieces)


def compare(value: int, relation: str, target: int) -> bool:
    """IFEval's two relations, and nothing else.

    Refuses an unknown relation rather than defaulting to one of them: a typo
    in a kwargs field would otherwise silently score every item that carries it
    under the wrong comparison.
    """
    if relation == "at least":
        return value >= target
    if relation == "less than":
        return value < target
    raise ValueError(f"unknown relation {relation!r}; expected one of {RELATIONS}")


def count_keyword(text: str, keyword: str) -> int:
    """Case-insensitive whole-word occurrences."""
    return len(re.findall(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE))


# --------------------------------------------------------------------------
# the verifiers -- one per instruction id
# --------------------------------------------------------------------------
#
# Each takes (response, kwargs) and returns whether the response obeys it.
# Signature kept uniform so the registry below is a plain dict and an unknown
# instruction id is a loud KeyError rather than a silent pass.


def _no_comma(response: str, kw: Mapping[str, Any]) -> bool:
    return "," not in response


def _number_words(response: str, kw: Mapping[str, Any]) -> bool:
    return compare(count_words(response), kw["relation"], int(kw["num_words"]))


def _number_sentences(response: str, kw: Mapping[str, Any]) -> bool:
    return compare(count_sentences(response), kw["relation"], int(kw["num_sentences"]))


def _forbidden_words(response: str, kw: Mapping[str, Any]) -> bool:
    return all(count_keyword(response, word) == 0 for word in kw["forbidden_words"])


def _existence(response: str, kw: Mapping[str, Any]) -> bool:
    return all(count_keyword(response, word) > 0 for word in kw["keywords"])


def _keyword_frequency(response: str, kw: Mapping[str, Any]) -> bool:
    return compare(
        count_keyword(response, kw["keyword"]), kw["relation"], int(kw["frequency"])
    )


def _letter_frequency(response: str, kw: Mapping[str, Any]) -> bool:
    """Character count, not word count.

    `letter` is not always a letter: items use '#' to ask for hashtags. So this
    counts occurrences of the character, lowercased on both sides so that a
    request for 'a' is not satisfied only by lowercase ones.
    """
    letter = str(kw["letter"]).lower()
    count = response.lower().count(letter)
    return compare(count, kw["let_relation"], int(kw["let_frequency"]))


def _number_highlighted_sections(response: str, kw: Mapping[str, Any]) -> bool:
    highlights = [h for h in _HIGHLIGHT.findall(response) if h.strip()]
    return len(highlights) >= int(kw["num_highlights"])


def _number_bullet_lists(response: str, kw: Mapping[str, Any]) -> bool:
    """Exactly N markdown bullets. Exact, not at-least: the prompt says exactly."""
    bullets = re.findall(r"^\s*[\*\-]\s+\S", response, re.MULTILINE)
    return len(bullets) == int(kw["num_bullets"])


def _title(response: str, kw: Mapping[str, Any]) -> bool:
    return any(t.strip() for t in _TITLE.findall(response))


def _number_placeholders(response: str, kw: Mapping[str, Any]) -> bool:
    return len(_PLACEHOLDER.findall(response)) >= int(kw["num_placeholders"])


def _postscript(response: str, kw: Mapping[str, Any]) -> bool:
    """A postscript marker, at the start of a line near the end.

    Case-insensitive and tolerant of the "P.P.S" variants the prompts ask for.
    Checked as a line start rather than anywhere, so a reply that merely
    mentions "P.S." mid-sentence does not pass.
    """
    marker = re.escape(str(kw["postscript_marker"]).strip())
    return re.search(rf"^\s*{marker}", response, re.IGNORECASE | re.MULTILINE) is not None


def _number_paragraphs(response: str, kw: Mapping[str, Any]) -> bool:
    """Paragraphs separated by the markdown divider `***`, exactly N of them."""
    parts = [p for p in re.split(r"\s?\*\*\*\s?", response) if p.strip()]
    return len(parts) == int(kw["num_paragraphs"])


def _nth_paragraph_first_word(response: str, kw: Mapping[str, Any]) -> bool:
    """These items separate paragraphs with a blank line, not with `***`."""
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) != int(kw["num_paragraphs"]):
        return False
    index = int(kw["nth_paragraph"]) - 1
    if not 0 <= index < len(paragraphs):
        return False
    words = _WORD.findall(paragraphs[index])
    return bool(words) and words[0].lower() == str(kw["first_word"]).lower()


def _multiple_sections(response: str, kw: Mapping[str, Any]) -> bool:
    spliter = re.escape(str(kw["section_spliter"]))
    found = re.findall(rf"{spliter}\s+\d+", response, re.IGNORECASE)
    return len(found) >= int(kw["num_sections"])


def _constrained_response(response: str, kw: Mapping[str, Any]) -> bool:
    return any(choice in response for choice in CONSTRAINED_RESPONSES)


def _json_format(response: str, kw: Mapping[str, Any]) -> bool:
    """The whole reply parses as JSON, once a markdown fence is peeled off.

    The fence is stripped because models wrap JSON in ```json by habit and the
    instruction is about the output being JSON, not about fences. Anything left
    over that does not parse is a failure.
    """
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def _two_responses(response: str, kw: Mapping[str, Any]) -> bool:
    parts = [p for p in response.split("******") if p.strip()]
    return len(parts) == 2


def _repeat_prompt(response: str, kw: Mapping[str, Any]) -> bool:
    return response.strip().lower().startswith(str(kw["prompt_to_repeat"]).strip().lower())


def _quotation(response: str, kw: Mapping[str, Any]) -> bool:
    text = response.strip()
    return len(text) >= 2 and text.startswith('"') and text.endswith('"')


def _end_checker(response: str, kw: Mapping[str, Any]) -> bool:
    return response.strip().lower().endswith(str(kw["end_phrase"]).strip().lower())


def _english_lowercase(response: str, kw: Mapping[str, Any]) -> bool:
    return response == response.lower()


def _english_capital(response: str, kw: Mapping[str, Any]) -> bool:
    return response == response.upper()


def _capital_word_frequency(response: str, kw: Mapping[str, Any]) -> bool:
    """Words written entirely in capitals.

    A single capital letter is not an all-caps word -- "I" and "A" would
    otherwise make almost every reply fail a `less than` constraint.
    """
    caps = [w for w in _WORD.findall(response) if len(w) > 1 and w.isupper()]
    return compare(len(caps), kw["capital_relation"], int(kw["capital_frequency"]))


def _response_language(response: str, kw: Mapping[str, Any]) -> bool:
    """The only verifier that needs a model, and it is seeded.

    Left unseeded, langdetect samples: the same reply verifies differently on
    two runs, and a config would appear to drift for reasons that are the
    scorer's. Seeded at import, as the original IFEval does.
    """
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    text = response.strip()
    if not text:
        return False
    try:
        return detect(text) == str(kw["language"]).lower()
    except Exception:  # noqa: BLE001 -- langdetect raises on text it cannot judge
        return False


VERIFIERS: dict[str, Callable[[str, Mapping[str, Any]], bool]] = {
    "punctuation:no_comma": _no_comma,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:existence": _existence,
    "keywords:frequency": _keyword_frequency,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:title": _title,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:json_format": _json_format,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:quotation": _quotation,
    "startend:end_checker": _end_checker,
    "change_case:english_lowercase": _english_lowercase,
    "change_case:english_capital": _english_capital,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "language:response_language": _response_language,
}


# --------------------------------------------------------------------------
# strict and loose
# --------------------------------------------------------------------------


def loose_variants(response: str) -> list[str]:
    """The reply, plus the lightly edited versions the loose score also accepts.

    Models routinely wrap a compliant answer in "Sure, here you go:" or a
    closing "Let me know if you'd like changes!", which fails a strict check for
    a reason that has nothing to do with the constraint. Dropping the first
    line, the last line, or both -- each with and without markdown stars --
    is the original's way of measuring the constraint rather than the wrapper.
    """
    text = response.strip()
    lines = text.split("\n")
    derived = [
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]

    # The strict text always leads, and is kept even when it is empty. Loose
    # must never be *stricter* than strict -- it is strict plus extra chances --
    # and dropping an empty original would do exactly that: several verifiers
    # are vacuously true of "" ("no commas", "fewer than 10 words"), so an empty
    # reply would pass strict and fail loose.
    variants: list[str] = [text, text.replace("*", "")]

    # The derived variants are dropped when empty, which is the case that
    # motivated the filter: a one-line reply must not pass by deleting its only
    # line and satisfying the constraint vacuously.
    for candidate in derived:
        if candidate:
            variants.append(candidate)
            variants.append(candidate.replace("*", ""))

    seen: set[str] = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def verify(instruction_id: str, kwargs: Mapping[str, Any], response: str) -> bool:
    """Strict: does this exact reply obey this one instruction?"""
    if instruction_id not in VERIFIERS:
        raise KeyError(
            f"no verifier for instruction {instruction_id!r}; known: {sorted(VERIFIERS)}"
        )
    return VERIFIERS[instruction_id](response, kwargs or {})


def verify_loose(instruction_id: str, kwargs: Mapping[str, Any], response: str) -> bool:
    """Loose: does any lightly edited variant of the reply obey it?"""
    return any(verify(instruction_id, kwargs, v) for v in loose_variants(response))


def evaluate(
    instruction_ids: list[str], kwargs_list: list[Mapping[str, Any]], response: str
) -> dict[str, Any]:
    """Both scores for one item, at both levels.

    Four numbers, as the original reports: an item is strict-correct only if
    *every* one of its instructions holds, while the instruction-level counts
    are summed across items -- which is why they leave here as a numerator and
    a denominator rather than as a ratio. Items carry between one and three
    instructions, so a mean of per-item ratios would weight them wrongly.
    """
    if len(instruction_ids) != len(kwargs_list):
        raise ValueError(
            f"{len(instruction_ids)} instructions but {len(kwargs_list)} kwargs entries"
        )
    strict = [verify(i, k, response) for i, k in zip(instruction_ids, kwargs_list)]
    loose = [verify_loose(i, k, response) for i, k in zip(instruction_ids, kwargs_list)]
    return {
        "strict": all(strict),
        "loose": all(loose),
        "n_instructions": len(strict),
        "n_strict_followed": sum(strict),
        "n_loose_followed": sum(loose),
        # Per-instruction verdicts, kept so the agreement join can pair two
        # configs on *which* instructions each obeyed rather than on a single
        # pass/fail. Two configs can both fail an item for opposite reasons,
        # and only the pattern sees that.
        "strict_flags": strict,
        "loose_flags": loose,
    }


def verdict_string(flags: list[bool]) -> str:
    """The per-instruction pattern, as the compact string `extracted` carries."""
    return "".join("1" if flag else "0" for flag in flags)
