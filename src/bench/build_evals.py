"""Build the frozen eval sets. Run once; the output is committed.

Deliberately outside the run path, the same way `bench.build_prompts` is: it
needs the network and pyarrow, and nothing that *measures* should need either.
A clone can run the whole quality pass from the committed `evals/` directory
with no dataset access at all.

    python -m bench.build_evals             # fetch, sample, write evals/
    python -m bench.build_evals --check     # digests only, no network

What is pinned, and why each one matters
----------------------------------------
*The commit.* `suites.yaml` names a dataset revision, not `main`. MMLU and GSM8K
have both been re-uploaded before; a set that changes underneath the study makes
configs measured weeks apart incomparable while every run still succeeds.

*The sample.* Fixed seed, and MMLU is stratified across all 57 subjects rather
than sampled flat -- a flat 250-of-14042 draw would over-weight the large
subjects and give a score that moves when the sample moves.

*The bytes.* Written with an explicit LF newline rather than the platform
default. `bench.scenarios.write_scenarios` opens in text mode, so on Windows
it would emit CRLF and a rebuild on the Linux GPU box would produce different
digests for identical content -- which is exactly the drift `manifest.json`
exists to catch, arriving from the tool that writes the manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import httpx

from bench.scenarios import Scenario, Turn
from bench.scoring import normalize_number
from bench.suites import MANIFEST_PATH, Suite, digest, load_suites

SEED = 0


def extra_body_for(suite: Suite) -> dict[str, Any]:
    """The per-request pins for one suite, from the suite's own fields.

    Passed through to vLLM on every quality request. Three pins, and the
    absence of a fourth:

      top_k 1 + temperature 0  greedy. Two configs must differ because their
                               weights differ, not because their samplers
                               rolled apart.
      seed                     belt and braces; greedy should not consult it.
      enable_thinking false    matches prompts/*.jsonl. Left on, Qwen3 reasons
                               for as long as it likes and GSM8K's output
                               length -- and therefore its truncation rate --
                               becomes a property of the config's verbosity
                               rather than of the suite.

    No `ignore_eos`. The speed sweep pins output length with it so content
    cannot affect timing; scoring needs the opposite, and a test asserts it
    stays absent.

    Derived rather than constant because the thinking arm needs the opposite of
    all that: Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k
    20) instead of greedy, because Qwen warns that greedy decoding in thinking
    mode can run away into repetition -- which would then be read as
    quantization damage.

    Key order matters and is not cosmetic. These dicts are serialized into the
    frozen prompt files, so re-ordering them changes the digests of eval sets
    that have already been measured. `top_p` is therefore appended rather than
    inserted, and a suite that does not set it emits exactly the three keys the
    original three suites were built with. A test asserts that byte-for-byte.
    """
    body: dict[str, Any] = {
        "top_k": suite.top_k,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": suite.thinking},
    }
    if suite.top_p is not None:
        body["top_p"] = suite.top_p
    return body


MMLU_INSTRUCTION = (
    'Reply with exactly "Answer: X", where X is A, B, C or D. Do not explain.'
)
GSM8K_INSTRUCTION = (
    "Solve the problem step by step. Then give the final numeric answer on its "
    'own last line, in the form "#### <number>".'
)
# Templated rather than fixed, because MMLU-Pro's option count is not constant:
# most items carry ten, but 2,051 of the 12,032 carry between three and nine
# after the authors dropped choices they judged unreasonable. Naming a range the
# item does not have would invite a letter that cannot be right.
MMLU_PRO_INSTRUCTION = (
    'Reply with exactly "Answer: X", where X is one of A-{last}. Do not explain.'
)

LETTERS = "ABCD"
LETTERS10 = "ABCDEFGHIJ"


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def source_url(suite: Suite) -> str:
    """Where this suite's items come from, pinned to a commit.

    Two fetched provenances, one URL shape each. Both resolve a commit rather
    than a branch, so re-running this a year from now rebuilds the same set or
    fails loudly -- which is the whole point of pinning.
    """
    if suite.provenance == "huggingface":
        return (
            f"https://huggingface.co/datasets/{suite.dataset}"
            f"/resolve/{suite.revision}/{suite.parquet}"
        )
    if suite.provenance == "repository":
        # Raw file at a commit. The Gorilla data files are not on the Hub, and
        # cloning a repository to read four files of it is a worse dependency
        # than one HTTP GET.
        return f"https://raw.githubusercontent.com/{suite.dataset}/{suite.revision}/{suite.parquet}"
    raise ValueError(f"suite {suite.id!r} is {suite.provenance} and has nothing to fetch")


def fetch_rows_at(suite: Suite, path: str) -> list[dict]:
    """One named file from this suite's pinned source, beside `parquet`.

    Same commit, same URL shape, different file. BFCL is split across a file per
    category plus a separate answer key for each, so its builder needs several.
    """
    return fetch_rows(replace(suite, parquet=path))


def fetch_rows(suite: Suite) -> list[dict]:
    """The pinned source file, as a list of row dicts.

    Dispatches on the file's extension rather than on the provenance: the
    benchmarks worth having are not all parquet. IFEval ships one JSONL, MMMLU
    a CSV per language, Belebele a JSONL per language variant, and the Gorilla
    data files are JSONL in a git repo. Only the parquet path needs pyarrow,
    which is why it is imported inside the branch -- a JSONL suite rebuilds
    without the `evals` extra installed at all.
    """
    url = source_url(suite)
    print(f"  fetching {url}")
    response = httpx.get(url, follow_redirects=True, timeout=180.0)
    response.raise_for_status()

    suffix = suite.parquet.rsplit(".", 1)[-1].lower()
    if suffix == "parquet":
        import pyarrow.parquet as pq

        return pq.read_table(io.BytesIO(response.content)).to_pylist()
    if suffix in ("jsonl", "json"):
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if suffix == "csv":
        return list(csv.DictReader(io.StringIO(response.text)))
    raise ValueError(f"suite {suite.id!r}: no reader for a {suffix!r} source file")


# --------------------------------------------------------------------------
# builders -- one per `source` in config/suites.yaml
# --------------------------------------------------------------------------


def build_mmlu(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """250 items, stratified round-robin across every subject.

    Stratified rather than flat: subject sizes range from 100 to 1534, so a flat
    draw would be most of a score about professional law and high-school
    psychology. Round-robin over subjects shuffled at a fixed seed gives an even
    spread and a reproducible one.
    """
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(SEED)
    for subject in sorted(by_subject):
        rng.shuffle(by_subject[subject])

    picked: list[dict] = []
    subjects = sorted(by_subject)
    depth = 0
    while len(picked) < suite.n_items:
        added = False
        for subject in subjects:
            if depth < len(by_subject[subject]):
                picked.append(by_subject[subject][depth])
                added = True
                if len(picked) == suite.n_items:
                    break
        if not added:
            break
        depth += 1

    scenarios, key = [], []
    for i, row in enumerate(picked):
        options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(row["choices"]))
        content = f"{row['question'].strip()}\n\n{options}\n\n{MMLU_INSTRUCTION}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": LETTERS[int(row["answer"])],
                "meta": {"subject": row["subject"]},
            }
        )

    return scenarios, key, {"subjects": len(subjects), "instruction": MMLU_INSTRUCTION}


def build_gsm8k(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """250 items sampled flat -- GSM8K has no subject axis to stratify over."""
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(suite.n_items, len(rows)))

    scenarios, key = [], []
    for i, row in enumerate(picked):
        content = f"{row['question'].strip()}\n\n{GSM8K_INSTRUCTION}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        # The gold answer sits after the "#### " marker in the reference
        # solution. Normalized at build time with the same function the scorer
        # uses, so "1,000" in the dataset and "1000" from the model are one
        # answer rather than a scoring bug discovered on the GPU box.
        gold = row["answer"].rsplit("####", 1)[-1]
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": normalize_number(gold),
                "meta": {},
            }
        )

    return scenarios, key, {"instruction": GSM8K_INSTRUCTION}


def build_mmlu_pro(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """250 items, stratified round-robin across all 14 categories.

    Same construction as `build_mmlu` and for the same reason -- category sizes
    run from 381 (history) to 1351 (math), so a flat draw would be mostly maths
    and law -- but over 14 categories rather than 57 subjects, which is how
    MMLU-Pro reorganised MMLU's subject list.

    The two suites overlap by construction: 6,810 of MMLU-Pro's 12,032 items are
    MMLU questions that survived its filtering pass, so a config's scores here
    and on `mmlu` are correlated rather than independent readings. Worth stating in
    any write-up that reports both.
    """
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    rng = random.Random(SEED)
    for category in sorted(by_category):
        rng.shuffle(by_category[category])

    picked: list[dict] = []
    categories = sorted(by_category)
    depth = 0
    while len(picked) < suite.n_items:
        added = False
        for category in categories:
            if depth < len(by_category[category]):
                picked.append(by_category[category][depth])
                added = True
                if len(picked) == suite.n_items:
                    break
        if not added:
            break
        depth += 1

    scenarios, key = [], []
    for i, row in enumerate(picked):
        choices = list(row["options"])
        index = int(row["answer_index"])
        # Both guards are re-upload detectors rather than defensive padding: on
        # the pinned commit every row satisfies them. A future revision that
        # renumbered options or truncated a list would otherwise build a key
        # that points at the wrong choice, and every config would score
        # against it equally -- a study-wide error that no downstream check
        # could see.
        if not 0 <= index < len(choices):
            raise ValueError(
                f"{suite.id} item {row['question_id']}: answer_index {index} is "
                f"outside its {len(choices)} options. The dataset revision has "
                f"changed shape; re-check the pin in config/suites.yaml."
            )
        if row["answer"] != LETTERS10[index]:
            raise ValueError(
                f"{suite.id} item {row['question_id']}: answer {row['answer']!r} "
                f"and answer_index {index} disagree. See above."
            )

        last = LETTERS10[len(choices) - 1]
        options = "\n".join(f"{LETTERS10[j]}. {c}" for j, c in enumerate(choices))
        instruction = MMLU_PRO_INSTRUCTION.format(last=last)
        content = f"{row['question'].strip()}\n\n{options}\n\n{instruction}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": LETTERS10[index],
                "meta": {"category": row["category"], "src": row["src"]},
            }
        )

    # `src` carries each item's provenance -- "ori_mmlu-*" for the inherited
    # questions, "stemez-*"/"theoremQA-*"/"scibench-*" for the new ones -- so the
    # overlap with the `mmlu` suite stays auditable from the frozen key alone.
    from_mmlu = sum(1 for r in key if str(r["meta"]["src"]).startswith("ori_mmlu"))
    return (
        scenarios,
        key,
        {
            "categories": len(categories),
            "instruction": MMLU_PRO_INSTRUCTION,
            "n_from_original_mmlu": from_mmlu,
        },
    )


def _stratified(rows: list[dict], key_of, n_items: int) -> list[dict]:
    """Round-robin over groups shuffled at a fixed seed.

    The draw `build_mmlu` uses, lifted out so the suites that need it share one
    implementation. Flat sampling would follow the source's group sizes, and a
    score that moves when the sample moves is not a measurement.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(key_of(row)), []).append(row)

    rng = random.Random(SEED)
    for group in sorted(grouped):
        rng.shuffle(grouped[group])

    picked: list[dict] = []
    groups = sorted(grouped)
    depth = 0
    while len(picked) < n_items:
        added = False
        for group in groups:
            if depth < len(grouped[group]):
                picked.append(grouped[group][depth])
                added = True
                if len(picked) == n_items:
                    break
        if not added:
            break
        depth += 1
    return picked


def build_ifeval(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """250 of IFEval's 541 items, stratified over the instruction they lead with.

    Stratified rather than flat for the usual reason, and for one specific to
    this suite: instruction types are unevenly represented (66 items carry
    `punctuation:no_comma`, 10 carry `detectable_format:constrained_response`),
    so a flat draw would under-sample exactly the rarer constraints that a
    degraded config is most likely to drop. A test asserts the drawn set still
    covers every type the verifiers implement.

    The prompt is the item's own text, unmodified. IFEval items already carry
    their instruction inside the prompt -- that is what makes them verifiable --
    so appending an answer-format line the way the MC suites do would add a
    constraint the key does not know about and score the model against it.
    """
    picked = _stratified(rows, lambda r: r["instruction_id_list"][0], suite.n_items)

    scenarios, key = [], []
    for i, row in enumerate(picked):
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": row["prompt"]}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        # The published kwargs carry explicit nulls for fields an instruction
        # does not use. Dropped here rather than in the verifiers, so the
        # verifiers can read a key and fail loudly when it is genuinely absent.
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in row["kwargs"]]
        key.append(
            {
                "scenario_id": scenario_id,
                # No gold answer exists: the constraint *is* the gold. This is
                # the display string, and the scorer reads `meta` instead.
                "answer": ",".join(row["instruction_id_list"]),
                "meta": {
                    "source_key": row["key"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": kwargs,
                },
            }
        )

    types = sorted({i for r in key for i in r["meta"]["instruction_id_list"]})
    return scenarios, key, {"instruction_types": len(types), "types": types}



# German answer-format instructions. In German on purpose.
#
# The alternative -- an English instruction wrapped around a German question --
# would isolate content language as the only variable against the `mmlu` suite,
# which is tidier. It is also not the thing being asked. An Atos deployment
# serving German-market customers prompts in German, so the realistic
# measurement is German end to end, and instruction-following in German is part
# of what degrades. The confound this accepts is stated in the suite
# description: `mmlu` vs `mmlu_de` differs in question language AND instruction
# language, and a gap between them is the two together.
MMLU_DE_INSTRUCTION = (
    'Antworte mit genau "Answer: X", wobei X A, B, C oder D ist. Keine Erklärung.'
)
BELEBELE_DE_INSTRUCTION = (
    'Antworte mit genau "Answer: X", wobei X A, B, C oder D ist. Keine Erklärung.'
)


def build_mmlu_de(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """MMLU in professionally translated German, item-matched to the `mmlu` suite.

    MMMLU is OpenAI's professional translation of the same 14,042 MMLU test
    items, in the same row order -- verified, not assumed: all 14,042 rows agree
    on subject with the pinned `cais/mmlu` commit. So running the identical
    stratified draw over this file selects the same questions the English suite
    drew, and `mmlu-0007` and `mmlu_de-0007` are the same question in two
    languages. That is what makes the English/German comparison paired per item
    rather than two independent samples of a benchmark, which at 250 items is
    the difference between a readable effect and noise.

    Professionally translated rather than machine-translated, which is the
    provenance the Occiglot maintainers themselves warn about: their German sets
    are machine-translated and sensitive to translation and prompt choices.

    Five of the 14,042 items carry a gold letter that disagrees with the English
    key. The options are in the same order in both files, so these are label
    differences rather than reordered choices, and on inspection the English key
    looks right in at least three of them. They are left exactly as the source
    has them -- this suite scores against its own published key -- and the
    disagreement is recorded per item in `meta` so it stays auditable. It costs
    the study nothing either way: every config is scored against the same key,
    so a wrong label subtracts the same amount from all of them and cancels in
    the paired comparison the study actually makes.
    """
    picked = _stratified(rows, lambda r: r["Subject"], suite.n_items)

    scenarios, key = [], []
    for i, row in enumerate(picked):
        choices = [row[letter] for letter in LETTERS]
        options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
        content = f"{row['Question'].strip()}\n\n{options}\n\n{MMLU_DE_INSTRUCTION}"
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        answer = row["Answer"].strip()
        if answer not in LETTERS:
            raise ValueError(
                f"{suite.id} item {i}: gold answer {answer!r} is not one of {LETTERS}. "
                f"The dataset revision has changed shape; re-check the pin."
            )
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": answer,
                "meta": {"subject": row["Subject"]},
            }
        )

    return (
        scenarios,
        key,
        {
            "subjects": len({r["meta"]["subject"] for r in key}),
            "instruction": MMLU_DE_INSTRUCTION,
            "instruction_language": "de",
            "translation": "professional (OpenAI MMMLU)",
        },
    )


def build_belebele_de(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """Belebele German: reading comprehension over a passage, natively parallel.

    A second German reading, and deliberately a different kind from `mmlu_de`.
    MMLU measures translated world knowledge, where a model can often answer
    from English-learned facts regardless of the question's language. Belebele
    gives the passage in German and asks a question that can only be answered by
    reading it, so it is much harder to pass on English knowledge alone -- which
    is the capability an enterprise German deployment actually needs.

    Flat sample rather than stratified: the 900 items carry no subject axis, and
    are already spread evenly over their source passages.
    """
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(suite.n_items, len(rows)))

    scenarios, key = [], []
    for i, row in enumerate(picked):
        choices = [row[f"mc_answer{n}"].strip() for n in (1, 2, 3, 4)]
        options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
        content = (
            f"{row['flores_passage'].strip()}\n\n"
            f"{row['question'].strip()}\n\n{options}\n\n{BELEBELE_DE_INSTRUCTION}"
        )
        scenario_id = f"{suite.id}-{i:04d}"
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": content}],
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        extra_body=extra_body_for(suite),
                    )
                ],
            )
        )
        index = int(row["correct_answer_num"]) - 1
        if not 0 <= index < len(LETTERS):
            raise ValueError(
                f"{suite.id} item {i}: correct_answer_num {row['correct_answer_num']!r} "
                f"is outside 1-4. The dataset revision has changed shape."
            )
        key.append(
            {
                "scenario_id": scenario_id,
                "answer": LETTERS[index],
                "meta": {"link": row["link"], "dialect": row["dialect"]},
            }
        )

    return (
        scenarios,
        key,
        {
            "instruction": BELEBELE_DE_INSTRUCTION,
            "instruction_language": "de",
            "passages": len({r["meta"]["link"] for r in key}),
        },
    )



# BFCL's single-turn categories, and what each is for. Drawn together and
# stratified so the suite measures all four rather than whichever is largest.
#
#   simple        one function offered, one call expected. The floor.
#   multiple      several functions offered, one is right. Tests selection.
#   parallel      one function, several calls expected from one instruction.
#   irrelevance   the offered functions cannot answer. The right move is to call
#                 NOTHING, and a model that invents a plausible call fails.
#
# Irrelevance is the category that makes this a tool-calling score rather than a
# call-formatting score: knowing when not to call is half of what makes an agent
# deployable, and it is not measured anywhere else in the suite.
BFCL_CATEGORIES = ("simple_python", "multiple", "parallel", "irrelevance")

BFCL_DATA = "berkeley-function-call-leaderboard/bfcl_eval/data"


def build_bfcl_ast(suite: Suite, rows: list[dict]) -> tuple[list[Scenario], list[dict], dict]:
    """250 single-turn tool-calling items, stratified over four categories.

    Served in NATIVE function-calling mode -- `tool_choice: auto` against the
    `tools` serve profile -- rather than by asking for JSON in the prompt. That
    is deliberate and it is what makes the structure-failure rate meaningful:
    the parser under test is then the one that would be deployed, and BFCL uses
    no guided decoding, so what is measured is the model's *unaided* ability to
    emit a well-formed call. Under guided decoding it would be ~100% by
    construction and the degradation signal would be masked.

    `tool_choice` travels in `extra_body` rather than as a Turn field because
    `Turn.to_payload` pins it to "none" for the speed sweep -- where tool
    schemas must render into the prompt without the parser buffering the
    stream -- and `extra_body` is applied last. Adding a field instead would
    change the serialized shape of every frozen prompt in `prompts/` and
    invalidate the speed sets.
    """
    scenarios, key = [], []
    index = 0
    per_category: dict[str, int] = {}

    # Drawn per category and interleaved, rather than pooled and stratified,
    # because each category has its own answer-key file to join against.
    drawn: list[tuple[str, dict, Any]] = []
    for position, category in enumerate(BFCL_CATEGORIES):
        items = fetch_rows_at(suite, f"{BFCL_DATA}/BFCL_v4_{category}.json")
        # Irrelevance has no answer key: there is no right call to record.
        truths: dict[str, Any] = {}
        if category != "irrelevance":
            truths = {
                row["id"]: row["ground_truth"]
                for row in fetch_rows_at(
                    suite, f"{BFCL_DATA}/possible_answer/BFCL_v4_{category}.json"
                )
            }
        rng = random.Random(SEED)
        rng.shuffle(items)
        # 250 does not divide by four. The remainder goes to the earliest
        # categories rather than being dropped, so the suite lands on exactly
        # `n_items` and the count does not silently become 248.
        share = suite.n_items // len(BFCL_CATEGORIES)
        if position < suite.n_items % len(BFCL_CATEGORIES):
            share += 1
        if len(items) < share:
            raise ValueError(
                f"{suite.id}: category {category} has {len(items)} items but "
                f"{share} were drawn from it"
            )
        for row in items[:share]:
            drawn.append((category, row, truths.get(row["id"])))

    for category, row, truth in drawn:
        # `question` is a list of turns, each a list of messages. Single-turn
        # categories carry exactly one turn; anything else belongs to the
        # multi-turn suite and would be silently truncated here.
        turns = row["question"]
        if len(turns) != 1:
            raise ValueError(
                f"{suite.id}: item {row['id']} has {len(turns)} turns, but this "
                f"suite is single-turn only"
            )
        messages = [dict(m) for m in turns[0]]

        scenario_id = f"{suite.id}-{index:04d}"
        index += 1
        per_category[category] = per_category.get(category, 0) + 1
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                class_id=suite.id,
                turns=[
                    Turn(
                        messages=messages,
                        max_tokens=suite.max_tokens,
                        temperature=suite.temperature,
                        tools=[
                            {"type": "function", "function": fn} for fn in row["function"]
                        ],
                        extra_body={**extra_body_for(suite), "tool_choice": "auto"},
                    )
                ],
            )
        )
        key.append(
            {
                "scenario_id": scenario_id,
                # Display only. The real gold is the acceptable-argument sets in
                # `meta`, and for irrelevance it is the absence of any call.
                "answer": "none" if truth is None else ";".join(sorted(t for g in truth for t in g)),
                "meta": {
                    "source_id": row["id"],
                    "category": category,
                    "ground_truth": truth,
                },
            }
        )

    return scenarios, key, {"categories": per_category}


BUILDERS: dict[str, Callable[[Suite, list[dict]], tuple[list[Scenario], list[dict], dict]]] = {
    "mmlu": build_mmlu,
    "mmlu_pro": build_mmlu_pro,
    "gsm8k": build_gsm8k,
    "ifeval": build_ifeval,
    "mmlu_de": build_mmlu_de,
    "belebele_de": build_belebele_de,
    "bfcl_ast": build_bfcl_ast,
}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    """Explicit LF, so the digest does not depend on which OS built the set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line + "\n")


def build(suite: Suite) -> dict:
    if suite.source not in BUILDERS:
        raise KeyError(
            f"suite {suite.id!r} names source {suite.source!r} with no builder; "
            f"known: {sorted(BUILDERS)}"
        )
    rows = fetch_rows(suite) if suite.provenance != "synthetic" else []
    scenarios, key, extra = BUILDERS[suite.source](suite, rows)

    if len(scenarios) != suite.n_items:
        print(
            f"  warning: {suite.id} produced {len(scenarios)} items, not "
            f"{suite.n_items}; the source split may be smaller than requested",
            file=sys.stderr,
        )

    _write_lines(suite.prompt_file, [s.to_json() for s in scenarios])
    _write_lines(suite.key_file, [json.dumps(row, ensure_ascii=False) for row in key])

    return {
        "suite_id": suite.id,
        "source": suite.source,
        "dataset": suite.dataset,
        "revision": suite.revision,
        "parquet": suite.parquet,
        "extra_sources": list(suite.extra_sources),
        "license": suite.license,
        "n_items": len(scenarios),
        "max_tokens": suite.max_tokens,
        "temperature": suite.temperature,
        "scorer": suite.scorer,
        "shots": 0,
        **extra,
        "sha256_16": digest(suite.prompt_file),
        "key_sha256_16": digest(suite.key_file),
    }


def check() -> int:
    """Re-hash the committed sets against the manifest. No network."""
    if not MANIFEST_PATH.exists():
        print(f"no {MANIFEST_PATH}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    suites = load_suites()
    bad = 0
    for suite_id, recorded in sorted((manifest.get("suites") or {}).items()):
        suite = suites[suite_id]
        for path, field in ((suite.prompt_file, "sha256_16"), (suite.key_file, "key_sha256_16")):
            actual = digest(path) if path.exists() else "MISSING"
            ok = actual == recorded[field]
            bad += 0 if ok else 1
            print(f"  {path.name:24s} {actual}  {'ok' if ok else '<- ' + recorded[field]}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", default="", help="Comma separated ids; default all enabled")
    parser.add_argument(
        "--check", action="store_true", help="Verify committed digests and exit; no network"
    )
    args = parser.parse_args()

    if args.check:
        return check()

    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    suites = load_suites()
    chosen = [suites[s] for s in requested] if requested else [
        s for s in suites.values() if s.enabled
    ]

    # Merge rather than replace: building one suite must not drop the other's
    # entry and leave a manifest that no longer describes what is on disk.
    manifest: dict[str, Any] = {"seed": SEED, "built_by": "bench.build_evals", "suites": {}}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest.setdefault("suites", {})

    for suite in chosen:
        print(f"{suite.id}: {suite.name}")
        manifest["suites"][suite.id] = build(suite)
        entry = manifest["suites"][suite.id]
        print(f"  wrote {entry['n_items']} items  {entry['sha256_16']} / {entry['key_sha256_16']}")

    manifest["seed"] = SEED
    manifest["built_by"] = "bench.build_evals"
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
