"""The quality pass: the frozen sets, the guards, and a whole cell end to end.

The mock has no model, so nothing here is a quality result. What it checks is
everything between the server and the number -- digest verification, the
answer-key join, extraction, the reference comparison, the merge rule -- which
would otherwise be exercised for the first time on the GPU box, at the end of a
long pass, with no way to tell a scorer bug from a bad checkpoint.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bench import quality as quality_mod
from bench.build_evals import extra_body_for
from bench.report import QUALITY_KEY, quality_problems, render_quality, select_cells
from bench.server import check_context_budget, load_configs
from bench.suites import (
    EVALS_DIR,
    MANIFEST_PATH,
    check_digests,
    digest,
    load_suite,
    load_suites,
    read_key,
    select_suites,
)

REFERENCE = quality_mod.DEFAULT_REFERENCE


@pytest.fixture(scope="module")
def suites():
    return load_suites()


@pytest.fixture(scope="module")
def configs():
    return load_configs()


# --------------------------------------------------------------------------
# the frozen eval sets
# --------------------------------------------------------------------------


def test_the_eval_files_are_the_bytes_the_manifest_recorded(suites):
    """The same guarantee prompts/manifest.json gives the speed sets.

    A regenerated or line-ending-mangled eval set would still load, still score,
    and still look plausible -- while making two configs measured either side of
    the change incomparable. Only the digest notices.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = manifest["suites"]
    assert recorded, "manifest lists no suites"

    for suite_id, stats in sorted(recorded.items()):
        suite = suites[suite_id]
        assert digest(suite.prompt_file) == stats["sha256_16"], f"{suite_id}.jsonl drifted"
        assert digest(suite.key_file) == stats["key_sha256_16"], f"{suite_id}.key.jsonl drifted"


def test_every_enabled_suite_has_its_files():
    for suite in select_suites():
        assert suite.prompt_file.exists(), f"{suite.id} has no prompt file"
        assert suite.key_file.exists(), f"{suite.id} has no key file"


def test_the_frozen_files_are_lf_so_a_rebuild_reproduces_the_digest(suites):
    """Written with an explicit newline rather than the platform default.

    `scenarios.write_scenarios` opens in text mode, so building on Windows
    would emit CRLF and building on the Linux GPU box would emit LF -- identical
    content, different digests, and the manifest would report drift that is
    really just a change of machine.
    """
    for suite in suites.values():
        raw = suite.prompt_file.read_bytes()
        assert b"\r\n" not in raw, f"{suite.prompt_file.name} has CRLF line endings"


def test_a_tampered_eval_set_is_refused(suites, tmp_path, monkeypatch):
    """The check has to fail closed, or it is decoration."""
    monkeypatch.setattr("bench.suites.EVALS_DIR", tmp_path)
    suite = suites["mmlu"]
    suite.prompt_file.write_text('{"scenario_id": "x"}\n', encoding="utf-8")
    suite.key_file.write_text('{"scenario_id": "x", "answer": "A"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not the frozen set"):
        check_digests(suite)


def test_prompts_and_key_describe_the_same_items(suites):
    """Two files that drift apart still run, and score noise."""
    for suite in suites.values():
        scenarios, answers = load_suite(suite)
        assert {s.scenario_id for s in scenarios} == set(answers)
        assert len(answers) == suite.n_items


def test_a_key_that_disagrees_with_the_prompts_is_refused(suites, tmp_path, monkeypatch):
    # Read the real set first: `prompt_file` resolves against EVALS_DIR at
    # access time, so everything after the patch points at tmp_path.
    scenarios, _ = load_suite(suites["mmlu"])

    monkeypatch.setattr("bench.suites.EVALS_DIR", tmp_path)
    suite = suites["mmlu"]
    suite.prompt_file.write_text(
        "\n".join(s.to_json() for s in scenarios[:4]) + "\n", encoding="utf-8"
    )
    suite.key_file.write_text(
        json.dumps({"scenario_id": "mmlu-9999", "answer": "A"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different items"):
        load_suite(suite, verify=False)


# --------------------------------------------------------------------------
# the pins that make two configs comparable
# --------------------------------------------------------------------------


def test_no_quality_prompt_pins_output_length(suites):
    """The exact inversion of the speed pass, and the reason for a separate set.

    `bench.run` sets ignore_eos so that content cannot affect timing. Scoring
    needs generation to stop where the model would stop; with ignore_eos on,
    every reply runs to max_tokens and is either truncated or padded past its
    answer, uniformly, in a way that looks like a bad checkpoint.
    """
    for suite in suites.values():
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                assert "ignore_eos" not in turn.extra_body, scenario.scenario_id


def test_every_quality_prompt_decodes_greedily(suites):
    """Two configs must differ because their weights differ, not their samplers.

    The thinking arm is the deliberate exception and is checked separately
    below: Qwen warns that greedy decoding with thinking on can run away into
    repetition, which would then be misread as quantization damage.
    """
    for suite in suites.values():
        if suite.thinking:
            continue
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                assert turn.temperature == 0.0
                assert turn.extra_body["top_k"] == 1
                # Left on, Qwen3 reasons for as long as it likes and GSM8K's
                # truncation rate becomes a property of the config's verbosity.
                assert turn.extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_a_thinking_suite_samples_the_way_qwen_recommends(suites):
    """The one place greedy is wrong, so it needs its own pin.

    Qwen's model card gives temperature 0.6 / top-p 0.95 / top-k 20 for thinking
    mode and explicitly warns against greedy there. A thinking suite left at
    temperature 0 would produce endless repetition that looks exactly like a
    damaged checkpoint, and `seeds` above 1 is what makes the resulting
    stochasticity reportable as a mean rather than hidden in a single draw.
    """
    for suite in suites.values():
        if not suite.thinking:
            continue
        assert suite.temperature == 0.6, f"{suite.id} is not at Qwen's thinking temperature"
        assert suite.top_p == 0.95, f"{suite.id} does not set Qwen's thinking top-p"
        assert suite.top_k == 20, f"{suite.id} does not set Qwen's thinking top-k"
        assert suite.seeds > 1, f"{suite.id} samples but draws once; report a mean, not a draw"
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                assert turn.extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_the_frozen_prompts_carry_the_sampling_their_suite_defines(suites):
    """The offline stand-in for a rebuild, and it guards a real hazard.

    `build_evals.extra_body_for` derives these pins from the suite's own fields
    rather than from a module constant. That is what lets the thinking arm ask
    for different sampling -- but it also means a future edit to that function,
    or to a suite's sampling fields, silently describes prompts that are not the
    ones on disk. Rebuilding would catch it; rebuilding needs the network, and
    this check does not.
    """
    for suite in suites.values():
        expected = extra_body_for(suite)
        scenarios, _ = load_suite(suite)
        for scenario in scenarios:
            for turn in scenario.turns:
                # Subset rather than equality: a suite may add keys of its
                # own (bfcl_ast sets tool_choice), but every sampling pin the
                # suite declares must be present and must have the declared value.
                carried = {k: turn.extra_body.get(k) for k in expected}
                assert carried == expected, (
                    f"{suite.id} {scenario.scenario_id}: the frozen prompt carries "
                    f"{carried}, but this suite now builds {expected}. "
                    f"Rebuild the set and re-measure every config, or revert the change."
                )


def test_the_quality_subset_covers_every_weight_change(configs):
    """The scored set is every config whose weights differ from the reference.

    The quantization ladder and the sparsity 2x2 are the comparisons the study
    makes; a config that changes the weights and is not scored has a speed
    number with no cost attached to it.
    """
    changes_weights = {c.id for c in configs.values() if c.spec_method == "none"}
    scored = {c.id for c in configs.values() if c.quality}
    assert changes_weights <= scored, f"not scored: {sorted(changes_weights - scored)}"


def test_the_reference_is_the_dense_bf16_config_and_is_scored(configs):
    """Everything is compared per item against it, so it has to be the one
    config with no compression at all, and it has to be in the subset."""
    reference = configs[REFERENCE]
    assert reference.quality
    assert reference.quant_method == "none"
    assert reference.bit_width == "16"
    assert reference.sparsity_pattern == "none"
    assert reference.spec_method == "none"


def test_speculative_configs_are_not_scored_by_default(configs):
    """Greedy speculative decoding answers what its verifier answers. Scoring it
    by default would spend a pass to re-measure the control."""
    for cfg in configs.values():
        if cfg.spec_method != "none":
            assert not cfg.quality, f"{cfg.id} is speculative but in the quality subset"


# --------------------------------------------------------------------------
# the suite config
# --------------------------------------------------------------------------


def test_unknown_key_in_a_suite_is_rejected(tmp_path):
    path = tmp_path / "suites.yaml"
    path.write_text(
        "defaults: {}\n"
        "suites:\n"
        "  - id: x\n"
        "    name: x\n"
        "    source: mmlu\n"
        "    dataset: d\n"
        "    revision: r\n"
        "    parquet: p\n"
        "    max_tokens: 16\n"
        "    scorer: mc\n"
        "    scorrer: mc\n",  # typo
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_suites(path)


def test_every_suite_names_a_scorer_that_exists(suites):
    from bench.scoring import SCORERS

    for suite in suites.values():
        assert suite.scorer in SCORERS


def test_a_single_suite_can_be_run_on_its_own(suites):
    """The pass must be selectable one benchmark at a time.

    Three suites, costing very different amounts -- mmlu and mmlu_pro are one
    letter of output, gsm8k is ~250 decode tokens an item. If selecting one
    were not possible, re-scoring a single cell would cost the whole pass on
    every config, which is how a study stops being re-run at all.
    """
    for suite_id in suites:
        assert [s.id for s in select_suites([suite_id])] == [suite_id]


def test_an_unknown_suite_id_is_refused_by_name():
    """A typo must not silently fall back to running everything."""
    with pytest.raises(KeyError, match="mmlupro"):
        select_suites(["mmlupro"])


def test_every_multiple_choice_gold_answer_is_offered_by_its_prompt(suites):
    """The key names a letter the item actually presents.

    `test_prompts_and_key_describe_the_same_items` checks that the two files
    agree on scenario ids; it cannot see a key whose letters have shifted
    against the rendered options -- an off-by-one in a builder, or a dataset
    re-upload that renumbered choices. That failure scores every config at
    chance equally, which looks like a result about the model rather than a
    bug, and nothing downstream of the scorer could distinguish the two.
    """
    for suite in suites.values():
        if not suite.scorer.startswith("mc"):
            continue
        scenarios, answers = load_suite(suite)
        for scenario in scenarios:
            prompt = scenario.turns[0].messages[0]["content"]
            offered = {
                line[0]
                for line in prompt.splitlines()
                if len(line) > 2 and line[0].isupper() and line[1] == "."
            }
            gold = answers[scenario.scenario_id]["answer"]
            assert gold in offered, (
                f"{suite.id} {scenario.scenario_id}: key says {gold}, but the "
                f"prompt offers {sorted(offered)}"
            )


def test_every_suite_pins_its_source(suites):
    """`main` would let the benchmark change underneath the study.

    Three provenance kinds, three things to pin. A fetched suite pins a commit;
    a generated one has no upstream commit to pin, so what has to be fixed
    instead is the generator and the corpus it drew from -- a suite generated
    at a fixed seed from changed source text is a changed suite, and only the
    corpus digest can see that.
    """
    for suite in suites.values():
        if suite.provenance == "synthetic":
            assert suite.generator, f"{suite.id} is synthetic but names no generator"
            assert len(suite.corpus_digest) == 16, (
                f"{suite.id} does not pin the corpus it was generated from"
            )
        else:
            assert len(suite.revision) == 40, f"{suite.id} is not pinned to a commit"


def test_a_suite_cannot_pin_its_source_to_a_branch(tmp_path):
    """The guard is in the loader, not only in this test file: a suite added on
    a branch pin must fail where it is defined, not silently measure."""
    path = tmp_path / "suites.yaml"
    path.write_text(
        "suites:\n"
        "  - id: drifty\n"
        "    name: Pinned to a branch\n"
        "    source: mmlu\n"
        "    dataset: cais/mmlu\n"
        "    revision: main\n"
        "    parquet: all/test.parquet\n"
        "    max_tokens: 16\n"
        "    scorer: mc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a 40-character commit"):
        load_suites(path)


def test_a_suite_records_itself_as_suite_id_not_class_id(suites):
    """Load-bearing. `report.load_cells` selects speed cells on class_id, so a
    quality cell carrying that name would be averaged into the speed tables."""
    params = suites["mmlu"].as_params()
    assert params["suite_id"] == "mmlu"
    assert "class_id" not in params


# --------------------------------------------------------------------------
# a whole cell, end to end
# --------------------------------------------------------------------------


def _expected_accuracy(letter: str, n: int) -> float:
    """What a server that always answers `letter` scores on the first n items."""
    rows = [json.loads(l) for l in (EVALS_DIR / "mmlu.key.jsonl").read_text().splitlines() if l]
    answers = [r["answer"] for r in rows[:n]]
    return answers.count(letter) / n


async def test_a_cell_scores_against_the_real_key(mock_server, configs, args, monkeypatch):
    """A server with a known, fixed answer has a known, computable accuracy."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    values, notes, results = await quality_mod.run_suite(
        suite, configs[REFERENCE], mock_server, args
    )

    assert len(results) == 8
    assert values["accuracy"] == pytest.approx(_expected_accuracy("C", 8))
    assert values["unparseable_rate"] == 0.0
    assert values["truncated_rate"] == 0.0
    assert values["accuracy_ci_lo"] < values["accuracy"] < values["accuracy_ci_hi"]


async def test_the_reference_join_is_paired_per_item(mock_server, configs, args, monkeypatch):
    """Score the reference, then a second config, and check the comparison.

    Same answers means agreement 1.0 even though accuracy is far from it -- the
    point of the metric is that it measures divergence from the reference, not
    correctness.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    reference_values, _, _ = await quality_mod.run_suite(
        suite, configs[REFERENCE], mock_server, args
    )
    # The reference has nothing to compare itself against.
    assert "agreement_with_reference" not in reference_values

    values, notes, _ = await quality_mod.run_suite(
        suite, configs["w4a16_gptq"], mock_server, args
    )
    assert values["agreement_with_reference"] == 1.0
    assert notes["reference_available"] is True

    monkeypatch.setenv("MOCK_REPLY", "Answer: A")
    diverged, _, _ = await quality_mod.run_suite(
        suite, configs["sparse24_w4a16"], mock_server, args
    )
    assert diverged["agreement_with_reference"] == 0.0
    # Accuracy moved too, but in the other direction: agreement and accuracy
    # are independent, which is exactly why both are reported.
    assert diverged["accuracy"] > values["accuracy"]


async def test_no_reference_yields_no_agreement_metric(mock_server, configs, args, monkeypatch):
    """A sweep that starts somewhere other than the reference still measures
    accuracy; it just cannot measure divergence, and must not claim zero."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    values, notes, _ = await quality_mod.run_suite(
        suite, configs["w4a16_gptq"], mock_server, args
    )
    assert "agreement_with_reference" not in values
    assert notes["reference_available"] is False


async def test_a_reference_from_different_eval_bytes_is_flagged(
    mock_server, configs, args, monkeypatch, capsys
):
    """`results/` is machine state: gitignored, regenerated, easily stale.

    An items file left over from a smoke test or from before the eval sets were
    rebuilt joins by scenario_id perfectly well and yields an agreement figure
    about nothing. Item ids cannot detect that; the sidecar recording what
    produced them can.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)

    # Rewrite the reference's sidecar as though it came from another eval set.
    reference = quality_mod.items_path(suite.id, REFERENCE)
    sidecar = quality_mod.meta_path(reference)
    meta = json.loads(sidecar.read_text())
    meta["eval_manifest"] = "mmlu:deadbeefdeadbeef"
    sidecar.write_text(json.dumps(meta), encoding="utf-8")

    _, notes, _ = await quality_mod.run_suite(suite, configs["w4a16_gptq"], mock_server, args)
    assert notes["reference_stale"] is True
    assert "were produced against eval sets" in capsys.readouterr().err


async def test_a_matching_reference_is_not_flagged(mock_server, configs, args, monkeypatch):
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    _, notes, _ = await quality_mod.run_suite(suite, configs["w4a16_gptq"], mock_server, args)
    assert notes["reference_stale"] is False


async def test_a_partial_reference_warns_about_the_overlap(
    mock_server, configs, args, monkeypatch, capsys
):
    """Agreement over 4 of 8 items is not agreement over the suite."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]

    args.n_items = 4
    await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    capsys.readouterr()

    args.n_items = 8
    values, _, _ = await quality_mod.run_suite(suite, configs["w4a16_gptq"], mock_server, args)
    assert values["agreement_n"] == 4
    assert "reference covers only 4 of 8" in capsys.readouterr().err


async def test_a_determinism_rerun_compares_against_the_main_pass(
    mock_server, configs, args, monkeypatch
):
    """`--suffix` on the reference itself: same weights, different batching.

    The rerun must read the reference's earlier file rather than the one it is
    about to write, or self-agreement would always come out at 1.0.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)

    args.suffix = "c1"
    monkeypatch.setenv("MOCK_REPLY", "Answer: B")
    values, _, _ = await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    assert values["agreement_with_reference"] == 0.0
    assert quality_mod.items_path(suite.id, REFERENCE, "c1").exists()


async def test_truncation_is_counted_rather_than_scored_wrong(
    mock_server, configs, args, monkeypatch
):
    """A reply cut off at max_tokens is an unmeasured item, not a wrong one."""
    monkeypatch.setenv("MOCK_REPLY", " ".join(str(i) for i in range(40)))
    suite = select_suites(["mmlu"])[0]
    values, _, _ = await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    assert values["truncated_rate"] == 1.0


async def test_a_cell_carries_no_boolean_into_the_metric_set(
    mock_server, configs, args, monkeypatch
):
    """MLflow metrics are floats; the booleans belong in tags."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    values, _, _ = await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    assert not any(isinstance(v, bool) for v in values.values())


async def test_every_item_is_scored_exactly_once(mock_server, configs, args, monkeypatch):
    """`run_load` cycles its scenario set to fill the request count. Asking for
    anything but len(scenarios) would score one item twice and another never."""
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    _, _, results = await quality_mod.run_suite(suite, configs[REFERENCE], mock_server, args)
    assert len({r.scenario_id for r in results}) == len(results)


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------


async def test_a_config_outside_the_subset_is_refused(args):
    """The scored set is a property of the study, not of whoever typed the loop."""
    args.config_id = "bf16_eagle3"  # measured for speed, not in the quality subset
    assert await quality_mod.main_async(args) == 2

    args.force = True
    args.dry_run = True
    assert await quality_mod.main_async(args) == 0


async def test_a_concurrency_above_max_num_seqs_is_refused(configs, args):
    """vLLM would queue the excess, and the pass would not know it had."""
    cfg = configs[REFERENCE]
    args.config_id = cfg.id
    args.dry_run = True
    args.concurrency = cfg.max_num_seqs + 1
    assert await quality_mod.main_async(args) == 2

    args.concurrency = cfg.max_num_seqs
    assert await quality_mod.main_async(args) == 0


def test_the_context_guard_names_both_failures(configs):
    cfg = configs[REFERENCE]
    assert "max-num-seqs" in check_context_budget(cfg, cfg.max_num_seqs + 1, 100)
    assert "max-model-len" in check_context_budget(cfg, 1, cfg.max_model_len + 1)
    assert check_context_budget(cfg, 1, cfg.max_model_len) is None


def test_every_suite_fits_the_pinned_context(configs, suites):
    """The default pass must never trip its own guard on a scored config."""
    for cfg in configs.values():
        if not cfg.quality:
            continue
        for suite in suites.values():
            budget = quality_mod.PROMPT_BUDGET + suite.max_tokens
            default = quality_mod.build_parser().parse_args([]).concurrency
            assert check_context_budget(cfg, default, budget) is None, (cfg.id, suite.id)


async def test_preflight_catches_a_server_forcing_length(
    mock_server, configs, args, monkeypatch
):
    """The mirror image of what bench.run relies on.

    The speed sweep needs ignore_eos honoured; this pass fails when generation
    never stops on its own, because then nothing stops where the model would
    and no reply can be scored honestly.
    """
    monkeypatch.setenv("MOCK_REPLY", " ".join(str(i) for i in range(40)))
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs[REFERENCE], suite, mock_server, args) == 1


async def test_preflight_passes_against_a_scoreable_server(
    mock_server, configs, args, monkeypatch
):
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs[REFERENCE], suite, mock_server, args) == 0


async def test_preflight_fails_when_nothing_parses(mock_server, configs, args, monkeypatch):
    monkeypatch.setenv("MOCK_REPLY", "I could not possibly say")
    suite = select_suites(["mmlu"])[0]
    assert await quality_mod.preflight(configs[REFERENCE], suite, mock_server, args) == 1


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _qcell(config_id, suite_id, start_time, *, acc=0.7, trunc=0.0, ok=250, total=250, suffix=""):
    return {
        "start_time": pd.Timestamp(start_time, unit="s"),
        "params.config_id": config_id,
        "params.suite_id": suite_id,
        "params.concurrency": "32",
        "params.suffix": suffix,
        "params.quant_method": "gptq",
        "params.bit_width": "4",
        "params.sparsity_pattern": "none",
        "params.spec_method": "none",
        "metrics.n_items": 250.0,
        "metrics.accuracy": acc,
        "metrics.accuracy_ci_lo": acc - 0.06,
        "metrics.accuracy_ci_hi": acc + 0.06,
        "metrics.agreement_with_reference": 0.9,
        "metrics.unparseable_rate": 0.0,
        "metrics.truncated_rate": trunc,
        "metrics.repetition_ratio": 0.02,
        "metrics.requests_ok": float(ok),
        "metrics.requests_total": float(total),
    }


def test_a_rescored_suite_is_merged_by_the_same_rule_as_a_remeasured_class():
    """The hazard is identical -- MLflow appends -- so the rule is reused."""
    df = pd.DataFrame(
        [_qcell("w4a16_gptq", "mmlu", 100, acc=0.60), _qcell("w4a16_gptq", "mmlu", 200, acc=0.71)]
    )
    winners, superseded = select_cells(df, QUALITY_KEY)
    assert len(winners) == 1
    assert winners.iloc[0]["metrics.accuracy"] == 0.71
    assert len(superseded) == 1


def test_a_determinism_rerun_does_not_supersede_the_measurement():
    """`--suffix c1` is a second reading meant to sit beside the first, not a
    correction of it, so the suffix is part of a cell's identity."""
    df = pd.DataFrame(
        [_qcell(REFERENCE, "mmlu", 100), _qcell(REFERENCE, "mmlu", 200, suffix="c1")]
    )
    winners, superseded = select_cells(df, QUALITY_KEY)
    assert len(winners) == 2
    assert len(superseded) == 0


def test_a_heavily_truncated_cell_is_flagged():
    """Its accuracy is a lower bound, not a measurement."""
    assert quality_problems(pd.Series(_qcell("sparse24_w4a16", "gsm8k", 100, trunc=0.4)))
    assert not quality_problems(pd.Series(_qcell("sparse24_w4a16", "gsm8k", 100, trunc=0.0)))


def test_failed_requests_are_flagged():
    assert quality_problems(pd.Series(_qcell("sparse24_w4a16", "gsm8k", 100, ok=240)))


def test_the_table_never_prints_accuracy_without_its_interval():
    """At 250 items the interval is wider than most pairs of configs differ by.
    A point estimate on its own invites a reader to find a difference in the
    noise."""
    df = pd.DataFrame([_qcell("w4a16_gptq", "mmlu", 100, acc=0.7)])
    lines = render_quality(df)
    body = [l for l in lines if "w4a16_gptq" in l]
    assert body and "0.700" in body[0]
    assert "[0.640, 0.760]" in body[0]


# --------------------------------------------------------------------------
# the German arm
# --------------------------------------------------------------------------


def test_the_german_and_english_mmlu_suites_are_item_matched(suites):
    """The claim the whole German/English comparison rests on.

    MMMLU is the same 14,042 MMLU items professionally translated in the same
    row order, so the identical stratified draw must select the identical
    questions. If a re-upload reorders either file, the two suites quietly
    become independent 250-item samples of the same benchmark -- still valid
    suites, still plausible numbers, but the English-German delta stops being
    paired and at 250 items is then mostly noise. Nothing downstream could tell.
    """
    en = read_key(suites["mmlu"])
    de = read_key(suites["mmlu_de"])
    assert len(en) == len(de) == 250

    for i in range(250):
        english = en[f"mmlu-{i:04d}"]
        german = de[f"mmlu_de-{i:04d}"]
        assert english["meta"]["subject"] == german["meta"]["subject"], (
            f"item {i}: {english['meta']['subject']} in English but "
            f"{german['meta']['subject']} in German -- the sources have drifted "
            f"out of row order and the two suites are no longer the same questions"
        )


def test_the_documented_gold_disagreements_are_still_only_a_handful(suites):
    """MMMLU's key differs from MMLU's on a few items, and that is written down.

    Left as the sources have them: each suite scores against its own published
    key, every config meets the same key, so a wrong label subtracts equally
    from all of them and cancels in the paired metric. What would matter is the
    count changing -- that would mean a re-upload moved options rather than just
    relabelling, and the two suites would no longer be asking the same thing.
    """
    en = read_key(suites["mmlu"])
    de = read_key(suites["mmlu_de"])
    differing = [
        i
        for i in range(250)
        if en[f"mmlu-{i:04d}"]["answer"] != de[f"mmlu_de-{i:04d}"]["answer"]
    ]
    assert len(differing) <= 2, (
        f"{len(differing)} of 250 drawn items disagree on the gold letter "
        f"({differing}); the published rate is 5 in 14,042"
    )


def test_the_german_suites_ask_in_german(suites):
    """A German capability slice must not be measured through an English scaffold.

    Not a style preference: instruction-following in German is part of what
    degrades, and an English answer-format line would measure the model's
    English scaffolding on a German question. The confound this accepts is
    documented in the suite description.
    """
    from bench.build_evals import BELEBELE_DE_INSTRUCTION, MMLU_DE_INSTRUCTION

    for suite_id, instruction in (
        ("mmlu_de", MMLU_DE_INSTRUCTION),
        ("belebele_de", BELEBELE_DE_INSTRUCTION),
    ):
        scenarios, _ = load_suite(suites[suite_id])
        for scenario in scenarios:
            assert instruction in scenario.turns[0].messages[0]["content"]
