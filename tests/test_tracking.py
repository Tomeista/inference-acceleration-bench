"""What actually reaches MLflow, checked by reading it back out.

Every other test in this repository stops at the value `run_suite` returns. That
is one function call short of the thing the study depends on: a metric that is
computed correctly and then silently dropped on the way to the store is
indistinguishable, months later, from a metric that was never measured -- and
the cell still succeeds, prints a healthy summary line, and logs everything
around the hole.

`tracking._clean_metrics` drops five kinds of value without a word: None,
booleans, NaN, infinity, and anything non-numeric. Every one of those is
reachable from a real cell. NaN and inf arrive whenever a cell produced no
successful request; booleans arrive the moment someone adds a pass/fail metric
to an `extra` dict, which is a natural thing to do and which the scorer contract
positively invites.

So these tests round-trip through a real sqlite store rather than asserting on
the dict that goes in. They are the only tests in the suite that do.
"""

from __future__ import annotations

import json
import math

import mlflow
import pytest

from bench import metrics as metrics_mod
from bench import quality as quality_mod
from bench import run as run_mod
from bench import tracking
from bench.scoring import ItemResult, aggregate
from bench.server import load_configs
from bench.suites import load_suites, select_suites


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real MLflow sqlite store, isolated per test.

    Sqlite rather than a mock: the drops this file is about happen inside
    MLflow's own validation and inside `_clean_metrics`, and a fake store would
    reproduce neither.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    tracking.setup("test-tracking", uri)
    return uri


def logged_metrics(run_name: str) -> dict[str, float]:
    """Metrics of the most recent run with this name, read back from the store."""
    runs = mlflow.search_runs(
        experiment_names=["test-tracking"],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list",
    )
    assert runs, f"no run named {run_name!r} in the store"
    return dict(runs[0].data.metrics)


# --------------------------------------------------------------------------
# what the cleaner drops
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,why",
    [
        (None, "a metric that was never computed"),
        (True, "a pass/fail metric added to an extra dict"),
        (float("nan"), "a cell in which no request succeeded"),
        (float("inf"), "a percentile over an empty sample"),
        ("n/a", "a placeholder where a number was expected"),
    ],
)
def test_the_cleaner_drops_these_kinds_of_value(value, why):
    """Documenting the hole rather than pretending it is not there.

    Each of these is silently absent from the store afterwards. That is the
    right call for NaN -- a zero p95 reads as an extraordinarily fast run, which
    is worse than a gap -- but it is a hole either way, and the tests below are
    what stop a real metric falling into it unnoticed.
    """
    assert "x" not in tracking._clean_metrics({"x": value}), why


def test_a_numeric_string_is_coerced_rather_than_dropped():
    """The cleaner tries `float()` before giving up, so "0.5" survives as 0.5.

    Worth pinning: it means a metric accidentally stringified upstream still
    lands, which is the forgiving direction, and it is why the drop list above
    names a non-numeric placeholder rather than any string at all.
    """
    assert tracking._clean_metrics({"x": "0.5"}) == {"x": 0.5}


def test_ordinary_numbers_survive():
    cleaned = tracking._clean_metrics({"accuracy": 0.73, "n_items": 250, "zero": 0.0})
    assert cleaned == {"accuracy": 0.73, "n_items": 250.0, "zero": 0.0}


def test_a_zero_is_not_confused_with_a_missing_value():
    """The distinction the whole "absent rather than zero" convention rests on.

    A metric that is genuinely 0.0 must reach the store, or the convention
    inverts: absent would stop meaning "not measured" and start meaning
    "measured, and it was zero or it was dropped", which is unreadable.
    """
    assert tracking._clean_metrics({"structure_failure": 0.0}) == {"structure_failure": 0.0}


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


async def test_every_metric_a_cell_computes_reaches_the_store(
    store, mock_server, configs_fixture, args, monkeypatch
):
    """The test that would have caught a number going missing.

    Runs a real cell against the mock, logs it exactly as `bench.quality` does,
    then reads the store back and compares key for key. Anything `_clean_metrics`
    refuses shows up here as a name in the difference, with the cell that
    produced it still reporting success.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites(["mmlu"])[0]
    cfg = configs_fixture["baseline_bf16"]

    values, notes, results = await quality_mod.run_suite(suite, cfg, mock_server, args)

    with tracking.config_run("parent", cfg.as_params()):
        with tracking.cell_run("cell", {**cfg.as_params(), **suite.as_params()}):
            tracking.log_cell_results(
                values, notes, [r.to_dict() for r in results], args_artifact_dir(args), "cell"
            )

    stored = logged_metrics("cell")
    numeric = {
        k: v
        for k, v in values.items()
        if v is not None and not isinstance(v, bool) and not isinstance(v, str)
        and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    }
    missing = sorted(set(numeric) - set(stored))
    assert not missing, (
        f"computed but never written to MLflow: {missing}. The cell succeeded and "
        f"every other metric landed, so nothing downstream would have said so."
    )
    for name, value in numeric.items():
        assert stored[name] == pytest.approx(float(value)), f"{name} changed in transit"


async def test_the_suite_specific_metrics_reach_the_store_too(
    store, mock_server, configs_fixture, args, monkeypatch
):
    """The regression risk the capability suite adds.

    `structure_failure`, `prompt_loose`, `instruction_strict` and the rest come
    from `ItemResult.extra`, which is a newer path than the headline metrics and
    is aggregated by a different rule. A scorer whose extras never reached the
    store would still produce a complete-looking accuracy column, and the
    failure split -- the entire reason BFCL and IFEval are in the suite -- would
    quietly be missing.
    """
    monkeypatch.setenv("MOCK_REPLY", "no commas and all lowercase")
    suite = select_suites(["ifeval"])[0]
    cfg = configs_fixture["baseline_bf16"]

    values, notes, results = await quality_mod.run_suite(suite, cfg, mock_server, args)

    with tracking.config_run("parent2", cfg.as_params()):
        with tracking.cell_run("ifeval_cell", {**cfg.as_params(), **suite.as_params()}):
            tracking.log_cell_results(
                values, notes, [r.to_dict() for r in results],
                args_artifact_dir(args), "ifeval_cell",
            )

    stored = logged_metrics("ifeval_cell")
    for name in ("prompt_loose", "instruction_strict", "instruction_loose"):
        assert name in stored, f"{name} was computed but did not reach MLflow"


def test_a_ratio_of_sums_reaches_the_store_as_one_number_not_two():
    """The `_num`/`_den` pair is an implementation detail of aggregation.

    Both halves landing in MLflow would put two meaningless columns next to the
    real one, and a later group-by would happily average denominators. Only the
    resolved ratio should be there.
    """
    results = [
        ItemResult("s0", "", None, True, "stop", False, 0.0, "",
                   {"instruction_strict_num": 1.0, "instruction_strict_den": 2.0}),
    ]
    stored = tracking._clean_metrics(aggregate(results))
    assert "instruction_strict" in stored
    assert "instruction_strict_num" not in stored
    assert "instruction_strict_den" not in stored


# --------------------------------------------------------------------------
# params and their limits
# --------------------------------------------------------------------------


def test_no_suite_param_is_long_enough_to_be_refused(store):
    """`bfcl_ast` records seven source paths in one param, at 487 characters.

    MLflow 3 caps a param value at 6000, so it fits -- but it fits by a margin
    nobody chose, and `pyproject.toml` only requires mlflow>=2.19. Adding two
    more sources, or pinning an older MLflow, would push it over and the run
    would fail at the point where a config's provenance is recorded.
    """
    from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

    for suite in load_suites().values():
        for name, value in tracking._clean_params(suite.as_params()).items():
            assert len(value) < MAX_PARAM_VAL_LENGTH, f"{suite.id}.{name} is too long"


def test_every_suite_and_config_param_actually_writes(store):
    """Params are strings, so they cannot be dropped for being NaN -- but they
    can be refused for their length or their characters, and that fails the run
    rather than silently omitting a column. Cheaper to find here."""
    configs = load_configs()
    for suite in load_suites().values():
        for cfg in configs.values():
            params = {**cfg.as_params(), **suite.as_params()}
            with tracking.config_run(f"p-{cfg.id}-{suite.id}", params):
                pass


def args_artifact_dir(args):
    """Where `log_cell_results` writes the per-request artifact."""
    from pathlib import Path

    path = Path(quality_mod.ARTIFACT_ROOT) / "test-tracking"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("suite_id", ["mmlu", "gsm8k", "ifeval", "bfcl_ast", "mmlu_de"])
async def test_no_metric_a_real_cell_computes_is_one_the_cleaner_would_drop(
    suite_id, store, mock_server, configs_fixture, args, monkeypatch
):
    """The guard with teeth, and the generalization of the boolean test.

    The round-trip test above cannot catch a dropped metric: to compare like
    with like it has to filter out exactly the values `_clean_metrics` refuses,
    so a metric that turns into a bool disappears from both sides and the
    comparison still passes. This is the other half -- it asserts that a real
    cell produces nothing the cleaner would refuse in the first place.

    Run per suite because each scorer contributes its own `extra` metrics, and
    a new scorer is precisely where a bool or a None would enter: returning
    `{"passed": True}` from an extra dict is the obvious thing to write and it
    vanishes without a word.
    """
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    suite = select_suites([suite_id])[0]
    values, _, _ = await quality_mod.run_suite(
        suite, configs_fixture["baseline_bf16"], mock_server, args
    )
    droppable = sorted(set(values) - set(tracking._clean_metrics(values)))
    assert not droppable, (
        f"{suite_id} computes {droppable}, which MLflow silently will not store. "
        f"Booleans belong in `notes` (they become tags); a value that can be "
        f"None or NaN needs the absent-rather-than-zero treatment at the source."
    )


def test_a_bool_in_an_extra_dict_cannot_reach_the_store_as_a_bool():
    """Why the `extra` path is safe from the drop, and it is not by design.

    Returning `{"passed": True}` from a scorer is the obvious thing to write and
    would be silently dropped anywhere else. It survives here only because
    `aggregate_extra` averages over items, and dividing a bool by a count
    produces a float before MLflow ever sees it.

    Pinned because it is load-bearing and invisible: if extras ever stop being
    averaged -- a max, a sum, a passthrough for single-item suites -- booleans
    become reachable again and this file's guarantee quietly weakens.
    """
    results = [
        ItemResult("s0", "", "", True, "stop", False, 0.0, "", {"flag": True}),
        ItemResult("s1", "", "", True, "stop", False, 0.0, "", {"flag": False}),
    ]
    value = aggregate(results)["flag"]
    assert isinstance(value, float) and not isinstance(value, bool)
    assert tracking._clean_metrics({"flag": value}) == {"flag": 0.5}


async def test_the_real_pass_writes_what_it_printed(
    tmp_path, mock_server, monkeypatch, capsys
):
    """End to end through `main_async`, not through a replica of it.

    The tests above call `run_suite` and then log the way `bench.quality` logs.
    That leaves the orchestration itself -- which suites it loops over, which
    params it attaches, whether `log_cell_results` is reached at all for every
    cell -- outside the check, and that block is where a "the numbers weren't
    written" bug actually lives: `run_suite` returns perfectly good values and
    the caller never stores them, or stores them under a parent run that the
    report's filter later excludes.

    So this drives the real entry point and then reads the store back, asserting
    one cell per suite with the headline metric present in each.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    monkeypatch.setattr(quality_mod, "QUALITY_ROOT", tmp_path / "items")
    monkeypatch.setattr(quality_mod, "ARTIFACT_ROOT", tmp_path / "artifacts")

    uri = f"sqlite:///{tmp_path}/real.db"
    args = quality_mod.build_parser().parse_args(
        [
            "--config-id", "baseline_bf16",
            "--server-url", mock_server,
            "--suites", "mmlu,gsm8k",
            "--concurrency", "2",
            "--n-items", "6",
            "--experiment", "real-pass",
            "--tracking-uri", uri,
        ]
    )
    assert await quality_mod.main_async(args) == 0

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["real-pass"], output_format="list")
    by_name = {r.data.tags.get("mlflow.runName"): r for r in runs}

    # One parent for the config, one nested cell per suite.
    assert "baseline_bf16-quality" in by_name, "the config's parent run was never created"
    for suite_id in ("mmlu", "gsm8k"):
        cell = by_name.get(f"{suite_id}__c2")
        assert cell is not None, f"{suite_id} ran but no cell run reached the store"
        assert "accuracy" in cell.data.metrics, f"{suite_id} stored no accuracy"
        assert cell.data.metrics["n_items"] == 6
        # The discriminator bench.report selects quality cells on. Without it the
        # cell is invisible to the report while sitting in the store, which looks
        # exactly like a number that was never written.
        assert cell.data.params.get("suite_id") == suite_id
        assert "class_id" not in cell.data.params

    # And what it printed is what it stored, so the console is not reassuring
    # anyone about a number that did not land.
    printed = capsys.readouterr().out
    for suite_id in ("mmlu", "gsm8k"):
        accuracy = by_name[f"{suite_id}__c2"].data.metrics["accuracy"]
        assert f"acc={accuracy:.3f}" in printed


# --------------------------------------------------------------------------
# the speed sweep's half of the store
# --------------------------------------------------------------------------


async def test_the_speed_sweep_writes_its_cells(tmp_path, mock_server, monkeypatch, capsys):
    """`bench.run` end to end, read back out of a real store.

    The speed sweep writes latency and throughput through the same
    `log_cell_results` the quality pass uses, and until now nothing tested that
    path at all -- `run.py` sat at 18% coverage. These are the numbers most
    likely to have gone missing before, because unlike an accuracy they are
    computed from percentiles that legitimately return None on an empty sample.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setattr(run_mod, "ARTIFACT_ROOT", tmp_path / "artifacts")

    uri = f"sqlite:///{tmp_path}/speed.db"
    args = run_mod.build_parser().parse_args(
        [
            "--config-id", "baseline_bf16",
            "--server-url", mock_server,
            "--classes", "c1_chat",
            "--concurrency", "2",
            "--requests-per-cell", "4",
            "--warmup", "1",
            "--experiment", "speed-pass",
            "--tracking-uri", uri,
        ]
    )
    assert await run_mod.main_async(args) == 0

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["speed-pass"], output_format="list")
    by_name = {r.data.tags.get("mlflow.runName"): r for r in runs}

    cell = by_name.get("c1_chat__c2")
    assert cell is not None, "the cell ran but never reached the store"
    # The headline numbers of the whole speed study. If any of these can go
    # missing while the cell succeeds, the sweep is not measuring anything.
    for name in ("ttft_s_p50", "ttft_s_mean", "tpot_s_p50", "output_tps", "requests_per_s"):
        assert name in cell.data.metrics, f"{name} was computed but not written"
    # And the discriminator the report selects speed cells on.
    assert cell.data.params.get("class_id") == "c1_chat"


async def test_a_baseline_records_no_acceptance_and_says_it_was_not_expected(
    tmp_path, mock_server, monkeypatch
):
    """Absence by design, and it must not be flagged as a problem.

    A server with no draft model publishes no speculative counters, and the repo
    logs nothing rather than zeros -- acceptance 0.0 would read as total
    rejection instead of "there was no draft model". So the acceptance metrics
    are absent here and that is correct; what matters is that nothing warns.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.delenv("MOCK_SPEC", raising=False)
    monkeypatch.setattr(run_mod, "ARTIFACT_ROOT", tmp_path / "artifacts")

    uri = f"sqlite:///{tmp_path}/base.db"
    args = run_mod.build_parser().parse_args(
        ["--config-id", "baseline_bf16", "--server-url", mock_server,
         "--classes", "c1_chat", "--concurrency", "2", "--requests-per-cell", "4",
         "--warmup", "1", "--experiment", "spec-absent", "--tracking-uri", uri]
    )
    assert await run_mod.main_async(args) == 0

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["spec-absent"], output_format="list")
    cell = {r.data.tags.get("mlflow.runName"): r for r in runs}["c1_chat__c2"]

    assert "spec_acceptance_rate" not in cell.data.metrics
    absent = json.loads(cell.data.tags.get("note.metrics_absent", "{}"))
    assert "spec_acceptance_rate" not in absent, (
        "a baseline was told to expect acceptance metrics; absence by design "
        "must not be recorded as a missing number"
    )


async def test_a_speculative_config_that_loses_its_counters_is_recorded_and_warned(
    tmp_path, mock_server, monkeypatch, capsys
):
    """Absence by bug, which used to be indistinguishable from absence by design.

    This is the failure the RUNBOOK warns about: vLLM's speculative counter
    names have moved between versions, so a speculative run can lose its
    acceptance column while every request succeeds and every other number lands.
    Simulated here by serving a speculative config from a mock that publishes no
    speculative counters.

    Before `expected`, the result was a cell that looked exactly like a baseline.
    Now the hole is named in the store and the sweep says so on the console.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.delenv("MOCK_SPEC", raising=False)  # the engine publishes nothing
    monkeypatch.setattr(run_mod, "ARTIFACT_ROOT", tmp_path / "artifacts")

    uri = f"sqlite:///{tmp_path}/spec.db"
    args = run_mod.build_parser().parse_args(
        ["--config-id", "bf16_eagle3", "--server-url", mock_server,
         "--classes", "c1_chat", "--concurrency", "2", "--requests-per-cell", "4",
         "--warmup", "1", "--experiment", "spec-lost", "--tracking-uri", uri]
    )
    assert await run_mod.main_async(args) == 0

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["spec-lost"], output_format="list")
    cell = {r.data.tags.get("mlflow.runName"): r for r in runs}["c1_chat__c2"]

    absent = json.loads(cell.data.tags["note.metrics_absent"])
    for name in metrics_mod.SPEC_METRICS:
        assert absent.get(name) == "expected_but_absent", (
            f"{name} is missing from a speculative cell and the store does not "
            f"say so -- which is indistinguishable from a baseline run"
        )
    assert "was expected to record" in capsys.readouterr().err


def test_the_expectation_list_matches_what_the_producer_emits(mock_server):
    """`SPEC_METRICS` and `spec_decode_metrics` must not drift apart.

    The expectation is only useful while it names exactly what a working
    speculative server produces. A key added to the producer and not to the list
    can still vanish silently; a key in the list the producer never emits would
    warn on every speculative cell until people learned to ignore the warning,
    which is worse than no warning at all.
    """
    dialect = metrics_mod.VLLM
    before = {dialect.spec_drafts: 0.0, dialect.spec_draft_tokens: 0.0, dialect.spec_accepted: 0.0}
    after = {dialect.spec_drafts: 10.0, dialect.spec_draft_tokens: 50.0, dialect.spec_accepted: 35.0}
    produced = metrics_mod.spec_decode_metrics(before, after)
    assert set(produced) == set(metrics_mod.SPEC_METRICS)
