"""The merge rule, which is the only thing standing between a rerun and a
silently wrong table.

Every test here builds its frame by hand rather than through MLflow. The
question being asked is "given two measurements of the same cell, which one
survives", and that question has nothing to do with a tracking store.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bench import tracking
from bench.report import cell_problems, render, select_cells


def cell(
    config_id: str,
    class_id: str,
    concurrency: int,
    start_time: int,
    *,
    output_tps: float = 100.0,
    ok: int = 16,
    total: int = 16,
    preemptions: float = 0.0,
    length_capped: str = "true",
) -> dict:
    return {
        "start_time": pd.Timestamp(start_time, unit="s"),
        "params.config_id": config_id,
        "params.class_id": class_id,
        "params.concurrency": str(concurrency),
        "params.quant_method": "none",
        "params.bit_width": "16",
        "params.sparsity_pattern": "none",
        "params.spec_method": "none",
        "metrics.output_tps": output_tps,
        "metrics.ttft_s_p50": 0.05,
        "metrics.ttft_s_p95": 0.08,
        "metrics.tpot_s_p50": 0.02,
        "metrics.kv_cache_usage_peak": 0.3,
        "metrics.preemptions": preemptions,
        "metrics.requests_ok": float(ok),
        "metrics.requests_total": float(total),
        "tags.note.length_capped": length_capped,
    }


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------
# the merge rule
# --------------------------------------------------------------------------


def test_the_later_measurement_of_a_cell_wins():
    df = frame(
        cell("fp8_dynamic", "c1_chat", 32, 1000, output_tps=111.0),
        cell("fp8_dynamic", "c1_chat", 32, 2000, output_tps=222.0),
    )
    winners, superseded = select_cells(df)

    assert len(winners) == 1
    assert len(superseded) == 1
    assert winners.iloc[0]["metrics.output_tps"] == 222.0
    assert superseded.iloc[0]["metrics.output_tps"] == 111.0


def test_row_order_does_not_decide_the_winner():
    """The frame arrives in whatever order the store returns it.

    If this ever regressed to "keep the last row" rather than "keep the latest
    start_time", the merge would depend on MLflow's result ordering, which is
    not something this code should be trusting.
    """
    newest = cell("fp8_dynamic", "c1_chat", 32, 2000, output_tps=222.0)
    oldest = cell("fp8_dynamic", "c1_chat", 32, 1000, output_tps=111.0)

    winners, _ = select_cells(frame(newest, oldest))
    assert winners.iloc[0]["metrics.output_tps"] == 222.0

    winners, _ = select_cells(frame(oldest, newest))
    assert winners.iloc[0]["metrics.output_tps"] == 222.0


def test_a_class_measured_once_is_untouched_by_a_rerun_of_others():
    """A class that is not re-measured in a corrective pass keeps its cell.

    Its original cell has to survive a merge whose other classes all gained a
    newer measurement -- otherwise the cheap rerun would cost the one class it
    deliberately skipped.
    """
    df = frame(
        cell("fp8_dynamic", "c2_longform", 1, 1000, output_tps=50.0),
        cell("fp8_dynamic", "c1_chat", 1, 1000, output_tps=10.0),
        cell("fp8_dynamic", "c1_chat", 1, 2000, output_tps=99.0),
    )
    winners, superseded = select_cells(df)

    by_class = {r["params.class_id"]: r["metrics.output_tps"] for _, r in winners.iterrows()}
    assert by_class == {"c2_longform": 50.0, "c1_chat": 99.0}
    assert len(superseded) == 1


def test_cells_differing_only_in_concurrency_are_not_merged():
    """c=1 and c=32 are separate measurements of separate questions."""
    df = frame(
        cell("fp8_dynamic", "c1_chat", 1, 1000, output_tps=50.0),
        cell("fp8_dynamic", "c1_chat", 32, 1000, output_tps=300.0),
    )
    winners, superseded = select_cells(df)
    assert len(winners) == 2
    assert superseded.empty


def test_cells_differing_only_in_config_are_not_merged():
    df = frame(
        cell("fp8_dynamic", "c1_chat", 1, 1000),
        cell("fp8_eagle3", "c1_chat", 1, 1000),
    )
    winners, superseded = select_cells(df)
    assert len(winners) == 2
    assert superseded.empty


def test_a_half_finished_rerun_leaves_untouched_configs_alone():
    """The rerun can die partway; that must not lose the configs it never reached."""
    df = frame(
        cell("baseline_bf16", "c1_chat", 32, 1000, output_tps=10.0),
        cell("baseline_bf16", "c1_chat", 32, 2000, output_tps=88.0),
        cell("w4a16_gptq", "c1_chat", 32, 1000, output_tps=20.0),
    )
    winners, _ = select_cells(df)

    got = {r["params.config_id"]: r["metrics.output_tps"] for _, r in winners.iterrows()}
    assert got == {"baseline_bf16": 88.0, "w4a16_gptq": 20.0}


def test_missing_identifying_columns_raise_rather_than_merge_wrongly():
    df = pd.DataFrame([{"start_time": pd.Timestamp(0), "params.config_id": "fp8_dynamic"}])
    with pytest.raises(KeyError, match="identifying columns"):
        select_cells(df)


def test_an_empty_frame_survives_the_merge():
    empty = pd.DataFrame()
    winners, superseded = select_cells(empty)
    assert winners.empty and superseded.empty


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------


def test_preemptions_are_a_problem():
    """Tail latency in that cell measures cache starvation, not the config."""
    row = pd.Series(cell("bf16_eagle3", "c3_rag", 32, 1000, preemptions=4.0))
    assert any("preempted=4" in p for p in cell_problems(row))


def test_failed_requests_are_a_problem():
    row = pd.Series(cell("fp8_dynamic", "c1_chat", 32, 1000, ok=14, total=16))
    assert any("failed=2" in p for p in cell_problems(row))


def test_generation_that_was_not_length_capped_is_a_problem():
    row = pd.Series(cell("fp8_dynamic", "c1_chat", 32, 1000, length_capped="false"))
    assert "not length-capped" in cell_problems(row)


def test_a_missing_length_capped_tag_is_not_treated_as_a_pass():
    """Absence must not read as 'checked, fine' -- nor as 'checked, failed'."""
    row = pd.Series(cell("fp8_dynamic", "c1_chat", 32, 1000, length_capped="true"))
    del row["tags.note.length_capped"]
    assert "not length-capped" not in cell_problems(row)


def test_a_clean_cell_has_no_problems():
    assert cell_problems(pd.Series(cell("fp8_dynamic", "c1_chat", 1, 1000))) == []


def test_nan_metrics_do_not_count_as_failures():
    """A cell that never logged a counter is not a cell that logged a bad one."""
    row = pd.Series(cell("fp8_dynamic", "c1_chat", 1, 1000))
    row["metrics.preemptions"] = float("nan")
    assert cell_problems(row) == []


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_rows_follow_the_config_file_order():
    """The file groups the configs into the study's three comparisons; any
    numeric sort would interleave them."""
    df = frame(
        cell("fp8_eagle3", "c1_chat", 1, 1000),
        cell("baseline_bf16", "c1_chat", 1, 1000),
        cell("w4a16_gptq", "c1_chat", 1, 1000),
    )
    winners, _ = select_cells(df)
    body = [ln for ln in render(winners) if ln.startswith("  ") and "config" not in ln]
    order = [ln.split()[0] for ln in body if not ln.strip().startswith("-")]
    assert order == ["baseline_bf16", "w4a16_gptq", "fp8_eagle3"]


def test_a_param_a_cell_never_logged_is_blank_not_nan():
    """Cells logged before a param existed come back with NaN in its column."""
    row = cell("baseline_bf16", "c1_chat", 1, 1000)
    row["params.sparsity_pattern"] = float("nan")
    row["params.spec_method"] = float("nan")
    line = next(ln for ln in render(frame(row)) if "baseline_bf16" in ln)
    assert "nan" not in line
    assert "none/16" in line


def test_invalid_cells_are_hidden_by_default_and_shown_with_all():
    df = frame(
        cell("bf16_eagle3", "c3_rag", 32, 1000, preemptions=3.0),
        cell("baseline_bf16", "c3_rag", 32, 1000),
    )
    winners, _ = select_cells(df)

    default = "\n".join(render(winners))
    assert "baseline_bf16" in default and "bf16_eagle3" not in default

    everything = "\n".join(render(winners, only_valid=False))
    assert "bf16_eagle3" in everything and "preempted=3" in everything


# --------------------------------------------------------------------------
# where the cells are read from
# --------------------------------------------------------------------------


def test_every_entry_point_defaults_to_the_same_store(monkeypatch):
    """The sweep, the quality pass and the report must agree on one file, or
    the report reads an empty store and says so with a straight face."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.resolve_tracking_uri() == tracking.DEFAULT_TRACKING_URI
    assert tracking.DEFAULT_TRACKING_URI.endswith("/mlflow.db")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///elsewhere.db")
    assert tracking.resolve_tracking_uri() == "sqlite:///elsewhere.db"
    assert tracking.resolve_tracking_uri("sqlite:///explicit.db") == "sqlite:///explicit.db"
