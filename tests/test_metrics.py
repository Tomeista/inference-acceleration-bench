from __future__ import annotations

import math

import pytest

from bench.client import LoadResult, RequestRecord
from bench.metrics import (
    aggregate,
    parse_prometheus,
    percentile,
    preemption_delta,
    spec_decode_metrics,
)


def test_percentile_matches_linear_interpolation():
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 4.0
    # rank = 0.5 * 3 = 1.5 -> midpoint of 2.0 and 3.0
    assert percentile(data, 50) == pytest.approx(2.5)
    assert percentile(data, 75) == pytest.approx(3.25)


def test_percentile_handles_degenerate_input():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1.0, float("nan"), 3.0], 50) == pytest.approx(2.0)


def _record(ttft: float, itls: list[float], tokens: int, ok: bool = True) -> RequestRecord:
    record = RequestRecord(
        scenario_id="s", class_id="c1_chat", turn_index=0, start_time=0.0
    )
    record.ttft = ttft
    record.itls = itls
    record.completion_tokens = tokens
    record.prompt_tokens = 64
    record.end_time = ttft + sum(itls)
    record.finish_reason = "length"
    record.success = ok
    return record


def test_tpot_excludes_the_first_token():
    record = _record(0.1, [0.01, 0.01, 0.01], tokens=4)
    # decode time is 0.03 over 3 gaps
    assert record.tpot == pytest.approx(0.01)


def test_tpot_is_none_for_a_single_token():
    assert _record(0.1, [], tokens=1).tpot is None


def test_aggregate_computes_throughput_over_the_window():
    records = [_record(0.1, [0.01] * 9, tokens=10) for _ in range(4)]
    result = LoadResult(records=records, started_at=0.0, finished_at=2.0)
    m = aggregate(result)

    assert m.values["requests_ok"] == 4
    assert m.values["requests_failed"] == 0
    assert m.values["output_tokens_total"] == 40
    # 40 tokens over a 2 second window, not the sum of per-request rates
    assert m.values["output_tps"] == pytest.approx(20.0)
    assert m.values["ttft_s_p50"] == pytest.approx(0.1)
    assert m.notes["length_capped"] is True


def test_aggregate_flags_a_run_that_did_not_stop_on_length():
    record = _record(0.1, [0.01], tokens=2)
    record.finish_reason = "stop"
    result = LoadResult(records=[record], started_at=0.0, finished_at=1.0)
    assert aggregate(result).notes["length_capped"] is False


def test_aggregate_survives_an_all_failed_cell():
    record = _record(0.0, [], tokens=0, ok=False)
    record.error = "HTTP 500"
    result = LoadResult(records=[record], started_at=0.0, finished_at=1.0)
    m = aggregate(result)

    assert m.values["requests_ok"] == 0
    assert m.values["requests_failed"] == 1
    assert "ttft_s_p50" not in m.values
    assert m.notes["first_error"] == "HTTP 500"


PROM = """
# HELP vllm:spec_decode_num_drafts_total Number of drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="qwen3-8b"} 100.0
vllm:spec_decode_num_draft_tokens_total{model_name="qwen3-8b"} 500.0
vllm:spec_decode_num_accepted_tokens_total{model_name="qwen3-8b"} 350.0
vllm:num_preemptions_total{model_name="qwen3-8b"} 2.0
vllm:some_other_metric{model_name="qwen3-8b"} 9.0
"""


def test_parse_prometheus_selects_and_sums_named_counters():
    parsed = parse_prometheus(PROM)
    assert parsed["vllm:spec_decode_num_drafts_total"] == 100.0
    assert parsed["vllm:spec_decode_num_accepted_tokens_total"] == 350.0
    assert "vllm:some_other_metric" not in parsed


def test_parse_prometheus_sums_across_label_sets():
    text = (
        'vllm:generation_tokens_total{model_name="a"} 10.0\n'
        'vllm:generation_tokens_total{model_name="b"} 5.0\n'
    )
    assert parse_prometheus(text)["vllm:generation_tokens_total"] == 15.0


def test_spec_decode_metrics_differences_cumulative_counters():
    before = parse_prometheus(PROM)
    after = dict(before)
    after["vllm:spec_decode_num_drafts_total"] = 200.0
    after["vllm:spec_decode_num_draft_tokens_total"] = 1000.0
    after["vllm:spec_decode_num_accepted_tokens_total"] = 700.0

    m = spec_decode_metrics(before, after)
    # Only the window counts: 100 drafts, 500 draft tokens, 350 accepted.
    assert m["spec_drafts"] == 100.0
    assert m["spec_acceptance_rate"] == pytest.approx(0.7)
    assert m["spec_mean_accepted_length"] == pytest.approx(4.5)


def test_spec_decode_metrics_absent_without_a_draft_model():
    # A baseline server exposes no speculative counters. Reporting zeros there
    # would read as total rejection rather than as not applicable.
    assert spec_decode_metrics({}, {"vllm:num_preemptions_total": 0.0}) == {}


def test_preemption_delta():
    before = {"vllm:num_preemptions_total": 2.0}
    after = {"vllm:num_preemptions_total": 5.0}
    assert preemption_delta(before, after)["preemptions"] == 3.0
