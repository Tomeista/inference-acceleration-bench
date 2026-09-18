"""End-to-end checks of the load client against the mock vLLM.

These are the tests that stand in for a GPU: they prove the client speaks the
protocol correctly, measures TTFT at the first content token, and actually runs
requests in parallel.
"""

from __future__ import annotations

import json
import time

import pytest

from bench.client import run_load
from bench.metrics import aggregate
from bench.scenarios import Scenario, Turn

MOCK_TTFT_S = 0.04
MOCK_ITL_S = 0.008


def _scenarios(n: int, max_tokens: int = 8, turns: int = 1) -> list[Scenario]:
    out = []
    for i in range(n):
        out.append(
            Scenario(
                scenario_id=f"s-{i}",
                class_id="c1_chat",
                turns=[
                    Turn(
                        messages=[{"role": "user", "content": f"question {i} turn {t}"}],
                        max_tokens=max_tokens,
                        temperature=0.0,
                        extra_body={"ignore_eos": True},
                    )
                    for t in range(turns)
                ],
            )
        )
    return out


def test_tool_requests_send_tool_choice_none():
    """Tools render into the prompt, but the call must stream as plain content.

    The implicit "auto" needs a vLLM tool-call parser, which buffers tokens until
    it can name the function and ends the request at the call, so TTFT and
    output length would stop measuring the engine.
    """
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    turn = Turn(messages=[{"role": "user", "content": "hi"}], max_tokens=8,
                temperature=0.0, tools=tools)
    payload = turn.to_payload("qwen3-8b")
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "none"

    plain = Turn(messages=[{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0)
    assert "tool_choice" not in plain.to_payload("qwen3-8b")


def test_warmup_shortens_replies_without_touching_the_prompt_set():
    from bench.run import WARMUP_MAX_TOKENS, shortened

    original = _scenarios(2, max_tokens=1024)
    short = shortened(original)
    assert all(t.max_tokens == WARMUP_MAX_TOKENS for s in short for t in s.turns)
    assert all(t.max_tokens == 1024 for s in original for t in s.turns)
    assert [s.turns[0].messages for s in short] == [s.turns[0].messages for s in original]


async def test_records_a_successful_request(mock_server):
    result = await run_load(
        _scenarios(1, max_tokens=8),
        base_url=mock_server,
        model="qwen3-8b",
        concurrency=1,
        num_requests=1,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.success, record.error
    assert record.completion_tokens == 8
    assert record.finish_reason == "length"
    # One TTFT plus one gap per subsequent token.
    assert len(record.itls) == 7
    assert record.ttft == pytest.approx(MOCK_TTFT_S, abs=0.05)


async def test_ttft_excludes_the_empty_role_chunk(mock_server):
    """The first streamed chunk carries role and empty content.

    Counting it would report a TTFT far below the truth, so the client waits for
    the first chunk that actually carries generated text.
    """
    result = await run_load(
        _scenarios(1, max_tokens=4),
        base_url=mock_server,
        model="qwen3-8b",
        concurrency=1,
        num_requests=1,
    )
    record = result.records[0]
    assert record.ttft is not None
    assert record.ttft >= MOCK_TTFT_S * 0.5


async def test_requests_actually_run_in_parallel(mock_server):
    per_request = MOCK_TTFT_S + 7 * MOCK_ITL_S
    started = time.perf_counter()
    result = await run_load(
        _scenarios(8, max_tokens=8),
        base_url=mock_server,
        model="qwen3-8b",
        concurrency=4,
        num_requests=8,
    )
    elapsed = time.perf_counter() - started

    assert len(result.successful) == 8
    # Serial execution would take 8x; a working pool of 4 takes roughly 2x.
    assert elapsed < per_request * 6


async def test_multi_turn_scenario_runs_turns_sequentially(mock_server):
    """The path c7 will take. Single-turn classes are the degenerate case."""
    result = await run_load(
        _scenarios(1, max_tokens=4, turns=3),
        base_url=mock_server,
        model="qwen3-8b",
        concurrency=1,
        num_requests=1,
    )

    assert [r.turn_index for r in result.records] == [0, 1, 2]
    assert all(r.success for r in result.records)
    assert all(r.scenario_id == "s-0" for r in result.records)


async def test_failed_request_is_recorded_not_raised(mock_server):
    result = await run_load(
        _scenarios(1),
        base_url=mock_server + "/nope",
        model="qwen3-8b",
        concurrency=1,
        num_requests=1,
    )
    record = result.records[0]
    assert not record.success
    assert record.error
    # A failing cell still aggregates, so one bad config cannot abort a sweep.
    assert aggregate(result).values["requests_failed"] == 1


async def test_aggregate_over_a_real_load(mock_server):
    result = await run_load(
        _scenarios(8, max_tokens=16),
        base_url=mock_server,
        model="qwen3-8b",
        concurrency=4,
        num_requests=8,
    )
    values = aggregate(result).values

    assert values["requests_ok"] == 8
    assert values["output_tokens_total"] == 8 * 16
    assert values["output_tps"] > 0
    assert values["ttft_s_p95"] >= values["ttft_s_p50"]
    assert values["tpot_s_p50"] == pytest.approx(MOCK_ITL_S, abs=0.02)


# --------------------------------------------------------------------------
# native function calling
# --------------------------------------------------------------------------


async def test_a_streamed_tool_call_is_reassembled(mock_server, monkeypatch):
    """The path BFCL depends on, and the only suite that exercises it.

    Under `--enable-auto-tool-choice` a tool call never arrives as content. The
    name comes in one chunk and the argument JSON streams in pieces, so a client
    that only read `delta.content` would record an empty reply and the whole
    BFCL cell would score as total structure failure -- a dramatic, entirely
    false, finding about quantization destroying tool calling.
    """
    monkeypatch.setenv(
        "MOCK_TOOL_CALLS",
        json.dumps([{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]),
    )
    scenario = Scenario(
        scenario_id="t0",
        class_id="tools",
        turns=[Turn(messages=[{"role": "user", "content": "weather?"}], max_tokens=32, temperature=0.0)],
    )
    result = await run_load(
        [scenario], base_url=mock_server, model="qwen3-8b",
        concurrency=1, num_requests=1, collect_output=True,
    )
    record = result.records[0]
    assert record.success
    assert record.tool_calls == [
        {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}
    ]
    assert json.loads(record.tool_calls[0]["arguments"]) == {"city": "Paris", "days": 3}


async def test_parallel_tool_calls_do_not_have_their_arguments_spliced(
    mock_server, monkeypatch
):
    """Two calls stream interleaved, keyed by index.

    Concatenating every fragment in arrival order -- which is what the timing
    path does deliberately, because for TTFT any token will do -- would splice
    the two argument blobs into one unparseable string. That would read as a
    structure failure caused by the model rather than by the client.
    """
    monkeypatch.setenv(
        "MOCK_TOOL_CALLS",
        json.dumps([
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Berlin"}},
        ]),
    )
    scenario = Scenario(
        scenario_id="t1",
        class_id="tools",
        turns=[Turn(messages=[{"role": "user", "content": "weather?"}], max_tokens=64, temperature=0.0)],
    )
    result = await run_load(
        [scenario], base_url=mock_server, model="qwen3-8b",
        concurrency=1, num_requests=1, collect_output=True,
    )
    calls = result.records[0].tool_calls
    assert len(calls) == 2
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    assert json.loads(calls[1]["arguments"]) == {"city": "Berlin"}


async def test_a_tool_call_still_counts_toward_ttft(mock_server, monkeypatch):
    """A tool-call chunk carries no content but is still a generated token.

    If it did not count, a tool-calling cell would report a TTFT equal to its
    whole latency -- which is the behaviour the timing path was written to
    avoid, and which the structured accumulator must not have regressed.
    """
    monkeypatch.setenv(
        "MOCK_TOOL_CALLS", json.dumps([{"name": "ping", "arguments": {"x": 1}}])
    )
    scenario = Scenario(
        scenario_id="t2",
        class_id="tools",
        turns=[Turn(messages=[{"role": "user", "content": "ping"}], max_tokens=16, temperature=0.0)],
    )
    result = await run_load(
        [scenario], base_url=mock_server, model="qwen3-8b",
        concurrency=1, num_requests=1, collect_output=True,
    )
    record = result.records[0]
    assert record.ttft is not None
    assert record.ttft < record.latency


async def test_an_ordinary_reply_records_no_tool_calls(mock_server, monkeypatch):
    """Absent rather than empty-but-present: every non-tool suite goes through
    the same client, and a stray call there would be a bug worth seeing."""
    monkeypatch.delenv("MOCK_TOOL_CALLS", raising=False)
    monkeypatch.setenv("MOCK_REPLY", "Answer: C")
    scenario = Scenario(
        scenario_id="t3",
        class_id="chat",
        turns=[Turn(messages=[{"role": "user", "content": "q"}], max_tokens=16, temperature=0.0)],
    )
    result = await run_load(
        [scenario], base_url=mock_server, model="qwen3-8b",
        concurrency=1, num_requests=1, collect_output=True,
    )
    assert result.records[0].tool_calls == []
    assert result.records[0].output_text == "Answer: C"
