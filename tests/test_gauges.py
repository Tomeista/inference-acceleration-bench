"""The gauge sampler.

Its reason for existing is the KV cache confound: `gpu_memory_utilization` is
pinned across configs, so a speculative server hands part of that budget to the
draft model and runs with a smaller KV cache than its own baseline. Peak cache
usage during the load is what makes that visible. Peak, not final, because by
the time the load drains every gauge has fallen back to zero.
"""

from __future__ import annotations

import pytest

from bench.client import run_load
from bench.metrics import GAUGES, GaugeSampler, parse_prometheus, scrape
from bench.scenarios import Scenario, Turn


def _scenarios(n: int, max_tokens: int = 24) -> list[Scenario]:
    return [
        Scenario(
            scenario_id=f"g-{i}",
            class_id="c1_chat",
            turns=[
                Turn(
                    messages=[{"role": "user", "content": f"q{i}"}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            ],
        )
        for i in range(n)
    ]


def test_parse_prometheus_reads_gauges_when_asked():
    text = (
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.4200\n'
        'vllm:num_requests_running{model_name="m"} 7\n'
    )
    parsed = parse_prometheus(text, GAUGES)
    assert parsed["vllm:gpu_cache_usage_perc"] == pytest.approx(0.42)
    assert parsed["vllm:num_requests_running"] == 7


async def test_sampler_captures_load_that_a_final_scrape_would_miss(mock_server):
    """The point of sampling: after the load, these gauges read zero."""
    async with GaugeSampler(mock_server, interval=0.02) as sampler:
        await run_load(
            _scenarios(16),
            base_url=mock_server,
            model="qwen3-8b",
            concurrency=8,
            num_requests=16,
        )

    peaks = sampler.metrics()
    assert peaks["requests_running_peak"] > 1, "sampler never observed concurrent load"
    assert peaks["kv_cache_usage_peak"] > 0

    # The same gauges scraped after the fact carry none of that signal, which is
    # exactly why a post-run read is not good enough.
    after = parse_prometheus(
        __import__("httpx").get(mock_server + "/metrics").text, GAUGES
    )
    assert after["vllm:num_requests_running"] == 0


async def test_sampler_reports_nothing_rather_than_a_zero_peak(mock_server):
    """A cell shorter than the polling interval yields no measurement at all.

    Emitting kv_cache_usage_peak = 0.0 here would be read months later as an
    empty cache, when in fact the sampler simply never looked during the load.
    """
    async with GaugeSampler(mock_server, interval=30.0) as sampler:
        await run_load(
            _scenarios(1, max_tokens=2),
            base_url=mock_server,
            model="qwen3-8b",
            concurrency=1,
            num_requests=1,
        )
    assert not sampler.observed_load
    assert sampler.metrics() == {}


async def test_sampler_is_quiet_when_metrics_are_unavailable(mock_server):
    """A missing endpoint degrades to no peaks, never to a failed run."""
    async with GaugeSampler(mock_server + "/absent", interval=0.02) as sampler:
        await run_load(
            _scenarios(2),
            base_url=mock_server,
            model="qwen3-8b",
            concurrency=1,
            num_requests=2,
        )
    assert sampler.metrics() == {}


def test_scrape_returns_empty_for_an_unreachable_server():
    assert scrape("http://127.0.0.1:1") == {}
