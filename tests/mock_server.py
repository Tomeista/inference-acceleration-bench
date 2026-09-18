"""A fake vLLM, so the harness can be verified without a GPU.

The GPU box is not available yet, and when it is, a wrong number has two
possible causes: the client or the server. This removes the first one. It
speaks enough of the OpenAI streaming protocol and enough of vLLM's Prometheus
output that `bench.run` cannot tell the difference, which means the concurrency
handling, TTFT measurement, percentile math, acceptance-rate differencing and
MLflow logging are all exercised end to end before anything is deployed.

It is a protocol stub, not a simulator. The latencies it emits are configured,
not modeled, so nothing it produces is a performance result.

    uvicorn tests.mock_server:app --port 8000
    MOCK_TTFT_MS=40 MOCK_ITL_MS=8 MOCK_SPEC=1 uvicorn tests.mock_server:app --port 8000
    MOCK_REPLY="Answer: C" uvicorn tests.mock_server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()

TTFT_MS = float(os.environ.get("MOCK_TTFT_MS", "40"))
ITL_MS = float(os.environ.get("MOCK_ITL_MS", "8"))
# Each additional in-flight request slows every request by this fraction, which
# is crude but enough to make a concurrency sweep produce a non-flat curve and
# therefore to prove the sweep is measuring what it thinks it is.
CONTENTION = float(os.environ.get("MOCK_CONTENTION", "0.02"))
SPEC_ENABLED = os.environ.get("MOCK_SPEC", "0") == "1"
SPEC_ACCEPT = float(os.environ.get("MOCK_SPEC_ACCEPT", "0.7"))
NUM_SPEC_TOKENS = int(os.environ.get("MOCK_SPEC_TOKENS", "5"))
MODEL_NAME = os.environ.get("MOCK_MODEL", "qwen3-8b")


def scripted_reply() -> str | None:
    """A literal reply to emit instead of filler, or None for the filler.

    The quality pass scores generated text, so a mock that always says "the
    quick brown fox" can exercise the request loop but not the scorer, the
    answer-key join, or the reference comparison. Setting this to "Answer: C"
    turns the mock into a server with a known, checkable accuracy against the
    real committed eval set -- which is what lets the whole scoring path be
    tested without a GPU.

    Read per request rather than at import, so one server instance can play
    several roles across a test session.
    """
    return os.environ.get("MOCK_REPLY")


def scripted_tool_calls() -> list[dict] | None:
    """Tool calls to stream instead of content, or None.

    `MOCK_TOOL_CALLS` holds a JSON list of {"name": ..., "arguments": {...}}.
    Set, the mock behaves the way vLLM does under `--enable-auto-tool-choice`:
    the reply carries NO content at all and the call arrives in `delta.tool_calls`
    instead, with its arguments split across chunks.

    That split is the point. A tool call does not arrive whole -- the name comes
    once and the argument JSON streams in pieces, interleaved across indices
    when there are several calls -- and reassembling it is the part of the client
    that BFCL depends on and that no other suite exercises.
    """
    raw = os.environ.get("MOCK_TOOL_CALLS")
    return json.loads(raw) if raw else None


_state = {
    "in_flight": 0,
    "prompt_tokens": 0.0,
    "generation_tokens": 0.0,
    "spec_drafts": 0.0,
    "spec_draft_tokens": 0.0,
    "spec_accepted_tokens": 0.0,
}

WORDS = "the quick brown fox jumps over a lazy dog while parsing tokens".split()


def _estimate_prompt_tokens(messages: list[dict], tools: list | None) -> int:
    """Rough character-based estimate. The real server tokenizes properly."""
    blob = json.dumps(messages) + (json.dumps(tools) if tools else "")
    return max(1, len(blob) // 4)


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("", status_code=200)


@app.get("/version")
async def version() -> JSONResponse:
    return JSONResponse({"version": "mock-0.0.0"})


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse({"data": [{"id": MODEL_NAME, "object": "model"}]})


@app.get("/metrics")
async def prometheus() -> PlainTextResponse:
    lines = [
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{model_name="{MODEL_NAME}"}} {_state["prompt_tokens"]}',
        "# TYPE vllm:generation_tokens_total counter",
        f'vllm:generation_tokens_total{{model_name="{MODEL_NAME}"}} {_state["generation_tokens"]}',
        "# TYPE vllm:num_preemptions_total counter",
        f'vllm:num_preemptions_total{{model_name="{MODEL_NAME}"}} 0.0',
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="{MODEL_NAME}"}} {_state["in_flight"]}',
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="{MODEL_NAME}"}} 0.0',
        # Stands in for KV cache pressure so the gauge sampler has something
        # that rises and falls with load, the way the real one does.
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        f'vllm:gpu_cache_usage_perc{{model_name="{MODEL_NAME}"}} '
        f'{min(1.0, _state["in_flight"] / 32.0):.4f}',
    ]
    if SPEC_ENABLED:
        lines += [
            "# TYPE vllm:spec_decode_num_drafts_total counter",
            f'vllm:spec_decode_num_drafts_total{{model_name="{MODEL_NAME}"}} {_state["spec_drafts"]}',
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            f'vllm:spec_decode_num_draft_tokens_total{{model_name="{MODEL_NAME}"}} {_state["spec_draft_tokens"]}',
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
            f'vllm:spec_decode_num_accepted_tokens_total{{model_name="{MODEL_NAME}"}} {_state["spec_accepted_tokens"]}',
        ]
    return PlainTextResponse("\n".join(lines) + "\n")


def _chunk(request_id: str, created: int, model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", MODEL_NAME)
    max_tokens = int(body.get("max_tokens", 16))
    messages = body.get("messages", [])
    tools = body.get("tools")
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    prompt_tokens = _estimate_prompt_tokens(messages, tools)

    # Filler always runs to max_tokens and finishes on length: ignore_eos is
    # what the speed sweep relies on for a fixed output length. A scripted
    # reply instead stops where it ends, one "token" per word, which is what
    # makes max_tokens bite the way it would on a real server -- a reply
    # longer than the cap comes back truncated and finishes on "length", so
    # the quality pass's truncation accounting has something real to measure.
    reply = scripted_reply()
    if reply is None:
        pieces = [WORDS[i % len(WORDS)] + " " for i in range(max_tokens)]
        finish = "length"
    else:
        words = reply.split(" ")
        pieces = [w + " " for w in words[:-1]] + [words[-1]]
        truncated = len(pieces) > max_tokens
        pieces = pieces[:max_tokens]
        finish = "length" if (truncated or body.get("ignore_eos")) else "stop"
    n_out = len(pieces)

    if not body.get("stream"):
        return JSONResponse(
            {"error": "this mock only implements streaming"}, status_code=400
        )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    async def generate():
        _state["in_flight"] += 1
        # Sampled once at admission so that the whole request is priced at the
        # contention level it was actually admitted under.
        factor = 1.0 + CONTENTION * max(0, _state["in_flight"] - 1)
        try:
            await asyncio.sleep(TTFT_MS / 1000.0 * factor)
            yield _chunk(request_id, created, model, {"role": "assistant", "content": ""})

            calls = scripted_tool_calls()
            if calls is not None:
                # Names first, then argument fragments round-robin across the
                # calls, so the indices interleave exactly as a real parallel
                # tool call does.
                for index, call in enumerate(calls):
                    yield _chunk(
                        request_id, created, model,
                        {"tool_calls": [{
                            "index": index,
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": call["name"], "arguments": ""},
                        }]},
                    )
                encoded = [json.dumps(c["arguments"]) for c in calls]
                for offset in range(max((len(e) for e in encoded), default=0)):
                    for index, blob in enumerate(encoded):
                        if offset < len(blob):
                            await asyncio.sleep(ITL_MS / 1000.0 * factor)
                            yield _chunk(
                                request_id, created, model,
                                {"tool_calls": [{
                                    "index": index,
                                    "function": {"arguments": blob[offset]},
                                }]},
                            )
            else:
                for i, piece in enumerate(pieces):
                    if i:
                        await asyncio.sleep(ITL_MS / 1000.0 * factor)
                    yield _chunk(request_id, created, model, {"content": piece})

            yield _chunk(request_id, created, model, {}, finish=finish)

            _state["prompt_tokens"] += prompt_tokens
            _state["generation_tokens"] += n_out
            if SPEC_ENABLED:
                drafts = n_out / max(1.0, 1.0 + SPEC_ACCEPT * NUM_SPEC_TOKENS)
                _state["spec_drafts"] += drafts
                _state["spec_draft_tokens"] += drafts * NUM_SPEC_TOKENS
                _state["spec_accepted_tokens"] += drafts * NUM_SPEC_TOKENS * SPEC_ACCEPT

            if include_usage:
                usage = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": n_out,
                        "total_tokens": prompt_tokens + n_out,
                    },
                }
                yield f"data: {json.dumps(usage)}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            _state["in_flight"] -= 1

    return StreamingResponse(generate(), media_type="text/event-stream")
