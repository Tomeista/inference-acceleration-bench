# bench

Latency and throughput measurement for quantized and speculatively decoded
Qwen3-8B served on vLLM, plus a quality pass that scores what each config gets
wrong in exchange for its speed. Results land in MLflow.

See `RUNBOOK.md` for the steps to run on the GPU server, and the
[experimental design summary](https://claude.ai/code/artifact/3408fccd-2da2-419b-b3ea-de0a802a4701)
for the same material written up for a reader rather than an operator.

## The test set

For latency and throughput, what a prompt *says* is close to irrelevant. Three
properties of its shape drive everything: input length, which sets prefill cost
and therefore TTFT; output length, which sets the number of decode steps; and
output entropy, which determines how often a speculative draft is accepted. A
fourth variable is not a property of the prompt at all but of the load:
concurrency.

Each class pins those to a fixed point, so a difference between two configs
cannot be prompt-length drift in disguise. The classes bracket the regimes where
acceleration behaves differently; they do not sample naturally occurring traffic.

| Class | In / out tokens | Temp | Why it is in the set |
|---|---|---|---|
| `c1_chat` | 64 / 128 | 0.0 | Balanced baseline, TTFT sensitive |
| `c2_longform` | 128 / 1024 | 0.7 | Pure decode, high entropy. Best case for weight-only quantization, worst for speculative decoding |
| `c3_rag` | 4000 / 128 | 0.0 | Pure prefill. Where 4-bit quantization loses on TTFT |
| `c5_toolcall` | 600 / 128 | 0.0 | Low-entropy structured JSON, three tool schemas. Best case for speculative decoding |

Concurrency is swept over {1, 32} by default. That range matters more than it
looks: quantization and speculative decoding both help most at batch 1 and can
both hurt at high batch, for different reasons. A suite that stays at one
concurrency will report a win that does not exist under real serving load.

**How a prompt is built.** Each is a seed instruction plus filler prose truncated
to the exact token budget. In `c3_rag` the filler is the document itself, with
the question appended after it so the resizing lands on the document rather than
on the question. `c5_toolcall` additionally carries one correct tool schema and
two distractors, with the intended tool name retained for the later quality pass.
The corpus is curated technical prose rather than ShareGPT or BFCL, so the set is
buildable and reviewable without network access.

**Reserved, not yet in scope.** Four further classes are defined but disabled and
extend the same framework without changing it: `c4_summarization` (8000 / 512,
copy-heavy decode), `c6_code` (300 / 512, the canonical speculative best case),
`c7_agent` (five sequential tool-using turns with growing KV), and `c8_thinking`
(200 / 2048, reasoning enabled).

## Server configurations

Seven servers, all of them Qwen3-8B, one per invocation:

| Config | Quant | Bits | Sparsity | Speculation |
|---|---|---|---|---|
| `baseline_bf16` | none | 16 | dense | none |
| `fp8_dynamic` | FP8 W8A8 dynamic | 8 | dense | none |
| `w4a16_gptq` | GPTQ W4A16 | 4 | dense | none |
| `sparse24_bf16` | none | 16 | 2:4 | none |
| `sparse24_w4a16` | GPTQ W4A16 | 4 | 2:4, via `marlin-24` | none |
| `bf16_eagle3` | none | 16 | dense | EAGLE-3 k=3 |
| `fp8_eagle3` | FP8 W8A8 dynamic | 8 | dense | EAGLE-3 k=3 |

These are not seven points on one axis. They form three deliberate comparisons,
and the value of the set comes from the pairs rather than from the individual
measurements.

**Quantization ladder:** `baseline_bf16` to `fp8_dynamic` to `w4a16_gptq`. Below
4 bits nothing is servable. vLLM has kernels for 4-bit and 8-bit integer, FP8 and
the FP4 microscaling formats, and nothing between or beneath.

**Sparsity 2x2** over {dense, 2:4} x {BF16, W4A16}: `baseline_bf16`,
`w4a16_gptq`, `sparse24_bf16`, `sparse24_w4a16`. Running all four rather than
only the combined config is what lets the analysis separate the sparsity effect
from the quantization effect and report the interaction. 2:4 is 50% sparse by
definition of the hardware pattern, so there is no sparsity ratio to sweep:
anything other than two zeros in four falls back to unstructured sparsity, which
vLLM stores but does not accelerate.

**Speculation 2x2** over {no draft, EAGLE-3} x {BF16, FP8}. Each speculative
config is paired with the config it differs from only by the draft model
(`bf16_eagle3` against `baseline_bf16`, `fp8_eagle3` against `fp8_dynamic`),
which is what lets a speedup be attributed to speculation rather than to the
quantization it is stacked on. A test asserts the pairing holds. The difference
between the rows is whether a quantized verifier changes acceptance.

The EAGLE-3 head is [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3),
trained with the speculators library, so no draft training is needed. `k=3` is
the value its card recommends. Use the non-thinking head unless `c8_thinking` is
enabled, in which case there is a separate Thinking variant.

**Peak KV cache usage is logged per cell** because `gpu_memory_utilization` is
pinned across configs. That pin stops a smaller checkpoint from quietly
receiving a larger cache, but it also means a draft model takes memory out of
the same budget, so a speculative server runs with a smaller cache than its own
control. Peak usage plus the preemption count is what separates "speculation was
slow" from "this config was starved". Gauges are sampled during the load, since
after it drains they all read zero, and the sampler reports nothing at all
rather than a zero peak when a cell was too short to sample.

## What each cell records

A cell is one prompt class at one concurrency against one config: 4 classes x 2
concurrency levels x 7 configs, or 56 cells. Each runs 32 requests at
concurrency 1 and 128 at concurrency 32, after a discarded warmup.

| Group | Recorded |
|---|---|
| Latency | TTFT, TPOT, inter-token latency, end-to-end, each at p50 / p90 / p95 / p99 and mean |
| Throughput | Output tokens/s, total tokens/s, requests/s, over the wall-clock window of the load phase |
| Speculation | Acceptance rate and mean accepted length, differenced across the run window. Absent rather than zero on a server with no draft model |
| Validity | Preemptions, peak KV cache usage, peak queue depth, finish reasons, observed against expected prompt length |

Two reporting rules the analysis depends on. **TTFT and throughput are always
reported separately**, because a single speedup figure would conceal the fact
that they can move in opposite directions, and that divergence is the result the
study exists to characterize. **Acceptance is reported next to every speculative
speedup**, because acceptance is the explanation, and a speedup without it is an
observation rather than a finding.

## Design decisions worth knowing

**Prompt sets are frozen.** `prompts/*.jsonl` is built once and committed. Every
run reads the same file, so a measurement from October is comparable to one from
December, and the later quality pass scores the exact prompts whose latency was
measured. `prompts/manifest.json` carries a hash per class, and every run records
it.

**Input lengths are exact.** A class asking for 4000 tokens gets prompts that are
4000 tokens after the Qwen3 chat template and any tool schemas are applied. All
512 prompts in the current set land exactly on budget. Without this, a difference
between two configs could be prompt-length drift.

**Output lengths are pinned.** Every request sets `ignore_eos` with a fixed
`max_tokens`, so a config that happens to stop early cannot look faster than it
is. Runs assert that every request finished on `length`. The quality pass runs
separately with natural stopping; see below.

**Thinking is off.** Qwen3 reasons by default, which would make output length a
property of the model's mood rather than of the class. `c8_thinking` exists to
measure that deliberately.

**Serving flags are pinned across configs.** `max_model_len` and
`gpu_memory_utilization` are held constant so that a smaller checkpoint does not
silently receive a larger KV cache and get credited for the difference.

**The harness never launches vLLM.** It attaches over HTTP and generates
`scripts/serve_*.sh` from `config/configs.yaml`, so the command that produced a
measurement lives in version control next to the measurement.

**Re-measurements are merged, not averaged.** MLflow appends, so a cell measured
twice has two runs, and a naive group-by blends a discarded measurement with the
one that replaced it. `bench.report` keeps the most recent cell per
`(config_id, class_id, concurrency)`, says how many it superseded, and hides
cells that failed a validity check (failed requests, output not length-capped,
preemptions) unless asked for them.

## The quality pass

Every number above gets *better* as compression increases. That is the half of
the trade the speed sweep can see, and on its own it argues for the most
aggressive config. So there is a second pass, `bench.quality`, that scores
what the configs actually say.

It inverts nearly every choice the speed sweep makes, because the two want
opposite things from a generation:

| | `bench.run` | `bench.quality` |
|---|---|---|
| stopping | `ignore_eos`, length pinned | natural, truncation counted |
| prompts | shapes with no right answer | benchmark items with a key |
| configs | all 7 | the 5 flagged `quality: true` |
| output | discarded | scored and kept per item |

What it does *not* invert is the server: the same `scripts/serve_<config>.sh`,
the same `max_model_len` and memory pin. Prefix caching changes how fast a
config answers, never what it answers.

**Five configs, not seven.** The quantization ladder and the sparsity 2x2 --
every config whose weights differ from `baseline_bf16`. The two EAGLE-3 configs
are left out because greedy speculative decoding is verified by the target
model, so it answers what its verifier answers; scoring one is a losslessness
check against its control (`--force --reference fp8_dynamic`), not a quality
measurement. A test asserts every weight-changing config stays in the subset.

**Three suites, all general, all zero-shot.** MMLU stratified across all 57
subjects (one letter of output), MMLU-Pro stratified across its 14 categories
(also one letter, and here for headroom -- see below), and GSM8K
chain-of-thought (~250 decode tokens an item, and the sensitive half:
quantization damage shows up in a chain of dependent steps long before it shows
up in a single recall lookup). All 250 items each, frozen into `evals/` with
digests exactly as `prompts/` is, and pinned to a dataset commit rather than to
`main`. Every request is greedy (`temperature 0`, `top_k 1`) with thinking off.

Any one suite can be run alone with `--suites`, which is what makes re-scoring a
single cell affordable:

```bash
uv run python -m bench.quality --config-id w4a16_gptq --suites mmlu_pro
```

**Why MMLU-Pro, and what it does not add.** Qwen3-8B sits high enough on MMLU
that there is little room between the BF16 reference and the ceiling for damage
to show. MMLU-Pro keeps the same cheap shape while moving the reference down and
the random floor to 10%, roughly four times the space to fall through. It is
*not* an independent second reading: 6,810 of its 12,032 items are MMLU
questions, so the two scores are correlated by construction (155 of the frozen
250). Each key row carries the item's `src`, so the overlap stays auditable. It
is also built to reward chain-of-thought, which this pass disables; if
`baseline_bf16` shows a high `truncated_rate` or `unparseable_rate` there, the
fix is a larger budget and a re-run of every config, not a scorer change.

**These numbers will not match published MMLU or GSM8K scores, and are not meant
to.** Published figures are few-shot and scored by log-likelihood ranking over
the options; these are zero-shot and scored by reading generated text. What the
pass measures is each config against *its own BF16 reference on identical
items*, which is the quantity the question asks for. Contamination cancels for
the same reason: every comparison is the same base model against itself.

**Accuracy is the headline and the blunt instrument; agreement is the sensitive
one.** `agreement_with_reference` is the fraction of items a config answers
identically to `baseline_bf16`, paired item by item. A config can hold its
accuracy while churning a third of its answers, and only the paired metric sees
that. The report never prints an accuracy without its Wilson interval, because
at 250 items that interval is about ±6 points -- wider than most pairs of
configs will differ by.

**Two failure modes are counted separately from being wrong.**
`unparseable_rate` is a config that has stopped answering in the requested
format -- instruction-following usually breaks before accuracy does -- and
`repetition_ratio` is a config stuck in a loop. **Truncation is counted, never
scored as wrong**: a reply cut off at `max_tokens` is an unmeasured item, and
above 5% the report flags the cell and its accuracy is a lower bound.

The honest limit: three benchmarks at 250 items is a coarse instrument. It is
enough to say whether a config's damage is visible, not to rank two configs
whose agreement differs by a point. See the RUNBOOK's determinism floor for how
small a difference is readable at all.

## Layout

```
config/classes.yaml     prompt class definitions (speed)
config/configs.yaml     server configurations; `quality: true` marks the scored subset
config/suites.yaml      benchmark suite definitions (quality)
prompts/                frozen prompt sets, committed
evals/                  frozen benchmark items + answer keys + manifest.json, committed
scripts/                generated serve scripts, committed
src/bench/
  classes.py            class config loader
  scenarios.py          Scenario / Turn: the unit of work
  corpus.py             offline seed material
  build_prompts.py      exact-token-budget prompt builder
  client.py             async load generator
  metrics.py            percentiles, aggregation, /metrics scrape
  server.py             config loader, serve scripts, readiness probe, context guard
  tracking.py           MLflow parent/child runs
  run.py                speed sweep driver
  quality.py            quality pass driver and preflight
  scoring.py            answer extraction, Wilson intervals, degeneracy measures
  suites.py             suite loader; pairs frozen items with their answer keys
  build_evals.py        one-time eval-set builder; output committed, not run-path
  report.py             the most-recent-wins merge rule, for both passes
tests/mock_server.py    fake vLLM, so the harness is testable without a GPU
```

## Extending it

**A new prompt class** is an entry in `config/classes.yaml` plus a builder
registered in `build_prompts.BUILDERS` under the same `source` name.

**A new benchmark suite** is an entry in `config/suites.yaml` plus a builder
registered in `build_evals.BUILDERS` under the same `source` name. Build it once
with `uv run --extra evals python -m bench.build_evals --suites <id>` and commit
the output; the manifest entries for the other suites are kept.

**A new quantization, sparsity, or speculative config** is an entry in
`config/configs.yaml`. No code changes; `--emit-scripts` regenerates the serve
script. Add `quality: true` if it changes the weights. Sparsity needed nothing beyond three metadata fields on `ServerConfig`,
because vLLM reads the pattern out of the checkpoint rather than from a flag.

**Checkpoints** are built in `../quantization-cpu/`: `quantize_qwen3_fp8.py`,
`quantize_qwen3_gptq.py`, and `prune_qwen3_24.py`. The last one prints a
reminder to run `verify_24_mask.py`, which is not optional: `GPTQModifier` has
no `preserve_sparsity_mask` flag, so a stacked prune-then-quantize recipe can
silently produce a dense checkpoint that still advertises 2:4. Overall sparsity
and mean zeros per group both read 0.50 and 2.00 for random 50% sparsity as well
as for true 2:4, so only the per-group exactness the verifier reports can tell
them apart.

**The multi-turn class (`c7_agent`)** is the one addition that needs more than a
config entry, and the client is already built for it: a `Scenario` is a sequence
of turns, and `Scenario.next_turn(history)` is the hook where a subclass builds
turn N+1 from turn N's tool result. Single-turn classes are the degenerate case
of the same loop.

**Acceptance-rate logging already works** and no-ops on servers without a draft
model, so nothing needs retrofitting when the speculative phase starts.

**Real datasets** replace `corpus.py` when the speculative phase starts. Synthetic
prose is fine for measuring BF16 against FP8, where only length matters, but
speculative acceptance depends on how predictable the model finds its own
continuation, so c5 and c6 need real prompts before those numbers mean anything.

## Development

```bash
uv sync --extra mock --extra dev
uv run pytest -q                            # 164 tests
uv run python -m bench.build_prompts
uv run python -m bench.build_evals --check  # eval-set digests, no network
```

The mock vLLM in `tests/mock_server.py` speaks enough of the OpenAI streaming
protocol and vLLM's Prometheus output that `bench.run` cannot tell the
difference. It is a protocol stub, not a simulator: its latencies are
configured, not modeled, so nothing it produces is a performance result.

With `MOCK_REPLY` set it answers every request with that literal text, which
makes it a server with a known and therefore checkable accuracy against the real
committed eval sets. That is how the whole quality path -- digest checks, the
answer-key join, scoring, the reference comparison and the MLflow nesting -- is
tested without a GPU. Nothing it produces is a quality result either.

All three entry points (`bench.run`, `bench.quality`, `bench.report`) default
to the same store, `mlflow.db` at the repository root, wherever they are
launched from. `MLFLOW_TRACKING_URI` or `--tracking-uri` overrides it.

`uv run --extra evals python -m bench.build_evals` reproduces all six committed
eval files byte-for-byte from the pinned dataset commits. That has been checked,
so a rebuild that reports different digests has changed something, and every
config scored before it is no longer comparable with every config scored after.
