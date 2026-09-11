# Runbook

Steps to run on the GPU server, in order. Each block is copy-pasteable and says
what to paste back. Everything above the line marked "GPU server" has already
been run and verified on the workstation.

## Verified locally (no action needed)

```bash
uv sync --extra mock --extra dev
uv run pytest -q                        # 164 passed
uv run python -m bench.build_prompts    # 512/512 prompts land on their exact budget
uv run python -m bench.build_evals --check   # all six eval files match the manifest
```

Smoke test of the whole sweep against the mock vLLM, which is how the client,
the percentile math, the acceptance-rate differencing and the MLflow structure
were checked without a GPU:

```bash
MOCK_TTFT_MS=15 MOCK_ITL_MS=1 uv run uvicorn --app-dir tests mock_server:app --port 8111 &
uv run python -m bench.run --config-id baseline_bf16 --server-url http://127.0.0.1:8111 \
    --requests-per-cell 24 --concurrency 1,8
```

And of the quality pass, with the mock answering "Answer: C" to everything:

```bash
MOCK_REPLY="Answer: C" uv run uvicorn --app-dir tests mock_server:app --port 8112 &
uv run python -m bench.quality --config-id baseline_bf16 --server-url http://127.0.0.1:8112 \
    --n-items 20 --no-mlflow
rm -r results/quality      # the mock's answers must never become the reference
```

The `rm` is not optional. The pass writes each config's per-item answers to
`results/quality/`, and a `baseline_bf16.jsonl` produced by the mock would be
joined against by the first real config scored after it.

---

# GPU server

## Step 0: record the hardware

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
free -g | head -2
```

**Paste back all three outputs.** `compute_cap` decides whether FP8 runs on
native tensor cores (8.9 and above) or through Marlin emulation, which changes
how the high-concurrency cells should be read and is worth knowing before the
numbers exist rather than after.

## Step 1: build the FP8 checkpoint

Needs roughly 20 GB of RAM on CPU, less wall time on GPU if it is idle.

```bash
cd quantization-cpu
uv run python quantize_qwen3_fp8.py --model-id Qwen/Qwen3-8B \
    --output-dir ../bench/models/Qwen3-8B-FP8-DYNAMIC
du -sh ../bench/models/Qwen3-8B-FP8-DYNAMIC
```

**Expected:** roughly 8 to 9 GB on disk, against about 16 GB for BF16. If it is
still near 16 GB the scheme did not apply and the rest of the comparison is
meaningless, so check this number before moving on.

## Step 2: start the baseline server

```bash
cd bench
bash scripts/serve_baseline_bf16.sh
```

In a second shell:

```bash
curl -s localhost:8000/health && echo OK
curl -s localhost:8000/v1/models | python -m json.tool
```

**Paste back** the startup log line reporting KV cache size (it looks like
`GPU KV cache size: N tokens`). That number has to be recorded for both configs:
if it differs between them, the pinned `gpu_memory_utilization` did not do its
job and the FP8 run is getting credit for a bigger cache.

## Step 3: validate the client against vLLM's own benchmark

Do this once. It is the evidence that the numbers in the thesis come from a
correct instrument, and it belongs in the methodology section.

```bash
vllm bench serve --backend openai-chat --model qwen3-8b \
    --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len 64 --random-output-len 128 \
    --num-prompts 32 --max-concurrency 1 --ignore-eos
```

Then the same shape through this harness:

```bash
uv run python -m bench.run --config-id baseline_bf16 \
    --classes c1_chat --concurrency 1 --requests-per-cell 32 --no-mlflow
```

**Paste back both.** Mean TTFT and mean TPOT should agree within a few percent.
If they do not, the client is the suspect, not vLLM.

## Step 4: measure the baseline

```bash
uv run python -m bench.run --config-id baseline_bf16
```

Runs all four classes at concurrency 1 and 32, 8 cells, roughly 10 to 20
minutes depending on the GPU. Watch for two warnings:

- `did not stop on length` means `ignore_eos` is not in force and output lengths
  are not controlled, which invalidates the cell.
- a non-zero `preemptions` metric means vLLM ran out of KV cache and evicted
  running sequences, which inflates tail latency for reasons unrelated to the
  config. Lower `--requests-per-cell` or `max_num_seqs` and rerun that cell.

## Step 5: measure FP8

Stop the baseline server, then:

```bash
bash scripts/serve_fp8_dynamic.sh
# second shell, after /health answers:
uv run python -m bench.run --config-id fp8_dynamic
```

At this point the MVP is complete and there is a result to look at. Step 9 works
already. Everything below extends it with the sparsity 2x2.

---

## Step 6: build the W4A16 checkpoint

The dense 4-bit cell, and the reference the sparse+quantized cell is measured
against.

```bash
cd quantization-cpu
uv run python quantize_qwen3_gptq.py --model-id Qwen/Qwen3-8B --scheme W4A16 \
    --output-dir ../bench/models/Qwen3-8B-W4A16-GPTQ
```

## Step 7: build the two 2:4 checkpoints

Both use the same calibration set as Step 6. A different one would turn the
sparsity comparison into a sparsity-plus-calibration comparison.

```bash
uv run python prune_qwen3_24.py --scheme none \
    --output-dir ../bench/models/Qwen3-8B-2of4
uv run python prune_qwen3_24.py --scheme W4A16 \
    --output-dir ../bench/models/Qwen3-8B-2of4-W4A16
```

Then the check that matters:

```bash
uv run python verify_24_mask.py ../bench/models/Qwen3-8B-2of4
uv run python verify_24_mask.py ../bench/models/Qwen3-8B-2of4-W4A16
```

**Paste back both.** `SparseGPTModifier` has a `preserve_sparsity_mask` flag and
`GPTQModifier` does not, so whether the pattern survives the second stage is an
open question, not a guarantee. The verifier prints `exact 2:4` per tensor and
exits non-zero if the pattern is broken.

Three outcomes to expect:

- **exit 0 on both.** Proceed.
- **exit 1 on the quantized one.** GPTQ overwrote the mask. The `sparse24_w4a16`
  cell is invalid and must be dropped or rebuilt with a different recipe. Do not
  benchmark it: vLLM may still serve it through `marlin-24` and produce a real
  latency number for a model that is not actually sparse.
- **exit 3 on the quantized one.** The weights are bit-packed and cannot be
  inspected directly. Treat the mask as unconfirmed, and rely on the exit 0 from
  the sparse-only checkpoint plus whatever vLLM logs at load time.

Note that overall sparsity and mean zeros per group cannot distinguish 2:4 from
random 50% sparsity: both read 0.50 and 2.00. Only the per-group exactness does,
which is why the verifier reports that column.

## Step 8: measure the two sparse configs

```bash
cd bench
bash scripts/serve_w4a16_gptq.sh      # then, in a second shell:
uv run python -m bench.run --config-id w4a16_gptq

bash scripts/serve_sparse24_bf16.sh
uv run python -m bench.run --config-id sparse24_bf16

bash scripts/serve_sparse24_w4a16.sh
uv run python -m bench.run --config-id sparse24_w4a16
```

Check the vLLM startup log for each sparse server: it should report a sparse or
`marlin-24` kernel. If it loads the checkpoint as dense, the cell measures
nothing and the mask or the config format is wrong.

## Step 9: speculative decoding

No draft model training. RedHatAI publishes an EAGLE-3 head for Qwen3-8B built
with the speculators library, so this is a download and a server restart.

```bash
bash scripts/serve_bf16_eagle3.sh
# second shell, after /health answers (first start also pulls the ~1 GB head):
uv run python -m bench.run --config-id bf16_eagle3
```

`baseline_bf16` from Step 4 is the matching control: same model path, same
quantization, differing only in the draft. Without that pair a speedup cannot be
attributed to speculation rather than to whatever else changed.

Then the quantized verifier, which is RQ2:

```bash
bash scripts/serve_fp8_eagle3.sh
uv run python -m bench.run --config-id fp8_eagle3
```

**Paste back the summary lines.** They now carry `accept=` and `len=` per cell.
Three things to check:

- **Instrument validation, free.** The model card reports mean accepted length
  at k=3 of 2.39 on HumanEval, 2.48 on GSM8K, 2.13 on CNN/DailyMail. Your `len=`
  uses the same definition (1 + accepted/drafts). Landing in that band confirms
  the speculative metrics the way Step 3 confirmed the latency client.
- **A predicted ordering.** `c5_toolcall` should exceed all three of those
  published numbers, and `c2_longform` should fall below them, because
  acceptance tracks how predictable the continuation is. Confirming a prediction
  made in advance is a stronger result than reporting an observed ordering.
- **KV cache pressure.** Compare `kv_cache_usage_peak` against the same cell in
  the non-speculative run. `gpu_memory_utilization` is pinned across configs, so
  the draft model takes memory out of the same budget the KV cache draws on, and
  the speculative server runs with a smaller cache than its own control. If
  `preemptions` is non-zero here and zero in the control, the c=32 tail latency
  is measuring cache starvation rather than speculation. The sweep warns when
  that happens.

If `len=` is absent from the output, the server exposed no speculative counters.
Check `curl -s localhost:8000/metrics | grep spec_decode` against
`metrics.COUNTERS`; the names have moved between vLLM versions.

## Step 10: compare

```bash
uv run python -m bench.report                 # one table per class and concurrency
uv run python -m bench.report --all           # including cells that failed a check
uv run python -m bench.report --csv speed.csv # the merged table, for plotting
```

**Use this rather than reading cells out of the UI by hand**, for one specific
reason: MLflow appends, so a cell that was measured twice has two runs, and
nothing in the UI stops a group-by from averaging a discarded measurement with
the one that replaced it. `bench.report` keeps the most recent cell per
`(config_id, class_id, concurrency)`, names the ones it superseded, and drops
cells with failed requests, output that was not length-capped, or preemptions
unless `--all` is given.

The UI is still the right tool for looking at one run:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Group by `params.config_id`, filter to one `params.class_id` at a time.

**Quantization ladder**, the four comparisons worth looking at first:

| Cell | What it should show |
|---|---|
| `c2_longform` at c=1 | Largest FP8 win. Decode is memory bound, weights halved. |
| `c3_rag` at c=1 | Smallest win, possibly negative on TTFT. Prefill is compute bound. |
| `c1_chat` c=1 vs c=32 | Whether the win survives batching. On Ampere it may not. |
| `c5_toolcall` | Baseline for the speculative phase, where this class should move most. |

**Sparsity 2x2**, holding the prompt class and concurrency fixed:

| | Dense | 2:4 | Effect of sparsity |
|---|---|---|---|
| **BF16** | `baseline_bf16` | `sparse24_bf16` | sparsity alone |
| **W4A16** | `w4a16_gptq` | `sparse24_w4a16` | sparsity given quantization |

The difference between those two rows is the interaction, and it is the part
worth reporting. Expect the sparsity effect to be small next to quantization,
roughly 1.1x to 1.3x, and largest in the bottom row where `marlin-24` combines
both.

**Speculative decoding**, with and without, at matched quantization:

| | No draft | EAGLE-3 k=3 |
|---|---|---|
| **BF16** | `baseline_bf16` | `bf16_eagle3` |
| **FP8** | `fp8_dynamic` | `fp8_eagle3` |

Read `spec_mean_accepted_length` alongside every speedup. Acceptance is the
explanation for the speedup, and a speedup reported without it is an observation
rather than a finding. The BF16 row is the clean measurement of what speculation
buys; the difference between the rows is whether a quantized verifier changes
acceptance, which is RQ2.

Expect the speedup to shrink or invert at c=32 in both rows. Rejected draft
tokens are wasted compute, and that is affordable at batch 1 and not at batch 32.

Report `ttft_s_p95` and `output_tps` separately throughout. A single speedup
number would hide the fact that they can move in opposite directions, which is
the finding the exposé is set up to make.

## Step 11: the quality pass

The speed sweep says a config got faster. It cannot say what that cost, because
every prompt in it runs with `ignore_eos` so that content cannot affect timing.
This pass scores instead, on the five configs flagged `quality: true`: the
quantization ladder and the sparsity 2x2.

**Same servers, same scripts.** Prefix caching changes how fast a config
answers, not what it answers, so there is no second serve-script set to keep in
sync.

```bash
uv run python -m bench.build_evals --check   # digests only, no network
bash scripts/serve_baseline_bf16.sh
# second shell, after /health answers:
uv run python -m bench.quality --config-id baseline_bf16 --preflight
```

**Paste back the preflight.** It checks the mirror image of what the speed sweep
relies on: that generation stops *naturally*. If every reply stops at
`max_tokens`, something is forcing length and nothing can be scored honestly. It
also fails when no answer parses out of any of the four replies, which at BF16
means the chat template is not rendering the prompt as it was built. The line
`extracted answers` should hold four letters.

Then the sweep, one server at a time. **`baseline_bf16` must go first** --
every other config's `agreement_with_reference` is a per-item join against the
answers it wrote:

```bash
uv run python -m bench.quality --config-id baseline_bf16
# stop the server, start the next one, and so on:
uv run python -m bench.quality --config-id fp8_dynamic
uv run python -m bench.quality --config-id w4a16_gptq
uv run python -m bench.quality --config-id sparse24_bf16
uv run python -m bench.quality --config-id sparse24_w4a16
```

`mmlu` and `mmlu_pro` are one letter of output per item and cost a few minutes
a config; `gsm8k` is ~250 decode tokens x 250 items and dominates the pass. It
runs at concurrency 32, pinned across configs. Do not delete `results/quality/`
mid-sweep -- that is where the reference answers live, and losing them costs
the agreement column.

**Running one suite at a time.** `--suites` takes a comma-separated list and
defaults to every enabled suite. Re-scoring one benchmark after a scorer fix, or
adding a suite to configs already measured, does not mean re-running the others:

```bash
uv run python -m bench.quality --config-id baseline_bf16 --suites mmlu_pro
uv run python -m bench.quality --config-id w4a16_gptq --suites mmlu_pro
```

The reference still goes first: its answers are stored per suite at
`results/quality/<suite>/baseline_bf16.jsonl`, so a suite never run on it has
nothing to join against. The report merges the new cells in beside the old ones
by the same most-recent-wins rule a re-measured speed cell uses.

### The determinism floor, once

vLLM batches, and batch composition changes floating-point reduction order, so
two runs of the *same* weights can differ by a token. Measure how much before
reading any small difference as a result:

```bash
bash scripts/serve_baseline_bf16.sh
uv run python -m bench.quality --config-id baseline_bf16 --concurrency 1 --suffix c1
```

Same config, same prompts, different batching. The `agree` figure on that row
is self-agreement, and **no agreement gap smaller than its complement is a
result.** If it comes back below ~0.99, say so in the write-up and treat it as
the noise floor. The `--suffix` keeps it as its own row rather than superseding
the c=32 measurement.

### Optional: speculative decoding is lossless

Greedy EAGLE-3 is verified by the target model, so it should answer what its
control answers. That is checkable with the same machinery, against the control
rather than against BF16:

```bash
bash scripts/serve_fp8_eagle3.sh
uv run python -m bench.quality --config-id fp8_eagle3 --force --reference fp8_dynamic
```

Expect `agree` at the determinism floor, not above it. Anything clearly below
it means the speculative server is not producing its verifier's output, and
every speedup measured on it in Step 9 is a speedup at a different quality.

### Reading it

```bash
uv run python -m bench.report --quality
uv run python -m bench.report --quality --csv quality.csv
```

**Accuracy is the headline and the blunter instrument.** At 250 items the 95%
interval is about ±6 points -- wider than most pairs of configs will differ by.
The table never prints the point estimate without the interval, for exactly that
reason.

**`agree` is where damage will actually show.** It is paired per item, so it
sees a config changing a third of its answers even when accuracy has not moved.
The two comparisons read the same way as the speed tables: down the
quantization ladder (`fp8_dynamic`, then `w4a16_gptq`), and across the sparsity
2x2, where the interaction is the part worth reporting.

**`unparse` and `rep` are the failure modes, separated on purpose.** A config
that stops emitting `Answer: C` has degraded even where its parseable answers
are right; one stuck in a loop has degraded differently again. Both usually
move on `gsm8k` before anything moves on `mmlu`.

### Warnings that mean stop

- **`truncated=NN%`** above ~5%. Replies are being cut off at `max_tokens`, so
  those items are unmeasured rather than wrong and the accuracy is a lower
  bound. Raise the suite's `max_tokens` and re-run **every** config -- a suite
  measured at two different caps is two different suites.
- **`unparse` high on `baseline_bf16` or `fp8_dynamic`.** This is a scorer bug,
  not a model result. Read `results/quality/<suite>/<config>.jsonl`, which keeps
  the raw text for this purpose, before believing the table.
- **digest failure from `build_evals --check`.** The eval sets have drifted from
  the manifest; configs measured either side of that are not comparable.
- **`the reference answers ... were produced against eval sets ...`.** A stale
  `results/quality/<suite>/baseline_bf16.jsonl`, from before the eval sets were
  rebuilt. It joins by item id perfectly well and the agreement column would be
  about nothing. Each items file carries a `.meta.json` recording what produced
  it, which is what catches this. Re-run the reference config.

## What is still unverified

The mock proves the harness is internally correct. It cannot prove that vLLM
streams exactly the chunk shapes assumed in `client._extract_delta`, that the
served chat template matches the one the prompts were built against, or that
the speculative counter names match this vLLM version. Step 3 catches the first
two. For the third, `curl -s localhost:8000/metrics | grep spec_decode` once a
draft model is running, and compare against `metrics.COUNTERS`.

For the quality pass, the mock cannot prove that vLLM honours the three
non-OpenAI fields every eval request carries (`top_k`, `seed`,
`chat_template_kwargs`). Step 11's preflight catches the one that matters most:
if `enable_thinking: false` were dropped, Qwen3 would open every reply with a
reasoning block, and the MMLU replies would run into their 16-token cap before
naming a letter.
