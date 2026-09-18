#!/usr/bin/env bash
# HAND-WRITTEN. Unlike serve_*.sh in this directory, bench.server does not
# generate this file -- it is the operator wrapper that starts the container,
# waits for health, and drives both passes. It is committed for the same reason
# the serve scripts are: the command that produced a measurement should sit in
# version control next to the measurement.
#
# Qwen3-4B DB Bahn SFT, before and after:
#
#     dbbahn_bf16         the "before" AND the quality reference
#     dbbahn_fp8_eagle3   the "after"  (FP8 W8A8 dynamic + EAGLE-3 k=3)
#
# Scope is deliberately two configs. Note what that costs: there is no
# dbbahn_fp8 control, so the speedup is quantization and speculation together
# and cannot be decomposed. tests/test_server.py records that gap explicitly.
#
# Modelled on ~/bench_quality_logs/run_eagle.sh, with three differences that
# matter:
#
#   1. --served-model-name qwen3-4b-dbbahn, not the qwen3-8b those scripts
#      hardcode. bench.run and bench.quality only warn when a config's served
#      name is ABSENT from /v1/models, so reusing the 8B name would let this
#      sweep attach to an 8B server and report a plausible, mislabelled result.
#
#   2. --reference dbbahn_bf16 is passed EVERY time, including on the reference
#      config itself. The flag defaults to baseline_bf16, which is the 8B
#      model; quality.py now refuses that join rather than silently reporting
#      "how often a 4B SFT model agrees with an 8B base model" as if it were a
#      compression-damage number.
#
#   3. --suites mmlu_pro only.
#
# Usage:  bash scripts/run_dbbahn.sh          (from the bench checkout)
set -u
export PATH=$HOME/.local/bin:$PATH
BENCH=$HOME/inference-acceleration-bench
LOG=$HOME/bench_quality_logs
mkdir -p "$LOG"
cd "$BENCH"

REFERENCE=dbbahn_bf16
# BOTH, and they measure different things -- keep both cells.
#
#   mmlu_pro      16-token budget. This model truncated 52.8% of replies and
#                 left 44.4% unparseable here, against 1.2% / ~0 for Qwen3-8B
#                 on identical items. That is not a failed measurement, it is
#                 the instruction-following result: the SFT answers generic
#                 multiple choice with chain-of-thought despite a "Do not
#                 explain." instruction. `unparseable_rate` is what that is
#                 for. Its ACCURACY, though, is a lower bound over a biased
#                 subset and must not be read as a quality number.
#
#   mmlu_pro_cot  the same suite at 512 tokens and nothing else changed --
#                 byte-identical prompts, same pinned commit, same seed. This
#                 is the cell that measures knowledge, and the one the FP8
#                 losslessness check should be read from.
#
# Neither is comparable with the 8B study's mmlu_pro figures: a suite measured
# at two different caps is two different suites.
SUITES=mmlu_pro,mmlu_pro_cot

model_of() { grep -oP '^vllm serve \K\S+' "scripts/serve_$1.sh"; }
spec_of()  { grep -oP "(?<=--speculative-config ')[^']+" "scripts/serve_$1.sh"; }

start_server() {
  local c=$1 spec extra=()
  spec=$(spec_of "$c" || true)
  [ -n "${spec:-}" ] && extra=(--speculative-config "$spec")

  docker rm -f bench-vllm >/dev/null 2>&1
  docker run -d --name bench-vllm --gpus all --ipc=host --network host \
    -v /data/hf_cache:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
    -e VLLM_USE_FLASHINFER_SAMPLER=0 \
    -v "$BENCH:$BENCH" -w "$BENCH" \
    vllm/vllm-openai:latest "$(model_of "$c")" \
    --served-model-name qwen3-4b-dbbahn --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 256 \
    --dtype auto --seed 0 "${extra[@]}" \
    --no-enable-prefix-caching >/dev/null || { echo "DOCKER RUN FAILED for $c"; return 1; }

  # The EAGLE-3 head downloads on first start, so allow a long window.
  local ok=0 i
  for i in $(seq 1 150); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    docker ps --filter name=bench-vllm --format '{{.Status}}' | grep -q Up || break
    sleep 10
  done
  docker logs bench-vllm > "$LOG/serve_$c.log" 2>&1
  if [ $ok -ne 1 ]; then
    echo "SERVER FAILED for $c -- last 30 lines:"; tail -30 "$LOG/serve_$c.log"; return 1
  fi

  # KV cache size must be recorded per config: gpu-memory-utilization is pinned,
  # so a draft model takes memory out of the same budget the cache draws on, and
  # the speculative server runs with a smaller cache than its own control.
  grep -E "GPU KV cache size|[Ss]peculative|EAGLE|eagle|draft" "$LOG/serve_$c.log" \
    | grep -v "Initializing a V1" | cut -c1-200 | head -6
  curl -s localhost:8000/v1/models | tr ',' '\n' | grep -i '"id"' | head -2
  return 0
}

echo "################ free memory before start"
free -h | head -2

################################################################################
# 1. dbbahn_bf16 -- the "before", and the reference every later join needs
################################################################################
echo; echo "################ dbbahn_bf16 start $(date +%T)"
start_server dbbahn_bf16 || exit 1

# Gate, not a formality. MMLU-Pro here is zero-shot, thinking off, 16-token
# budget; a model SFT'd on one narrow domain is exactly the kind that stops
# answering generic multiple choice in the requested format. If this fails,
# that is a protocol problem to fix BEFORE scoring anything, not a result.
echo "--- preflight (gate)"
uv run python -m bench.quality --config-id dbbahn_bf16 --suites "$SUITES" --preflight \
  || { echo "PREFLIGHT FAILED -- stopping before any number is produced"; exit 1; }

echo "--- quality: the reference itself"
uv run python -m bench.quality --config-id dbbahn_bf16 \
  --suites "$SUITES" --reference "$REFERENCE" 2>&1 | tail -6

echo "--- speed c=1 (16 req/cell, matching the 8B study)"
uv run python -m bench.run --config-id dbbahn_bf16 --concurrency 1 --requests-per-cell 16 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"
echo "--- speed c=32"
uv run python -m bench.run --config-id dbbahn_bf16 --concurrency 32 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"

################################################################################
# 2. dbbahn_fp8_eagle3 -- the "after"
################################################################################
echo; echo "################ dbbahn_fp8_eagle3 start $(date +%T)"
start_server dbbahn_fp8_eagle3 || exit 1

# --force because speculative configs are deliberately outside the quality
# subset: greedy EAGLE-3 is verified by the target, so this is a losslessness
# check against its own family, not a quality measurement. Expect `agree` at
# the determinism floor. Clearly below it means the speculative server is not
# reproducing its verifier's output, and every speedup below is at a different
# quality.
echo "--- losslessness vs $REFERENCE"
uv run python -m bench.quality --config-id dbbahn_fp8_eagle3 --force \
  --reference "$REFERENCE" --suites "$SUITES" 2>&1 | tail -6

echo "--- speed c=1 (16 req/cell)"
uv run python -m bench.run --config-id dbbahn_fp8_eagle3 --concurrency 1 --requests-per-cell 16 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"
echo "--- speed c=32"
uv run python -m bench.run --config-id dbbahn_fp8_eagle3 --concurrency 32 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"

# Acceptance is the explanation for whatever speedup the rows above show. If
# `len=` was absent from them, the counters did not register and the names have
# moved between vLLM versions -- compare against metrics.COUNTERS.
echo "--- speculative counters"
curl -s localhost:8000/metrics | grep -E "^vllm:spec_decode" | cut -d'{' -f1 | sort -u | head -8

docker logs bench-vllm > "$LOG/serve_dbbahn_fp8_eagle3.log" 2>&1
docker rm -f bench-vllm >/dev/null 2>&1
echo; echo "################ ALL DONE $(date +%T)"
echo "read with:  uv run python -m bench.report --csv speed_dbbahn.csv"
echo "            uv run python -m bench.report --quality --csv quality_dbbahn.csv"
