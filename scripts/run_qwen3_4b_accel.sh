#!/usr/bin/env bash
# HAND-WRITTEN, like run_dbbahn.sh and run_qwen3_4b_base.sh. bench.server
# generates serve_*.sh; this is the operator wrapper that starts the container,
# waits for health, and drives both passes. Committed for the same reason those
# are: the command that produced a measurement belongs next to the measurement.
#
# THE QUESTION
# ------------
# "FP8 + EAGLE-3 costs a domain-SFT'd model more accuracy than it costs a stock
# model." That is the interesting claim, and as of now the study cannot support
# it. The two measurements it rests on differ in THREE ways at once:
#
#     Qwen3-8B stock   baseline_bf16 -> fp8_eagle3         mmlu_pro   @16
#     Qwen3-4B SFT     dbbahn_bf16   -> dbbahn_fp8_eagle3  mmlu_pro_cot @2048
#
# size (8B vs 4B), training (stock vs three epochs of domain SFT), and suite
# (two different token budgets, which per suites.yaml are two different
# suites). Any one of the three explains the difference in drop, so none of
# them is attributable, and the SFT reading is simply the most flattering of
# several.
#
# This run pins size and suite and varies only the training:
#
#     Qwen3-4B stock   qwen3_4b_bf16_eagle3 -> qwen3_4b_fp8_eagle3   mmlu_pro_cot @2048
#
# Against the dbbahn pair that is a difference-in-differences. Same scale, same
# benchmark, same FP8_DYNAMIC recipe, same EAGLE-3 head at the same k. What is
# left varying between the two pairs is the fine-tune, which is the thing the
# claim is about.
#
# WHY BOTH ARMS CARRY THE DRAFT HEAD
# ----------------------------------
# The dbbahn pair's "before" has no draft head, so its delta is quantization
# and speculation confounded. Here the head is on BOTH sides, so it cancels and
# FP8 is the only difference. This costs nothing at greedy: EAGLE-3 proposes
# but the target model verifies every token, so it answers what its verifier
# answers -- measured at agreement 1.000 on the 8B pair.
#
# THE REFERENCE CHAIN, AND WHY IT IS TWO STEPS
# --------------------------------------------
#   qwen3_4b_bf16_eagle3  --reference qwen3_4b_base         (step 1)
#   qwen3_4b_fp8_eagle3   --reference qwen3_4b_bf16_eagle3  (step 2)
#
# Step 2 is the headline. Step 1 is not ceremony: qwen3_4b_base is the SAME
# weights with no draft head, already measured (mmlu_pro_cot acc 0.500,
# trunc 0.000, 2026-09-14), so any disagreement it shows is EAGLE-3 plus run to
# run nondeterminism and nothing else. That is this model's noise FLOOR, and
# without it the step 2 agreement number cannot be read -- on the SFT model the
# floor was 0.896, i.e. more than half of what looked like a 24-point FP8 wound
# was the harness breathing. All three cells share
# `served_model_name: qwen3-4b-base`, which is the field quality.py joins on.
#
# Usage:  bash scripts/run_qwen3_4b_accel.sh
set -u
export PATH=$HOME/.local/bin:$PATH
BENCH=$HOME/inference-acceleration-bench
LOG=$HOME/bench_quality_logs
mkdir -p "$LOG"
cd "$BENCH"

SUITE=mmlu_pro_cot
BASE=qwen3_4b_base
BF16=qwen3_4b_bf16_eagle3
FP8=qwen3_4b_fp8_eagle3

# Max acceptable DIFFERENCE between the two arms' truncation rates. This is the
# gate that actually protects the agreement number: agreement is paired per
# item, so truncation is harmless while both arms truncate at the same rate and
# fatal when they do not, because the two are then compared over different
# subsets of each reply. At the old 16-token budget the dbbahn pair sat at
# 0.528 vs 0.352 and the resulting agree=0.792 measured nothing at all.
SPREAD_LIMIT=0.05

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
    --served-model-name qwen3-4b-base --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 256 \
    --dtype auto --seed 0 "${extra[@]}" \
    --no-enable-prefix-caching >/dev/null || { echo "DOCKER RUN FAILED for $c"; return 1; }

  # Qwen3-4B (~8 GB) and the EAGLE-3 head both download on first start into
  # /data/hf_cache, which this repo's cache was cleared of. Allow a long window.
  local ok=0 i
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    docker ps --filter name=bench-vllm --format '{{.Status}}' | grep -q Up || break
    sleep 10
  done
  docker logs bench-vllm > "$LOG/serve_$c.log" 2>&1
  if [ $ok -ne 1 ]; then
    echo "SERVER FAILED for $c -- last 30 lines:"; tail -30 "$LOG/serve_$c.log"; return 1
  fi

  # KV cache size must be recorded per config. gpu-memory-utilization is pinned
  # across configs, so the FP8 arm's smaller weights leave more room for cache
  # and the draft head takes room back out of the same budget. Neither effect
  # is visible in the latency numbers without this line.
  grep -E "GPU KV cache size" "$LOG/serve_$c.log" | cut -c1-200 | head -2
  curl -s localhost:8000/v1/models | tr ',' '\n' | grep -i '"id"' | head -2
  return 0
}

trunc_of() { grep -oP 'trunc=\K[0-9.]+' <<<"$1" | tail -1; }

echo "################ free memory before start"; free -h | head -2

################################################################################
# RUN 1 + 3 : qwen3_4b_bf16_eagle3 -- quality, then speed
################################################################################
echo; echo "################ $BF16 start $(date +%T)"
start_server "$BF16" || exit 1

# --force because speculative configs sit outside the default quality subset by
# design (see configs.yaml). Against qwen3_4b_base -- the same weights without a
# draft head -- this is a losslessness check AND the noise floor every later
# agreement number is read against.
echo "--- quality: $SUITE, vs $BASE (losslessness + noise floor)"
bf16_out=$(uv run python -m bench.quality --config-id "$BF16" --force \
             --reference "$BASE" --suites "$SUITE" 2>&1)
echo "$bf16_out" | tail -8
bf16_trunc=$(trunc_of "$bf16_out")
[ -n "${bf16_trunc:-}" ] || { echo "GATE FAILED: no trunc= from $BF16"; exit 1; }

echo "--- speed c=1 (16 req/cell, matching the 8B study and the dbbahn pair)"
uv run python -m bench.run --config-id "$BF16" --concurrency 1 --requests-per-cell 16 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"
echo "--- speed c=32"
uv run python -m bench.run --config-id "$BF16" --concurrency 32 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"

# Acceptance explains whatever speedup the rows above show, and this head was
# trained against STOCK Qwen3-4B -- so this row is the matched-head ceiling and
# is the thing dbbahn_fp8_eagle3's 0.20-0.28 acceptance was always missing.
echo "--- speculative counters"
curl -s localhost:8000/metrics | grep -E "^vllm:spec_decode_num_(draft|accepted)_tokens_total" | head -4

################################################################################
# RUN 2 + 4 : qwen3_4b_fp8_eagle3 -- the headline arm
################################################################################
echo; echo "################ $FP8 start $(date +%T)"
start_server "$FP8" || exit 1

echo "--- quality: $SUITE, vs $BF16 (the FP8 delta)"
fp8_out=$(uv run python -m bench.quality --config-id "$FP8" --force \
            --reference "$BF16" --suites "$SUITE" 2>&1)
echo "$fp8_out" | tail -8
fp8_trunc=$(trunc_of "$fp8_out")

echo "--- speed c=1 (16 req/cell)"
uv run python -m bench.run --config-id "$FP8" --concurrency 1 --requests-per-cell 16 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"
echo "--- speed c=32"
uv run python -m bench.run --config-id "$FP8" --concurrency 32 2>&1 \
  | grep -vE "^[0-9]{4}/[0-9]{2}/[0-9]{2} .* INFO"

echo "--- speculative counters"
curl -s localhost:8000/metrics | grep -E "^vllm:spec_decode_num_(draft|accepted)_tokens_total" | head -4

docker logs bench-vllm > "$LOG/serve_$FP8.log" 2>&1
docker rm -f bench-vllm >/dev/null 2>&1

################################################################################
# Gate + paired statistics
################################################################################
echo; echo "################ truncation spread gate"
echo "    $BF16  trunc=$bf16_trunc"
echo "    $FP8   trunc=${fp8_trunc:-?}"
if [ -n "${fp8_trunc:-}" ]; then
  spread=$(awk "BEGIN{d=$fp8_trunc-$bf16_trunc; print (d<0?-d:d)}")
  echo "    spread=$spread (limit $SPREAD_LIMIT)"
  if awk "BEGIN{exit !($spread > $SPREAD_LIMIT)}"; then
    echo "    *** DO NOT REPORT THE AGREEMENT FIGURE. *** The two arms stopped"
    echo "    early at materially different rates, so they are being compared"
    echo "    over different subsets of each reply. Investigate before use."
  else
    echo "    OK: like-for-like."
  fi
fi

echo; echo "################ paired tests (McNemar, per item)"
echo "--- FLOOR: $BASE -> $BF16   (same weights, +draft head; expect no effect)"
uv run python scripts/paired_test.py --suite "$SUITE" --a "$BASE" --b "$BF16"
echo
echo "--- HEADLINE: $BF16 -> $FP8   (the FP8 delta, head on both sides)"
uv run python scripts/paired_test.py --suite "$SUITE" --a "$BF16" --b "$FP8"
echo
echo "--- FOR REFERENCE: the SFT pair, same suite (measured 2026-09-14)"
uv run python scripts/paired_test.py --suite "$SUITE" --a dbbahn_bf16 --b dbbahn_fp8_eagle3

echo; echo "################ ALL DONE $(date +%T)"
echo "read with:  uv run python -m bench.report --csv speed_qwen3_4b.csv"
echo "            uv run python -m bench.report --quality --csv quality_qwen3_4b.csv"
