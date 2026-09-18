#!/usr/bin/env bash
# HAND-WRITTEN, like run_dbbahn.sh. Quality pass ONLY, on the mmlu_pro_cot
# suite. Speed is not re-measured: run_dbbahn.sh already recorded it and those
# cells are unaffected by anything below.
#
# Why this exists
# ---------------
# The first pass scored `mmlu_pro`, whose 16-token budget cannot measure this
# model. dbbahn_bf16 truncated 52.8% of replies and left 44.4% unparseable,
# against 1.2% / ~0 for Qwen3-8B on identical items: the SFT answers generic
# multiple choice with real chain-of-thought and runs out of budget
# mid-sentence. Accuracy over the measurable minority is a lower bound on a
# subset biased toward whatever was answerable in three tokens.
#
# `mmlu_pro_cot` is that suite with max_tokens 512 and NOTHING else changed --
# same pinned commit, same draw, same seed, same scorer, byte-identical prompts
# including the "Do not explain." instruction the model ignores. max_tokens is
# baked into the frozen items at build time, so this had to be a new suite with
# rebuilt items; editing suites.yaml alone would have changed only the MLflow
# param while every request still sent 16.
#
# Keep BOTH cells. The 16-token one is not a failed measurement, it is the
# instruction-following result (`unparseable_rate` is what that is for); this
# one measures knowledge given that the model ignores the format. Neither is
# comparable with the 8B study's mmlu_pro figures.
#
# Usage:  bash scripts/run_dbbahn_quality.sh
set -u
export PATH=$HOME/.local/bin:$PATH
BENCH=$HOME/inference-acceleration-bench
LOG=$HOME/bench_quality_logs
mkdir -p "$LOG"
cd "$BENCH"

REFERENCE=dbbahn_bf16
SUITE=mmlu_pro_cot

# Above this, ACCURACY is a lower bound (the report's own flagging threshold).
# Not fatal: at 2048 this model sits near 0.10 and the probe showed that tail is
# genuine non-termination on hard STEM items, not an under-budget artifact.
TRUNC_LIMIT=0.05
# Above this, the scoreable subset is too small for agreement to mean anything.
CATASTROPHE_LIMIT=0.35
# Max acceptable DIFFERENCE between the two configs' truncation rates. This is
# the one that actually protects the agreement number: at the old 16-token
# budget the two sat at 0.528 and 0.352, and the resulting agree=0.792 was
# computed over largely cut-off text and meant nothing.
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
    --served-model-name qwen3-4b-dbbahn --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 256 \
    --dtype auto --seed 0 "${extra[@]}" \
    --no-enable-prefix-caching >/dev/null || { echo "DOCKER RUN FAILED for $c"; return 1; }

  local ok=0 i
  for i in $(seq 1 150); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    docker ps --filter name=bench-vllm --format '{{.Status}}' | grep -q Up || break
    sleep 10
  done
  docker logs bench-vllm > "$LOG/quality_serve_$c.log" 2>&1
  [ $ok -eq 1 ] || { echo "SERVER FAILED for $c"; tail -30 "$LOG/quality_serve_$c.log"; return 1; }
  curl -s localhost:8000/v1/models | tr ',' '\n' | grep -i '"id"' | head -1
  return 0
}

################################################################################
# 1. dbbahn_bf16 -- the reference, and the gate
################################################################################
echo "################ $REFERENCE ($SUITE) $(date +%T)"
start_server "$REFERENCE" || exit 1

out=$(uv run python -m bench.quality --config-id "$REFERENCE" \
        --suites "$SUITE" --reference "$REFERENCE" 2>&1)
echo "$out" | tail -8

# The real gate. bench.quality's own --preflight samples four items, which
# cannot see a RATE: on the 16-token pass it returned "preflight OK" while 53%
# of the full set was truncating. So gate on what the finished reference cell
# actually reports, before spending a second server on a comparison that would
# be measured over a biased subset.
ref_trunc=$(echo "$out" | grep -oP 'trunc=\K[0-9.]+' | tail -1)
ref_unparse=$(echo "$out" | grep -oP 'unparse=\K[0-9.]+' | tail -1)
if [ -z "${ref_trunc:-}" ]; then
  echo "GATE FAILED: could not read trunc= from the reference run."; exit 1
fi
echo "--- reference: trunc=$ref_trunc unparse=$ref_unparse"

# Two different questions, and the first version of this script conflated them.
#
# For ACCURACY, truncation above ~5% means the number is a lower bound over a
# biased subset. That is true here (~10% projected at 2048) and is reported,
# not fixed: a probe at 2048 showed the residual tail is this model failing to
# terminate on hard STEM items, not a budget artifact.
#
# For AGREEMENT -- which is what this pass is actually for -- truncation is
# largely harmless, because the comparison is paired per item: if both configs
# truncate the same item at the same point with the same text, they agree. What
# destroys it is the two configs truncating at DIFFERENT rates, which is
# exactly what happened at the 16-token budget (52.8% vs 35.2%) and produced a
# meaningless agree=0.792. So the real gate is the SPREAD between the two
# configs, checked after both have run, not the reference's rate alone.
if awk "BEGIN{exit !($ref_trunc > $CATASTROPHE_LIMIT)}"; then
  echo "GATE FAILED: $ref_trunc truncated. Above $CATASTROPHE_LIMIT the scoreable"
  echo "subset is too small for agreement to mean anything. Raise the budget and"
  echo "REBUILD the suite -- max_tokens is frozen into the items, so editing"
  echo "suites.yaml alone changes only the logged parameter."
  exit 1
fi
if awk "BEGIN{exit !($ref_trunc > $TRUNC_LIMIT)}"; then
  echo "    NOTE: above $TRUNC_LIMIT, so ACCURACY from this suite is a LOWER BOUND."
  echo "    Report it with the truncation rate beside it. Agreement is unaffected"
  echo "    as long as the second config truncates at a similar rate (checked below)."
fi

################################################################################
# 2. dbbahn_fp8_eagle3 -- losslessness against its own family
################################################################################
echo; echo "################ dbbahn_fp8_eagle3 ($SUITE) $(date +%T)"
start_server dbbahn_fp8_eagle3 || exit 1

# --force because speculative configs sit outside the quality subset by design:
# greedy EAGLE-3 is verified by the target, so this is a losslessness check,
# not a quality measurement. Expect `agree` at the determinism floor. Clearly
# below it means the speculative server is not reproducing its verifier's
# output, and every speedup run_dbbahn.sh measured is at a different quality.
spec_out=$(uv run python -m bench.quality --config-id dbbahn_fp8_eagle3 --force \
             --reference "$REFERENCE" --suites "$SUITE" 2>&1)
echo "$spec_out" | tail -8

# The check the first version of this script was missing, and the one that
# actually protects the agreement number. Agreement is paired per item, so
# truncation is harmless while both configs truncate at the SAME rate -- and
# fatal when they do not, because the two are then compared over different
# subsets of each reply. At the old 16-token budget they sat at 0.528 and
# 0.352, and the resulting agree=0.792 measured nothing at all.
spec_trunc=$(echo "$spec_out" | grep -oP 'trunc=\K[0-9.]+' | tail -1)
agree=$(echo "$spec_out" | grep -oP 'agree=\K[0-9.]+' | tail -1)
if [ -n "${spec_trunc:-}" ]; then
  spread=$(awk "BEGIN{d=$spec_trunc-$ref_trunc; print (d<0?-d:d)}")
  echo
  echo "--- truncation spread check"
  echo "    reference   ($REFERENCE)      trunc=$ref_trunc"
  echo "    speculative (dbbahn_fp8_eagle3) trunc=$spec_trunc"
  echo "    spread=$spread (limit $SPREAD_LIMIT)   agree=${agree:-?}"
  if awk "BEGIN{exit !($spread > $SPREAD_LIMIT)}"; then
    echo "    *** DO NOT REPORT agree=${agree:-?} AS A LOSSLESSNESS FIGURE. ***"
    echo "    The two configs stopped early at materially different rates, so this"
    echo "    compares different subsets of each reply. Investigate before use."
  else
    echo "    OK: like-for-like. Expect agreement at the determinism floor;"
    echo "    clearly below it means the speculative server is not reproducing"
    echo "    its verifier's output, and every speedup measured is at a"
    echo "    different quality."
  fi
fi

docker rm -f bench-vllm >/dev/null 2>&1
echo; echo "################ DONE $(date +%T)"
echo "read with:  uv run python -m bench.report --quality --csv quality_dbbahn.csv"
