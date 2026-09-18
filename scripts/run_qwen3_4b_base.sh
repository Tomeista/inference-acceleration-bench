#!/usr/bin/env bash
# HAND-WRITTEN. The stock-Qwen3-4B control.
#
# Question: is the DB Bahn SFT model's collapse on terse multiple-choice a
# property of the FINE-TUNE or of the 4B SCALE? Every comparison so far has
# been against Qwen3-8B, which differs in both, so neither explanation has been
# excluded.
#
# Both budgets are scored, because they answer different questions:
#
#   mmlu_pro      (16 tokens)  INSTRUCTION-FOLLOWING. The decisive cell. The SFT
#                              model truncated 52.8% / unparseable 44.4% here;
#                              Qwen3-8B on identical items, 1.2% / ~0.
#   mmlu_pro_cot  (2048)       KNOWLEDGE, with room to reason. Gives a second
#                              reading: did three epochs of domain SFT also cost
#                              general capability? SFT model scored 0.596 here.
#
# --reference is this config ITSELF in both cases, deliberately. Agreement
# against the SFT model would be meaningless (different models, and quality.py
# now refuses that join on served_model_name anyway). What is being compared is
# the RATES, read side by side afterwards.
set -u
export PATH=$HOME/.local/bin:$PATH
BENCH=$HOME/inference-acceleration-bench
LOG=$HOME/bench_quality_logs
mkdir -p "$LOG"
cd "$BENCH"

CFG=qwen3_4b_base

docker rm -f bench-vllm >/dev/null 2>&1
docker run -d --name bench-vllm --gpus all --ipc=host --network host \
  -v /data/hf_cache:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -v "$BENCH:$BENCH" -w "$BENCH" \
  vllm/vllm-openai:latest Qwen/Qwen3-4B \
  --served-model-name qwen3-4b-base --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 256 \
  --dtype auto --seed 0 --no-enable-prefix-caching >/dev/null \
  || { echo "DOCKER RUN FAILED"; exit 1; }

ok=0
for i in $(seq 1 150); do
  curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
  docker ps --filter name=bench-vllm --format '{{.Status}}' | grep -q Up || break
  sleep 10
done
docker logs bench-vllm > "$LOG/serve_$CFG.log" 2>&1
[ $ok -eq 1 ] || { echo "SERVER FAILED"; tail -30 "$LOG/serve_$CFG.log"; exit 1; }
curl -s localhost:8000/v1/models | tr ',' '\n' | grep -i '"id"' | head -1
grep -E "GPU KV cache size" "$LOG/serve_$CFG.log" | cut -c1-160 | head -2

echo
echo "################ mmlu_pro (16 tokens) -- the instruction-following cell"
uv run python -m bench.quality --config-id "$CFG" --suites mmlu_pro \
  --reference "$CFG" 2>&1 | tail -5

echo
echo "################ mmlu_pro_cot (2048 tokens) -- the knowledge cell"
uv run python -m bench.quality --config-id "$CFG" --suites mmlu_pro_cot \
  --reference "$CFG" 2>&1 | tail -5

docker rm -f bench-vllm >/dev/null 2>&1

echo
echo "################ THE COMPARISON ################"
echo "  mmlu_pro @16 tokens        trunc   unparse"
echo "  Qwen3-8B base              0.012   0.012      (measured earlier)"
printf "  Qwen3-4B base (this run)   "
python3 - <<'PY'
import json, os
p = "results/quality/mmlu_pro/qwen3_4b_base.jsonl"
if os.path.exists(p):
    r = [json.loads(l) for l in open(p)]
    t = sum(1 for x in r if x.get("truncated")) / len(r)
    u = sum(1 for x in r if x.get("extracted") is None) / len(r)
    print(f"{t:.3f}   {u:.3f}")
else:
    print("  (not written)")
PY
echo "  Qwen3-4B DB Bahn SFT       0.528   0.444      (measured earlier)"
echo
echo "  low middle row  => the SFT eroded instruction-following"
echo "  high middle row => it is the 4B scale, not the fine-tune"
echo "################ BASE CONTROL DONE $(date +%T)"
