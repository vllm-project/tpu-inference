#!/usr/bin/env bash
# Drive a measurement against a server started by run_server.sh. Two
# evaluation styles, chosen like the weight form. The default reproduces
# the original commands these results have always used, 640 prompts at
# concurrency 64, no warmups, no chat template. EVAL=inferencex
# reproduces the public InferenceX protocol instead, 128 warmup requests
# discarded before the clock opens and the chat template applied.
# Results land in ./results/<style>-<input>x<output>/ per cell and the
# per-chip figure prints at the end.
#
#   ./run_client.sh                      the 8K/1K prefill heavy point
#   ISL=1024 OSL=8192 ./run_client.sh    the 1K/8K decode heavy point
#   EVAL=inferencex ./run_client.sh      the InferenceX protocol
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
EVAL="${EVAL:-current}"
case "$EVAL" in
  current)    WARMUPS=0;   TEMPLATE=();                    TPL=off;;
  inferencex) WARMUPS=128; TEMPLATE=(--use-chat-template); TPL=on;;
  *) echo "EVAL must be current or inferencex, got $EVAL" >&2; exit 1;;
esac
OUT="${OUT:-$SCRIPT_DIR/results/${EVAL}-${ISL}x${OSL}}"

# Fail in seconds on a missing piece rather than after a long server start.
for mod in aiohttp tqdm transformers; do
  PYTHONPATH="" python3 -c "import $mod" 2>/dev/null || {
    echo "python module $mod is missing, pip install aiohttp tqdm transformers lm-eval" >&2; exit 1; }
done

# The InferenceX repository carries the benchmark client at
# utils/bench_serving. Fetched once, pinned.
IX_DIR="${IX_DIR:-$SCRIPT_DIR/InferenceX}"
IX_PIN=d089a91
if [ ! -f "$IX_DIR/utils/bench_serving/benchmark_serving.py" ]; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/SemiAnalysisAI/InferenceX "$IX_DIR"
  git -C "$IX_DIR" checkout "$IX_PIN"
fi
[ -f "$IX_DIR/utils/bench_serving/benchmark_serving.py" ] \
  || { echo "benchmark client missing under $IX_DIR"; exit 1; }
[ "$(git -C "$IX_DIR" rev-parse --short=7 HEAD 2>/dev/null)" = "$IX_PIN" ] \
  || git -C "$IX_DIR" checkout "$IX_PIN"
BENCH_DIR="$IX_DIR/utils/bench_serving"

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "no server answering on port $PORT, start run_server.sh first" >&2
  exit 1
fi

# When the server log is visible, prove the right server before measuring.
SERVER_LOG="${SERVER_LOG:-$SCRIPT_DIR/server.log}"
if [ -f "$SERVER_LOG" ]; then
  OK=1
  grep -q "Sliced rope cache" "$SERVER_LOG" || OK=0
  grep -q "USE_MOE_FUSED_EP_KERNEL=1" "$SERVER_LOG" || OK=0
  grep -q "fused MoE FFN kernel is off" "$SERVER_LOG" && OK=0
  if [ "$OK" != 1 ]; then
    echo "the server log at $SERVER_LOG is missing the branch's engagement" >&2
    echo "lines, refusing to measure." >&2
    exit 1
  fi
else
  echo "server log not found at $SERVER_LOG, engagement not checked"
fi


mkdir -p "$OUT"
echo "cell ${ISL}/${OSL}, eval=$EVAL, concurrency 64, 640 prompts, warmups $WARMUPS"
PYTHONPATH="" python3 "$BENCH_DIR/benchmark_serving.py" \
  --model "$MODEL" --backend vllm \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random \
  --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio=0.8 \
  --num-prompts=640 --max-concurrency=64 --request-rate inf \
  --ignore-eos --num-warmups "$WARMUPS" "${TEMPLATE[@]}" \
  --save-result --result-dir "$OUT"
{ echo "seed=0 default"; echo "eval=$EVAL isl=$ISL osl=$OSL ratio=0.8 prompts=640 conc=64 warmups=$WARMUPS template=$TPL ignore_eos=on"; } > "$OUT/protocol.txt"

python3 - "$OUT" <<'PY'
import json, sys, glob, os
for p in sorted(glob.glob(f"{sys.argv[1]}/*.json"), key=os.path.getmtime, reverse=True):
    try: d = json.load(open(p))
    except Exception: continue
    tot = d.get("total_token_throughput"); out = d.get("output_throughput") or 0
    if tot:
        print(f"total {tot/4:.2f} per chip, output {out/4:.2f} per chip")
        break
else:
    print("no result file found, the run did not complete")
PY
