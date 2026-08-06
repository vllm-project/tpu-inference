#!/usr/bin/env bash
# The concurrency sweep for the 8K/1K workload, a fresh server per tier,
# tiers 64, 128, 256, 512. Two evaluation styles, chosen like the weight
# form. The default reproduces the original commands, 640 prompts at
# every tier, no warmups, no chat template, server cap fixed at 64
# sequences per rank. EVAL=inferencex reproduces the public InferenceX
# harness instead, 10x prompts, 2x warmups, chat template, and the
# per-rank cap rising with the tier the way their harness sizes servers.
#
#   ./run_sweep.sh                        eight-bit weights
#   WEIGHTS=fp4 ./run_sweep.sh            four-bit MoE weights
#   EVAL=inferencex ./run_sweep.sh        the InferenceX protocol
#
# Results land in ./sweep-results/<style>-<weights>/ and a summary prints at the end. The
# compile cache lives in ./jax-cache beside this script. Each new tier cap
# compiles its own shapes once. The script starts its own servers, stop any running
# server on the port first.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

WEIGHTS="${WEIGHTS:-fp8}"
case "$WEIGHTS" in
  fp4|fp8) ;;
  *) echo "run_sweep.sh: WEIGHTS must be fp4 or fp8, got '$WEIGHTS'" >&2; exit 2 ;;
esac
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
EVAL="${EVAL:-current}"
case "$EVAL" in
  current|inferencex) ;;
  *) echo "run_sweep.sh: EVAL must be current or inferencex, got '$EVAL'" >&2; exit 2 ;;
esac
OUT="${OUT:-$SCRIPT_DIR/sweep-results/${EVAL}-${WEIGHTS}}"
ISL=8192; OSL=1024

# The InferenceX repository carries the benchmark client at
# utils/bench_serving. Fetched once, pinned.
IX_DIR="${IX_DIR:-$SCRIPT_DIR/InferenceX}"
IX_PIN=d089a91
if [ ! -f "$IX_DIR/utils/bench_serving/benchmark_serving.py" ]; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/SemiAnalysisAI/InferenceX "$IX_DIR" || exit 1
  git -C "$IX_DIR" checkout "$IX_PIN" || exit 1
fi
CLIENT="$IX_DIR/utils/bench_serving/benchmark_serving.py"
[ -f "$CLIENT" ] || { echo "benchmark client missing under $IX_DIR"; exit 1; }
[ "$(git -C "$IX_DIR" rev-parse --short=7 HEAD 2>/dev/null)" = "$IX_PIN" ] \
  || git -C "$IX_DIR" checkout "$IX_PIN" || exit 1

export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export USE_MOE_EP_KERNEL=0
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256
export USE_MOE_FUSED_EP_KERNEL=1
export USE_MOE_FUSED_GMM_KERNEL=1
export MOE_FUSED_EP_KERNEL_MIN_TOKENS=1024
export GDN_BF16_RECURRENT_STATE=1
export SLICE_ROPE_CACHE=1
export LOGITS_ALL_GATHER_CONSERVATIVE=1
export DP_SCHED_BATCH_PREFILL=1
if [ "$WEIGHTS" = fp4 ]; then
  export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
  export MOE_REQUANTIZE_BLOCK_SIZE=512
else
  unset MOE_REQUANTIZE_WEIGHT_DTYPE MOE_REQUANTIZE_BLOCK_SIZE || true
fi
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'
export JAX_EXPLAIN_CACHE_MISSES=true
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"

# Persistent compile cache. The first server start pays the full compile,
# every later start reuses it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRIPT_DIR/jax-cache}"
export VLLM_XLA_CACHE_PATH="$JAX_COMPILATION_CACHE_DIR"
# Fail in seconds on a missing piece rather than after a long server start.
for mod in aiohttp tqdm transformers; do
  PYTHONPATH="" python3 -c "import $mod" 2>/dev/null || {
    echo "python module $mod is missing, pip install aiohttp tqdm transformers lm-eval" >&2; exit 1; }
done

mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$OUT"

compiles() {
  local n
  n=$(grep -c "PERSISTENT COMPILATION CACHE MISS" "$1" 2>/dev/null)
  echo "${n:-0}"
}

stop_server() { # $1 server process group leader
  [ "${1:-0}" -gt 1 ] || return 0
  kill -TERM -"$1" 2>/dev/null
  sleep 20
  for q in $(pgrep -g "$1" 2>/dev/null); do kill -9 "$q" 2>/dev/null; done
  for i in $(seq 1 12); do
    pgrep -g "$1" >/dev/null 2>&1 || break
    sleep 5
  done
}
SPID=0
trap 'stop_server "$SPID"; exit 130' INT TERM HUP

for CONC in 64 128 256 512; do
  PD="$OUT/conc${CONC}"
  mkdir -p "$PD"
  rm -f "$PD"/*.json "$PD/CLIENT_FAILED" "$PD/SERVER_DIED" \
    "$PD/KERNELS_NOT_PROVEN" "$PD/COMPILE_DURING_CLIENT"
  if [ "$EVAL" = inferencex ]; then
    SRVCAP="$CONC"
    case "$CONC" in
      128) export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128 ;;
      256) export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256 ;;
      512) export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256,512 ;;
      *)   export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ;;
    esac
  else
    SRVCAP=64
    export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64
  fi
  echo "=== tier conc=$CONC, eval=$EVAL, fresh server, cap $SRVCAP per rank ==="

  setsid vllm serve "$MODEL" \
    --max-model-len=$((ISL + OSL + 20)) --max-num-batched-tokens=1024 --max-num-seqs="$SRVCAP" \
    --no-enable-prefix-caching --gpu-memory-utilization=0.88 \
    --tensor-parallel-size=8 --async-scheduling --port="$PORT" \
    --language-model-only --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 \
    '--limit-mm-per-prompt={"image": 0, "video": 0}' \
    --kv-cache-dtype=fp8 --enable-expert-parallel \
    '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
    --block-size=256 > "$PD/server.log" 2>&1 &
  SPID=$!

  # Health, bounded at 3 hours. A new tier cap compiles shapes that have
  # never existed, which can take well over an hour the first time.
  HEALTHY=0
  DIED=0
  for i in $(seq 1 1080); do
    kill -0 "$SPID" 2>/dev/null || { DIED=1; break; }
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
    sleep 10
  done
  if [ "$HEALTHY" != 1 ]; then
    if [ "$DIED" = 1 ]; then
      echo "server exited during startup, see $PD/server.log"
    else
      echo "server not healthy after 3 hours, see $PD/server.log"
    fi
    stop_server "$SPID"
    exit 1
  fi

  # A server that lost these settings still starts and serves, at stock
  # speed, so prove the right server before measuring it.
  OK=1
  grep -q "Sliced rope cache" "$PD/server.log" || OK=0
  grep -q "USE_MOE_FUSED_EP_KERNEL=1" "$PD/server.log" || OK=0
  grep -q "fused MoE FFN kernel is off" "$PD/server.log" && OK=0
  if [ "$WEIGHTS" = fp8 ]; then
    grep -q "re-quantizing MoE weights to float8_e4m3fn" "$PD/server.log" || OK=0
    grep -q "re-quantizing MoE weights to float4_e2m1fn" "$PD/server.log" && OK=0
  else
    grep -q "re-quantizing MoE weights to float4_e2m1fn" "$PD/server.log" || OK=0
  fi
  if [ "$OK" != 1 ]; then
    echo "verification lines missing in $PD/server.log, refusing to measure this server"
    stop_server "$SPID"
    exit 1
  fi

  if [ "$EVAL" = inferencex ]; then
    NP=$((CONC * 10)); WARMUPS=$((CONC * 2)); TEMPLATE=(--use-chat-template); TPL=on
  else
    NP=640; WARMUPS=0; TEMPLATE=(); TPL=off
  fi
  CB=$(compiles "$PD/server.log")
  PYTHONPATH="" python3 "$CLIENT" \
    --model "$MODEL" --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --dataset-name random \
    --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio=0.8 \
    --num-prompts="$NP" --max-concurrency="$CONC" --request-rate inf \
    --ignore-eos --num-warmups "$WARMUPS" "${TEMPLATE[@]}" \
    --save-result --result-dir "$PD" 2>&1 | tee "$PD/client.log"
  RC=${PIPESTATUS[0]}
  { echo "seed=0 default"; echo "eval=$EVAL isl=$ISL osl=$OSL ratio=0.8 prompts=$NP conc=$CONC warmups=$WARMUPS template=$TPL ignore_eos=on"; } > "$PD/protocol.txt"
  if [ "$RC" != 0 ]; then
    echo "client failed at conc=$CONC, see $PD/client.log"
    touch "$PD/CLIENT_FAILED"
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "server died during the cell at conc=$CONC"; touch "$PD/SERVER_DIED"; }
  CA=$(compiles "$PD/server.log")
  if [ $((CA - CB)) -gt 0 ]; then
    echo "compiles landed while the client ran at conc=$CONC. Compiles during the"
    echo "warmup phase are fine, one after the clock opened invalidates the tier."
    echo "Check the compile timestamps in $PD/server.log."
    touch "$PD/COMPILE_DURING_CLIENT"
  fi

  if ! grep -q "runs on the fused expert-parallel MoE kernel" "$PD/server.log"; then
    echo "the branch kernels did not serve this tier, see $PD/server.log"
    touch "$PD/KERNELS_NOT_PROVEN"
  fi

  stop_server "$SPID"
  SPID=0
done

echo "=== sweep summary, per chip ==="
python3 - "$OUT" <<'PY2'
import json, sys, glob, os
for c in (64, 128, 256, 512):
    pd = f"{sys.argv[1]}/conc{c}"
    flags = []
    if os.path.exists(f"{pd}/COMPILE_DURING_CLIENT"):
        flags.append("compiles during the client run, check server.log timestamps")
    if os.path.exists(f"{pd}/CLIENT_FAILED"):
        flags.append("client failed, see client.log")
    if os.path.exists(f"{pd}/KERNELS_NOT_PROVEN"):
        flags.append("the branch kernels did not serve, see server.log")
    if os.path.exists(f"{pd}/SERVER_DIED"):
        flags.append("the server died during the cell, number invalid")
    bad = "".join(", " + f for f in flags)
    for p in sorted(glob.glob(f"{pd}/*.json"), key=os.path.getmtime, reverse=True):
        try: d = json.load(open(p))
        except Exception: continue
        t = d.get("total_token_throughput"); o = d.get("output_throughput") or 0
        if t:
            print(f"conc {c:>3}: total {t/4:8.2f}, output {o/4:7.2f}{bad}")
            break
    else: print(f"conc {c:>3}: no result{bad}")
PY2
