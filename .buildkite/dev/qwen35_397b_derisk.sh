#!/bin/bash
# De-risk arm for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup, run on 8 devices.
#
# Qwen3.5-397B-A17B-FP8 is the right stand-in: it is the same hybrid Gated
# DeltaNet + MoE architecture as the 2.4T (full_attention_interval=4, 512
# experts top-10, block-wise fp8) and, critically, has the *identical*
# linear_num_key_heads=16 that caps attention-head parallelism and forces the
# 2.4T onto attn_dp in the first place. It just fits on one host.
#
# CI already runs this model at plain tp=8 (pipeline_jax.yml:515+). The two
# things CI does NOT exercise, and that the 2.4T plan depends on, are:
#   1. attn_dp > 1 on a Gated DeltaNet model -- gdn_attention.py shards the
#      recurrent state on ShardingAxisName.ATTN_HEAD, and nothing in the repo
#      has ever combined that with a replicated attention batch split.
#   2. the batched RPA kernel with SEQ_ALONG_LANE layout on that same model.
# ARM_TAG=attndp2 turns both on; ARM_TAG=control leaves attn_dp at 1 so a
# failure can be attributed to one or the other rather than to the model.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
TAG="${ARM_TAG:-arm}"
mkdir -p "${ART}"

TP="${TENSOR_PARALLEL_SIZE:-8}"
ATTN_DP="${ATTN_DP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.88}"
# 128, not 256: the SEQ_ALONG_LANE layout hard-requires page_size==128
# (batched_rpa/configs.py:360-363). CI runs this model at --block-size 256, but
# CI does not use the batched RPA kernel.
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
LOAD_WAIT="${LOAD_WAIT_SECONDS:-3600}"

SERVE_LOG="${ART}/serve_${TAG}.log"

echo "=== arm=${TAG} model=${MODEL} tp=${TP} attn_dp=${ATTN_DP} -> model_axis=$((TP / ATTN_DP)) ==="
echo "--- effective env ---"
env | grep -E '^(VLLM_|MODEL_IMPL_TYPE|NEW_MODEL_DESIGN|USE_|ATTN_|SKIP_JAX|MOE_|ONEHOT_|TPU_|JAX_)' | sort || true
echo "--- jax cache ---"
echo "dir=${VLLM_XLA_CACHE_PATH:-unset} entries_before=$(find "${VLLM_XLA_CACHE_PATH:-/nonexistent}" -type f 2>/dev/null | wc -l)"

# attn_dp=1 is the control arm: leave --additional-config off entirely so the
# sharding config manager takes exactly the path CI takes today.
declare -a SHARDING_ARGS=()
if [ "${ATTN_DP}" -gt 1 ]; then
  SHARDING_ARGS=(--additional-config "{\"sharding\": {\"sharding_strategy\": {\"enable_dp_attention\": true, \"attn_dp_size\": ${ATTN_DP}}}}")
fi

vllm serve "${MODEL}" \
  --port "${PORT}" \
  --seed 42 \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  "${SHARDING_ARGS[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --no-enable-prefix-caching \
  --async-scheduling \
  > "${SERVE_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ~425 GB of weights off the persistent-disk HF cache, so the wait is long.
# Poll the process too: if vllm dies on a sharding assertion the health check
# would otherwise just burn the whole timeout.
echo "--- waiting up to ${LOAD_WAIT}s for /health ---"
for i in $(seq 1 "${LOAD_WAIT}"); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "!!! server exited after ${i}s -- last 100 lines:"
    tail -n 100 "${SERVE_LOG}"
    exit 1
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "--- server up after ${i}s ---"
    break
  fi
  if [ $((i % 60)) -eq 0 ]; then
    echo "[${i}s] $(tail -n 1 "${SERVE_LOG}")"
  fi
  sleep 1
done

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "!!! /health never came up in ${LOAD_WAIT}s -- last 100 lines:"
  tail -n 100 "${SERVE_LOG}"
  exit 1
fi

# First request compiles the whole graph (SKIP_JAX_PRECOMPILE=1). This is the
# call that actually executes the GDN shard_map under attn_dp, so it is the
# real pass/fail of the de-risk -- the benchmark below is just numbers.
#
# --fail-with-body, not plain -sS: curl exits 0 on an HTTP 500 and would let a
# dead engine sail through as a pass. Build #940 did exactly that -- the engine
# died on a kernel assertion, the smoke test recorded the 500 body, the bench
# then failed all 64 requests with connection-refused, and the job still
# reported `passed`.
echo "--- smoke test (first request = full compile) ---"
time curl -sS --fail-with-body --max-time 3600 "http://localhost:${PORT}/v1/completions" \
  -X POST -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 32, \"temperature\": 0}" \
  | tee "${ART}/smoke_${TAG}.json"
echo

echo "--- benchmark ---"
vllm bench serve \
  --backend vllm \
  --model "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN:-1024}" \
  --random-output-len "${OUTPUT_LEN:-128}" \
  --num-prompts "${NUM_PROMPTS:-64}" \
  --max-concurrency "${MAX_CONCURRENCY:-16}" \
  --request-rate inf --seed 42 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-dir "${ART}" --result-filename "bench_${TAG}.json" \
  2>&1 | tee "${ART}/bench_${TAG}.log"

# `vllm bench serve` exits 0 even when every request failed, so check the result.
python3 - "${ART}/bench_${TAG}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
done, failed = d.get("completed", 0), d.get("failed", 0)
print(f"[bench] completed={done} failed={failed} "
      f"out_tok/s={d.get('output_throughput', 0):.1f} "
      f"median_ttft_ms={d.get('median_ttft_ms', 0):.0f} "
      f"median_tpot_ms={d.get('median_tpot_ms', 0):.1f}")
if done == 0:
    sys.exit("[bench] FAILED: no request completed")
PY

# HBM accounting from the server log: how much did the weights actually take
# per device, and how much was left for KV + the GDN recurrent state. This is
# the number to extrapolate from when sizing the 2.4T run.
echo "--- memory lines from serve log ---"
grep -iE "available|kv cache|memory|hbm|GiB" "${SERVE_LOG}" | tail -n 40 || true

echo "--- jax cache after ---"
echo "dir=${VLLM_XLA_CACHE_PATH:-unset} entries_after=$(find "${VLLM_XLA_CACHE_PATH:-/nonexistent}" -type f 2>/dev/null | wc -l)"
