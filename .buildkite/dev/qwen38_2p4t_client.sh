#!/bin/bash
# Client leg for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup.
#
# Runs inside the head node's Ray container after run_multihost.sh has seen
# /health come up. The first request is the expensive one: the pipeline runs
# with SKIP_JAX_PRECOMPILE=1, so the whole 92-layer graph is compiled lazily on
# the first prefill. Do the smoke test before the benchmark and give it room.
set -euo pipefail

MODEL="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
mkdir -p "${ART}"

echo "--- smoke test (triggers first-request compilation) ---"
time curl -sS --max-time 5400 "http://localhost:${PORT}/v1/completions" \
  -X POST -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 32, \"temperature\": 0}" \
  | tee "${ART}/smoke.json"
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
  --save-result --result-dir "${ART}" --result-filename bench.json \
  2>&1 | tee "${ART}/bench.log"

# Did the local-disk compilation cache actually get written? This is the head
# host's copy only; each of the other three hosts has its own under the same
# bind mount. A non-zero count here means the next run on this slice starts
# warm -- that is the whole point of the /tmp/jax_cache_tpu7x mount.
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
echo "--- jax compilation cache (head host) ---"
echo "dir=${JAX_CACHE_DIR} entries=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)"
du -sh "${JAX_CACHE_DIR}" 2>/dev/null || true
