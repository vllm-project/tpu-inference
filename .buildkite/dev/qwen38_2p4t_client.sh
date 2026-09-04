#!/bin/bash
# Client leg for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup.
#
# Runs inside the head node's Ray container after run_multihost.sh has seen
# /health come up. The smoke test still goes first and still gets a long
# deadline: precompilation is on now, so the first prefill should no longer pay
# for the whole 92-layer graph, but it is the cheapest place to find out if a
# shape slipped through the buckets.
set -euo pipefail

MODEL="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
DEV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ART}"

# RandomDataset defaults range_ratio to 0.0, so every request is exactly
# INPUT_LEN in and, with --ignore-eos, exactly OUTPUT_LEN out. The shape is
# therefore identical across runs and the only variance left is the server's.
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"

# --fail-with-body, not plain -sS: curl exits 0 on an HTTP 500, which on build
# tc#940 let a dead engine report the job as passed.
echo "--- smoke test (triggers first-request compilation) ---"
time curl -sS --fail-with-body --max-time 5400 "http://localhost:${PORT}/v1/completions" \
  -X POST -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 32, \"temperature\": 0}" \
  | tee "${ART}/smoke.json"
echo

# --num-warmups: build tc#947 ran with 0 and the means were unusable -- mean TTFT
# 44.5s against a 1.35s median, because the first few requests absorbed
# just-in-time compilation and the percentile machinery counts them like any
# other. serve.py fires the warmups concurrently under the same
# --max-concurrency semaphore before timing starts, so a count at or above
# MAX_CONCURRENCY also drives the scheduler to steady-state batch shapes rather
# than only the single-request path.
NUM_WARMUPS="${NUM_WARMUPS:-32}"

# Will the requested shape fit the KV cache, or will the scheduler preempt to
# make it fit? At 8192+1024 a single request holds 72 of the pool's 660 blocks,
# so this is a live constraint rather than a formality, and an oversubscribed
# run still reports numbers -- they just measure re-prefill. Fail before the
# benchmark rather than publish those.
python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
  --tokens-per-request $((INPUT_LEN + OUTPUT_LEN)) \
  --concurrency "${MAX_CONCURRENCY}" \
  --block-size "${BLOCK_SIZE:-128}"

echo "--- benchmark (in=${INPUT_LEN} out=${OUTPUT_LEN} n=${NUM_PROMPTS} conc=${MAX_CONCURRENCY} num_warmups=${NUM_WARMUPS}) ---"
vllm bench serve \
  --backend vllm \
  --model "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --num-warmups "${NUM_WARMUPS}" \
  --request-rate inf --seed 42 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-dir "${ART}" --result-filename bench.json \
  2>&1 | tee "${ART}/bench.log"

# `vllm bench serve` exits 0 even when every request failed, so check the result.
python3 - "${ART}/bench.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
done, failed = d.get("completed", 0), d.get("failed", 0)
print(f"[bench] completed={done} failed={failed} "
      f"out_tok/s={d.get('output_throughput', 0):.1f} "
      f"median_ttft_ms={d.get('median_ttft_ms', 0):.0f} "
      f"median_tpot_ms={d.get('median_tpot_ms', 0):.1f}")
# With warmups the means should have converged on the medians. A mean TTFT
# still an order of magnitude above the median means compilation leaked into
# the timed run, i.e. the warmups missed a shape the benchmark hits.
mt, md = d.get("mean_ttft_ms", 0), d.get("median_ttft_ms", 0)
print(f"[bench] mean_ttft_ms={mt:.0f} median_ttft_ms={md:.0f} "
      f"ratio={mt / md if md else float('nan'):.1f}x "
      f"mean_tpot_ms={d.get('mean_tpot_ms', 0):.1f}")
if done == 0:
    sys.exit("[bench] FAILED: no request completed")
PY

# Did the local-disk compilation cache actually get written? This is the head
# host's copy only; each of the other three hosts has its own under the same
# bind mount. A non-zero count here means the next run on this slice starts
# warm -- that is the whole point of the /tmp/jax_cache_tpu7x mount.
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
echo "--- jax compilation cache (head host) ---"
echo "dir=${JAX_CACHE_DIR} entries=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)"
du -sh "${JAX_CACHE_DIR}" 2>/dev/null || true
# Post-run disk state. With precompilation on, this is the number that says
# whether the 20GB cap and the 25GB floor are set anywhere near right.
echo "--- disk after run (head host) ---"
df -h "${JAX_CACHE_DIR}" / 2>/dev/null || true
