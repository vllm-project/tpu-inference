#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Benchmark tpu-raiden Multihost (no disagg) KV Cache Offloading (shared prefix benchmark)
# and report TTFT, TPOT, ITL, Token Throughput, and Host KV Cache Hit Rate.
set -euo pipefail

PORT="8000"
MODEL_NAME="${MODEL:-Qwen/Qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
PREFIX_LEN="${PREFIX_LEN:-8192}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RPS="${RPS:-2}"

echo "=== Step 1: Polling vLLM Multihost Engine Readiness ==="
START_TIME=$(date +%s)
while true; do
  ENGINE_READY=$(kubectl logs vllm-serving-0 -c vllm-leader 2>/dev/null | grep -c -E "Application startup complete\." || true)

  if [ "$ENGINE_READY" -gt 0 ]; then
    echo "✅ vLLM engine is ready!"
    break
  fi

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo -ne "Waiting for engine... (${ELAPSED}s elapsed)\r"
  sleep 5
done
echo ""

echo "=== Step 2: Running vLLM Serving Benchmark inside Leader Pod ==="
echo "Configuration: ${NUM_PROMPTS} prompts, ${PREFIX_LEN} shared prefix + ${INPUT_LEN} input tokens, ${OUTPUT_LEN} output tokens @ ${RPS} RPS..."
echo ""

kubectl exec vllm-serving-0 -c vllm-leader -- vllm bench serve \
  --backend vllm \
  --host localhost \
  --port "${PORT}" \
  --model "${MODEL_NAME}" \
  --dataset-name random \
  --random-prefix-len "${PREFIX_LEN}" \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${RPS}" \
  --seed 42 \
  --trust-remote-code \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "95,97,99"

echo ""
echo "=== Step 3: Fetching KV Cache Hit Rate & Transfer Metrics from Multihost Engine ==="
kubectl logs vllm-serving-0 -c vllm-leader --tail=100 | grep -E "Offload Metrics Snapshot|Prefix cache hit rate|GPU KV cache usage" | tail -n 10 || true
echo ""
echo "=== Benchmark Complete ==="
