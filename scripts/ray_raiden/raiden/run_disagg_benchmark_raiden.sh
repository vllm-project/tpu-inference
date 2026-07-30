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

# Benchmark tpu-raiden Disaggregated Serving (30 prompts, 8k-in/1k-out @ 2 RPS)
# and report TTFT, TPOT, ITL, Token Throughput, and Decode KV Cache Hit Rate.
set -euo pipefail

PROXY_PORT="10000"
MODEL_NAME="${MODEL:-Qwen/Qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
RPS="${RPS:-2}"

echo "=== Step 1: Running vLLM Serving Benchmark inside Proxy Pod ==="
echo "Configuration: ${NUM_PROMPTS} prompts, ${INPUT_LEN} input tokens, ${OUTPUT_LEN} output tokens @ ${RPS} RPS..."
echo ""

kubectl exec deployment/vllm-disagg-proxy -- vllm bench serve \
  --backend vllm \
  --host localhost \
  --port "${PROXY_PORT}" \
  --model "${MODEL_NAME}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${RPS}" \
  --trust-remote-code \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "95,97,99"

echo ""
echo "=== Step 2: Fetching KV Cache Hit Rate & Transfer Metrics from Decode Engine ==="
kubectl logs vllm-decode-0 --tail=30 | grep -E "External prefix cache hit rate|GPU KV cache usage" | tail -n 5 || true
echo ""
echo "=== Benchmark Complete ==="
