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

# Benchmark tpu-raiden Disaggregated Serving MMLU Correctness Test for Qwen3-8B
# and report MMLU evaluation accuracy and Decode KV Cache Hit Rate.
set -euo pipefail

PROXY_PORT="10000"
MODEL_NAME="${MODEL:-Qwen/Qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RPS="${RPS:-2}"


echo "=== Step 1: Checking and Downloading MMLU Dataset inside Proxy Pod ==="
kubectl exec deployment/vllm-disagg-proxy -- bash -c '
if [ ! -d "mmlu/data/test" ]; then
    echo "Downloading MMLU dataset..."
    mkdir -p mmlu
    cd mmlu
    wget -q https://people.eecs.berkeley.edu/~hendrycks/data.tar -P .
    tar -xf data.tar
    cd ..
    echo "MMLU dataset downloaded successfully."
else
    echo "MMLU dataset already exists."
fi
'
echo "=== Step 2: Syncing updated benchmark_utils.py to Proxy Pod ==="
PROXY_POD=$(kubectl get pod -l app=vllm-proxy -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$PROXY_POD" ]; then
    kubectl cp "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../vllm/benchmarking/benchmark_utils.py" "${PROXY_POD}:/workspace/tpu_inference/scripts/vllm/benchmarking/benchmark_utils.py" || true
fi
echo ""

echo "=== Step 3: Running MMLU Correctness Benchmark inside Proxy Pod ==="
echo "Configuration: ${NUM_PROMPTS} prompts for ${MODEL_NAME} @ ${RPS} RPS..."
echo ""

kubectl exec deployment/vllm-disagg-proxy -- python3 /workspace/tpu_inference/scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --host localhost \
  --port "${PROXY_PORT}" \
  --model "${MODEL_NAME}" \
  --dataset-name mmlu \
  --dataset-path mmlu/data/test \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${RPS}" \
  --run-eval \
  --warmup-mode none \
  --trust-remote-code \
  --mmlu-num-shots 5 \
  --mmlu-output-len 8

echo ""
echo "=== Step 4: Fetching KV Cache Hit Rate & Transfer Metrics from Decode Engine ==="
kubectl logs vllm-decode-0 --tail=500 | grep -E "External prefix cache hit rate|GPU KV cache usage" | tail -n 5 || true
echo ""
echo "=== Benchmark Complete ==="
