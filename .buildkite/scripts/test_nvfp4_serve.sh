#!/bin/bash
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

# Test script for NVFP4 model serving on TPU.
# Starts vllm serve with an NVFP4 model, waits for it to be ready,
# and runs a smoke test.

set -euo pipefail

MODEL="${TEST_MODEL:-nvidia/Qwen3-30B-A3B-NVFP4}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PORT=8000
SERVER_LOG=/tmp/vllm_server.log

echo "--- Starting vllm serve with NVFP4 model: ${MODEL}"
MODEL_IMPL_TYPE=vllm vllm serve "${MODEL}" \
  --port "${PORT}" \
  --seed 42 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --tensor-parallel-size "${TP_SIZE}" \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Server PID: ${SERVER_PID}"

echo "--- Waiting for server to start (up to 30 min)"
for i in $(seq 1 180); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Server is up after $((i * 10)) seconds"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: Server process died. Server log:"
    cat "${SERVER_LOG}"
    exit 1
  fi
  sleep 10
done

if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: Server did not start within 30 minutes. Server log (last 500 lines):"
  tail -500 "${SERVER_LOG}"
  kill "${SERVER_PID}" 2>/dev/null || true
  exit 1
fi

echo "--- Running smoke test"
RESPONSE=$(curl -s "http://localhost:${PORT}/v1/completions" \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}")
echo "Response: ${RESPONSE}"

echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert 'choices' in data, f'No choices in response: {data}'
text = data['choices'][0]['text']
assert len(text) > 0, 'Empty response text'
print(f'Generated text: {text}')
print('NVFP4 smoke test PASSED')
"

echo "--- Cleaning up"
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
