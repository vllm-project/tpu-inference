#!/bin/bash
# Copyright 2025 Google LLC
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

# Demo script: start vllm serve, wait for health, run vllm bench serve.
# Uses --dataset-name random so no dataset file is needed.
# Customize MODEL, PORT, tensor-parallel-size, and bench params as needed.

set -e

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-256}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-20}
NUM_WARMUPS=${NUM_WARMUPS:-2}

echo "--- Starting vllm serve in background ---"
vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  > /tmp/vllm_serve.log 2>&1 &
SERVER_PID=$!

echo "--- Waiting for server to be healthy ---"
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Server is healthy after ${i}0s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process died. Logs:"
    cat /tmp/vllm_serve.log
    exit 1
  fi
  sleep 10
done

echo "--- Running vllm bench serve ---"
vllm bench serve \
  --model "${MODEL}" \
  --host localhost \
  --port "${PORT}" \
  --dataset-name random \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --num-warmups "${NUM_WARMUPS}" \
  --ignore-eos

echo "--- Stopping server ---"
kill "${SERVER_PID}" || true
