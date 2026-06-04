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

set -ex

# Initialize variables
MODEL_NAME=""
THRESHOLD=""
TP_SIZE=""
LIMIT=100

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --tp_size) TP_SIZE="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required parameters are provided
if [ -z "$MODEL_NAME" ] || [ -z "$THRESHOLD" ] || [ -z "$TP_SIZE" ]; then
    echo "Usage: $0 --model <name> --threshold <float> --tp_size <int> [--limit <int>]"
    exit 1
fi

# 1. Start Server in the background
# We use OOM-safe parameters verified on TPU VM
# We set max_pixels to 401408 (approx 632x632) to balance readability and memory
# We set max_num_batched_tokens and max_model_len to 4096 to avoid VMEM OOMs on TPU
vllm serve "$MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --mm-processor-kwargs '{"max_pixels": 401408}' \
  --trust-remote-code \
  --disable-chunked-mm-input \
  --port 8000 &
SERVER_PID=$!

# Ensure server and temp files are killed/removed on exit
BENCHMARK_OUTPUT=$(mktemp)
trap 'kill $SERVER_PID || true; rm -f $BENCHMARK_OUTPUT' EXIT

# 2. Wait for health check to pass (up to 15 minutes for large models)
echo "Waiting for vLLM server to start..."
set +x # Turn off echoing for the loop
for _ in $(seq 1 90); do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "Server is healthy!"
    HEALTHY=1
    break
  fi
  echo -n "."
  sleep 10
done
echo ""
set -x # Turn echoing back on


if [ -z "$HEALTHY" ]; then
  echo "Error: Server failed to start within timeout."
  exit 1
fi

# 3. Run the uLLM-native benchmark script
# This script handles dataset loading and accuracy evaluation internally
# We use a system prompt that aligns with the internal dataset's parsing logic
# We set output length to 32 to allow for the 'Final Answer: (X)' format
# We use --mmmu-pro-prompt-footer and --mmmu-pro-strip-reasoning to optimize Qwen accuracy
python3 scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm-chat \
  --model "$MODEL_NAME" \
  --dataset-name mmmu_pro \
  --num-prompts "$LIMIT" \
  --mmmu-pro-output-len 32 \
  --mmmu-pro-prompt-footer "" \
  --mmmu-pro-strip-reasoning \
  --run-eval \
  --chat-template-system-prompt "IMPORTANT: Directly output the final answer in the format: 'Final Answer: (X)' where X is the correct letter choice. Do not include any reasoning or explanation. Your response must start with 'Final Answer:'." \
  --endpoint /v1/chat/completions \
  --port 8000 > "$BENCHMARK_OUTPUT" 2>&1

cat "$BENCHMARK_OUTPUT"

# 4. Parse the accuracy score from the JSON-like output
# Pattern matches: {'accuracy': 0.4, ...}
score=$(grep -oP "'accuracy': \K[0-9.]+" "$BENCHMARK_OUTPUT" | head -n 1)

if [ -z "$score" ]; then
  # Fallback for double quotes
  score=$(grep -oP '"accuracy": \K[0-9.]+' "$BENCHMARK_OUTPUT" | head -n 1)
fi

echo "Extracted accuracy score: $score"

if [ -z "$score" ]; then
  echo "Error: Could not extract accuracy score from output."
  exit 1
fi

# 5. Check against threshold
is_ok=$(awk -v val="$score" -v threshold="$THRESHOLD" 'BEGIN {print (val >= threshold)}')

if [ "$is_ok" -eq 1 ]; then
  echo "Accuracy check passed! ($score >= $THRESHOLD)"
  exit 0
else
  echo "Accuracy check failed! Score $score is below threshold $THRESHOLD"
  exit 1
fi
