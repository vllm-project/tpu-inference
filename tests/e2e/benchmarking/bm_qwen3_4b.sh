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

set -ex

# Default values
MODEL="Qwen/Qwen3-4B"
DATASET_PATH="/mnt/disks/persist/dataset/vllm-cb-storage2/bench-dataset-copy/Qwen3-32B/inlen1024_outlen4096_prefixlen0.jsonl"
HOST=127.0.0.1
PORT=8000

# Download dataset if missing
if [ ! -f "$DATASET_PATH" ]; then
  echo "Dataset not found at $DATASET_PATH. Attempting to download from GCS..."
  # Ensure the directory exists. We might need sudo if /mnt/disks/persist is restricted.
  mkdir -p "$(dirname "$DATASET_PATH")"
  gcloud storage cp gs://vllm-cb-storage2/bench-dataset-copy/Qwen3-32B/inlen1024_outlen4096_prefixlen0.jsonl "$DATASET_PATH" || \
  gsutil cp gs://vllm-cb-storage2/bench-dataset-copy/Qwen3-32B/inlen1024_outlen4096_prefixlen0.jsonl "$DATASET_PATH"
fi

# Logs
LOG_FILE="vllm_server_out.txt"
BENCHMARK_LOG_FILE="vllm_benchmark_out.txt"

# Cleanup function to ensure server is killed on exit
cleanup() {
  echo "Cleaning up: killing the server..."
  pkill -f "vllm serve $MODEL" || true
  sleep 10
}
trap cleanup EXIT

echo "Starting vLLM server for $MODEL..."

# Start the server in the background
vllm serve "$MODEL" \
  --max-model-len=5120 \
  --max-num-seqs=256 \
  --disable-log-requests \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --additional_config='{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.99 \
  --async-scheduling 2>&1 | tee "$LOG_FILE" &

# Wait for the server to be ready
TIMEOUT_SECONDS=$((30 * 60)) # 30 minutes
wait_start_time=$(date +%s)
echo "Waiting for the server to start on $HOST:$PORT..."
while ! nc -zv $HOST $PORT; do
  current_time=$(date +%s)
  elapsed=$((current_time - wait_start_time))
  if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
    echo "Timeout: Server did not start within 30 minutes."
    exit 1
  fi
  sleep 10
done

echo "Server is up and running. Starting benchmark..."

# Run the benchmark
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --dataset-name custom \
  --dataset-path "$DATASET_PATH" \
  --num-prompts 1000 \
  --custom-output-len 4096 \
  --ignore-eos \
  --skip-chat-template \
  --temperature=0 2>&1 | tee "$BENCHMARK_LOG_FILE"

echo "Benchmark complete. Results saved to $BENCHMARK_LOG_FILE"

# Copy logs to mounted volume for Buildkite artifact upload
if [ -d "/tmp/hf_home" ]; then
  cp "$LOG_FILE" "/tmp/hf_home/$LOG_FILE"
  cp "$BENCHMARK_LOG_FILE" "/tmp/hf_home/$BENCHMARK_LOG_FILE"
  echo "Logs copied to /tmp/hf_home."
fi
