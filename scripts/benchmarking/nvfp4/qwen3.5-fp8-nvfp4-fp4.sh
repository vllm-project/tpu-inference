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


# This script benchmarks Qwen3.5-397B for FP8, NVFP4, and FP4 (99.9% clip) on the
# InferenceX setup for 1k input / 8k output and 8k input / 1k output
# setups for the latest setup, as of June 16, 2026.

# Usage: bash qwen3.5-bm-nvfp4-fp8.sh

DEFAULT_HOST=0.0.0.0
DEFAULT_PORT=8000
BASE_RESULTS_DIR="./benchmark_results-nvfp4-fp8-fp4_clip"

# ----------------------------------------------------------------------
# 0. Initial Setup
# ----------------------------------------------------------------------

if [ ! -d "bench_serving" ]; then
  echo "bench_serving directory not found. Cloning repository..."
  git clone https://github.com/kimbochen/bench_serving.git
fi

# ----------------------------------------------------------------------
# 1. Server Launch Function
# ----------------------------------------------------------------------

run_server() {
  local target_model=$1

  # Environment variables exported in the main loop (like FP4 quantization)
  # will be automatically inherited here.
  MODEL_IMPL_TYPE=vllm \
  USE_MOE_EP_KERNEL=0 \
  ATTN_BUCKETIZED_NUM_REQS=true \
  ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 \
  ONEHOT_MOE_PERMUTE_THRESHOLD=32768 \
  RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d \
  NEW_MODEL_DESIGN=1 \
  exec vllm serve "$target_model" \
    --max-model-len=9216 \
    --max-num-batched-tokens=1024 \
    --max-num-seqs=64 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization=0.9 \
    --tensor-parallel-size=8 \
    --async-scheduling \
    --port=$DEFAULT_PORT \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder \
    --reasoning-parser=qwen3 \
    --limit-mm-per-prompt='{"image": 0, "video": 0}' \
    --kv-cache-dtype=fp8 \
    --enable-expert-parallel \
    --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
    --mamba-ssm-cache-dtype=bfloat16 \
    --block-size=256
}

# ----------------------------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------------------------

wait_for_port() {
  echo "Checking if port $DEFAULT_PORT is open..."
  while ! nc -zv $DEFAULT_HOST $DEFAULT_PORT >/dev/null 2>&1; do
    echo "Waiting for the server to start (checking again in 15 seconds)..."
    sleep 15
  done
  echo "Server is up! Waiting 10 additional seconds for warm-up..."
  sleep 10
}

cleanup() {
  echo "Cleaning up any remaining server processes..."
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "vllm" 2>/dev/null
  sleep 5 # Allow ports to clear up
}

# Trap exit signals to run cleanup on script termination
trap cleanup EXIT INT TERM

# ----------------------------------------------------------------------
# 3. Main Execution Loop
# ----------------------------------------------------------------------

CONFIGS=(
  "nvfp4"
  "fp8"
  "fp4_clip"
)

for CONFIG in "${CONFIGS[@]}"; do

  # Reset potentially lingering quantization environment variables
  unset DISABLE_WEIGHT_REQUANTIZATION
  unset MOE_REQUANTIZE_WEIGHT_DTYPE
  unset MOE_REQUANTIZE_BLOCK_SIZE
  unset MOE_REQUANTIZE_CLIP_PERCENTILE

  # Configure Model Path, Env Vars, and Result Dir based on CONFIG
  if [[ "$CONFIG" == "nvfp4" ]]; then
    MODEL="nvidia/Qwen3.5-397B-A17B-NVFP4"
    DIR_NAME="nvfp4"
    export DISABLE_WEIGHT_REQUANTIZATION=1

  elif [[ "$CONFIG" == "fp8" ]]; then
    MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
    DIR_NAME="fp8"

  elif [[ "$CONFIG" == "fp4_clip" ]]; then
    MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
    DIR_NAME="fp4_clip"
    export MOE_REQUANTIZE_WEIGHT_DTYPE="float4_e2m1fn"
    export MOE_REQUANTIZE_BLOCK_SIZE="512"
    export MOE_REQUANTIZE_CLIP_PERCENTILE="99.9"
  fi

  RESULTS_DIR="${BASE_RESULTS_DIR}/${DIR_NAME}"
  mkdir -p "$RESULTS_DIR"

  echo "========================================="
  echo "Starting vLLM Server: $MODEL (Config: $CONFIG)"
  echo "Logging to: $RESULTS_DIR"
  echo "========================================="

  server_log="$RESULTS_DIR/qwen_vllm_server.log"
  run_server "$MODEL" > "$server_log" 2>&1 &
  SERVER_PID=$!

  # Wait for port to become active
  wait_for_port

  echo "========================================="
  echo "Running Benchmark 1: Input 8192 / Output 1024"
  echo "========================================="

  bench1_log="$RESULTS_DIR/bench_8192_1024.log"
  python3 bench_serving/benchmark_serving.py \
    --model "$MODEL" \
    --dataset-name random \
    --backend vllm \
    --random-input-len=8192 \
    --random-output-len=1024 \
    --num-prompts=640 \
    --random-range-ratio=0.8 \
    --ignore-eos \
    --save-result \
    --max-concurrency=512 > "$bench1_log" 2>&1

  echo "Benchmark 1 complete. Log saved to: $bench1_log"
  echo "Sleeping for 5 seconds to let server settle..."
  sleep 5

  echo "========================================="
  echo "Running Benchmark 2: Input 1024 / Output 8192"
  echo "========================================="

  bench2_log="$RESULTS_DIR/bench_1024_8192.log"
  python3 bench_serving/benchmark_serving.py \
    --model "$MODEL" \
    --dataset-name random \
    --backend vllm \
    --random-input-len=1024 \
    --random-output-len=8192 \
    --num-prompts=640 \
    --random-range-ratio=0.8 \
    --ignore-eos \
    --save-result \
    --max-concurrency=512 > "$bench2_log" 2>&1

  echo "Benchmark 2 complete. Log saved to: $bench2_log"


  # ----------------------------------------------------------------------
  # 4. Reporting the Results
  # ----------------------------------------------------------------------
  echo ""
  echo "========================================================================"
  echo "                 BENCHMARK REPORT: $DIR_NAME                            "
  echo "========================================================================"

  for log_file in "$bench1_log" "$bench2_log"; do
    echo ">>> Results for: $(basename "$log_file") <<<"
    if [ -f "$log_file" ]; then
      # Print key metrics output by the benchmark script
      grep -E "(Successful requests|Benchmark duration|throughput|TTFT|TPOT|ITL|E2EL)" "$log_file" || \
        echo "  [No benchmark summary found in log. Please check: $log_file]"
    else
      echo "  [Error: Log file not found!]"
    fi
    echo "------------------------------------------------------------------------"
  done

  # Shut down the current server to free up VRAM and Port before the next model runs
  echo "Cleaning up server for config [$CONFIG] to prepare for next run..."
  cleanup
  SERVER_PID="" # Clear PID for the next loop iteration

done

echo "All configurations successfully benchmarked. Script complete."