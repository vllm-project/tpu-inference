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

# This script benchmarks the performance of a vLLM model and checks if the
# performance metrics are above the given limits.
#
# The script takes the model name, tensor parallel size, and the performance
# limits as input. It then runs the benchmark with input and output
# lengths and checks if the performance metrics are above the given limits.
#
# If any of the performance metrics are below the given limits, the script
# exits with a non-zero status code.
set -ex


# Usage:
# bash tests/e2e/benchmarking/bm_qwen3_coder.sh --model BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic --tp 8 --req_tput_limit 1.05  --output_token_tput_limit 1926 --total_token_tput_limit 1948 --input_len 1024 --output_len 1024 --use_moe_ep_kernel 1
# bash tests/e2e/benchmarking/bm_qwen3_coder.sh --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --tp 8 --req_tput_limit 1.05  --output_token_tput_limit 1926 --total_token_tput_limit 1948 --input_len 1024 --output_len 1024 --use_moe_ep_kernel 1


OPTIONS=""
LONGOPTS=model:,tp:,req_tput_limit:,output_token_tput_limit:,total_token_tput_limit:,input_len:,output_len:,use_moe_ep_kernel:

# Parse arguments
if ! PARSED=$(getopt --options="$OPTIONS" --longoptions=$LONGOPTS --name "$0" -- "$@"); then
  exit 2
fi
eval set -- "$PARSED"
# Option parsing
while true; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --tp)
      tp=$2
      shift 2
      ;;
    --req_tput_limit)
      req_tput_limit=$2
      shift 2
      ;;
    --output_token_tput_limit)
      output_token_tput_limit=$2
      shift 2
      ;;
    --total_token_tput_limit)
      total_token_tput_limit=$2
      shift 2
      ;;
    --input_len)
      input_len=$2
      shift 2
      ;;
    --output_len)
      output_len=$2
      shift 2
      ;;
    --use_moe_ep_kernel)
      use_moe_ep_kernel=$2
      shift 2
      ;;

    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      exit 3
      ;;
  esac
done
if [ -z "$model" ] || [ -z "$tp" ] || [ -z "$req_tput_limit" ] || [ -z "$output_token_tput_limit" ] || [ -z "$total_token_tput_limit" ] || [ -z "$input_len" ] || [ -z "$output_len" ] || [ -z "$use_moe_ep_kernel" ]; then
  echo "Error: All parameters are required."
  echo "model=$model"
  echo "tp=$tp"
  echo "req_tput_limit=$req_tput_limit"
  echo "output_token_tput_limit=$output_token_tput_limit"
  echo "total_token_tput_limit=$total_token_tput_limit"
  echo "input_len=$input_len"
  echo "output_len=$output_len"
  echo "use_moe_ep_kernel=$use_moe_ep_kernel"

  exit 1
fi

if [ ! -e ./bench_serving ]; then
  echo "git clone bench_serving"
  git clone https://github.com/kimbochen/bench_serving.git
else
  echo "bench_serving exists. Skip git cloning the repo."
fi

DEFAULT_HOST=127.0.0.1
DEFAULT_PORT=8000

start_time=$(date +%s)

# If needed, replace "--async-scheduling" with "--no-async-scheduling"
export USE_MOE_EP_KERNEL=${use_moe_ep_kernel}
export MODEL_IMPL_TYPE=vllm

echo "bench_serving commit: $(git -C bench_serving rev-parse HEAD)"

vllm serve --seed=42 --model="$model" --max-model-len=10240 --max-num-batched-tokens=8192 --max-num-seqs=512 --no-enable-prefix-caching --disable-log-requests --tensor-parallel-size="$tp" --kv-cache-dtype=fp8 --gpu-memory-utilization=0.95 --async-scheduling --enable-expert-parallel   2>&1 | tee vllm_server_out.txt &

# Trap registers the cleanup as a handler for the EXIT. Whenever the shell exits, it runs `cleanup` before terminating.
# The trap does not affect the exit status. When an EXIT trap fires, the script's exit status is whatever it was before the trap was triggered.
cleanup() {
  echo "Cleaning up: killing the server..."
  kill %1 2>/dev/null || true
}
trap cleanup EXIT

# Need to put the nc command in a condition.
# If we assign it to a variable, the nc command is supposed to fail at first because it takes some time for the server to be ready. But the "set -e" will cause the script to exit immediately so the while loop will not run.
TIMEOUT_SECONDS=$((40 * 60))  # 40 minutes
wait_start_time=$(date +%s)
while ! nc -zv $DEFAULT_HOST $DEFAULT_PORT; do
  current_time=$(date +%s)
  elapsed=$((current_time - wait_start_time))
  if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
    echo "Timeout: Server did not start within 40 minutes."
    exit 1
  fi
  echo "Waiting for the server to start... (${elapsed}s elapsed)"
  sleep 15
done

echo "Server is up and running"

perf_regressed=0

# This function checks if the performance metrics are above the given limits.
check_metrics() {
    local benchmark_output="$1"
    local req_tput_limit="$2"
    local output_token_tput_limit="$3"
    local total_token_tput_limit="$4"
    local config_name="$5"

    request_throughput=$(echo "$benchmark_output" | grep "Request throughput (req/s):" | awk '{print $4}')
    output_token_throughput=$(echo "$benchmark_output" | grep "Output token throughput (tok/s):" | awk '{print $5}')
    total_token_throughput=$(echo "$benchmark_output" | grep "Total Token throughput (tok/s):" | awk '{print $5}')

    # 'bc -l' evaluates the floating point comparison before the pipe.
    if (( $(echo "$request_throughput < $req_tput_limit" | bc -l) )); then
        echo "Request throughput ($request_throughput) is below the limit ($req_tput_limit) for $config_name"
        perf_regressed=1
    fi

    if (( $(echo "$output_token_throughput < $output_token_tput_limit" | bc -l) )); then
        echo "Output token throughput ($output_token_throughput) is below the limit ($output_token_tput_limit) for $config_name"
        perf_regressed=1
    fi

    if (( $(echo "$total_token_throughput < $total_token_tput_limit" | bc -l) )); then
        echo "Total token throughput ($total_token_throughput) is below the limit ($total_token_tput_limit) for $config_name"
        perf_regressed=1
    fi
}


echo "----------------------------------------------------------------"
echo "Running benchmark with input_len=$input_len and output_len=$output_len"
echo "----------------------------------------------------------------"
benchmark_output=$(python3 bench_serving/benchmark_serving.py \
  --model="$model" \
  --backend=vllm \
  --host="$DEFAULT_HOST" \
  --port="$DEFAULT_PORT" \
  --dataset-name=random \
  --random-input-len="$input_len" \
  --random-output-len="$output_len" \
  --random-range-ratio=0.8 \
  --num-prompts=320 \
  --max-concurrency=64 \
  --request-rate=inf \
  --ignore-eos 2>&1 | tee vllm_benchmark_out.txt)
echo "benchmark_output: $benchmark_output"
check_metrics "$benchmark_output" "$req_tput_limit" "$output_token_tput_limit" "$total_token_tput_limit" "1k_1k"


end_time=$(date +%s)
echo "Elapsed time: $((end_time - start_time)) seconds"

echo "All done."

echo "perf_regressed: $perf_regressed"
if [ "$perf_regressed" -eq 1 ]; then
    exit 1
fi
