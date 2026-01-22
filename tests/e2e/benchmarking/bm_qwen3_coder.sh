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
# limits as input. It then runs the benchmark with different input and output
# lengths and checks if the performance metrics are above the given limits.
#
# If any of the performance metrics are below the given limits, the script
# exits with a non-zero status code.
set -ex


# Usage:
# bash bm_qwen3_coder.sh --model BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic --tp 8 --req_tput_limit_1k_1k 1.05  --output_token_tput_limit_1k_1k 1926 --total_token_tput_limit_1k_1k 1948  --req_tput_limit_1k_8k 0.15  --output_token_tput_limit_1k_8k 1100  --total_token_tput_limit_1k_8k 1236  --req_tput_limit_8k_1k 0.418  --output_token_tput_limit_8k_1k 382  --total_token_tput_limit_8k_1k 3425


OPTIONS=""
LONGOPTS=model:,tp:,req_tput_limit_1k_1k:,output_token_tput_limit_1k_1k:,total_token_tput_limit_1k_1k:,req_tput_limit_1k_8k:,output_token_tput_limit_1k_8k:,total_token_tput_limit_1k_8k:,req_tput_limit_8k_1k:,output_token_tput_limit_8k_1k:,total_token_tput_limit_8k_1k:

# Parse arguments
if [[ ! PARSED = $(getopt --options="$OPTIONS" --longoptions=$LONGOPTS --name "$0" -- "$@") ]]; then
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
    --req_tput_limit_1k_1k)
      req_tput_limit_1k_1k=$2
      shift 2
      ;;
    --output_token_tput_limit_1k_1k)
      output_token_tput_limit_1k_1k=$2
      shift 2
      ;;
    --total_token_tput_limit_1k_1k)
      total_token_tput_limit_1k_1k=$2
      shift 2
      ;;
    --req_tput_limit_1k_8k)
      req_tput_limit_1k_8k=$2
      shift 2
      ;;
    --output_token_tput_limit_1k_8k)
      output_token_tput_limit_1k_8k=$2
      shift 2
      ;;
    --total_token_tput_limit_1k_8k)
      total_token_tput_limit_1k_8k=$2
      shift 2
      ;;
    --req_tput_limit_8k_1k)
      req_tput_limit_8k_1k=$2
      shift 2
      ;;
    --output_token_tput_limit_8k_1k)
      output_token_tput_limit_8k_1k=$2
      shift 2
      ;;
    --total_token_tput_limit_8k_1k)
      total_token_tput_limit_8k_1k=$2
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
if [ -z "$model" ] || [ -z "$tp" ] || [ -z "$req_tput_limit_1k_1k" ] || [ -z "$output_token_tput_limit_1k_1k" ] || [ -z "$total_token_tput_limit_1k_1k" ] || [ -z "$req_tput_limit_1k_8k" ] || [ -z "$output_token_tput_limit_1k_8k" ] || [ -z "$total_token_tput_limit_1k_8k" ] || [ -z "$req_tput_limit_8k_1k" ] || [ -z "$output_token_tput_limit_8k_1k" ] || [ -z "$total_token_tput_limit_8k_1k" ]; then
  echo "Error: All parameters are required."
  echo "model=$model"
  echo "tp=$tp"
  echo "req_tput_limit_1k_1k=$req_tput_limit_1k_1k"
  echo "output_token_tput_limit_1k_1k=$output_token_tput_limit_1k_1k"
  echo "total_token_tput_limit_1k_1k=$total_token_tput_limit_1k_1k"
  echo "req_tput_limit_1k_8k=$req_tput_limit_1k_8k"
  echo "output_token_tput_limit_1k_8k=$output_token_tput_limit_1k_8k"
  echo "total_token_tput_limit_1k_8k=$total_token_tput_limit_1k_8k"
  echo "req_tput_limit_8k_1k=$req_tput_limit_8k_1k"
  echo "output_token_tput_limit_8k_1k=$output_token_tput_limit_8k_1k"
  echo "total_token_tput_limit_8k_1k=$total_token_tput_limit_8k_1k"
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

export USE_MOE_EP_KERNEL=1
export MODEL_IMPL_TYPE=vllm
vllm serve --seed=42 --model="$model" --max-model-len=10240 --max-num-batched-tokens=8192 --max-num-seqs=512 --no-enable-prefix-caching --disable-log-requests --tensor-parallel-size="$tp" --kv-cache-dtype=fp8 --gpu-memory-utilization=0.95 --async-scheduling --enable-expert-parallel  2>&1 | tee vllm_server_out.txt &

# Need to put the nc command in a condition.
# If we assign it to a variable, the nc command is supposed to fail at first. But the "set -e" will cause the script to exit immediately.
while ! nc -zv $DEFAULT_HOST $DEFAULT_PORT; do
  echo "Waiting for the server to start..."
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

{
    for config in "1024 1024" "8192 1024" "1024 8192"; do
        # Use set -- to parse the config string into $1 (input) and $2 (output)
        read -r input_len output_len <<< "$config"

        echo "----------------------------------------------------------------"
        echo "Running benchmark with input_len=$input_len and output_len=$output_len"
        echo "----------------------------------------------------------------"

        benchmark_output=$(python3 bench_serving/benchmark_serving.py \
          --model="$model" \
          --backend=vllm \
          --host=127.0.0.1 \
          --port=8000 \
          --dataset-name=random \
          --random-input-len="$input_len" \
          --random-output-len="$output_len" \
          --random-range-ratio=0.8 \
          --num-prompts=320 \
          --max-concurrency=64 \
          --request-rate=inf \
          --ignore-eos)

        echo "benchmark_output: $benchmark_output"

        if [ "$input_len" == 1024 ] && [ "$output_len" == 1024 ]; then
            check_metrics "$benchmark_output" "$req_tput_limit_1k_1k" "$output_token_tput_limit_1k_1k" "$total_token_tput_limit_1k_1k" "1k_1k"
        elif [ "$input_len" == 1024 ] && [ "$output_len" == 8192 ]; then
            check_metrics "$benchmark_output" "$req_tput_limit_1k_8k" "$output_token_tput_limit_1k_8k" "$total_token_tput_limit_1k_8k" "1k_8k"
        elif [ "$input_len" == 8192 ] && [ "$output_len" == 1024 ]; then
            check_metrics "$benchmark_output" "$req_tput_limit_8k_1k" "$output_token_tput_limit_8k_1k" "$total_token_tput_limit_8k_1k" "8k_1k"
        fi
    done
}


end_time=$(date +%s)
echo "Elapsed time: $((end_time - start_time)) seconds"

echo "All done. Killing the server..."
kill %1

echo "perf_regressed: $perf_regressed"
if [ "$perf_regressed" -eq 1 ]; then
    exit 1
fi
