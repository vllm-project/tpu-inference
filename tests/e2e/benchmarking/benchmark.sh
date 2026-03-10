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


# Log the vLLM server output to a file
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
# The sentinel message that indicates the server is ready (in LOG_FILE)
export READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
export TIMEOUT_SECONDS=3600

test_model="meta-llama/Llama-3.1-8B-Instruct"
if [ "$USE_V6E8_QUEUE" == "True" ]; then
    test_model="meta-llama/Llama-3.1-70B-Instruct"
fi

root_dir=/workspace
dataset_name=sonnet
dataset_path="benchmarks/sonnet.txt"
vllm_download_dir=/tmp/hf_home
input_len=1024
output_len=1024
prefix_len=0
minimum_throughput_threshold=1500
tensor_parallel_size=1
exit_code=0

helpFunction()
{
    echo ""
    echo "Usage: $0 [-r root-dir-path] [-d dataset-name] [-p dataset-path] [-v vllm-download-dir] [-h]"
    echo -e "\t-r, --root-dir-path\tThe path to your root directory (default: /workspace/, which is used in the Dockerfile)"
    echo -e "\t-d, --dataset-name\tThe name of the dataset to use (default: sonnet)"
    echo -e "\t-p, --dataset-path\tThe path to the processed dataset. This is required when using a custom model (default: benchmarks/sonnet.txt)"
    echo -e "\t-v, --vllm-download-dir\tThe directory to download vLLM into (default: /tmp/hf_home)"
    echo -e "\t-h, --help\tShow this help message"
    echo ""
    echo "================================================================"
    echo "REQUIRED ENVIRONMENT VARIABLES:"
    echo "================================================================"
    echo "  MAX_MODEL_LEN           Model context length (prompt and output)"
    echo "  MAX_NUM_SEQS            Maximum number of sequences to be processed in a single iteration"
    echo "  MAX_NUM_BATCHED_TOKENS  Maximum number of tokens to be processed in a single iteration"
    echo ""
    echo "Example: export MAX_MODEL_LEN=...; export MAX_NUM_SEQS=...; export MAX_NUM_BATCHED_TOKENS=...; bash $0"
    exit 1
}

# Access shared benchmarking functionality
# shellcheck disable=SC1091
source "$(dirname "$0")/bench_utils.sh"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -r|--root-dir-path)
            root_dir="$2"
            shift
            shift
            ;;
        -d|--dataset-name)
            dataset_name="$2"
            shift
            shift
            ;;
        -p|--dataset-path)
            dataset_path="$2"
            shift
            shift
            ;;
        -v|--vllm-download-dir)
            vllm_download_dir="$2"
            shift
            shift
            ;;
        -h|--help)
            helpFunction
            ;;
        *)
            echo "Unknown option: $1"
            helpFunction
            ;;
    esac
done

if [ -n "${TEST_MODEL:-}" ]; then
    test_model="$TEST_MODEL"
fi

if [ -n "${INPUT_LEN:-}" ]; then
    input_len="$INPUT_LEN"
fi

if [ -n "${OUTPUT_LEN:-}" ]; then
    output_len="$OUTPUT_LEN"
fi

if [ -n "${PREFIX_LEN:-}" ]; then
    prefix_len="$PREFIX_LEN"
fi

if [ -n "${MINIMUM_THROUGHPUT_THRESHOLD:-}" ]; then
    minimum_throughput_threshold="$MINIMUM_THROUGHPUT_THRESHOLD"
fi

if [ -n "${TENSOR_PARALLEL_SIZE:-}" ]; then
    tensor_parallel_size="$TENSOR_PARALLEL_SIZE"
fi

echo "Using the root directory at $root_dir"
echo "Using $input_len input len"
echo "Using $output_len output len"
echo "Using $prefix_len prefix len"
echo "Using minimum throughput threshold $minimum_throughput_threshold"
echo "Using tensor parallel size $tensor_parallel_size"
echo "Using the dataset name $dataset_name"
echo "Using the dataset at $dataset_path"

cd "$root_dir"/vllm || exit
echo "Current working directory: $(pwd)"
echo "Using vLLM hash: $(git rev-parse HEAD)"

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp -r "$root_dir"/tpu_inference/scripts/vllm/benchmarking/*.py "$root_dir"/vllm/benchmarks/
echo "Using TPU Inference hash: $(git -C "$root_dir"/tpu_inference rev-parse HEAD)"

checkThroughput() {
    # This function checks whether the Request throughput from a benchmark
    # log file meet specified target value. It validates the presence and
    # accessibility of the log file, extracts Request throughput, and
    # compares them against predefined targets.
    # The function outputs the results of these comparisons and exits with a
    # status code indicating overall success or failure.

    # Check if the inputs are valid
    if [ -z "$BENCHMARK_LOG_FILE" ]; then
        echo "Error: BENCHMARK_LOG_FILE environment variable is not set." >&2
        exit_code=2
        return
    fi
    if [ ! -f "$BENCHMARK_LOG_FILE" ]; then
        echo "Error: Benchmark log file '$BENCHMARK_LOG_FILE' not found." >&2
        exit_code=2
        return
    fi

    # Extract Request throughput
    actual_throughput=$(awk '/Request throughput \(req\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$actual_throughput" ]; then
        echo "Request throughput: NOT FOUND"
        throughput_pass=0
    else
        echo "Request throughput: $actual_throughput"
        if awk -v actual="$actual_throughput" -v target="$minimum_throughput_threshold" 'BEGIN { exit !(actual >= target) }'; then
            echo "Request throughput comparison (>= $minimum_throughput_threshold): PASSED"
            throughput_pass=1
        else
            echo "Request throughput comparison (>= $minimum_throughput_threshold): FAILED"
            throughput_pass=0
        fi
    fi
    echo

    echo "--- Summary ---"
    # Ensure pass flag is initialized if extraction fails
    : "${throughput_pass:=0}"

    if [ "$throughput_pass" -eq 1 ]; then
        echo "Overall: PASSED"
    else
        echo "Overall: FAILED"
        [ "$throughput_pass" -eq 0 ] && echo "Reason: Throughput check failed or value not found."
        exit_code=1
    fi
}


echo "--------------------------------------------------"
echo "Running benchmark for model: $test_model"
echo "--------------------------------------------------"

# If we have multiple args, we can add them in extra_serve_args.
extra_serve_args=()
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    echo "Using customized max-model-len: $MAX_MODEL_LEN"
    extra_serve_args+=(--max-model-len "$MAX_MODEL_LEN")
else
    echo "Error: The environment variable MAX_MODEL_LEN must be specified."
    exit 1
fi

if [ -n "${MAX_NUM_SEQS:-}" ]; then
    echo "Using customized max-num-seqs: $MAX_NUM_SEQS"
    extra_serve_args+=(--max-num-seqs "$MAX_NUM_SEQS")
else
    echo "Error: The environment variable MAX_NUM_SEQS must be specified."
    exit 1
fi

if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    echo "Using customized max-num-batched-tokens: $MAX_NUM_BATCHED_TOKENS"
    extra_serve_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
else
    echo "Error: The environment variable MAX_NUM_BATCHED_TOKENS must be specified."
    exit 1
fi


# Spin up the vLLM server
echo "Spinning up the vLLM server..."
if [[ "$test_model" == "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ]]; then
    (vllm serve "$test_model" --disable-log-requests --download_dir "$vllm_download_dir" --tensor-parallel-size "$tensor_parallel_size" "${extra_serve_args[@]}" 2>&1 | tee -a "$LOG_FILE") &
else
    (vllm serve "$test_model" --disable-log-requests --download_dir "$vllm_download_dir" --tensor-parallel-size "$tensor_parallel_size" "${extra_serve_args[@]}" 2>&1 | tee -a "$LOG_FILE") &
fi

# Set trap to ensure cleanup happens even on immediate or normal exit
trap 'cleanUp "$test_model"' EXIT

waitForServerReady

# Implement other dataset's args here
dataset_args=()
if [ "$dataset_name" = "sonnet" ]; then
    dataset_args+=(
        "--sonnet-input-len" "$input_len"
        "--sonnet-output-len" "$output_len"
        "--sonnet-prefix-len" "$prefix_len"
    )
elif [ "$dataset_name" = "random" ]; then
    dataset_args+=(
        "--random-input-len" "$input_len"
        "--random-output-len" "$output_len"
        "--random-prefix-len" "$prefix_len"
    )
fi

echo "Starting the benchmark for $test_model..."
echo "Current working directory: $(pwd)"
python benchmarks/benchmark_serving.py \
--backend vllm \
--model "$test_model" \
--dataset-name "$dataset_name" \
--dataset-path "$dataset_path" \
"${dataset_args[@]}" 2>&1 | tee -a "$BENCHMARK_LOG_FILE"

checkThroughput
if [ "$exit_code" -ne 0 ]; then
    exit_code=1
fi

exit $exit_code
