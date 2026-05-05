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


# This script, by default, will test running one of the listed models on 1000 MMLU prompts and gage whether accuracy & perf is reasonable.
# Specifically, it will do the following:
# 1. Download the MMLU dataset
# 2. Spin up a vLLM server
# 2. Once the server is ready, run 1000 prompts of the MMLU benchmark
# 3. Parse out the accuracy score and throughput and exit cleanly if and only if they are both reasonably high

# Example default usage: bash tests/e2e/benchmarking/mmlu.sh -r /local/root_dir
# Example local docker + JAX TPU usage: BUILDKITE_COMMIT=0f199f1 .buildkite/scripts/run_in_docker.sh bash /workspace/tpu_inference/tests/e2e/benchmarking/mmlu.sh

# Logs the vLLM server output to a file
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
# The sentinel message that indicates the server is ready (in LOG_FILE)
export READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
export TIMEOUT_SECONDS=1800

# The minimum accuracy and throughput scores we expect
# TODO (jacobplatin): these are very low, so we'll want to boost them eventually
# We need to support different targets for different models.
TARGET_ACCURACY="0.6"
TARGET_THROUGHPUT="1500"

model_list="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct meta-llama/Llama-3.1-8B-Instruct"
if [ "$USE_V6E8_QUEUE" == "True" ]; then
    model_list="meta-llama/Llama-4-Scout-17B-16E-Instruct"
fi

extra_serve_args=()
if [ "$QUANTIZATION" = "True" ]; then
    echo "QUANTIZATION is True. Running with quantization."
    extra_serve_args+=("--additional_config")
    extra_serve_args+=('{"quantization": "int8_default.yaml"}')
else
    echo "QUANTIZATION is False. Running without quantization."
fi

root_dir=/workspace
dataset_name=mmlu
dataset_path=""
num_prompts=1000
mmlu_output_len=1
exit_code=0

helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_inference' (default: /workspace/, which is used in the Dockerfile)"
   echo -e "\t-d The dataset name (default: mmlu, which will download the dataset)"
   echo -e "\t-p The path to the processed MMLU dataset (default: None, which will download the dataset)"
   echo -e "\t-m A space-separated list of HuggingFace model ids to use (default: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct, meta-llama/Llama-3.1-8B-Instruct and meta-llama/Llama-4-Scout-17B-16E-Instruct)"
   echo -e "\t-n Number of prompts to use for the benchmark (default: 10)"
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
        -m|--model)
            model_list="$2"
            shift
            shift
            ;;
        -n|--num-prompts)
            num_prompts="$2"
            shift
            shift
            ;;
        -l|--mmlu-output-len)
            mmlu_output_len="$2"
            shift
            shift
            ;;
        -h|--help)
            helpFunction
            ;;
        *) # unknown option
            echo "Unknown option: $1"
            helpFunction
            ;;
    esac
done

echo "Using the root directory at $root_dir"
echo "Using $num_prompts prompts"


cd "$root_dir" || exit
# If the user doesn't specify a dataset_path then download the MMLU data (if it doesn't already exist)
if [ -z "$dataset_path" ]; then
    if [[ "$dataset_name" != "mmlu" ]]; then
        echo "Dataset name must be mmlu if dataset_path is not specified."
        exit 1
    fi
    # Only download the dataset if it doesn't already exist
    dataset_path="$root_dir"/mmlu/
    mkdir -p "$dataset_path"
    cd "$dataset_path" || exit
    if [ ! -d "$dataset_path/data/test" ]; then
        wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P .
        tar -xvf data.tar
    fi
    dataset_path=$dataset_path/data/test/
fi

echo extra_serve_args: "${extra_serve_args[@]}"


echo "Using the dataset at $dataset_path"

cd "$root_dir"/tpu_inference || exit
echo "Current working directory: $(pwd)"
echo "Using vLLM hash: $(git rev-parse HEAD)"

# Overwrite a few of the vLLM benchmarking scripts with the TPU Inference ones
echo "Using TPU Inference hash: $(git -C "$root_dir"/tpu_inference rev-parse HEAD)"

checkThroughputAndAccuracy() {
    # This function checks whether the accuracy score and total token throughput
    # from a benchmark log file meet specified target values. It validates the
    # presence and accessibility of the log file, extracts the accuracy score and
    # total token throughput, and compares them against predefined targets.
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

    # Extract accuracy score
    actual_accuracy=$(grep -oP "'accuracy': \K[0-9.]+" "$BENCHMARK_LOG_FILE")

    # Extract Total token throughput
    actual_throughput=$(awk '/Total token throughput \(tok\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    echo "--- Extracted Values ---"
    if [ "$SKIP_ACCURACY_TESTS" = "True" ]; then
        echo "skipping accuracy test"
        accuracy_pass=1
    elif [ -z "$actual_accuracy" ]; then
        echo "Accuracy score: NOT FOUND"
        accuracy_pass=0
    else
        echo "Accuracy score: $actual_accuracy"
        # awk exits with 0 if the condition (actual >= target) is true, making the 'if' statement succeed.
        # We achieve this by 'exit !(condition)' because awk's boolean true is 1, false is 0.
        # So, if (actual >= target) is true (1), !(1) is 0. exit 0.
        # If (actual >= target) is false (0), !(0) is 1. exit 1.
        if awk -v actual="$actual_accuracy" -v target="$TARGET_ACCURACY" 'BEGIN { exit !(actual >= target) }'; then
            echo "Accuracy comparison (>= $TARGET_ACCURACY): PASSED"
            accuracy_pass=1
        else
            echo "Accuracy comparison (>= $TARGET_ACCURACY): FAILED"
            accuracy_pass=0
        fi
    fi
    echo

    if [ -z "$actual_throughput" ]; then
        echo "Total token throughput: NOT FOUND"
        throughput_pass=0
    else
        echo "Total token throughput: $actual_throughput"
        if awk -v actual="$actual_throughput" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
            echo "Total token throughput comparison (>= $TARGET_THROUGHPUT): PASSED"
            throughput_pass=1
        else
            echo "Total token throughput comparison (>= $TARGET_THROUGHPUT): FAILED"
            throughput_pass=0
        fi
    fi
    echo

    echo "--- Summary ---"
    # Ensure pass flags are initialized if extraction fails
    : "${accuracy_pass:=0}"
    : "${throughput_pass:=0}"

    if [ "$accuracy_pass" -eq 1 ] && [ "$throughput_pass" -eq 1 ]; then
        echo "Overall: PASSED"
    else
        echo "Overall: FAILED"
        [ "$accuracy_pass" -eq 0 ] && echo "Reason: Accuracy check failed or value not found."
        [ "$throughput_pass" -eq 0 ] && echo "Reason: Throughput check failed or value not found."
        exit_code=1
    fi
}

for model_name in $model_list; do
    echo "--------------------------------------------------"
    echo "Running benchmark for model: $model_name"
    echo "--------------------------------------------------"

    if [ "$NEW_MODEL_DESIGN" = "True" ] && { [ "$model_name" == "Qwen/Qwen2.5-1.5B-Instruct" ] || [ "$model_name" == "Qwen/Qwen2.5-0.5B-Instruct" ];}; then
       echo "Skipping $model_name for NEW_MODEL_DESIGN: True"
        continue
    fi

    if [ "$MODEL_IMPL_TYPE" == "vllm" ] && [ "$model_name" == "Qwen/Qwen2.5-0.5B-Instruct" ]; then
        echo "Skipping $model_name for MODEL_IMPL_TYPE: vllm"
        continue
    fi

    # TODO (jacobplatin): remove when Qwen2.5 uses new model implementation
    if [ "$QUANTIZATION" = "True" ] && { [ "$model_name" == "Qwen/Qwen2.5-1.5B-Instruct" ] || [ "$model_name" == "Qwen/Qwen2.5-0.5B-Instruct" ];}; then
       echo "Skipping $model_name for QUANTIZATION: True"
        continue
    fi

    # Define model-specific arguments
    current_serve_args=("${extra_serve_args[@]}")

    current_device_count=${DEVICE_COUNT}
    if [ -z "$current_device_count" ]; then
        # Set to 8 if job is in 8 chips queue.
        # TODO (Qiliang Cui) Rename USE_V6E8_QUEUE to USE_8_CHIPS_QUEUE
        if [ "$USE_V6E8_QUEUE" == "True" ]; then
            current_device_count=8
        elif [ "$TPU_VERSION" == "tpu7x" ]; then
            # Set the default value to 2 for tpu v7x for most models
            current_device_count=2
            # If using large model (e.g. DeepSeek, then set the count to 8)
            if [[ "${model_name,,}" == *"deepseek"* ]]; then
                current_device_count=8
            fi
        else
            current_device_count=1
        fi
    fi
    current_serve_args+=(--tensor-parallel-size "${current_device_count}")

    max_batched_tokens=8192
    served_name=$model_name
    max_num_seqs=64
    max_model_len=1024
    if [ "$USE_V6E8_QUEUE" == "True" ]; then
        max_batched_tokens=1024
        if [ "$model_name" == "meta-llama/Llama-4-Scout-17B-16E-Instruct" ]; then
            current_serve_args+=(--hf-overrides '{"architectures": ["Llama4ForCausalLM"]}')
        fi
    else
        if [[ "${model_name,,}" == *"deepseek"* ]]; then
            export TIMEOUT_SECONDS=3600 # DeepSeek needs a longer timeout
            max_batched_tokens=512
            max_model_len=9216
            served_name=deepseek-ai/DeepSeek-R1
            TARGET_ACCURACY="0.84"
            current_serve_args+=(--served-model-name "${served_name}" --load-format=runai_streamer --trust-remote-code)
            current_serve_args+=(--kv-cache-dtype=fp8)
            if [ "$MODEL_IMPL_TYPE" == "vllm" ]; then
                current_serve_args+=(--enable-expert-parallel)
                current_serve_args+=(--additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}')
            elif [ "$MODEL_IMPL_TYPE" == "flax_nnx" ]; then
                current_serve_args+=(--additional_config '{"sharding": {"sharding_strategy": 
                    {"enable_dp_attention": true, "expert_parallelism": '"${current_device_count}"', "tensor_parallelism": 1}},
                        "replicate_attn_weights": "True", "sparse_matmul": "True"}')
            fi
        fi
    fi

    # Spin up the vLLM server
    echo "Spinning up the vLLM server..."
    (vllm serve "$model_name" --no-enable-prefix-caching --gpu-memory-utilization=0.95 --max-model-len $max_model_len --max-num-seqs $max_num_seqs --max-num-batched-tokens "$max_batched_tokens" "${current_serve_args[@]}" 2>&1 | tee -a "$LOG_FILE") &

    # Set initial trap to ensure cleanup happens even on immediate exit
    trap 'cleanUp "$model_name"' EXIT

    waitForServerReady

    echo "Starting the benchmark for $served_name..."
    echo "Current working directory: $(pwd)"
    python scripts/vllm/benchmarking/benchmark_serving.py \
    --backend vllm \
    --model "$served_name" \
    --dataset-name "$dataset_name" \
    --dataset-path "$dataset_path" \
    --num-prompts "$num_prompts" \
    --mmlu-output-len "${mmlu_output_len}" \
    --run-eval 2>&1 | tee -a "$BENCHMARK_LOG_FILE"

    checkThroughputAndAccuracy
    if [ "$exit_code" -ne 0 ]; then
        exit_code=1
    fi

    # Call cleanUp normally instead of using a trap
    cleanUp "$model_name"
done

# We successfully cleanUp every model, so cancel the trap.
trap - EXIT

exit $exit_code
