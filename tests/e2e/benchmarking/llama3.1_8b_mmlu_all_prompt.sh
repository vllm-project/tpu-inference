#!/bin/bash

# This script will test running Llama3.1-8B-Instruct on a single prompt to check that the ROUGE score and overall throughput are reasonable.  Specifically, it will do the following:
# 1. Spin up a vLLM server
# 2. Once the server is ready, run the single prompt MMLU benchmark
# 3. Check the ROUGE score and throughput and exit cleanly if and only if they are both reasonably high

# Example usage: bash tests/e2e/benchmarking/llama3.1_8b_mmlu_few_prompt.sh
# Logs the vLLM server output to a file
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
# The sentinel message that indicates the server is ready (in LOG_FILE)
READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
TIMEOUT_SECONDS=300

# The minimum ROUGE1 and throughput scores we expect
# TODO (jacobplatin): these are very low, so we'll want to boost them eventually
TARGET_ROUGE1="30"
TARGET_THROUGHPUT="90"

model_name=meta-llama/Llama-3.1-8B-Instruct
root_dir=/workspace
dataset_path=""
helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_commons' (default: /workspace/, which is used in the Dockerfile)"
   echo -e "\t-d The path to the processed MMLU dataset (default: None, which will download the dataset)"
   echo -e "\t-m The HuggingFace model id to use (default: meta-llama/Llama-3.1-8B-Instruct)"
   exit 1
}

for arg in "$@"; do
    case $arg in
        -r|--root-dir-path) shift; root_dir="$1" ;;
        -d|--dataset-path) shift; dataset_path="$1" ;;
        -m|--model) shift; model_name="$1" ;;
        -h|--help) helpFunction ;;
    esac
    shift
done

cd "$root_dir" || exit
# If the user doesn't specify a dataset_path then download the MLPerf data
if [ -z "$dataset_path" ]; then
    echo "Downloading the MMLU dataset since 'dataset_path' was not specified"
    curl https://rclone.org/install.sh | bash
    rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
    rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
    gzip -d open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
    dataset_path="$root_dir"/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
fi

echo "Using the dataset at $dataset_path"

cd "$root_dir"/vllm || exit
echo "Current working directory: $(pwd)"

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp -r "$root_dir"/tpu_commons/scripts/vllm/benchmarking/*.py "$root_dir"/vllm/benchmarks/

# Spin up the vLLM server
echo "Spinning up the vLLM server..."
(TPU_BACKEND_TYPE=jax vllm serve "$model_name" --max-model-len=1024 --disable-log-requests --max-num-batched-tokens 8192 --max-num-seqs=1 2>&1 | tee -a "$LOG_FILE") &

# Run a busy loop to block until the server is ready to receive requests
did_find_ready_message=false
timeout_hit=false
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    sleep 5

    # Check for timeout so we don't wait forever
    if [[ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]]; then
        echo "TIMEOUT: Waited $elapsed_time seconds (limit was $TIMEOUT_SECONDS). The string '$READY_MESSAGE' was NOT found."
        timeout_hit=true
        exit 1
    fi

    if grep -q "$READY_MESSAGE" "$LOG_FILE" ; then
        did_find_ready_message=true
        break
    fi
done

checkThroughputAndRouge() {
    # This function checks whether the ROUGE1 score and total token throughput
    # from a benchmark log file meet specified target values. It validates the
    # presence and accessibility of the log file, extracts the ROUGE1 score and
    # total token throughput, and compares them against predefined targets.
    # The function outputs the results of these comparisons and exits with a
    # status code indicating overall success or failure.

    # Check if the inputs are valid
    if [ -z "$BENCHMARK_LOG_FILE" ]; then
        echo "Error: BENCHMARK_LOG_FILE environment variable is not set." >&2
        exit 2
    fi
    if [ ! -f "$BENCHMARK_LOG_FILE" ]; then
        echo "Error: Benchmark log file '$BENCHMARK_LOG_FILE' not found." >&2
        exit 2
    fi

    # Extract ROUGE1 score
    actual_rouge1=$(grep -oP "'rouge1': \K[0-9.]+" "$BENCHMARK_LOG_FILE")

    # Extract Total Token throughput
    actual_throughput=$(awk '/Total Token throughput \(tok\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    echo "--- Extracted Values ---"
    if [ -z "$actual_rouge1" ]; then
        echo "Rouge1 score: NOT FOUND"
        rouge1_pass=0
    else
        echo "Rouge1 score: $actual_rouge1"
        # awk exits with 0 if the condition (actual >= target) is true, making the 'if' statement succeed.
        # We achieve this by 'exit !(condition)' because awk's boolean true is 1, false is 0.
        # So, if (actual >= target) is true (1), !(1) is 0. exit 0.
        # If (actual >= target) is false (0), !(0) is 1. exit 1.
        if awk -v actual="$actual_rouge1" -v target="$TARGET_ROUGE1" 'BEGIN { exit !(actual >= target) }'; then
            echo "Rouge1 comparison (>= $TARGET_ROUGE1): PASSED"
            rouge1_pass=1
        else
            echo "Rouge1 comparison (>= $TARGET_ROUGE1): FAILED"
            rouge1_pass=0
        fi
    fi
    echo

    if [ -z "$actual_throughput" ]; then
        echo "Total Token throughput: NOT FOUND"
        throughput_pass=0
    else
        echo "Total Token throughput: $actual_throughput"
        if awk -v actual="$actual_throughput" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
            echo "Total Token throughput comparison (>= $TARGET_THROUGHPUT): PASSED"
            throughput_pass=1
        else
            echo "Total Token throughput comparison (>= $TARGET_THROUGHPUT): FAILED"
            throughput_pass=0
        fi
    fi
    echo

    echo "--- Summary ---"
    # Ensure pass flags are initialized if extraction fails
    : "${rouge1_pass:=0}"
    : "${throughput_pass:=0}"

    if [ "$rouge1_pass" -eq 1 ] && [ "$throughput_pass" -eq 1 ]; then
        echo "Overall: PASSED"
        exit 0
    else
        echo "Overall: FAILED"
        [ "$rouge1_pass" -eq 0 ] && echo "Reason: Rouge1 check failed or value not found."
        [ "$throughput_pass" -eq 0 ] && echo "Reason: Throughput check failed or value not found."
        exit 1
    fi
}

if $did_find_ready_message && ! $timeout_hit; then
    echo "Running the single prompt MMLU benchmark..."
    echo "Current working directory: $(pwd)"
    python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model "$model_name" \
    --dataset-name mlperf \
    --dataset-path "$dataset_path" \
    --num-prompts 24576 \
    --run_eval 2>&1 | tee -a "$BENCHMARK_LOG_FILE"

    checkThroughputAndRouge
else
    echo "vLLM server did not start successfully."
    exit 1
fi
