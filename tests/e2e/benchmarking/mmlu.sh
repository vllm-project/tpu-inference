#!/bin/bash

# This script, by default, will test running Llama3.1-8B-Instruct on 10 prompts on the MLPerf dataset to check that the ROUGE score and overall throughput are reasonable.
# Specifically, it will do the following:
# 1. Spin up a vLLM server
# 2. Once the server is ready, run 10 prompts of the MLPerf benchmark
# 3. Check the ROUGE score and throughput and exit cleanly if and only if they are both reasonably high

# You can also manually invoke this script to run an E2E benchmark on any given model and dataset, making sure
# you specify the --dataset-name, --dataset-path, and --root-dir flags
# Example usage: bash tests/e2e/benchmarking/mmlu.sh -r /mnt/disks/jacobplatin -d mmlu -p /mnt/disks/jacobplatin/mmlu/data/test

# Logs the vLLM server output to a file
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
# The sentinel message that indicates the server is ready (in LOG_FILE)
READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
TIMEOUT_SECONDS=300

# The minimum ROUGE1 and throughput scores we expect
# TODO (jacobplatin): these are very low, so we'll want to boost them eventually
# We need to support different targets for different models.
TARGET_ROUGE1="40"
TARGET_THROUGHPUT="1500"

model_list="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.1-8B-Instruct"
root_dir=/workspace
dataset_name=mlperf
dataset_path=""
num_prompts=1000
exit_code=0

helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_commons' (default: /workspace/, which is used in the Dockerfile)"
   echo -e "\t-d The dataset name (default: mlperf, which will download the dataset)"
   echo -e "\t-p The path to the processed MLPerf dataset (default: None, which will download the dataset)"
   echo -e "\t-m A space-separated list of HuggingFace model ids to use (default: Qwen/Qwen2.5-1.5B-Instruct and meta-llama/Llama-3.1-8B-Instruct)"
   echo -e "\t-n Number of prompts to use for the benchmark (default: 10)"
   exit 1
}

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
# If the user doesn't specify a dataset_path then download the MLPerf data (if it doesn't already exist)
if [ -z "$dataset_path" ]; then
    if [ "$dataset_name" != "mlperf" ]; then
        echo "Dataset name must be mlperf if dataset_path is not specified.  We only support downloading the MLPerf dataset at this time."
        exit 1
    fi
    # Only download the dataset if it doesn't already exist
    dataset_path="$root_dir"/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
    if [ ! -f "$dataset_path" ]; then
        echo "Downloading the MLPerf dataset"
        curl https://rclone.org/install.sh | bash
        rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
        rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
        gzip -d open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
    else
        echo "Not downloading the MLPerf dataset because it already exists"
    fi
fi

echo "Using the dataset at $dataset_path"

cd "$root_dir"/vllm || exit
echo "Current working directory: $(pwd)"

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp -r "$root_dir"/tpu_commons/scripts/vllm/benchmarking/*.py "$root_dir"/vllm/benchmarks/

cleanUp() {
    echo "Stopping the vLLM server and cleaning up log files..."
    pkill -f "vllm serve $1"

    # Clean up log files. Use -f to avoid errors if files don't exist.
    rm -f "$LOG_FILE"
    rm -f "$BENCHMARK_LOG_FILE"
    echo "Cleanup complete."
}

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
        exit_code=2
        return
    fi
    if [ ! -f "$BENCHMARK_LOG_FILE" ]; then
        echo "Error: Benchmark log file '$BENCHMARK_LOG_FILE' not found." >&2
        exit_code=2
        return
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
    else
        echo "Overall: FAILED"
        [ "$rouge1_pass" -eq 0 ] && echo "Reason: Rouge1 check failed or value not found."
        [ "$throughput_pass" -eq 0 ] && echo "Reason: Throughput check failed or value not found."
        exit_code=1
    fi
}


for model_name in $model_list; do
    echo "--------------------------------------------------"
    echo "Running benchmark for model: $model_name"
    echo "--------------------------------------------------"
    # Spin up the vLLM server
    echo "Spinning up the vLLM server..."
    (vllm serve "$model_name" --max-model-len=1024 --disable-log-requests --max-num-batched-tokens 8192 2>&1 | tee -a "$LOG_FILE") &



    # Run a busy loop to block until the server is ready to receive requests
    did_find_ready_message=false
    start_time=$(date +%s)
    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        sleep 5

        # Check for timeout so we don't wait forever
        if [[ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]]; then
            echo "TIMEOUT: Waited $elapsed_time seconds (limit was $TIMEOUT_SECONDS). The string '$READY_MESSAGE' was NOT found."
            cleanUp "$model_name"
            exit 1
        fi

        if grep -q "$READY_MESSAGE" "$LOG_FILE" ; then
            did_find_ready_message=true
            break
        fi
    done



    if $did_find_ready_message; then
        echo "Starting the benchmark for $model_name..."
        echo "Current working directory: $(pwd)"
        python benchmarks/benchmark_serving.py \
        --backend vllm \
        --model "$model_name" \
        --dataset-name "$dataset_name" \
        --dataset-path "$dataset_path" \
        --num-prompts "$num_prompts" \
        --run_eval 2>&1 | tee -a "$BENCHMARK_LOG_FILE"

        # TODO (jacobplatin): probably want to add an option to skip this in the future
        if [ "$dataset_name" == "mlperf" ]; then
            checkThroughputAndRouge
            if [ "$exit_code" -ne 0 ]; then
                exit_code=1
            fi
        fi
    else
        echo "vLLM server did not start successfully."
        exit_code=1
    fi
    cleanUp "$model_name"
done


exit $exit_code
