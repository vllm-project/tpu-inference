#!/bin/bash

# Log the vLLM server output to a file
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
# The sentinel message that indicates the server is ready (in LOG_FILE)
READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
TIMEOUT_SECONDS=3600

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
   echo -e "\t-p, --dataset-path\tThe path to the processed dataset. This is required when using a custom model (default: benchmarks/sonnet.txt)."
   echo -e "\t-v, --vllm-download-dir\tThe directory to download vLLM into (default: /tmp/hf_home)"
   echo -e "\t-h, --help\tShow this help message"
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

if [ -n "$TEST_MODEL" ]; then
  test_model="$TEST_MODEL"
fi

if [ -n "$INPUT_LEN" ]; then
  input_len="$INPUT_LEN"
fi

if [ -n "$OUTPUT_LEN" ]; then
  output_len="$OUTPUT_LEN"
fi

if [ -n "$PREFIX_LEN" ]; then
  prefix_len="$PREFIX_LEN"
fi

if [ -n "$MINIMUM_THROUGHPUT_THRESHOLD" ]; then
  minimum_throughput_threshold="$MINIMUM_THROUGHPUT_THRESHOLD"
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
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

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp -r "$root_dir"/tpu_inference/scripts/vllm/benchmarking/*.py "$root_dir"/vllm/benchmarks/

cleanUp() {
    echo "Stopping the vLLM server and cleaning up log files..."
    pkill -f "vllm serve $1"
    # Kill all processes related to vllm.
    pgrep -f -i vllm | xargs -r kill -9

    # Clean up log files. Use -f to avoid errors if files don't exist.
    rm -f "$LOG_FILE"
    rm -f "$BENCHMARK_LOG_FILE"
    echo "Cleanup complete."
}

checkThroughput() {
    # This function checks whether the total token throughput from a benchmark
    # log file meet specified target value. It validates the presence and
    # accessibility of the log file, extracts total token throughput, and
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

    # Extract Total Token throughput
    actual_throughput=$(awk '/Total Token throughput \(tok\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$actual_throughput" ]; then
        echo "Total Token throughput: NOT FOUND"
        throughput_pass=0
    else
        echo "Total Token throughput: $actual_throughput"
        if awk -v actual="$actual_throughput" -v target="$minimum_throughput_threshold" 'BEGIN { exit !(actual >= target) }'; then
            echo "Total Token throughput comparison (>= $minimum_throughput_threshold): PASSED"
            throughput_pass=1
        else
            echo "Total Token throughput comparison (>= $minimum_throughput_threshold): FAILED"
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

# If we have multiple args, we can add them in extra_serve_args.
extra_serve_args=()

echo "--------------------------------------------------"
echo "Running benchmark for model: $test_model"
echo "--------------------------------------------------"

# Define model-specific arguments
current_serve_args=("${extra_serve_args[@]}")
max_batched_tokens=8192
if [ "$USE_V6E8_QUEUE" == "True" ]; then
    max_batched_tokens=1024
fi

# Spin up the vLLM server
echo "Spinning up the vLLM server..."
(vllm serve "$test_model" --max-model-len="$((input_len + output_len + prefix_len))" --disable-log-requests --max-num-batched-tokens "$max_batched_tokens" --download_dir "$vllm_download_dir" --tensor-parallel-size "$tensor_parallel_size" "${current_serve_args[@]}" 2>&1 | tee -a "$LOG_FILE") &


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
        cleanUp "$test_model"
        break
    fi

    if grep -q "$READY_MESSAGE" "$LOG_FILE" ; then
        did_find_ready_message=true
        break
    fi
done


if $did_find_ready_message; then
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
else
    echo "vLLM server did not start successfully."
    exit_code=1
fi

cleanUp "$test_model"


exit $exit_code
