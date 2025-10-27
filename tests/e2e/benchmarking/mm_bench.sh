#!/bin/bash

# Example offline benchmark:

# vllm bench throughput \
#   --model Qwen/Qwen2.5-VL-3B-Instruct \
#   --backend vllm-chat \
#   --dataset-name hf \
#   --dataset-path lmarena-ai/VisionArena-Chat \
#   --num-prompts 10 \
#   --hf-split train \
#   --max_num_batched_tokens 98304

# Spin up the vLLM server
model_name="Qwen/Qwen2.5-VL-3B-Instruct"
max_batched_tokens=98304
max_model_len=98304
max_num_seqs=128
dataset_name="hf"
dataset_path="lmarena-ai/VisionArena-Chat"
num_prompts=10

TIMEOUT_SECONDS=600
READY_MESSAGE="Application startup complete."
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
TARGET_THROUGHPUT="4" # Set it to a reasonably low value for now. Can set it higher when get optimized later.
exit_code=0




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

checkThroughputAndRouge() {

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

    echo "--- Extracted Values ---"
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
    : "${throughput_pass:=0}"

    if [ "$throughput_pass" -eq 1 ]; then
        echo "Overall: PASSED"
    else
        echo "Overall: FAILED"
        [ "$throughput_pass" -eq 0 ] && echo "Reason: Throughput check failed or value not found."
        exit_code=1
    fi
}

echo "Spinning up the vLLM server..."
(SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 vllm serve "$model_name" --max-model-len "$max_model_len" --max-num-seqs "$max_num_seqs" --disable-log-requests --max-num-batched-tokens "$max_batched_tokens" 2>&1 | tee -a "$LOG_FILE") &


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
    vllm bench serve \
    --backend "openai-chat" \
    --model "$model_name" \
    --dataset-name "$dataset_name" \
    --dataset-path "$dataset_path" \
    --num-prompts "$num_prompts" \
    --endpoint /v1/chat/completions 2>&1 | tee -a "$BENCHMARK_LOG_FILE"


    checkThroughputAndRouge
    if [ "$exit_code" -ne 0 ]; then
        exit_code=1
    fi

else
    echo "vLLM server did not start successfully."
    exit_code=1
fi
cleanUp "$model_name"

exit $exit_code
