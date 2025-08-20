#!/bin/bash

# vllm bench throughput \
#   --model Qwen/Qwen2.5-VL-3B-Instruct \
#   --backend vllm-chat \
#   --dataset-name hf \
#   --dataset-path lmarena-ai/VisionArena-Chat \
#   --num-prompts 10 \
#   --hf-split train \
#   --max_num_batched_tokens 98304


# vllm bench serve \
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
dataset_name="hf"
dataset_path="lmarena-ai/VisionArena-Chat"
num_prompts=10

TIMEOUT_SECONDS=600
READY_MESSAGE="Application startup complete."
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
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

echo "Spinning up the vLLM server..."
(vllm serve "$model_name" --max-model-len=98304 --disable-log-requests --max-num-batched-tokens "$max_batched_tokens" 2>&1 | tee -a "$LOG_FILE") &


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
    python /workspace/vllm/benchmarks/benchmark_serving.py \
    --backend vllm-chat \
    --model "$model_name" \
    --dataset-name "$dataset_name" \
    --dataset-path "$dataset_path" \
    --num-prompts "$num_prompts" 2>&1 | tee -a "$BENCHMARK_LOG_FILE"
else
    echo "vLLM server did not start successfully."
    exit_code=1
fi
cleanUp "$model_name"

exit $exit_code
