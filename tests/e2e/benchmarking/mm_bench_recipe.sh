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


# Spin up the vLLM server
model_name="Qwen/Qwen2.5-VL-7B-Instruct"
max_model_len=16384
dataset_name="random-mm"
backend="openai-chat"
num_prompts=128

TIMEOUT_SECONDS=600
READY_MESSAGE="Application startup complete."
LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
TARGET_THROUGHPUT="0.76"
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

checkThroughput() {

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

    # Extract request throughput
    actual_throughput=$(awk '/Request throughput \(req\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    echo "--- Extracted Values ---"
    echo

    if [ -z "$actual_throughput" ]; then
        echo "Total token throughput: NOT FOUND"
        throughput_pass=0
    else
        echo "Request throughput: $actual_throughput"
        if awk -v actual="$actual_throughput" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
            echo "Request throughput comparison (>= $TARGET_THROUGHPUT): PASSED"
            throughput_pass=1
        else
            echo "Request throughput comparison (>= $TARGET_THROUGHPUT): FAILED"
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
(vllm serve "$model_name" --tensor-parallel-size 1 --pipeline-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.98 --max-model-len "$max_model_len" --limit-mm-per-prompt '{"image": 10, "video": 0}' --mm-processor-kwargs '{"size": {"longest_edge": 1003520, "shortest_edge": 3136}}' --disable-chunked-mm-input  2>&1 | tee -a "$LOG_FILE") &


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
    --model "$model_name" \
    --dataset-name "$dataset_name" \
    --num-prompts "$num_prompts" \
    --backend "$backend" \
    --endpoint "/v1/chat/completions" \
    --random-mm-bucket-config '{(736, 736, 1): 1.0}' \
    --random-mm-base-items-per-request 6 \
    --random-mm-num-mm-items-range-ratio 0.67 \
    --random-mm-limit-mm-per-prompt '{"image": 10, "video": 0}' 2>&1 | tee -a "$BENCHMARK_LOG_FILE"


    checkThroughput
    if [ "$exit_code" -ne 0 ]; then
        exit_code=1
    fi

else
    echo "vLLM server did not start successfully."
    exit_code=1
fi
cleanUp "$model_name"

exit $exit_code
