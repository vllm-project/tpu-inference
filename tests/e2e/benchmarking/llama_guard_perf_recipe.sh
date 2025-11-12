#!/bin/bash
set -e

# --- Configuration ---
# Read from CI environment variables (set in the YAML)
MODEL_NAME="${TEST_MODEL:-meta-llama/Llama-Guard-4-12B}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# Paths and Files
# --- CRITICAL CHANGE: Point directly to the final JSONL file in GCS ---
# This GCS path is now the *dataset-path* argument for vllm bench serve
GCS_DATASET_URI="gs://jiries/datasets/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.jsonl" 

LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"

# Server/Benchmark Settings
MAX_MODEL_LEN=4096
NUM_PROMPTS=500 # Use a large subset of the dataset for stable metrics
# Llama Guard generates a short, deterministic output (e.g., "unsafe\nS4")
OUTPUT_LEN_OVERRIDE=20 
# Target for checking minimum acceptable performance (You must measure this!)
TARGET_THROUGHPUT="40.00" #NOTE: Setting low intentionally to see the test pass first.  

backend="vllm"

TIMEOUT_SECONDS=600
READY_MESSAGE="Application startup complete."
exit_code=0
# ---------------------


cleanUp() {
    echo "Stopping the vLLM server and cleaning up..."
    pkill -f "vllm serve $MODEL_NAME" || true
    pgrep -f -i vllm | xargs -r kill -9 || true
    rm -f "$LOG_FILE" "$BENCHMARK_LOG_FILE"
    echo "Cleanup complete."
}

checkThroughput() {
    # Check benchmark logs for 'Output token throughput (tok/s):'
    actual_throughput=$(awk '/Output token throughput \(tok\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$actual_throughput" ]; then
        echo "Error: Output token throughput NOT FOUND in benchmark logs."
        exit_code=1
        return
    fi
    
    echo "Actual Output Token Throughput: $actual_throughput tok/s"
    
    # Compare with the TARGET_THROUGHPUT
    if awk -v actual="$actual_throughput" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
        echo "PERFORMANCE CHECK (>= $TARGET_THROUGHPUT): PASSED"
        exit_code=0
    else
        echo "PERFORMANCE CHECK (>= $TARGET_THROUGHPUT): FAILED"
        echo "Target throughput ($TARGET_THROUGHPUT) not met."
        exit_code=1
    fi
}

# --- Trap cleanup function to run on exit or error ---
trap cleanUp EXIT

echo "Using GCS dataset at: $GCS_DATASET_URI"

# --- 2. SPIN UP VLLM SERVER ---
echo "Spinning up the vLLM server for $MODEL_NAME (TP=$TP_SIZE)..."
# Using the standard model load command. I used to have SKIP_JAX_PRECOMPILE=1 in the following command
(vllm serve "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 3072 \
    --hf-overrides '{"architectures": ["LlamaForCausalLM"]}' \
    2>&1 | tee -a "$LOG_FILE") &

# Wait loop
start_time=$(date +%s)
echo "Waiting for server ready message: '$READY_MESSAGE'"
while ! grep -q "$READY_MESSAGE" "$LOG_FILE" ; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    if [[ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]]; then
        echo "TIMEOUT: Server did not start within $TIMEOUT_SECONDS seconds."
        exit 1
    fi
    sleep 5
done
echo "Server is ready."


# --- 3. RUN BENCHMARK ---
echo "Starting the benchmark using AILuminate prompts..."
vllm bench serve \
    --model "$MODEL_NAME" \
    --endpoint "/v1/completions" \
    --dataset-name custom \
    --dataset-path "$GCS_DATASET_URI" \
    --num-prompts "$NUM_PROMPTS" \
    --backend "$backend" \
    --custom-output-len "$OUTPUT_LEN_OVERRIDE" \
    2>&1 | tee -a "$BENCHMARK_LOG_FILE"


# --- 4. CHECK THROUGHPUT AND SET EXIT CODE ---
checkThroughput

exit $exit_code