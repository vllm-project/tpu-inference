#!/bin/bash
# -----------------------------------------------------------------------------
# Llama Guard 4 Performance Benchmark Recipe
# -----------------------------------------------------------------------------
# DESCRIPTION:
# This script runs a rigorous serving benchmark for the JAX Llama-Guard-4-12B
# model using vLLM's API server and bench client. It loads a pre-processed
# AILuminate JSONL dataset from a GCS URI to measure Output Token Throughput
# (tok/s) against a performance baseline.
#
# USAGE (CI/Docker Environment):
# This script is intended to be executed inside the Buildkite Docker container
# via the CI YAML, which injects necessary environment variables (TEST_MODEL, TP_SIZE).
#
# USAGE (Local Testing):
# To run locally, set the environment variables and execute:
# export TEST_MODEL="meta-llama/Llama-Guard-4-12B"
# export TENSOR_PARALLEL_SIZE=8
# bash llama_guard_perf_recipe.sh
# -----------------------------------------------------------------------------
set -e

# --- Configuration ---
# Read from CI environment variables (set in the YAML)
MODEL_NAME="${TEST_MODEL:-meta-llama/Llama-Guard-4-12B}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# Paths and Files
GCS_DATASET_URI="gs://jiries/datasets/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.jsonl" 

LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"

# Server/Benchmark Settings
MAX_MODEL_LEN=4096
NUM_PROMPTS=500 # Use a large subset of the dataset for stable metrics
# Llama Guard generates a short, deterministic output (e.g., "unsafe\nS4")
OUTPUT_LEN_OVERRIDE=20 
# Target for checking minimum acceptable performance (You must measure this!)
TARGET_THROUGHPUT="450.00" 

backend="vllm"

TIMEOUT_SECONDS=600
READY_MESSAGE="Application startup complete."
exit_code=0

SHARED_UTILS_PATH="/workspace/tests/e2e/benchmarking/bench_utils.sh" #THIS PATH IS ERRORING IN THE CI

# Source the shared functions (cleanUp, waitForServerReady)
. "$SHARED_UTILS_PATH"

# ---------------------


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
trap 'cleanUp "$MODEL_NAME"' EXIT

echo "Using GCS dataset at: $GCS_DATASET_URI"

# --- 2. SPIN UP VLLM SERVER ---
echo "Spinning up the vLLM server for $MODEL_NAME (TP=$TP_SIZE)..."
# Using the standard model load command.
(vllm serve "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4096 \
    2>&1 | tee -a "$LOG_FILE") &

# --- 3. WAIT FOR SERVER (Shared Function Call) ---
waitForServerReady


# --- 4. RUN BENCHMARK ---
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


# --- 5. CHECK THROUGHPUT AND SET EXIT CODE ---
checkThroughput

exit $exit_code