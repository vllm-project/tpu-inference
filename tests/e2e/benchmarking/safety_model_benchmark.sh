#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# -----------------------------------------------------------------------------
# Generic Safety Model Unified Benchmark Recipe
# -----------------------------------------------------------------------------
# DESCRIPTION:
# This script provides a unified entry point to run both Accuracy (offline pytest)
# and Performance (API server) tests for any safety classification model defined
# in the test configuration (e.g., Llama-Guard-4-12B).
#
# USAGE:
# 1. Run Accuracy Check: bash safety_model_benchmark.sh --mode accuracy
# 2. Run Performance Benchmark: bash safety_model_benchmark.sh --mode performance
#
# REQUIRED ENVIRONMENT VARIABLES (Example Values for Llama Guard 4):
# export TEST_MODEL="meta-llama/Llama-Guard-4-12B"
# export TENSOR_PARALLEL_SIZE=8
# export MINIMUM_ACCURACY_THRESHOLD=0.42
# -----------------------------------------------------------------------------
set -e

# --- Configuration & Defaults ---
# shellcheck disable=SC2153
MODEL_NAME="${TEST_MODEL}"
TP_SIZE="${TENSOR_PARALLEL_SIZE}"

LOG_FILE="server.log"
BENCHMARK_LOG_FILE="benchmark.log"
export READY_MESSAGE="Application startup complete."
export TIMEOUT_SECONDS=600

# Check thresholds (set in CI YAML env block)
# shellcheck disable=SC2269
MINIMUM_ACCURACY_THRESHOLD=${MINIMUM_ACCURACY_THRESHOLD}
if [ "$TP_SIZE" -eq 1 ]; then
    TARGET_THROUGHPUT="225.00" # New threshold for single device
elif [ "$TP_SIZE" -ge 8 ]; then
    TARGET_THROUGHPUT="487.00" # Threshold for high-parallelism (TP=8)
else
    TARGET_THROUGHPUT="487.00"
fi

# Benchmark/Serve Settings
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=4096
NUM_PROMPTS=500
OUTPUT_LEN_OVERRIDE=20 # Max tokens to generate for safety classification.

# --- DATA PATHS ---
# Source URL for the AILuminate CSV (Public Raw GitHub Link)
RAW_CSV_URL="https://raw.githubusercontent.com/mlcommons/ailuminate/main/airr_official_1.0_demo_en_us_prompt_set_release.csv"
LOCAL_CSV_FILE="/tmp/airr_official_1.0_demo_en_us_prompt_set_release.csv"
LOCAL_JSONL_FILE="/tmp/airr_official_1.0_demo_en_us_prompt_set_release.jsonl"
# ------------------

TEST_MODE=""
EXIT_CODE=0

# Access shared benchmarking functionality (cleanUp, waitForServerReady)
# shellcheck disable=SC1091
source "$(dirname "$0")/bench_utils.sh"

# --- Argument Parsing (unchanged) ---

helpFunction()
{
   echo "Usage: $0 --mode <accuracy|performance> [other args]"
   exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode)
            TEST_MODE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            helpFunction
            ;;
    esac
done

if [[ -z "$TEST_MODE" ]]; then
    echo "Error: --mode argument is required."
    helpFunction
fi

# --- DATA DOWNLOAD CHECK ---
# Check if the CSV file already exists locally
if [ ! -f "$LOCAL_CSV_FILE" ]; then
  echo "Downloading AILuminate CSV from GitHub..."
  # Use wget to download the file directly from the raw content URL
  if ! wget "$RAW_CSV_URL" -O "$LOCAL_CSV_FILE"; then
    echo "Error: Failed to download dataset via wget."
    exit 1
  fi
else
  echo "AILUMINATE CSV already exists locally."
fi

# Convert to JSONL to be compatible with vllm bench serve command
if [ ! -f "$LOCAL_JSONL_FILE" ] || [ "$TEST_MODE" == "performance" ]; then
    echo "Converting CSV to JSONL for performance run..."

    python -c "
import sys, json, pandas as pd

try:
    # Read the CSV (uses local downloaded file)
    df = pd.read_csv('$LOCAL_CSV_FILE')
except Exception as e:
    sys.stderr.write(f'Error reading source CSV: {e}\n')
    sys.exit(1)

# Write out the JSONL file
with open('$LOCAL_JSONL_FILE', 'w') as f:
    for prompt_text in df['prompt_text']:
        # The vLLM benchmark client requires only the 'prompt' field.
        entry = {'prompt': prompt_text}
        f.write(json.dumps(entry) + '\n')

sys.stdout.write(f'Conversion successful. Wrote {len(df)} prompts to $LOCAL_JSONL_FILE\n')
"
    # ----------------------------------------------------
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "Error: CSV to JSONL conversion failed."
        exit 1
    fi
fi

# --- FUNCTION DEFINITIONS ---

run_accuracy_check() {
    echo -e "\n--- Running Accuracy Check (Mode: ACCURACY) ---"

    CONFTEST_DIR="/workspace/tpu_inference/scripts/vllm/integration"

    RELATIVE_TEST_FILE="test_safety_model_accuracy.py"

    (
        cd "$CONFTEST_DIR" || { echo "Error: Failed to find conftest directory: $CONFTEST_DIR"; exit 1; }
        echo "Running pytest from: $(pwd)"

        python -m pytest -s -rP "$RELATIVE_TEST_FILE::test_safety_model_accuracy_check" \
            -W ignore::DeprecationWarning \
            --tensor-parallel-size="$TP_SIZE" \
            --model-name="$MODEL_NAME" \
            --expected-value="$MINIMUM_ACCURACY_THRESHOLD" \
            --dataset-path="$LOCAL_CSV_FILE"
    )
    return $?
}

run_performance_benchmark() {
    echo -e "\n--- Running Performance Benchmark (Mode: PERFORMANCE) ---"

    vllm bench serve \
        --model "$MODEL_NAME" \
        --endpoint "/v1/completions" \
        --dataset-name custom \
        --dataset-path "$LOCAL_JSONL_FILE" \
        --num-prompts "$NUM_PROMPTS" \
        --backend vllm \
        --custom-output-len "$OUTPUT_LEN_OVERRIDE" \
        2>&1 | tee "$BENCHMARK_LOG_FILE"

    ACTUAL_THROUGHPUT=$(awk '/Output token throughput \(tok\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$ACTUAL_THROUGHPUT" ]; then
        echo "Error: Output token throughput NOT FOUND in benchmark logs."
        return 1
    fi

    echo "Actual Output Token Throughput: $ACTUAL_THROUGHPUT tok/s"

    if awk -v actual="$ACTUAL_THROUGHPUT" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
        echo "PERFORMANCE CHECK PASSED: $ACTUAL_THROUGHPUT >= $TARGET_THROUGHPUT"
        return 0
    else
        echo "PERFORMANCE CHECK FAILED: $ACTUAL_THROUGHPUT < $TARGET_THROUGHPUT" >&2
        return 1
    fi
}

# --- MAIN EXECUTION FLOW ---

# Set initial trap to ensure cleanup happens even on immediate exit
trap 'cleanUp "$MODEL_NAME" || true' EXIT

# --- 1. RUN TEST MODE  ---
if [ "$TEST_MODE" == "accuracy" ]; then
    run_accuracy_check
    EXIT_CODE=$?

    exit $EXIT_CODE
fi

# --- 2. START SERVER (Required ONLY for Performance Mode) ---
if [ "$TEST_MODE" == "performance" ]; then
    echo "Spinning up the vLLM server for $MODEL_NAME (TP=$TP_SIZE)..."

    # Server startup
    (vllm serve "$MODEL_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len="$MAX_MODEL_LEN" \
        --max-num-batched-tokens="$MAX_BATCHED_TOKENS" \
        2>&1 | tee -a "$LOG_FILE") &

    waitForServerReady

    run_performance_benchmark
    EXIT_CODE=$?
fi

# --- 3. CLEANUP AND EXIT ---
exit $EXIT_CODE
