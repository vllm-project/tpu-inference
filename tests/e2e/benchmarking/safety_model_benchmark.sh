#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# -----------------------------------------------------------------------------
# Generic Safety Model Unified Benchmark Recipe
# -----------------------------------------------------------------------------
# DESCRIPTION:
# This script provides a unified entry point to run both Accuracy (offline pytest)
# and Performance (API server) tests for any safety classification model defined
# in the test configuration (e.g., Llama-Guard-4-12B). It now supports multimodal
# inputs using a custom dataset for performance benchmarks.
#
# USAGE:
# 1. Run Text-Only Accuracy Check: bash tests/e2e/benchmarking/safety_model_benchmark.sh --mode accuracy --benchmark text-only
# 2. Run Multimodal Accuracy Check: bash tests/e2e/benchmarking/safety_model_benchmark.sh --mode accuracy --benchmark multimodal
# 3. Run Text-Only Performance Benchmark: bash tests/e2e/benchmarking/safety_model_benchmark.sh --mode performance --benchmark text-only
# 4. Run Multimodal Performance Benchmark: bash tests/e2e/benchmarking/safety_model_benchmark.sh --mode performance --benchmark multimodal
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
CI_MAX_TEST_CASES=100
TEXT_ONLY_REQUEST_THROUGHPUT_TP1="50.00"
TEXT_ONLY_REQUEST_THROUGHPUT_TP8="125.00"
MULTIMODAL_REQUEST_THROUGHPUT_TP1="30"
MULTIMODAL_REQUEST_THROUGHPUT_TP8="70"


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

# MM-SafetyBench Data Paths
MM_SAFETYBENCH_REPO="https://github.com/isXinLiu/MM-SafetyBench.git"

MM_SAFETYBENCH_DIR="/tmp/MM-SafetyBench"
MM_SAFETYBENCH_IMAGE_DIR="${MM_SAFETYBENCH_DIR}/data/imgs"
# ------------------

TEST_MODE=""
BENCHMARK_TYPE=""
EXIT_CODE=0

# Access shared benchmarking functionality (cleanUp, waitForServerReady)
# shellcheck disable=SC1091
source "$(dirname "$0")/bench_utils.sh"

# --- Argument Parsing (unchanged) ---

helpFunction()
{
   echo "Usage: $0 --mode <accuracy|performance> --benchmark <text-only|multimodal> [other args]"
   exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode)
            TEST_MODE="$2"
            shift
            shift
            ;; 
        --benchmark)
            BENCHMARK_TYPE="$2"
            shift
            shift
            ;;        *)
            echo "Unknown option: $1"
            helpFunction
            ;;    esac
done

if [[ -z "$TEST_MODE" ]]; then
    echo "Error: --mode argument is required."
    helpFunction
fi

if [[ -z "$BENCHMARK_TYPE" ]]; then
    echo "Error: --benchmark argument is required."
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

create_multimodal_dataset() {
    echo "Creating multimodal dataset..."

    NUM_PROMPTS=$NUM_PROMPTS python -c "
import json
import os
from pathlib import Path

# The path to the processed_questions directory
questions_dir = Path('$MM_SAFETYBENCH_DIR') / 'data' / 'processed_questions'
image_dir = Path('$MM_SAFETYBENCH_IMAGE_DIR') / 'MM-SafetyBench(imgs)'
output_file = Path('/tmp/mm_safety_bench.jsonl')
num_prompts = int(os.environ['NUM_PROMPTS'])

with output_file.open('w') as f:
    for json_file in sorted(questions_dir.glob('*.json')):
        scenario_name = json_file.stem
        with json_file.open('r') as qf:
            data = json.load(qf)
        for question_id, entry in data.items():
            if num_prompts <= 0:
                break
            prompt_text = entry.get('Rephrased Question(SD)') or entry.get('Rephrased Question')
            img_filename = f'{question_id}.jpg'
            possible_paths = [
                image_dir / scenario_name / 'SD' / img_filename,
                image_dir / scenario_name / img_filename,
                image_dir / scenario_name / 'SD_TYPO' / img_filename,
            ]
            final_image_path = None
            for p in possible_paths:
                if p.exists():
                    final_image_path = p
                    break
            if final_image_path:
                entry = {
                    'prompt': prompt_text,
                    'image': str(final_image_path)
                }
                f.write(json.dumps(entry) + '\n')
                num_prompts -= 1
        if num_prompts <= 0:
            break
"
}

run_accuracy_check() {
    echo -e "\n--- Running Accuracy Check (Mode: ACCURACY) ---"

    CONFTEST_DIR="./scripts/vllm/integration"

    RELATIVE_TEST_FILE="test_safety_model_accuracy.py"

    (
        cd "$CONFTEST_DIR" || { echo "Error: Failed to find conftest directory: $CONFTEST_DIR"; exit 1; }
        echo "Running pytest from: $(pwd)"

        python -m pytest -s -rP "$RELATIVE_TEST_FILE"::test_safety_model_accuracy_check \
            -W ignore::DeprecationWarning \
            --tensor-parallel-size "$TP_SIZE" \
            --model-name "$MODEL_NAME" \
            --expected-value "$MINIMUM_ACCURACY_THRESHOLD" \
            --dataset-path "$LOCAL_CSV_FILE"
    )
    return $?
}

run_performance_benchmark() {
    echo -e "\n--- Running Performance Benchmark (Mode: PERFORMANCE) ---"

    local TARGET_THROUGHPUT
    if [ "$TP_SIZE" -eq 1 ]; then
        TARGET_THROUGHPUT="$TEXT_ONLY_REQUEST_THROUGHPUT_TP1"
    elif [ "$TP_SIZE" -ge 8 ]; then
        TARGET_THROUGHPUT="$TEXT_ONLY_REQUEST_THROUGHPUT_TP8"
    else
        TARGET_THROUGHPUT="$TEXT_ONLY_REQUEST_THROUGHPUT_TP8" # Default to TP8 if not 1
    fi

    vllm bench serve \
        --model "$MODEL_NAME" \
        --endpoint "/v1/completions" \
        --dataset-name custom \
        --dataset-path "$LOCAL_JSONL_FILE" \
        --num-prompts "$NUM_PROMPTS" \
        --backend vllm \
        --custom-output-len "$OUTPUT_LEN_OVERRIDE" \
        2>&1 | tee "$BENCHMARK_LOG_FILE"

    ACTUAL_THROUGHPUT=$(awk '/Request throughput \(req\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$ACTUAL_THROUGHPUT" ]; then
        echo "Error: Request throughput NOT FOUND in benchmark logs."
        return 1
    fi

    echo "Actual Request Throughput: $ACTUAL_THROUGHPUT req/s"

    if awk -v actual="$ACTUAL_THROUGHPUT" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
        echo "PERFORMANCE CHECK PASSED: $ACTUAL_THROUGHPUT >= $TARGET_THROUGHPUT"
        return 0
    else
        echo "PERFORMANCE CHECK FAILED: $ACTUAL_THROUGHPUT < $TARGET_THROUGHPUT" >&2
        return 1
    fi
}

run_multimodal_performance_benchmark() {
    echo -e "\n--- Running Multimodal Performance Benchmark (Mode: PERFORMANCE, Benchmark: MULTIMODAL) ---"

    local TARGET_THROUGHPUT
    if [ "$TP_SIZE" -eq 1 ]; then
        TARGET_THROUGHPUT="$MULTIMODAL_REQUEST_THROUGHPUT_TP1"
    elif [ "$TP_SIZE" -ge 8 ]; then
        TARGET_THROUGHPUT="$MULTIMODAL_REQUEST_THROUGHPUT_TP8"
    else
        TARGET_THROUGHPUT="$MULTIMODAL_REQUEST_THROUGHPUT_TP8" # Default to TP8 if not 1
    fi

    download_mm_safetybench_dataset || return 1
    create_multimodal_dataset

    echo "Spinning up the vLLM server for $MODEL_NAME (TP=$TP_SIZE)..."

    # Server startup
    (vllm serve "$MODEL_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --limit-mm-per-prompt '{"image": 1}' \
        --mm-processor-kwargs '{"max_pixels": 1003520}' \
        --disable-chunked-mm-input \
        2>&1 | tee -a "$LOG_FILE") &

    waitForServerReady
    
    # Test run
    vllm bench serve \
        --model "$MODEL_NAME" \
        --dataset-name custom \
        --dataset-path /tmp/mm_safety_bench.jsonl \
        --num-prompts "$NUM_PROMPTS" \
        --backend "openai-chat" \
        --endpoint "/v1/chat/completions" \
        2>&1 | tee -a "$BENCHMARK_LOG_FILE"


    ACTUAL_THROUGHPUT=$(awk '/Request throughput \(req\/s\):/ {print $NF}' "$BENCHMARK_LOG_FILE")

    if [ -z "$ACTUAL_THROUGHPUT" ]; then
        echo "Error: Request throughput NOT FOUND in benchmark logs."
        return 1
    fi

    echo "Actual Request Throughput: $ACTUAL_THROUGHPUT req/s"

    if awk -v actual="$ACTUAL_THROUGHPUT" -v target="$TARGET_THROUGHPUT" 'BEGIN { exit !(actual >= target) }'; then
        echo "PERFORMANCE CHECK PASSED: $ACTUAL_THROUGHPUT >= $TARGET_THROUGHPUT"
        return 0
    else
        echo "PERFORMANCE CHECK FAILED: $ACTUAL_THROUGHPUT < $TARGET_THROUGHPUT" >&2
        return 1
    fi
}

download_mm_safetybench_dataset() {
    echo -e "\n--- Setting up MM-SafetyBench Dataset ---"

    # 1. Clone the MM-SafetyBench repository
    if [ ! -d "$MM_SAFETYBENCH_DIR" ]; then
        echo "Cloning MM-SafetyBench repository..."
        if ! git clone "$MM_SAFETYBENCH_REPO" "$MM_SAFETYBENCH_DIR"; then
            echo "Error: Failed to clone MM-SafetyBench repository." >&2
            return 1
        fi
    else
        echo "MM-SafetyBench repository already exists locally: $MM_SAFETYBENCH_DIR"
        # Ensure it's up to date
        (cd "$MM_SAFETYBENCH_DIR" && git pull) || \
            { echo "Warning: Failed to update MM-SafetyBench repository."; }
    fi

    # 2. Download and unzip the images from Google Drive
    if [ ! -d "$MM_SAFETYBENCH_IMAGE_DIR" ]; then
        echo "Downloading MM-SafetyBench images from Google Drive..."
        local zip_file="${MM_SAFETYBENCH_DIR}/MM-SafetyBench_imgs.zip"
        
        # Ensure gdown is installed
        if ! command -v gdown &> /dev/null; then
            echo "Installing gdown..."
            pip install gdown
        fi

        # Ensure unzip is installed
        if ! command -v unzip &> /dev/null; then
            echo "Installing unzip..."
            sudo apt-get update && sudo apt-get install -y unzip
        fi

        # Use gdown to handle Google Drive download
        if ! gdown --fuzzy --id 1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_ -O "$zip_file"; then
            echo "Error: Failed to download MM-SafetyBench images." >&2
            return 1
        fi

        echo "Unzipping images..."
        mkdir -p "${MM_SAFETYBENCH_DIR}/data/imgs"
        if ! unzip -q "$zip_file" -d "${MM_SAFETYBENCH_DIR}/data/imgs"; then
            echo "Error: Failed to unzip MM-SafetyBench images." >&2
            rm -f "$zip_file" # Clean up incomplete zip
            return 1
        fi
        rm -f "$zip_file" # Clean up zip file after extraction
        echo "MM-SafetyBench images downloaded and unzipped."
    else
        echo "MM-SafetyBench images already exist locally: $MM_SAFETYBENCH_IMAGE_DIR"
    fi
}

run_multimodal_accuracy_check() {
    echo -e "\n--- Running Multimodal Accuracy Check (Mode: ACCURACY, Benchmark: MULTIMODAL) ---"

    download_mm_safetybench_dataset || return 1

    CONFTEST_DIR="./scripts/vllm/integration"

    RELATIVE_TEST_FILE="test_multimodal_safety_model_accuracy.py"

    (
        cd "$CONFTEST_DIR" || { echo "Error: Failed to find conftest directory: $CONFTEST_DIR"; exit 1; }
        echo "Running pytest from: $(pwd)"

        python -m pytest -s -rP "${RELATIVE_TEST_FILE}::test_multimodal_safety_model_accuracy_check" \
            -W ignore::DeprecationWarning \
            --tensor-parallel-size "$TP_SIZE" \
            --model-name "$MODEL_NAME" \
            --expected-value "$MINIMUM_ACCURACY_THRESHOLD" \
            --image-dir "$MM_SAFETYBENCH_IMAGE_DIR" \
            --num-test-cases "$CI_MAX_TEST_CASES"
    )
    return $?
}

# --- MAIN EXECUTION FLOW ---

# Set initial trap to ensure cleanup happens even on immediate exit
trap 'cleanUp "$MODEL_NAME" || true' EXIT

# --- 1. RUN TEST MODE  ---
if [ "$TEST_MODE" == "accuracy" ]; then
    if [ "$BENCHMARK_TYPE" == "text-only" ]; then
        run_accuracy_check
        EXIT_CODE=$?
    elif [ "$BENCHMARK_TYPE" == "multimodal" ]; then
        run_multimodal_accuracy_check
        EXIT_CODE=$?
    else
        echo "Error: Invalid benchmark type for accuracy mode: $BENCHMARK_TYPE" >&2
        EXIT_CODE=1
    fi
    exit $EXIT_CODE
fi

# --- 2. START SERVER (Required ONLY for Performance Mode) ---
if [ "$TEST_MODE" == "performance" ]; then
    if [ "$BENCHMARK_TYPE" == "text-only" ]; then
        echo "Spinning up the vLLM server for $MODEL_NAME (TP=$TP_SIZE)..."

        # Server startup
        (vllm serve "$MODEL_NAME" \
            --tensor-parallel-size "$TP_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
            2>&1 | tee -a "$LOG_FILE") &

        waitForServerReady

        run_performance_benchmark
        EXIT_CODE=$?
    elif [ "$BENCHMARK_TYPE" == "multimodal" ]; then
        run_multimodal_performance_benchmark
        EXIT_CODE=$?
    else
        echo "Error: Invalid benchmark type for performance mode: $BENCHMARK_TYPE" >&2
        EXIT_CODE=1
    fi
fi

# --- 3. CLEANUP AND EXIT ---
exit $EXIT_CODE