#!/bin/bash
# Copyright 2026 Google LLC
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

set -Eeuo pipefail

# ==============================================================================
# 0. Global Panic Handler (Crash Interceptor)
# ==============================================================================
# shellcheck disable=SC2317
on_crash() {
    local exit_code=$?
    local line_no=$1
    local command="$2"
    
    # Ignore normal exits (Fixed SC2086 by adding double quotes)
    if [ "$exit_code" -eq 0 ]; then
        return
    fi

    echo ""
    echo "================================================================"
    echo "🚨 [FATAL ERROR] Bash Script Crashed Unexpectedly!"
    echo "================================================================"
    echo "File:     $(basename "$0")"
    echo "Line:     $line_no"
    echo "Command:  $command"
    echo "ExitCode: $exit_code"
    echo "================================================================"
    echo ""
}

# Bind the ERR signal: Triggers on_crash immediately if any command fails 
# and is not explicitly caught by an 'if' statement or '||' operator.
trap 'on_crash ${LINENO} "$BASH_COMMAND"' ERR

CASE_FILE="$1"
TARGET_CASE_NAME=${2:-""}
VLLM_PID=""
CLEANUP_DONE="false"

if [ -z "$CASE_FILE" ]; then
    echo "Usage: $0 <case.json> [TARGET_CASE_NAME]"
    exit 1
fi

# shellcheck disable=SC2317
cleanup() {
    local exit_code=$?

    # Only perform cleanup if NOT in Buildkite (Local only)
    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        return
    fi

    # Prevent multiple executions of the cleanup logic
    if [[ "${CLEANUP_DONE}" == "true" ]]; then
        return
    fi
    
    CLEANUP_DONE="true"

    # Only show cleanup info if exiting with an error or interrupted
    if [[ $exit_code -ne 0 ]]; then
        echo -e "\n[INFO] Running cleanup procedure (Exit code: $exit_code)..."
    fi

    if [[ -n "${VLLM_PID:-}" ]]; then
        # Check if the process is still running
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[INFO] Stopping vLLM server (PID: $VLLM_PID)..."
            # Send TERM to the process group (using negative PID) to ensure all children close
            kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
            
            # Wait up to 10 seconds for resources (HBM) to be released
            for _ in {1..10}; do
                if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
        fi
        
        # Force kill if still alive after timeout
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[WARN] Server not responding, force terminating..."
            kill -9 -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null
        fi
    fi
}

trap cleanup EXIT INT TERM

if [[ "${BUILDKITE:-false}" == "true" ]]; then
  # TODO: Re-enable and check for compatible versions if running accuracy or lm-eval tasks.
  # pip install evaluate==0.4.5 || true
  # pip install rouge-score==0.1.2 || true
  # # Install lm_eval with dependencies, version is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
  # pip install "lm-eval[api,math]>=0.4.9.2" || true

  # Set umask so that any newly created files/directories have 777/666 permissions by default.
  # This ensures that the host user can delete artifacts created by the docker root user.
  umask 000
fi

if ! command -v gcloud &> /dev/null; then
    echo "Warning: gcloud is not installed. Some dataset or generation config downloads from GCS may be skipped or fail."
    # We do not exit here anymore, to allow local runs without gcloud to proceed if datasets are already present or not needed.
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ARTIFACT_FOLDER is provided by Buildkite via environment variable. 
# Default to a local path relative to the script for local runs.
ARTIFACT_FOLDER="${ARTIFACT_FOLDER:-$SCRIPT_DIR/artifacts}"
LOG_FOLDER="$ARTIFACT_FOLDER/temp_logs"
VLLM_TORCH_PROFILER_DIR="$LOG_FOLDER/profile"
export ARTIFACT_FOLDER
export LOG_FOLDER
export VLLM_TORCH_PROFILER_DIR

report_and_exit() {
  local exit_code=${1:-0}
  local record_id="${RECORD_ID:-local}"
  local report_exit_code

  echo "--- Calling report_result.sh for RECORD_ID=${record_id}"
  bash "$SCRIPT_DIR/report_result.sh" "$record_id"
  report_exit_code=$?

  # Exit with the reporting script's failure code if it did not succeed.
  if [ "$report_exit_code" -ne 0 ]; then
    exit "$report_exit_code"
  fi

  # Exit with the originally provided exit code.
  exit "$exit_code"
}

echo "--- Preparing Local Artifacts Folder"
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"
mkdir -p "$VLLM_TORCH_PROFILER_DIR"

PYTHON_PARSER="$SCRIPT_DIR/parser_case.py"
# Evaluate the Python output to set variables in the current shell context
eval "$(python3 "$PYTHON_PARSER" "$CASE_FILE" "$TARGET_CASE_NAME")"

VLLM_LOG="$LOG_FOLDER/vllm_log.txt"
BM_LOG="$LOG_FOLDER/bm_log.txt"
BEST_BM_LOG="$LOG_FOLDER/best_bm_log.txt"
printf "[INFO] %-25s = %s\n" "VLLM_LOG" "$VLLM_LOG"
printf "[INFO] %-25s = %s\n" "BM_LOG" "$BM_LOG"
printf "[INFO] %-25s = %s\n" "ARTIFACT_FOLDER" "$ARTIFACT_FOLDER"

echo "model: $MODEL"

# Helper function to check if a value is in an array
contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Download Datasets
DATASET_DIR="$ARTIFACT_FOLDER/dataset"
mkdir -p "$DATASET_DIR"

DATASETS=("custom" "custom-token" "mmlu" "mlperf" "math500" "sharegpt")
# shellcheck disable=SC2153
if contains_element "$DATASET" "${DATASETS[@]}"; then
  if [[ -z "${GCS_BUCKET:-}" ]]; then
    echo "[INFO] GCS_BUCKET is not set. Skipping dataset download. Ensure datasets are present in $DATASET_DIR if needed."
  elif command -v gsutil &> /dev/null; then
    echo "Syncing dataset for $DATASET from gs://$GCS_BUCKET"
    case "$DATASET" in
      "custom-token")
        gsutil -m cp gs://"$GCS_BUCKET"/dataset/*.* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
      "mmlu")
        gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/mmlu/* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
      "mlperf")
        gsutil -m cp gs://"$GCS_BUCKET"/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DIR/mlperf.jsonl" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
      "math500")
        gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/math500/math500.jsonl "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
      "custom")
        gsutil -m cp -r gs://"$GCS_BUCKET"/bench-dataset/* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
      "sharegpt")
        gsutil -m cp -r gs://"$GCS_BUCKET"/sharegpt/* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
        ;;
    esac
  else
    echo "Warning: gsutil not found. Skipping dataset download from GCS."
  fi
fi

# Prep specialized configurations (DeepSeek)
if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  if command -v gsutil &> /dev/null; then
    echo "Syncing generation configs for DeepSeek-R1"
    GENERATION_CONFIG_FOLDER="$ARTIFACT_FOLDER/generation_configs"
    mkdir -p "$GENERATION_CONFIG_FOLDER"
    gsutil -m cp -r gs://gpolovets-inference/deepseek/generation_configs/* "$GENERATION_CONFIG_FOLDER" || echo "Warning: failed to sync generation configs ${DATASET}"
  else
    echo "Warning: gsutil not found. Skipping DeepSeek-R1 generation configs download from GCS."
  fi
fi

if [ "$COMMAND_TYPE" = "lm_eval" ]; then
  {
    ".buildkite/benchmark/lm_eval/$DATASET/run.sh"
    printf "AccuracyMetrics: "
    tr -d '\n' < "/workspace/${DATASET}_accuracy.json"
    echo ""
  } >> "$BM_LOG"
  echo "Finished running $DATASET benchmark."
  report_and_exit 0
fi

# For Sonnet
if [ "$DATASET" = "sonnet" ]; then
  echo "Create sonnet_4x.txt"
  echo "" > benchmarks/sonnet_4x.txt
  for _ in {1..4}
    do
     cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
  done
fi

#
# start vllm service in backend
#
echo "lanching vllm..."
echo "logging to $VLLM_LOG"
echo

# Command from parser case json
echo "[INFO] Starting vLLM Server in background..."

echo "Printing the vllm serve command used to start the server:"
printf "[DEBUG] Executing server_cmd: %s %s > \"%s\" 2>&1 &\n" "${SERVER_CMD_ENVS[*]}" "${SERVER_CMD[*]}" "$VLLM_LOG"

# Start the server and capture its PID
env "${SERVER_CMD_ENVS[@]}" "${SERVER_CMD[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Immediate check to see if it crashed on startup
sleep 2
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM Server failed to start immediately. Check log: $VLLM_LOG"
    exit 1
fi

# ---------------------------------------------------------
# Server startup wait logic
# ---------------------------------------------------------
SERVER_WAIT_MINS=${SERVER_WAIT_MINS:-60}

MAX_WAIT_SECONDS=$((SERVER_WAIT_MINS * 60))
WAIT_START_TIME=$(date +%s)
ELAPSED=0

echo "Waiting up to ${SERVER_WAIT_MINS} minutes for server to start (PID: ${VLLM_PID})..."

# Initial state set to not started
SERVER_STARTED="false"

# Loop continues as long as elapsed time is within the maximum allowed
while (( ELAPSED <= MAX_WAIT_SECONDS )); do
    
    # 1. [Fail-Fast Check] Ask the OS if the process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[ERROR] vLLM process (PID=$VLLM_PID) has exited unexpectedly!"
        echo "--- Dumping VLLM_LOG for debugging ---"
        cat "$VLLM_LOG"
        exit 1
    fi

    # 2. [Success Check] Look for the startup completion flag
    if grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started successfully."
        SERVER_STARTED="true"
        break
    fi

    # 3. Print progress approximately every 1 minute (every 6 iterations) to keep logs clean
    ITERATION=$((ELAPSED / 10))
    if (( ITERATION % 6 == 0 )); then
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        printf "Still waiting... Elapsed: %02d:%02d / %02d:00\n" "$ELAPSED_MIN" "$ELAPSED_SEC" "$SERVER_WAIT_MINS"
    fi

    # 4. Wait 10 seconds before the next check
    sleep 10

    # 5. Update elapsed time for the next while loop condition evaluation
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - WAIT_START_TIME))
done

# Direct exit if server is not started to prevent fetching the dirty bm result
if [[ "$SERVER_STARTED" == "false" ]]; then
    echo "[ERROR] Server failed to start within ${SERVER_WAIT_MINS} minutes! Timeout reached."
    echo "--- Dumping VLLM_LOG for debugging ---"
    cat "$VLLM_LOG"
    exit 1
fi

# Set Default
EXPECTED_ETEL=${EXPECTED_ETEL:-3600000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
PREFIX_LEN=${PREFIX_LEN:-0}

run_benchmark(){
  echo "running benchmark..."
  echo "logging to $BM_LOG"
  echo

  local request_rate=${1:-""}

  if [[ -n "$request_rate" ]]; then
    local found=false

    # Iterate through array indices to find and update the parameter
    for i in "${!CLIENT_CMD[@]}"; do
      if [[ "${CLIENT_CMD[$i]}" == "--request-rate" ]]; then
        # Update the next element (the value) for separated format: --flag value
        CLIENT_CMD[i+1]="$request_rate"
        found=true
        break
      elif [[ "${CLIENT_CMD[$i]}" == --request-rate=* ]]; then
        # Update the element itself for combined format: --flag=value
        CLIENT_CMD[i]="--request-rate=$request_rate"
        found=true
        break
      fi
    done

    # Append the flag and value as separate array elements if not found
    if [[ "$found" == false ]]; then
      CLIENT_CMD+=( "--request-rate" "$request_rate" )
    fi
  fi

  echo "[DEBUG] Executing client_cmd: ${CLIENT_CMD_ENVS[*]} ${CLIENT_CMD[*]} > $BM_LOG" >&2
  set +e
  # Execute the array directly, preserving strict argument boundaries
  env "${CLIENT_CMD_ENVS[@]}" "${CLIENT_CMD[@]}" > "$BM_LOG" 2>&1
  local client_exit_code=$?
  set -e

  if [ $client_exit_code -ne 0 ]; then
      return $client_exit_code
  fi

  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  echo "throughput: $throughput, P99 E2EL:$p99_e2el"
  echo "$throughput $p99_e2el"
}

if [ "${BUILDKITE:-false}" == "true" ]; then
  ENV_CONTEXT="Buildkite environment"
else
  ENV_CONTEXT="Local environment"
fi
printf "[DEBUG] Checking folder structure (Environment: %s)...\n" "$ENV_CONTEXT"
printf "[DEBUG] pwd=%s\n\nls $ARTIFACT_FOLDER=\n%s\n" "$(pwd)" "$(ls "$ARTIFACT_FOLDER")" || true
printf "[DEBUG] ls $ARTIFACT_FOLDER/temp_logs=\n%s\n" "$(ls "$ARTIFACT_FOLDER"/temp_logs)" || true

# ---------------------------------------------------------
# Helper Function: Safely execute benchmark and validate metrics
# ---------------------------------------------------------
# Define global variables for the main workflow to read
VALID_THROUGHPUT=""
VALID_P99_E2EL=""

execute_benchmark_safely() {
    local rate_arg="${1:-}" # Accept the request_rate argument; default to empty string if not provided
    local output
    local bm_exit_code

    # 1. Execute the benchmark and intercept the exit code from the subshell pipeline
    set +e  
    output=$(run_benchmark "$rate_arg" | tail -n 1)
    bm_exit_code=$?
    set -e
    if [[ "$bm_exit_code" -ne 0 ]]; then
        echo "[ERROR] Benchmark client crashed with exit code $bm_exit_code (rate=${rate_arg:-initial})!"
        echo "--- Dumping BM_LOG for debugging ---"
        cat "$BM_LOG"
        report_and_exit 1
    fi

    # 2. Parse the extracted string into respective variables safely
    local temp_throughput
    local temp_p99
    read -r temp_throughput temp_p99 <<< "$output"

    # 3. Validate that the extracted variables are strictly numerical (float or int)
    if ! [[ "$temp_throughput" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! [[ "$temp_p99" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] Failed to parse valid metrics (rate=${rate_arg:-initial})! Output was: '$output'"
        report_and_exit 1
    fi

    # 4. Validation passed, assign to global variables
    VALID_THROUGHPUT="$temp_throughput"
    VALID_P99_E2EL="$temp_p99"
}


# =========================================================
# Main Flow: Benchmark Starts
# =========================================================

# Step 1: Initial run
echo "Starting initial run..."
execute_benchmark_safely  # Call the helper without arguments
throughput="$VALID_THROUGHPUT"
p99_e2el="$VALID_P99_E2EL"

echo "throughput:$throughput"
echo "p99_e2el:$p99_e2el"

# Step 1.5: check if initial run meets the E2EL requirement
p99_int=$(printf "%.0f" "$p99_e2el")
goal_int=$(printf "%.0f" "$EXPECTED_ETEL")

if (( p99_int <= goal_int )); then
  echo "Initial run: P99 E2EL ($p99_e2el ms) <= EXPECTED_ETEL ($EXPECTED_ETEL ms), good enough. Exiting 0."
  report_and_exit 0
fi

echo "Initial run failed: P99 E2EL ($p99_e2el ms) > EXPECTED_ETEL ($EXPECTED_ETEL ms)"
echo "Starting binary search to lower request rate..."

# Step 2: Binary search
low=0
high=$(printf "%.0f" "$throughput")
goal=$EXPECTED_ETEL

# Round goal to nearest int
goal_int=$(printf "%.0f" "$goal")

best_rate=0
best_throughput=0
best_e2el=0

while (( high - low > 0 )); do
  mid=$(( (low + high + 1) / 2 ))
  echo "Trying request_rate=$mid"

  # Single function call with double-layer defense (exit code interception + regex validation)
  execute_benchmark_safely "$mid"
  throughput="$VALID_THROUGHPUT"
  p99_e2el="$VALID_P99_E2EL"

  # Convert p99_e2el to integer
  p99_int=$(printf "%.0f" "$p99_e2el")

  if (( p99_int <= goal_int )); then
    echo "PASS: p99_e2el=$p99_e2el <= $goal"
    best_rate=$mid
    best_throughput=$throughput
    best_e2el=$p99_e2el
    low=$mid

    # Backup best log
    cp "$BM_LOG" "$BEST_BM_LOG"
  else
    echo "FAIL: p99_e2el=$p99_e2el > $goal"
    high=$((mid - 1))
  fi
done

if (( best_rate == 0 )); then
  echo "Could not find a valid request_rate >= 1 that meets EXPECTED_ETEL=$EXPECTED_ETEL" | tee -a "$BM_LOG"
  report_and_exit 1
fi

# Restore the best log to BM_LOG
cp "$BEST_BM_LOG" "$BM_LOG"

echo
echo "======================================"
echo "✓ Final best request_rate: $best_rate"
echo "✓ Throughput: $best_throughput"
echo "✓ P99 E2EL: $best_e2el"
echo "======================================"

report_and_exit 0
