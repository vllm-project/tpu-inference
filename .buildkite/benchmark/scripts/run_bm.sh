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

set -euo pipefail

if ! command -v gcloud &> /dev/null; then
  echo "gcloud not found. Installing Google Cloud SDK..."
  apt-get update && apt-get install -y gnupg curl
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  apt-get update && apt-get install -y google-cloud-cli
fi

ARTIFACT_FOLDER="$(pwd)/artifacts"
LOG_FOLDER="$ARTIFACT_FOLDER/temp_logs"
PROFILE_FOLDER="$LOG_FOLDER/profile"
export ARTIFACT_FOLDER
export LOG_FOLDER
export PROFILE_FOLDER

cleanup_artifact_log() {
  echo "deleting artifacts: ${ARTIFACT_FOLDER}"
  rm -rf "${ARTIFACT_FOLDER}"
}

# If Running on Buildkite, do cleanup artifact
if [[ "${BUILDKITE:-false}" == "true" ]]; then
  trap cleanup_artifact_log EXIT
  cleanup_artifact_log
fi

echo "--- Preparing Local Artifacts Folder"
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"
mkdir -p "$PROFILE_FOLDER"


CASE_FILE="$1"
TARGET_CASE_NAME=${2:-""}

if [ -z "$CASE_FILE" ]; then
    echo "Usage: $0 <case.json> [TARGET_CASE_NAME]"
    exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PARSER="$DIR/parser_case.py"
# Evaluate the Python output to set variables in the current shell context
eval "$(python3 "$PYTHON_PARSER" "$CASE_FILE" "$TARGET_CASE_NAME")"

# TODO: Move to image building.
# Ingore the error because in case of using uv, the packages are installed outside this script.
pip install evaluate==0.4.5 || true
pip install rouge-score==0.1.2 || true
# Install lm_eval with dependencies, version is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
pip install "lm-eval[api,math]>=0.4.9.2" || true

VLLM_LOG="$LOG_FOLDER/vllm_log.txt"
BM_LOG="$LOG_FOLDER/bm_log.txt"
BEST_BM_LOG="$LOG_FOLDER/best_bm_log.txt"
VLLM_TORCH_PROFILER_DIR="$PROFILE_FOLDER"
export VLLM_TORCH_PROFILER_DIR
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

DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500" "bench-custom-mm")
# shellcheck disable=SC2153
if contains_element "$DATASET" "${DATASETS[@]}"; then
  echo "Syncing dataset for $DATASET"
  case "$DATASET" in
    "custom-token")
      gsutil -m cp gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/*.* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
      ;;
    "mmlu")
      gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/mmlu/* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
      ;;
    "mlperf")
      gsutil -m cp gs://vllm-cb-storage2/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DIR/mlperf.jsonl" || echo "Warning: failed to sync dataset ${DATASET}"
      ;;
    "math500")
      gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/math500/math500.jsonl "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
      ;;
    "bench-custom-token"|"bench-custom-mm")
      gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/bench-dataset/* "$DATASET_DIR/" || echo "Warning: failed to sync dataset ${DATASET}"
      ;;
  esac
fi

# Prep specialized configurations (DeepSeek)
if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  echo "Syncing generation configs for DeepSeek-R1"
  GENERATION_CONFIG_FOLDER="$ARTIFACT_FOLDER/generation_configs"
  mkdir -p "$GENERATION_CONFIG_FOLDER"
  gsutil -m cp -r gs://gpolovets-inference/deepseek/generation_configs/* "$GENERATION_CONFIG_FOLDER" || echo "Warning: failed to sync generation configs ${DATASET}"
fi

if [ "$RUN_TYPE" = "lm_eval" ]; then
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
printf "[DEBUG] Executing server_cmd: %s\n" "${SERVER_CMD[*]} > \"$VLLM_LOG\" 2>&1 &"

"${SERVER_CMD[@]}" > "$VLLM_LOG" 2>&1 &
echo "wait for 60 minutes.."
echo
for _ in {1..360}; do
    # TODO: detect other type of errors.
    if grep -Fq "raise RuntimeError" "$VLLM_LOG"; then
        echo "Detected RuntimeError, exiting."
        cat "$VLLM_LOG"
        exit 1
    elif grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

# Set Default
EXPECTED_ETEL=${EXPECTED_ETEL:-3600000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
PREFIX_LEN=${PREFIX_LEN:-0}

run_benchmark(){
  echo "running benchmark..."
  echo "logging to $BM_LOG"
  echo

  local request_rate=$1

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

  printf "[DEBUG] Executing client_cmd: %s\n" "${CLIENT_CMD[*]} > \"$BM_LOG\" 2>&1"
  # Execute the array directly, preserving strict argument boundaries
  "${CLIENT_CMD[@]}" > "$BM_LOG" 2>&1

  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  echo "throughput: $throughput, P99 E2EL:$p99_e2el"
  echo "$throughput $p99_e2el"
}

printf "[DEBUG] Checking folder structure in container...\n"
printf "[DEBUG] pwd=%s\n\nls $ARTIFACT_FOLDER=\n%s\n" "$(pwd)" "$(ls "$ARTIFACT_FOLDER")" || true
printf "[DEBUG] ls $ARTIFACT_FOLDER/temp_logs=\n%s\n" "$(ls "$ARTIFACT_FOLDER"/temp_logs)" || true

# request_rate use default value (inf)
read -r throughput p99_e2el < <(run_benchmark | tail -n 1)

echo "throughput:$throughput"
echo "p99_e2el:$p99_e2el"

# Step 1: check if initial run meets the E2EL requirement
p99_int=$(printf "%.0f" "$p99_e2el")
goal_int=$(printf "%.0f" "$EXPECTED_ETEL")

if (( p99_int <= goal_int )); then
  echo "Initial run: P99 E2EL ($p99_e2el ms) <= EXPECTED_ETEL ($EXPECTED_ETEL ms), good enough. Exiting 0."
  report_and_exit 0
fi

echo "Initial run failed: P99 E2EL ($p99_e2el ms) > EXPECTED_ETEL ($EXPECTED_ETEL ms)"
echo "Starting binary search to lower request rate..."

# Step 2: Begin binary search
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

  read -r throughput p99_e2el < <(run_benchmark "$mid" | tail -n 1)

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

echo "--- Calling report_result.sh for RECORD_ID=${RECORD_ID}"
bash .buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"
