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

readonly EXIT_SUCCESS=0
readonly EXIT_FAILURE=1

echo "--- Generating Record ID"
: "${BUILDKITE_STEP_ID:?[ERROR] The BUILDKITE_STEP_ID variable is missing or empty!}"
# Use Buildkite step ID to ensure retries map to the same RecordId
RECORD_ID="${BUILDKITE_STEP_ID}"
export RECORD_ID

: "${MODEL:?Error: Environment variable MODEL is strictly required but not set. Exiting.}"
export MODEL

echo "--- Prepare benchmark Record ID: ${RECORD_ID}, Model: ${MODEL}"

# --- Prepare Default Configuration ---
export EXPECTED_ETEL="${EXPECTED_ETEL:-3600000}"
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MODELTAG="${MODELTAG:-PROD}"
export PREFIX_LEN="${PREFIX_LEN:-0}"

CODE_HASH=$(buildkite-agent meta-data get "CODE_HASH")
export CODE_HASH
JOB_REFERENCE=$(buildkite-agent meta-data get "JOB_REFERENCE")
export JOB_REFERENCE
export RUN_TYPE="${RUN_TYPE:-DAILY}"

# Dynamically update the Buildkite step label
BK_RECORD_ID_LABEL=" RecordId: ${RECORD_ID}"
buildkite-agent step update "label" "$BK_RECORD_ID_LABEL" --append

ARTIFACT_FOLDER="$(pwd)/artifacts"
LOG_FOLDER="${ARTIFACT_FOLDER}/temp_logs"
PROFILE_FOLDER="${LOG_FOLDER}/profile"
export ARTIFACT_FOLDER
export LOG_FOLDER
export PROFILE_FOLDER

cleanup_artifact_log() {
  echo "deleting artifacts: $ARTIFACT_FOLDER"
  rm -rf "$ARTIFACT_FOLDER"
}
trap cleanup_artifact_log EXIT

echo "--- Verifying Submodule Commit"
git submodule status

# Do cleanup before create config
cleanup_artifact_log
echo "--- Preparing Local Artifacts Folder"
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"
mkdir -p "$PROFILE_FOLDER"

echo "--- Configuring Docker Arguments for benchmark"
# Prepare environment variables for the Docker container.
declare -a BENCHMARK_DOCKER_ARGS=(
  "-v" "$ARTIFACT_FOLDER:/workspace/artifacts"
  "-v" "/dev/shm:/dev/shm"
  "-e" "DOCKER_ARTIFACT_FOLDER=/workspace/artifacts"
  "-e" "DOCKER_LOG_FOLDER=/workspace/artifacts/temp_logs"
  "-e" "RECORD_ID=$RECORD_ID"
  "-e" "DEVICE=$DEVICE"
  "-e" "MODEL=$MODEL"
  "-e" "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  "-e" "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  "-e" "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
  "-e" "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  "-e" "DATASET=$DATASET"
  "-e" "INPUT_LEN=$INPUT_LEN"
  "-e" "OUTPUT_LEN=$OUTPUT_LEN"
  "-e" "EXPECTED_ETEL=$EXPECTED_ETEL"
  "-e" "NUM_PROMPTS=$NUM_PROMPTS"
  "-e" "MODELTAG=$MODELTAG"
  "-e" "PREFIX_LEN=$PREFIX_LEN"
  "-e" "ADDITIONAL_CONFIG=$ADDITIONAL_CONFIG"
  "-e" "EXTRA_ARGS=$EXTRA_ARGS"
)

if [ -n "${EXTRA_ENVS:-}" ]; then
  echo "--- Parsing EXTRA_ENVS into Docker arguments"

  # Strip leading and trailing single or double quotes
  CLEANED_EXTRA_ENVS=$(printf "%s\n" "$EXTRA_ENVS" | sed -e "s/^'//" -e "s/'$//" -e 's/^"//' -e 's/"$//')
  
  IFS=';' read -ra ENV_PAIRS <<< "$CLEANED_EXTRA_ENVS"
  
  for pair in "${ENV_PAIRS[@]}"; do
    if [ -n "$pair" ]; then
      BENCHMARK_DOCKER_ARGS+=("-e" "$pair")
    fi
  done
fi

BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' "${BENCHMARK_DOCKER_ARGS[@]}")"
export BENCHMARK_DOCKER_ARGS_STR

# Sync datasets (Copied from original logic)
DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500" "bench-custom-mm")
if [[ " ${DATASETS[*]} " == *" $DATASET "* ]]; then
  echo "--- Syncing dataset for $DATASET"
  DATASET_DIR="$ARTIFACT_FOLDER/dataset"
  mkdir -p "$DATASET_DIR"
  case "$DATASET" in
    "custom-token") gsutil -m cp gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/*.* "$DATASET_DIR/" ;;
    "mmlu")         gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/mmlu/* "$DATASET_DIR/" ;;
    "mlperf")       gsutil -m cp gs://vllm-cb-storage2/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DIR/mlperf.jsonl" ;;
    "math500")      gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/dataset/math500/math500.jsonl "$DATASET_DIR/" ;;
    "bench-custom-token"|"bench-custom-mm") gsutil -m cp -r gs://"${GCS_BUCKET:-vllm-cb-storage2}"/bench-dataset/* "$DATASET_DIR/" ;;
  esac
fi

# Prep specialized configurations (DeepSeek)
if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  GENERATION_CONFIG_FOLDER="$ARTIFACT_FOLDER/generation_configs"
  export GENERATION_CONFIG_FOLDER
  mkdir -p "$GENERATION_CONFIG_FOLDER"
  gsutil -m cp -r gs://gpolovets-inference/deepseek/generation_configs/* "$GENERATION_CONFIG_FOLDER"
fi

echo "--- Running job in docker via run_in_docker.sh"
BM_JOB_STATUS=$EXIT_SUCCESS

.buildkite/scripts/run_in_docker.sh bash -c "
  echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
  chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
  .buildkite/benchmark/scripts/run_bm.sh" || {
    echo "Error running benchmark job in docker."
    BM_JOB_STATUS=$EXIT_FAILURE
}

# report_result.sh determines whether to Insert or Update data by checking if the same RecordId already exists in the DB
echo "--- Reporting result"
.buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"

exit $BM_JOB_STATUS
