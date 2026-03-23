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

ARTIFACT_FOLDER="$(pwd)/artifacts"
LOG_FOLDER="${ARTIFACT_FOLDER}/temp_logs"
export ARTIFACT_FOLDER
export LOG_FOLDER

cleanup_artifact_log() {
  echo "deleting artifacts: $ARTIFACT_FOLDER"
  rm -rf "$ARTIFACT_FOLDER"
}

trap cleanup_artifact_log EXIT

#
# Check input argument
#
if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
echo "Record ID: $RECORD_ID"

# Note: Removed the failure log upload feature. These logs are now accessible directly via the Buildkite interface.

if [ -z "$RECORD_ID" ] || [ "$RECORD_ID" == "null" ]; then
  echo "Invalid or missing record_id."
  exit 1
fi

echo "--- Verifying Submodule Commit"
git submodule status

# Query TryCount as JSON
echo "Query $RECORD_ID..."
TRY_COUNT_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --format=json \
  --sql="SELECT TryCount FROM RunRecord WHERE RecordId='$RECORD_ID'")

# Parse TryCount using jq, convert to number, default to 0 if missing
TRY_COUNT=$(echo "$TRY_COUNT_JSON" | jq -r '.rows[0][0] | tonumber // 0')

# Increment TryCount
NEW_TRY_COUNT=$((TRY_COUNT + 1))

# Update record
echo "Updating record $RECORD_ID: Status=RUNNING, TryCount=$NEW_TRY_COUNT, RunBy=$BUILDKITE_AGENT_NAME..."
gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="UPDATE RunRecord SET Status='RUNNING', LastUpdate=CURRENT_TIMESTAMP(), TryCount=$NEW_TRY_COUNT, RunBy='${BUILDKITE_AGENT_NAME}' WHERE RecordId='$RECORD_ID'"

# Do cleanup before create config
cleanup_artifact_log

#
# Prepare artifacts folder to save log and another need shared file, mount to docker container
#
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"

#
# Create running config
#
BM_JOB_STATUS=$EXIT_SUCCESS
echo "Creating running config..."
.buildkite/benchmark/scripts/create_config.sh "$RECORD_ID" || {
  echo "Error creating running config."
  BM_JOB_STATUS=$EXIT_FAILURE
}

: "${DATASET:=}"

#
# This makes GCS_BUCKET and other vars available to the whole script.
#
ENV_FILE="artifacts/${RECORD_ID}.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
else
  echo "Error: Config file $ENV_FILE not found after create_config.sh"
  BM_JOB_STATUS=$EXIT_FAILURE
fi

#
# Run job in docker
#
if (( BM_JOB_STATUS == EXIT_SUCCESS )); then
  echo "Code_Hash: $CODE_HASH"
  # shellcheck disable=SC2034
  IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH _ <<< "$CODE_HASH"

  # Prepare Dataset (Temp solution)
  DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500" "bench-custom-mm")
  if [[ " ${DATASETS[*]} " == *" $DATASET "* ]]; then
    echo "--- Syncing dataset for $DATASET"
    DATASET_DIR="./artifacts/dataset"
    mkdir -p "$DATASET_DIR"
    case "$DATASET" in
      "custom-token") gsutil -m cp gs://"$GCS_BUCKET"/dataset/*.* "$DATASET_DIR/" ;;
      "mmlu")         gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/mmlu/* "$DATASET_DIR/" ;;
      "mlperf")       gsutil -m cp gs://vllm-cb-storage2/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DIR/mlperf.jsonl" ;;
      "math500")      gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/math500/math500.jsonl "$DATASET_DIR/" ;;
      "bench-custom-token"|"bench-custom-mm") gsutil -m cp -r gs://"$GCS_BUCKET"/bench-dataset/* "$DATASET_DIR/" ;;
    esac
  fi

  # Prep specialized configurations (DeepSeek)
  if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
    GENERATION_CONFIG_FOLDER="$ARTIFACT_FOLDER/generation_configs"
    export GENERATION_CONFIG_FOLDER
    mkdir -p "$GENERATION_CONFIG_FOLDER"
    gsutil -m cp -r gs://gpolovets-inference/deepseek/generation_configs/* "$GENERATION_CONFIG_FOLDER"
  fi

  echo "Running job in docker via run_in_docker.sh..."

  # Configuration docker parameter for execute benchmark
  declare -a BENCHMARK_DOCKER_ARGS=(
    "-v" "$ARTIFACT_FOLDER:/workspace/artifacts"
    "-e" "DOCKER_ARTIFACT_FOLDER=/workspace/artifacts"
    "-e" "DOCKER_LOG_FOLDER=/workspace/artifacts/temp_logs"
    "--env-file" "$ENV_FILE"
  )
  # Join the array elements using newline as the delimiter and export as a single string.
  if [ ${#BENCHMARK_DOCKER_ARGS[@]} -gt 0 ]; then
    BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' "${BENCHMARK_DOCKER_ARGS[@]}")"
    export BENCHMARK_DOCKER_ARGS_STR
  else
    export BENCHMARK_DOCKER_ARGS_STR=""
  fi

  # Restore the SKIP_JAX_PRECOMPILE logic
  [ -n "${SKIP_JAX_PRECOMPILE:-}" ] && export SKIP_JAX_PRECOMPILE

  .buildkite/scripts/run_in_docker.sh bash -c "
    echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
    chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
    .buildkite/benchmark/scripts/run_bm.sh" || {
      echo "Error running benchmark job in docker."
      BM_JOB_STATUS=$EXIT_FAILURE
  }
else
  echo "Skipping benchmark job because BM_JOB_STATUS is non-zero ($BM_JOB_STATUS)."
fi

echo "Benchmark script completed."

#
# Report result
#
echo "Reporting result..."
.buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"

echo ".buildkite/benchmark/scripts/cleanup_docker.sh"
.buildkite/benchmark/scripts/cleanup_docker.sh

exit $BM_JOB_STATUS
