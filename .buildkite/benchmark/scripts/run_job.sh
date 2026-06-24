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

CASE_FILE="$1"
TARGET_CASE_NAME="$2"

if [ -z "$CASE_FILE" ] || [ -z "$TARGET_CASE_NAME" ]; then
    echo "Usage: $0 <case.json> <TARGET_CASE_NAME>"
    exit 1
fi

echo "--- Generating Record ID"
: "${BUILDKITE_STEP_ID:?[ERROR] The BUILDKITE_STEP_ID variable is missing or empty!}"
# Use Buildkite step ID to ensure retries map to the same RecordId
RECORD_ID="${BUILDKITE_STEP_ID}"
export RECORD_ID
echo "--- Prepare benchmark Record ID: ${RECORD_ID}"

CODE_HASH=$(buildkite-agent meta-data get "CODE_HASH")
export CODE_HASH
JOB_REFERENCE=$(buildkite-agent meta-data get "JOB_REFERENCE")
export JOB_REFERENCE
export RUN_TYPE="${RUN_TYPE:-DAILY}"

# Dynamically update the Buildkite step label
BK_RECORD_ID_LABEL=" RecordId: ${RECORD_ID}"
buildkite-agent step update "label" "$BK_RECORD_ID_LABEL" --append

echo "--- Verifying Submodule Commit"
git submodule status

# Device value convert to db value
# Dynamic mapping for TPU version and chip count
# This regex matches patterns like 'tpu_v7x_8_queue' or 'tpu_v6e_16_queue'
if [[ "$BUILDKITE_AGENT_META_DATA_QUEUE" =~ ^tpu_v(7x|6e)_([0-9]+)_queue$ ]]; then
    # Extract hardware version (7x or 6e) and chip count from regex capture groups
    VERSION="${BASH_REMATCH[1]}"
    COUNT="${BASH_REMATCH[2]}"
    
    if [ "$VERSION" == "7x" ]; then
        # Map v7x patterns to "tpu7x-N"
        DEVICE="tpu7x-$COUNT"
    else
        # Map v6e patterns to "v6e-N"
        DEVICE="v6e-$COUNT"
    fi
elif [ "$BUILDKITE_AGENT_META_DATA_QUEUE" == "tpu_v6e_queue" ]; then
    DEVICE="v6e-1"
else
    DEVICE="$BUILDKITE_AGENT_META_DATA_QUEUE"
fi

echo "[INFO] Dynamic mapping complete: $BUILDKITE_AGENT_META_DATA_QUEUE -> $DEVICE"

echo "--- Configuring Docker Arguments for benchmark"

ARTIFACT_FOLDER="$(pwd)/artifacts"
LOG_FOLDER="${ARTIFACT_FOLDER}/temp_logs"
PROFILE_FOLDER="${LOG_FOLDER}/profile"

# Do cleanup before create config
cleanup_artifact_log() {
  echo "deleting artifacts: $ARTIFACT_FOLDER"
  rm -rf "$ARTIFACT_FOLDER"
}
cleanup_artifact_log

echo "--- Preparing Local Artifacts Folder"
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"
mkdir -p "$PROFILE_FOLDER"
trap cleanup_artifact_log EXIT

# Prepare environment variables for the Docker container.
declare -a BENCHMARK_DOCKER_ARGS=(
  "-v" "/dev/shm:/dev/shm"
  "-v" "/etc/boto.cfg:/etc/boto.cfg"
  "-v" "$ARTIFACT_FOLDER:/workspace/tpu_inference/artifacts"
  "-e" "ARTIFACT_FOLDER=/workspace/tpu_inference/artifacts"
  "-e" "DEVICE=$DEVICE"
  "-e" "RECORD_ID=$RECORD_ID"
  "-e" "RUN_TYPE=$RUN_TYPE"
  "-e" "CODE_HASH=${CODE_HASH}"
  "-e" "JOB_REFERENCE=${JOB_REFERENCE}"
  "-e" "BUILDKITE=${BUILDKITE}"
  "-e" "BUILDKITE_AGENT_NAME=${BUILDKITE_AGENT_NAME}"
  "-e" "BUILDKITE_AGENT_META_DATA_QUEUE=${BUILDKITE_AGENT_META_DATA_QUEUE}"
  "-e" "BUILDKITE_BUILD_NUMBER=${BUILDKITE_BUILD_NUMBER}"
  "-e" "BUILDKITE_JOB_ID=${BUILDKITE_JOB_ID}"
  "-e" "UPLOAD_DB=${UPLOAD_DB:-true}"
  "-e" "MLCOMPASS_EXECUTION_MODE=${MLCOMPASS_EXECUTION_MODE:-}"
  "-e" "MLCOMPASS_EXPORT_ENABLED=${MLCOMPASS_EXPORT_ENABLED:-}"
  "-e" "MLCOMPASS_TEST_NAME=${MLCOMPASS_TEST_NAME:-}"
  "-e" "MLCOMPASS_TRACKING_ID=${MLCOMPASS_TRACKING_ID:-}"
  "-e" "MLCOMPASS_SPONGE_ID=${MLCOMPASS_SPONGE_ID:-}"
)

BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' "${BENCHMARK_DOCKER_ARGS[@]}")"
export BENCHMARK_DOCKER_ARGS_STR

# Determine if it is a multi-host run.
IS_MULTI_HOST="false"
if [[ "${VERSION:-}" == "7x" && ${COUNT:-0} -gt 4 ]]; then
    IS_MULTI_HOST="true"
fi
if [[ -n "${WORKER_IPS:-}" || "${TPU_MULTIHOST_BACKEND:-}" == "ray" ]]; then
    IS_MULTI_HOST="true"
fi

BM_JOB_STATUS=$EXIT_SUCCESS
export BM_INFRA="true"

if [[ "$IS_MULTI_HOST" == "true" ]]; then
    echo "--- Multi-host environment detected. Running via run_multihost.sh on host..."

    # Export Docker environment variables as EXTRA_DOCKER_ARGS
    export EXTRA_DOCKER_ARGS="-v $ARTIFACT_FOLDER:/workspace/tpu_inference/artifacts \
      -e DEVICE=$DEVICE \
      -e RECORD_ID=$RECORD_ID \
      -e RUN_TYPE=$RUN_TYPE \
      -e CODE_HASH=${CODE_HASH} \
      -e JOB_REFERENCE=${JOB_REFERENCE} \
      -e GCP_PROJECT_ID=${GCP_PROJECT_ID:-} \
      -e GCP_INSTANCE_ID=${GCP_INSTANCE_ID:-} \
      -e GCP_DATABASE_ID=${GCP_DATABASE_ID:-} \
      -e GCP_REGION=${GCP_REGION:-} \
      -e GCS_BUCKET=${GCS_BUCKET:-} \
      -e UPLOAD_DB=${UPLOAD_DB:-true}"

    # Re-use parser_case.py to resolve the server command for this case on host.
    PYTHON_PARSER="$(pwd)/.buildkite/benchmark/scripts/parser_case.py"
    eval "$(python3 "$PYTHON_PARSER" "$CASE_FILE" "$TARGET_CASE_NAME")"

    # Convert SERVER_CMD array to a single string for run_multihost.sh.
    FULL_SERVER_CMD=""
    for env_item in "${SERVER_CMD_ENVS[@]}"; do
        FULL_SERVER_CMD+="$(printf '%q ' "$env_item")"
    done
    for cmd_item in "${SERVER_CMD[@]}"; do
        FULL_SERVER_CMD+="$(printf '%q ' "$cmd_item")"
    done

    # The client command will execute run_bm.sh inside the container
    CLIENT_CMD="SERVER_ALREADY_RUNNING=true ARTIFACT_FOLDER=/workspace/tpu_inference/artifacts bash /workspace/tpu_inference/.buildkite/benchmark/scripts/run_bm.sh '$CASE_FILE' '$TARGET_CASE_NAME'"

    echo "Executing run_multihost.sh on host..."
    .buildkite/scripts/run_multihost.sh "$FULL_SERVER_CMD" "$CLIENT_CMD" || {
        echo "Error running multihost benchmark."
        BM_JOB_STATUS=$EXIT_FAILURE
    }
else
    echo "--- Running job in docker via run_in_docker.sh (Single-Host Mode)"
    .buildkite/scripts/run_in_docker.sh bash -c "
      echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
      chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
      .buildkite/benchmark/scripts/run_bm.sh $CASE_FILE $TARGET_CASE_NAME" || {
        echo "Error running benchmark job in docker."
        BM_JOB_STATUS=$EXIT_FAILURE
    }
fi


(
  # Handle log file
  VLLM_LOG="$LOG_FOLDER/vllm_log.txt"
  BM_LOG="$LOG_FOLDER/bm_log.txt"

  # Upload vllm and bm log to Buildkite aritfact
  ARTIFACT_VLLM="${RECORD_ID}_vllm_log.txt"
  ARTIFACT_BM="${RECORD_ID}_bm_log.txt"

  # Re-enable set -e inside the subshell because it is disabled by the || operator
  set -e

  echo "Preparing Buildkite artifacts..."
  if [ -f "$VLLM_LOG" ]; then
    cp "$VLLM_LOG" "$ARTIFACT_VLLM"
    buildkite-agent artifact upload "$ARTIFACT_VLLM"
    rm -f "$ARTIFACT_VLLM"
  else
    echo "Warning: $VLLM_LOG not found, skipping upload."
  fi

  if [ -f "$BM_LOG" ]; then
    cp "$BM_LOG" "$ARTIFACT_BM"
    buildkite-agent artifact upload "$ARTIFACT_BM"
    rm -f "$ARTIFACT_BM"
  else
    echo "Warning: $BM_LOG not found, skipping upload."
  fi
) || {
  echo "Error uploading artifacts to Buildkite."
  BM_JOB_STATUS=$EXIT_FAILURE
}

exit $BM_JOB_STATUS
