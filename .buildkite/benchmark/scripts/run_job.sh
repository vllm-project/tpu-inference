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
TARGET_CASE_NAME=${2:-""}

if [ -z "$CASE_FILE" ]; then
    echo "Usage: $0 <case.json> [TARGET_CASE_NAME]"
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
)

BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' "${BENCHMARK_DOCKER_ARGS[@]}")"
export BENCHMARK_DOCKER_ARGS_STR

echo "--- Running job in docker via run_in_docker.sh"
BM_JOB_STATUS=$EXIT_SUCCESS

export BM_INFRA="true"

.buildkite/scripts/run_in_docker.sh bash -c "
  echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
  chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
  .buildkite/benchmark/scripts/run_bm.sh $CASE_FILE $TARGET_CASE_NAME" || {
    echo "Error running benchmark job in docker."
    BM_JOB_STATUS=$EXIT_FAILURE
}


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
