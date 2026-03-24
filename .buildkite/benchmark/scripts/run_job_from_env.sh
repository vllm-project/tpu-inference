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

# shellcheck disable=SC2317
cleanup_artifact_log() {
  echo "deleting artifacts: $ARTIFACT_FOLDER"
  rm -rf "$ARTIFACT_FOLDER"
}

# shellcheck disable=SC2317
function defer_cleanup_and_report() {
  # Capture the exit code of the last command executed before the trap was triggered.
  # If the task is canceled too quickly, cleanup might not finish completely. Do not rely on it.
  local exit_code=$?
  echo "--- Running defer (cleanup and report)"
  
  # Best-effort cleanup of Docker resources.
  # We use '|| true' to ensure that a cleanup failure doesn't overwrite our final exit code.
  .buildkite/benchmark/scripts/cleanup_docker.sh || true
  cleanup_artifact_log || true

  # Exit with the final determined status code.
  exit "$exit_code"
}
trap defer_cleanup_and_report EXIT

echo "--- Generating Record ID"
if [ -n "${BUILDKITE_STEP_ID:-}" ]; then
  # Use Buildkite step ID to ensure retries map to the same RecordId
  export RECORD_ID="bk-${BUILDKITE_STEP_ID}"
else
  export RECORD_ID
  RECORD_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
fi
echo "Record ID: $RECORD_ID"


echo "--- Verifying Submodule Commit"
git submodule status

# --- Prepare Configuration Metadata (Exported for report_result.sh) ---
export EXPECTED_ETEL="${EXPECTED_ETEL:-3600000}"
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MODELTAG="${MODELTAG:-PROD}"
export PREFIX_LEN="${PREFIX_LEN:-0}"
export DATASET="${DATASET:-}"
export ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"
export EXTRA_ENVS="${EXTRA_ENVS:-}"

export CODE_HASH=$(buildkite-agent meta-data get "CODE_HASH")
export JOB_REFERENCE=$(buildkite-agent meta-data get "JOB_REFERENCE")
export RUN_TYPE="${RUN_TYPE:-DAILY}"
export DEVICE="${DEVICE:-}"
export MODEL="${MODEL:-}"

# Numeric values exported for SQL insertion
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-NULL}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-NULL}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-NULL}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-NULL}"
export INPUT_LEN="${INPUT_LEN:-NULL}"
export OUTPUT_LEN="${OUTPUT_LEN:-NULL}"

echo "--- Preparing Local Artifacts Folder"
ARTIFACT_FOLDER="$(pwd)/artifacts"
export ARTIFACT_FOLDER
LOG_FOLDER="${ARTIFACT_FOLDER}/temp_logs"
export LOG_FOLDER
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"

echo "--- Configuring Docker Arguments"
# Prepare environment variables for the Docker container. 
declare -a BENCHMARK_DOCKER_ARGS=(
  "-v" "$ARTIFACT_FOLDER:/workspace/artifacts"
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
  "-e" "CODE_HASH=$CODE_HASH"
  "-e" "RUN_TYPE=$RUN_TYPE"
  "-e" "GCP_REGION=${GCP_REGION:-}"
  "-e" "GCP_PROJECT_ID=${GCP_PROJECT_ID:-}"
  "-e" "ARTIFACT_REPO=${ARTIFACT_REPO:-}"
  "-e" "GCS_BUCKET=${GCS_BUCKET:-}"
  "-e" "SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-}"
)

if [ -n "${EXTRA_ENVS:-}" ]; then
  echo "--- Parsing EXTRA_ENVS into Docker arguments"
  
  IFS=';' read -ra ENV_PAIRS <<< "$EXTRA_ENVS"
  
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
  DATASET_DIR="./artifacts/dataset"
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

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

echo "--- Running job in docker via run_in_docker.sh"
BM_JOB_STATUS=$EXIT_SUCCESS

.buildkite/scripts/run_in_docker.sh bash -c "
  echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
  chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
  .buildkite/benchmark/scripts/run_bm.sh" || {
    echo "Error running benchmark job in docker."
    BM_JOB_STATUS=$EXIT_FAILURE
}

echo "Benchmark script completed."

#
# Report result
#
echo "Reporting result..."
.buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"

exit $BM_JOB_STATUS
