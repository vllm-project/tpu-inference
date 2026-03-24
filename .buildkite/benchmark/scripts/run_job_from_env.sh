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
if [ -n "${BUILDKITE_STEP_ID:-}" ]; then
  # Use Buildkite step ID to ensure retries map to the same RecordId
  export RECORD_ID="bk-${BUILDKITE_STEP_ID}"
else
  export RECORD_ID
  RECORD_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
fi
echo "Record ID: $RECORD_ID"

# Function to handle cleanup and final reporting to the database
# shellcheck disable=SC2317
function defer_cleanup_and_report() {
  local exit_code=$?
  echo "--- Running defer (cleanup and report)"
  
  # Call report_result.sh which now handles the "Select -> Insert/Update" logic.
  # All exported variables are inherited by this sub-script.
  if ! .buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"; then
    echo "Error: report_result.sh failed!"
    if [ "$exit_code" -eq 0 ]; then
      exit_code=1
    fi
  fi
  
  # Best-effort cleanup of Docker resources.
  .buildkite/benchmark/scripts/cleanup_docker.sh || true

  # Cleanup artifact log folder
  if [ -d "${ARTIFACT_FOLDER:-}" ]; then
    echo "--- Cleaning up artifact folder: $ARTIFACT_FOLDER"
    rm -rf "$ARTIFACT_FOLDER"
  fi
  
  exit "$exit_code"
}
trap defer_cleanup_and_report EXIT

echo "--- Verifying Submodule Commit"
git submodule status

# --- Prepare Configuration Metadata (Exported for report_result.sh) ---
export EXPECTED_ETEL="${EXPECTED_ETEL:-3600000}"
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MODELTAG="${MODELTAG:-PROD}"
export PREFIX_LEN="${PREFIX_LEN:-0}"
export DATASET="${DATASET:-}"
export ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-"{}"}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

# Reconstruct EXTRA_ENVS string for the database registration
GENERATED_EXTRA_ENVS=""
EXTRA_ENV_KEYS=(
  "VLLM_MLA_DISABLE" "NEW_MODEL_DESIGN" "MOE_REQUANTIZE_BLOCK_SIZE" 
  "MOE_REQUANTIZE_WEIGHT_DTYPE" "TPU_BACKEND_TYPE" "MODEL_IMPL_TYPE"
  "USE_MOE_EP_KERNEL" "USE_BENCHMARK_SERVING" "MAX_CONCURRENCY"
)
for key in "${EXTRA_ENV_KEYS[@]}"; do
  if [ -n "${!key:-}" ]; then
    GENERATED_EXTRA_ENVS+="${key}=${!key};"
  fi
done
export EXTRA_ENVS="${EXTRA_ENVS:-};${GENERATED_EXTRA_ENVS}"

export CODE_HASH="${CODE_HASH:-$(buildkite-agent meta-data get 'CODE_HASH' || echo '')}"
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

# Inject dynamic EXTRA_ENVS into the Docker container
for key in "${EXTRA_ENV_KEYS[@]}"; do
  if [ -n "${!key:-}" ]; then
    BENCHMARK_DOCKER_ARGS+=("-e" "${key}=${!key}")
  fi
done

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

[ -n "${SKIP_JAX_PRECOMPILE:-}" ] && export SKIP_JAX_PRECOMPILE

.buildkite/scripts/run_in_docker.sh bash -c "
  echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
  chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
  .buildkite/benchmark/scripts/run_bm.sh" || {
    echo "Error running benchmark job in docker."
    BM_JOB_STATUS=$EXIT_FAILURE
}

exit $BM_JOB_STATUS