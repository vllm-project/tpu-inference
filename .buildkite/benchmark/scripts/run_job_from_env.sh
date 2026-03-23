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

# shellcheck disable=SC2317
function defer_cleanup_and_report() {
  # Capture the exit code of the last command executed before the trap was triggered.
  local exit_code=$?
  echo "--- Running defer (cleanup and report)"
  
  # Attempt to report the benchmark results to the database.
  # If the reporting script fails (returns non-zero), we log an error.
  if ! .buildkite/benchmark/scripts/report_result.sh "$RECORD_ID"; then
    echo "Error: report_result.sh failed!"
    
    # If the main job was successful (exit_code 0) but reporting failed,
    # we override the exit code to 1 so the Buildkite job is marked as failed.
    if [ "$exit_code" -eq 0 ]; then
      exit_code=1
    fi
  fi
  
  # Best-effort cleanup of Docker resources.
  # We use '|| true' to ensure that a cleanup failure doesn't overwrite our final exit code.
  .buildkite/benchmark/scripts/cleanup_docker.sh || true

  # Cleanup artifact log folder
  if [ -d "${ARTIFACT_FOLDER:-}" ]; then
    echo "--- Cleaning up artifact folder: $ARTIFACT_FOLDER"
    rm -rf "$ARTIFACT_FOLDER"
  fi
  
  # Exit with the final determined status code.
  exit "$exit_code"
}
trap defer_cleanup_and_report EXIT

echo "--- Verifying Submodule Commit"
git submodule status

echo "--- Registering Run to Spanner DB"
# Ensure all required envs are present. Some are provided by Buildkite directly or from the step.
# For missing optional fields we provide defaults.
EXPECTED_ETEL="${EXPECTED_ETEL:-3600000}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MODELTAG="${MODELTAG:-PROD}"
PREFIX_LEN="${PREFIX_LEN:-0}"
DATASET="${DATASET:-}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-"{}"}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Reconstruct EXTRA_ENVS for the database based on the environment variables
# that are present. This allows the YAML to remain clean while keeping the DB populated.
GENERATED_EXTRA_ENVS=""
EXTRA_ENV_KEYS=(
  "VLLM_MLA_DISABLE" "NEW_MODEL_DESIGN" "MOE_REQUANTIZE_BLOCK_SIZE" 
  "MOE_REQUANTIZE_WEIGHT_DTYPE" "TPU_BACKEND_TYPE" "MODEL_IMPL_TYPE"
  "USE_MOE_EP_KERNEL" "USE_BENCHMARK_SERVING" "MAX_CONCURRENCY"
)
for key in "${EXTRA_ENV_KEYS[@]}"; do
  # Check if the environment variable is set
  if [ -n "${!key:-}" ]; then
    GENERATED_EXTRA_ENVS+="${key}=${!key};"
  fi
done

# If EXTRA_ENVS was passed directly, append the generated ones to it
if [ -n "${EXTRA_ENVS:-}" ]; then
  EXTRA_ENVS="${EXTRA_ENVS};${GENERATED_EXTRA_ENVS}"
else
  EXTRA_ENVS="${GENERATED_EXTRA_ENVS}"
fi

# We will need CODE_HASH and JOB_REFERENCE from Buildkite env
CODE_HASH="${CODE_HASH:-$(buildkite-agent meta-data get 'CODE_HASH' || echo '')}"
JOB_REFERENCE="${JOB_REFERENCE:-${BUILDKITE_BUILD_URL:-}}"
RUN_TYPE="${RUN_TYPE:-DAILY}"

prepare_sql_val() {
  local val="$1"
  local default="$2"
  if [ -z "$val" ]; then
    echo "$default"
    return
  fi
  # Escape internal single quotes
  local escaped_val="${val//\'/\'\'}"
  echo "'$escaped_val'"
}

SQL_ADDITIONAL_CONFIG=$(prepare_sql_val "$ADDITIONAL_CONFIG" "'{}'")
SQL_EXTRA_ARGS=$(prepare_sql_val "$EXTRA_ARGS" "''")
SQL_EXTRA_ENVS=$(prepare_sql_val "$EXTRA_ENVS" "''")

echo "--- Querying existing TryCount for $RECORD_ID"
TRY_COUNT_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --format=json \
  --sql="SELECT TryCount FROM RunRecord WHERE RecordId='$RECORD_ID'" 2>/dev/null || echo '{"rows":[]}')

# Parse TryCount using jq. Returns null if empty.
TRY_COUNT=$(echo "$TRY_COUNT_JSON" | jq -r 'if .rows then (if (.rows | length) > 0 then .rows[0][0] else "null" end) else "null" end' || echo "null")

if [ "$TRY_COUNT" == "null" ] || [ -z "$TRY_COUNT" ]; then
  echo "--- Inserting new RunRecord (TryCount=1)"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, LastUpdate, CreatedBy, JobReference, RunBy,
      Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen,
      ExpectedETEL, NumPrompts, ModelTag, PrefixLen, ExtraEnvs,
      AdditionalConfig, ExtraArgs, TryCount
    ) VALUES (
      '$RECORD_ID', 'RUNNING', PENDING_COMMIT_TIMESTAMP(), PENDING_COMMIT_TIMESTAMP(), '${USER:-buildkite-agent}', '$JOB_REFERENCE', '${BUILDKITE_AGENT_NAME:-}',
      '$DEVICE', '$MODEL', '$RUN_TYPE', '$CODE_HASH',
      $MAX_NUM_SEQS,
      $MAX_NUM_BATCHED_TOKENS,
      $TENSOR_PARALLEL_SIZE,
      $MAX_MODEL_LEN,
      '$DATASET',
      $INPUT_LEN,
      $OUTPUT_LEN,
      $EXPECTED_ETEL,
      $NUM_PROMPTS,
      '$MODELTAG',
      $PREFIX_LEN,
      $SQL_EXTRA_ENVS,
      $SQL_ADDITIONAL_CONFIG,
      $SQL_EXTRA_ARGS,
      1
    );" || {
      echo "Warning: Failed to insert RunRecord for $RECORD_ID. Skipping DB registration."
  }
else
  NEW_TRY_COUNT=$((TRY_COUNT + 1))
  echo "--- Updating existing RunRecord (TryCount=$NEW_TRY_COUNT)"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="UPDATE RunRecord SET Status='RUNNING', LastUpdate=PENDING_COMMIT_TIMESTAMP(), TryCount=$NEW_TRY_COUNT, RunBy='${BUILDKITE_AGENT_NAME:-}' WHERE RecordId='$RECORD_ID'" || {
      echo "Warning: Failed to update RunRecord for $RECORD_ID. Skipping DB update."
  }
fi

echo "--- Preparing Config Environment"
ARTIFACT_FOLDER="$(pwd)/artifacts"
export ARTIFACT_FOLDER
LOG_FOLDER="${ARTIFACT_FOLDER}/temp_logs"
export LOG_FOLDER
mkdir -p "$ARTIFACT_FOLDER"
mkdir -p "$LOG_FOLDER"

# We safely dump the specific environment variables
ENV_FILE="artifacts/${RECORD_ID}.env"
cat <<EOF > "$ENV_FILE"
DEVICE='${DEVICE}'
MODEL='${MODEL}'
MAX_NUM_SEQS='${MAX_NUM_SEQS}'
MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}'
TENSOR_PARALLEL_SIZE='${TENSOR_PARALLEL_SIZE}'
MAX_MODEL_LEN='${MAX_MODEL_LEN}'
DATASET='${DATASET}'
INPUT_LEN='${INPUT_LEN}'
OUTPUT_LEN='${OUTPUT_LEN}'
EXPECTED_ETEL='${EXPECTED_ETEL}'
NUM_PROMPTS='${NUM_PROMPTS}'
MODELTAG='${MODELTAG}'
PREFIX_LEN='${PREFIX_LEN}'
ADDITIONAL_CONFIG='${ADDITIONAL_CONFIG}'
EXTRA_ARGS='${EXTRA_ARGS}'
CODE_HASH='${CODE_HASH}'
RECORD_ID='${RECORD_ID}'
GCP_REGION='${GCP_REGION:-}'
GCP_PROJECT_ID='${GCP_PROJECT_ID:-}'
ARTIFACT_REPO='${ARTIFACT_REPO:-}'
GCS_BUCKET='${GCS_BUCKET:-}'
SKIP_JAX_PRECOMPILE='${SKIP_JAX_PRECOMPILE:-}'
EOF

# Inject dynamic EXTRA_ENVS generated ones
for key in "${EXTRA_ENV_KEYS[@]}"; do
  if [ -n "${!key:-}" ]; then
    echo "${key}='${!key}'" >> "$ENV_FILE"
  fi
done

# Sync datasets (Copied from run_job.sh)
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

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

echo "--- Running job in docker via run_in_docker.sh"
BM_JOB_STATUS=$EXIT_SUCCESS

# Configuration docker parameter for execute benchmark
declare -a BENCHMARK_DOCKER_ARGS=(
  "-v" "$ARTIFACT_FOLDER:/workspace/artifacts"
  "-e" "DOCKER_ARTIFACT_FOLDER=/workspace/artifacts"
  "-e" "DOCKER_LOG_FOLDER=/workspace/artifacts/temp_logs"
  "--env-file" "$ENV_FILE"
)
# Join the array elements using newline as the delimiter and export as a single string.
BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' "${BENCHMARK_DOCKER_ARGS[@]}")"
export BENCHMARK_DOCKER_ARGS_STR

# Restore the SKIP_JAX_PRECOMPILE logic
[ -n "${SKIP_JAX_PRECOMPILE:-}" ] && export SKIP_JAX_PRECOMPILE

.buildkite/scripts/run_in_docker.sh bash -c "
  echo always > /sys/kernel/mm/transparent_hugepage/enabled && \
  chmod +x .buildkite/benchmark/scripts/run_bm.sh && \
  .buildkite/benchmark/scripts/run_bm.sh" || {
    echo "Error running benchmark job in docker."
    BM_JOB_STATUS=$EXIT_FAILURE
}

exit $BM_JOB_STATUS