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

# === Usage ===
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input.csv|gs://path/to/input.csv> VLLM_COMMIT_HASH TPU_INFERENCE_HASH"
  exit 1
fi

CSV_FILE_ARG="$1"
VLLM_COMMIT_HASH="$2"
TPU_INFERENCE_HASH="$3"
CODE_HASH="${VLLM_COMMIT_HASH}-${TPU_INFERENCE_HASH}-"

# Config
TIMEZONE="America/Los_Angeles"
JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
RUN_TYPE="HOURLY_JAX"

# ./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv "" $TAG HOURLY_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax"
# ./scripts/scheduler/create_job.sh ./cases/hourly_jax_new.csv "" $TAG HOURLY_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True"
declare -A EXTRA_ENVS_MAP=(
  ["hourly_jax.csv"]="TPU_BACKEND_TYPE=jax"
  ["hourly_jax_dev.csv"]="TPU_BACKEND_TYPE=jax"
  ["hourly_jax_new.csv"]="TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True"
)

if [[ "$CSV_FILE_ARG" == gs://* ]]; then
  echo "GCS path detected. Downloading from $CSV_FILE_ARG"
  CSV_FILE=$(mktemp)
  if ! gsutil cp "$CSV_FILE_ARG" "$CSV_FILE"; then
    echo "Failed to download from GCS: $CSV_FILE_ARG"
    rm "$CSV_FILE"
    exit 1
  fi
  # Schedule cleanup of the temporary file on exit
  trap 'rm -f "$CSV_FILE"' EXIT
else
  CSV_FILE="$CSV_FILE_ARG"
fi

if [ ! -f "$CSV_FILE" ]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

# milliseconds: one hour
VERY_LARGE_EXPECTED_ETEL=3600000

# === Config ===
# Make sure these environment variables are set or export here
# export GCP_PROJECT_ID="your-project"
# export GCP_INSTANCE_ID="your-instance"
# export GCP_DATABASE_ID="your-database"

CSV_FILE_NAME=${basename "$CSV_FILE"}

if [[ -v EXTRA_ENVS_MAP["$CSV_FILE_NAME"] ]]; then
    EXTRA_ENVS=${EXTRA_ENVS_MAP["$CSV_FILE_NAME"]}
    echo "Key exists! Value: ${EXTRA_ENVS_MAP[$CSV_FILE_NAME]}"
else
    echo "Key '$CSV_FILE_NAME' not found in MAP."
    EXTRA_ENVS=""
fi

# === Read CSV and skip header ===
tail -n +2 "$CSV_FILE" | while read -r line || [ -n "${line}" ]; do

  line=$(echo "$line" | tr -d '\r')
    # Safely split CSV line into variables
  IFS=',' read -r \
    DEVICE \
    MODEL \
    MAX_NUM_SEQS \
    MAX_NUM_BATCHED_TOKENS \
    TENSOR_PARALLEL_SIZE \
    MAX_MODEL_LEN \
    DATASET \
    INPUT_LEN \
    OUTPUT_LEN \
    EXPECTED_ETEL \
    NUM_PROMPTS \
    MODELTAG \
    PREFIX_LEN <<< "$line"

  RECORD_ID=$(uuidgen | tr 'A-Z' 'a-z')

  # ** TODO: Using DEVICE to depart in which buildkite queue
  
  echo "Inserting Run: $RECORD_ID"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen, LastUpdate, CreatedBy,JobReference, ExpectedETEL, NumPrompts, ModelTag, PrefixLen, ExtraEnvs
    ) VALUES (
      '$RECORD_ID', 'CREATED', PENDING_COMMIT_TIMESTAMP(), '$DEVICE', '$MODEL', '$RUN_TYPE', '$CODE_HASH',
      $MAX_NUM_SEQS,
      $MAX_NUM_BATCHED_TOKENS,
      $TENSOR_PARALLEL_SIZE,
      $MAX_MODEL_LEN,
      '$DATASET',
      $INPUT_LEN,
      $OUTPUT_LEN,
      PENDING_COMMIT_TIMESTAMP(),
      '$USER',
      '$JOB_REFERENCE',
      ${EXPECTED_ETEL:-$VERY_LARGE_EXPECTED_ETEL},
      ${NUM_PROMPTS:-1000},
      '${MODELTAG:-PROD}',
      ${PREFIX_LEN:-0},
      '$EXTRA_ENVS'
    );"
  
  # If insert failed, just continue without publishing
  if [ $? -ne 0 ]; then
    echo "Insert failed for $RECORD_ID — skipping publish." >&2
    continue
  fi

  # Publishing to Buildkite step (main parameter: $RECORD_ID)



  # echo "Publishing to Pub/Sub queue: $GCP_QUEUE"
  # # Construct key-value string
  # MESSAGE_BODY="RecordId=$RECORD_ID"
  # # Publish the message
  # gcloud pubsub topics publish $QUEUE_TOPIC \
  #   --project="$GCP_PROJECT_ID" \
  #   --message="$MESSAGE_BODY" > /dev/null

  echo "$RECORD_ID scheduled."
done