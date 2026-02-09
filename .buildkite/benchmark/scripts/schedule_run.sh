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
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <input.csv|gs://path/to/input.csv> CODE_HASH JOB_REFERENCE RUN_TYPE EXTRA_ENVS"
  exit 1
fi

CSV_FILE_ARG="$1"
CODE_HASH="$2"
JOB_REFERENCE="$3"
RUN_TYPE="$4"
EXTRA_ENVS="$5"

if [[ "$CSV_FILE_ARG" == gs://* ]]; then
  echo "GCS path detected. Downloading from $CSV_FILE_ARG"
  CSV_FILE=$(mktemp)
  # Schedule cleanup of the temporary file on exit
  trap 'rm -f "$CSV_FILE"' EXIT
  if ! gsutil cp "$CSV_FILE_ARG" "$CSV_FILE"; then
    echo "Failed to download from GCS: $CSV_FILE_ARG"
    rm "$CSV_FILE"
    exit 1
  fi
else
  CSV_FILE="$CSV_FILE_ARG"
fi

if [ ! -f "$CSV_FILE" ]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

# milliseconds: one hour
VERY_LARGE_EXPECTED_ETEL=3600000

declare -a pipeline_steps

declare -A BUILDKITE_QUEUE_DEVICE_MAP
BUILDKITE_QUEUE_DEVICE_MAP=(
    ["tpu7x-8"]="tpu_v7x_8_queue"
    ["tpu7x-2"]="tpu_v7x_2_queue"
    ["v6e-1"]="tpu_v6e_queue"
    ["v6e-8"]="tpu_v6e_8_queue"
)

# === Config ===
# Make sure these environment variables are set or export here
# export GCP_PROJECT_ID="your-project"
# export GCP_INSTANCE_ID="your-instance"
# export GCP_DATABASE_ID="your-database"

# === Read CSV and skip header ===
while read -r line || [ -n "${line}" ]; do

  line=$(echo "$line" | tr -d '\r')
  [ -z "$line" ] && continue

  # Use Python to safely parse the CSV line.
  # This handles internal commas inside quoted fields (like JSON in AdditionalConfig).
  # mapfile captures the output into an array called 'fields'.
  mapfile -t fields < <(python3 -c "import csv, sys; line = sys.stdin.read(); reader = csv.reader([line], quotechar=\"'\"); print('\n'.join(next(reader)))" <<< "$line")

  # Map array indices to descriptive variables
  DEVICE="${fields[0]}"
  MODEL="${fields[1]}"
  MAX_NUM_SEQS="${fields[2]}"
  MAX_NUM_BATCHED_TOKENS="${fields[3]}"
  TENSOR_PARALLEL_SIZE="${fields[4]}"
  MAX_MODEL_LEN="${fields[5]}"
  DATASET="${fields[6]}"
  INPUT_LEN="${fields[7]}"
  OUTPUT_LEN="${fields[8]}"
  EXPECTED_ETEL="${fields[9]}"
  NUM_PROMPTS="${fields[10]}"
  MODELTAG="${fields[11]}"
  PREFIX_LEN="${fields[12]}"
  ADDITIONAL_CONFIG="${fields[13]}"
  EXTRA_ARGS="${fields[14]}"

  RECORD_ID=$(uuidgen | tr 'A-Z' 'a-z')

  # Helper to handle SQL quoting for JSON/String fields.
  # This now properly escapes internal single quotes by doubling them (' -> '').
  prepare_sql_val() {
    local val="$1"
    local default="$2"

    # if empty, return default
    if [ -z "$val" ]; then
      echo "$default"
      return
    fi

    # Remove leading/trailing single quotes if the CSV parser preserved them
    val="${val#\'}"
    val="${val%\'}"

    # Escape internal single quotes for Spanner SQL (replace ' with '')
    local escaped_val="${val//\'/\'\'}"

    # Wrap the escaped value in single quotes for the SQL statement
    echo "'$escaped_val'"
  }

  SQL_ADDITIONAL_CONFIG=$(prepare_sql_val "$ADDITIONAL_CONFIG" "'{}'")
  SQL_EXTRA_ARGS=$(prepare_sql_val "$EXTRA_ARGS" "''")
  
  echo "Inserting Run: $RECORD_ID"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen, LastUpdate, CreatedBy, JobReference,
      ExpectedETEL, NumPrompts, ModelTag, PrefixLen, ExtraEnvs,
      AdditionalConfig, ExtraArgs
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
      '$EXTRA_ENVS',
      $SQL_ADDITIONAL_CONFIG,
      $SQL_EXTRA_ARGS
    );"
  
  # If insert failed, just continue without publishing
  if [ $? -ne 0 ]; then
    echo "Insert failed for $RECORD_ID — skipping publish." >&2
    continue
  fi

  # Check if the DEVICE exists in the buildkite queue map
  if [[ -v BUILDKITE_QUEUE_DEVICE_MAP["$DEVICE"] ]]; then
    BUILDKITE_AGENT_QUEUE="${BUILDKITE_QUEUE_DEVICE_MAP[$DEVICE]}"
  else
    echo "No suitable Buildkite queue was found for the $DEVICE_ID, recordId: ${RECORD_ID} - skipping publish." >&2
    continue
  fi

  # Publishing to Buildkite step (main parameter: $RECORD_ID)
  # For each line in csv file, generate a command step
  pipeline_yaml=$(cat <<EOF
- label: "Publish: benchmark - RecordId: ${RECORD_ID}"
  depends_on: "build-vllm-tpu-image"
  env:
    GCP_PROJECT_ID: "cloud-tpu-inference-test"
    GCP_REGION: "southamerica-west1"
    GCS_BUCKET: "vllm-cb-storage2"
    ARTIFACT_REPO: "vllm-tpu-bm-bk"
    GCP_INSTANCE_ID: "vllm-bm-inst"
    GCP_DATABASE_ID: "vllm-bm-bk-runs"
  agents:
    queue: $BUILDKITE_AGENT_QUEUE
  command: 
    - ".buildkite/benchmark/scripts/agent/run_job.sh ${RECORD_ID}"
EOF
)

  pipeline_steps+=("${pipeline_yaml}")

  echo "$RECORD_ID handled."
done < <(tail -n +2 "$CSV_FILE")

# --- Upload Benchmark Dynamic Pipeline ---
if [[ "${#pipeline_steps[@]}" -gt 0 ]]; then
  echo "--- Uploading Benchmark Dynamic Pipeline Steps"
  final_pipeline_yaml="steps:"$'\n'
  final_pipeline_yaml+=$(printf "%s\n" "${pipeline_steps[@]}")
  echo "Upload YML: ${final_pipeline_yaml}"
  echo -e "${final_pipeline_yaml}" | buildkite-agent pipeline upload
else
  echo "--- No benchmark records found, no new Pipeline Steps to upload."
  exit 0
fi
