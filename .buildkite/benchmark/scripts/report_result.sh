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

# Helper function to escape single quotes and handle defaults for SQL
prepare_sql_val() {
  local val="$1"
  local default="$2"
  if [ -z "$val" ]; then
    echo "$default"
    return
  fi
  val="${val#\'}"
  val="${val%\'}"
  local escaped_val="${val//\'/\'\'}"
  echo "'$escaped_val'"
}

if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
RESULT_FILE="artifacts/${RECORD_ID}.result"
printf "[INFO] Pre-check LOG_FOLDER=%s\n" "$LOG_FOLDER"
LOG_FOLDER=${LOG_FOLDER:-"artifacts/temp_logs"}
# Temp write to another bucket
# REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"
REMOTE_LOG_ROOT="gs://vllm-bm-bk-storage/job_logs/$RECORD_ID/"

(
  printf "[DEBUG] Start scan artifacts folder...\n"
  printf "[DEBUG] pwd=%s\n\nls artifacts=\n%s\n" "$(pwd)" "$(ls -al artifacts)"
  printf "[DEBUG] ls artifacts/temp_logs=\n%s\n" "$(ls -al artifacts/temp_logs)"
  printf "[INFO] LOG_FOLDER=\n%s\n" "$LOG_FOLDER"

  # Handle log file
  VLLM_LOG="$LOG_FOLDER/vllm_log.txt"
  BM_LOG="$LOG_FOLDER/bm_log.txt"
  # PROFILE_FOLDER="$LOG_ROOT/$TEST_NAME"_profile
  echo "gsutil cp $LOG_FOLDER/* $REMOTE_LOG_ROOT"
  gsutil cp -r "$LOG_FOLDER"/* "$REMOTE_LOG_ROOT"

  # Upload vllm and bm log to Buildkite aritfact
  ARTIFACT_VLLM="${RECORD_ID}_vllm_log.txt"
  ARTIFACT_BM="${RECORD_ID}_bm_log.txt"

  echo "Preparing Buildkite artifacts..."
  cp "$VLLM_LOG" "$ARTIFACT_VLLM"
  cp "$BM_LOG" "$ARTIFACT_BM"
  echo "Uploading artifacts to Buildkite..."
  buildkite-agent artifact upload "$ARTIFACT_VLLM"
  buildkite-agent artifact upload "$ARTIFACT_BM"
  echo "Cleaning up temporary artifact files..."
  rm -f "$ARTIFACT_VLLM" "$ARTIFACT_BM"

  # Metric data extraction from log file

  # Set internal error handling for this scope
  set -e

  AccuracyMetricsJSON=$(grep -a "AccuracyMetrics:" "$BM_LOG" | sed 's/AccuracyMetrics: //')
  echo "AccuracyMetricsJSON: $AccuracyMetricsJSON"

  if [[ "$RUN_TYPE" == *"ACCURACY"* ]]; then
    # Accuracy run logic
    echo "Accuracy run ($RUN_TYPE) detected. Parsing accuracy metrics."
    if [ -n "$AccuracyMetricsJSON" ]; then
      echo "AccuracyMetrics=$AccuracyMetricsJSON" > "$RESULT_FILE"
    else
      echo "Error: Accuracy run but no AccuracyMetrics found."
      exit 1
    fi
  else
    # Performance run logic
    throughput=$(grep -i "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    echo "throughput: $throughput"

    output_token_throughput=$(grep -i "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    total_token_throughput=$(grep -i "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')

    if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Failed to get the throughput and this is not an accuracy run."
      exit 1
    fi

    if (( $(echo "$throughput < ${EXPECTED_THROUGHPUT:-0}" | bc -l) )); then
      echo "Error: throughput($throughput) is less than expected($EXPECTED_THROUGHPUT)"
    fi
    echo "Throughput=$throughput" > "$RESULT_FILE"

    extract_value() {
      local section="$1"
      local label="$2"  # Mean, Median, or P99
      grep "$section (ms):" "$BM_LOG" | \
      awk -v label="$label" '$0 ~ label { print $NF }'
    }

    # Median values
    MedianITL=$(extract_value "ITL" "Median")
    MedianTPOT=$(extract_value "TPOT" "Median")
    MedianTTFT=$(extract_value "TTFT" "Median")
    MedianETEL=$(extract_value "E2EL" "Median")

    # P99 values
    P99ITL=$(extract_value "ITL" "P99")
    P99TPOT=$(extract_value "TPOT" "P99")
    P99TTFT=$(extract_value "TTFT" "P99")
    P99ETEL=$(extract_value "E2EL" "P99")

    # Write results to file
    (
      printf '%s=%s\n' \
      "MedianITL" "$MedianITL" \
      "MedianTPOT" "$MedianTPOT" \
      "MedianTTFT" "$MedianTTFT" \
      "MedianETEL" "$MedianETEL" \
      "P99ITL" "$P99ITL" \
      "P99TPOT" "$P99TPOT" \
      "P99TTFT" "$P99TTFT" \
      "P99ETEL" "$P99ETEL" \
      "OutputTokenThroughput" "$output_token_throughput" \
      "TotalTokenThroughput" "$total_token_throughput"
    ) >> "$RESULT_FILE"
  fi
) || {
  # Handle the error or log it before continuing
  echo "Warning: Metric extraction block failed. Continuing with script execution."
}

# Database Reporting Logic (ON CONFLICT (RecordId) DO UPDATE SET)
: "${BUILDKITE_AGENT_NAME:?Need to set BUILDKITE_AGENT_NAME}"

# 1. Parse metric assignments for dynamic columns
FINAL_STATUS="FAILED"
insert_cols=""
insert_vals=""
update_metrics=""

if [ -f "$RESULT_FILE" ]; then
  while IFS='=' read -r key value; do
    if [[ -n "$key" && -n "$value" ]]; then
      insert_cols+=", $key"
      if [[ "$key" == "AccuracyMetrics" ]]; then
        val_str="JSON '${value}'"
      elif [[ "$value" =~ ^[0-9.]+$ ]]; then
        val_str="${value}"
      else
        val_str="'${value//\'/\'\'}'"
      fi
      insert_vals+=", $val_str"
      # Use excluded keyword to refer to the proposed insert value
      update_metrics+=", ${key}=excluded.${key}"
      FINAL_STATUS="COMPLETED"
    fi
  done < "$RESULT_FILE"
fi

# fo test
# FINAL_STATUS="FAILED"

# 2. Prepare Base SQL Values
SQL_ADDITIONAL_CONFIG=$(prepare_sql_val "${ADDITIONAL_CONFIG:-}" "'{}'")
SQL_EXTRA_ARGS=$(prepare_sql_val "${EXTRA_ARGS:-}" "''")
SQL_EXTRA_ENVS=$(prepare_sql_val "${EXTRA_ENVS:-}" "''")
SQL_RECORD_ID=$(prepare_sql_val "$RECORD_ID" "")
SQL_STATUS=$(prepare_sql_val "$FINAL_STATUS" "FAILED")
SQL_USER=$(prepare_sql_val "${USER:-buildkite-agent}" "buildkite-agent")
SQL_JOB_REFERENCE=$(prepare_sql_val "${JOB_REFERENCE:-}" "")
SQL_AGENT_NAME=$(prepare_sql_val "${BUILDKITE_AGENT_NAME:-}" "")
SQL_DEVICE=$(prepare_sql_val "${DEVICE:-}" "")
SQL_MODEL=$(prepare_sql_val "${MODEL:-}" "")
SQL_RUN_TYPE=$(prepare_sql_val "${RUN_TYPE:-DAILY}" "DAILY")
SQL_CODE_HASH=$(prepare_sql_val "${CODE_HASH:-}" "")
SQL_DATASET=$(prepare_sql_val "${DATASET:-}" "")
SQL_MODELTAG=$(prepare_sql_val "${MODELTAG:-PROD}" "PROD")

# 3. Construct the atomic Upsert (Insert or Update) SQL statement
SQL="INSERT INTO RunRecord (
    RecordId, Status, CreatedTime, LastUpdate, CreatedBy, JobReference, RunBy,
    Device, Model, RunType, CodeHash,
    MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
    Dataset, InputLen, OutputLen,
    ExpectedETEL, NumPrompts, ModelTag, PrefixLen,
    ExtraEnvs, AdditionalConfig, ExtraArgs, TryCount $insert_cols
  ) VALUES (
    $SQL_RECORD_ID, $SQL_STATUS, PENDING_COMMIT_TIMESTAMP(), PENDING_COMMIT_TIMESTAMP(), $SQL_USER, $SQL_JOB_REFERENCE, $SQL_AGENT_NAME,
    $SQL_DEVICE, $SQL_MODEL, $SQL_RUN_TYPE, $SQL_CODE_HASH,
    ${MAX_NUM_SEQS:-NULL}, ${MAX_NUM_BATCHED_TOKENS:-NULL}, ${TENSOR_PARALLEL_SIZE:-NULL}, ${MAX_MODEL_LEN:-NULL},
    $SQL_DATASET, ${INPUT_LEN:-NULL}, ${OUTPUT_LEN:-NULL},
    ${EXPECTED_ETEL:-3600000}, ${NUM_PROMPTS:-1000}, $SQL_MODELTAG, ${PREFIX_LEN:-0},
    $SQL_EXTRA_ENVS, $SQL_ADDITIONAL_CONFIG, $SQL_EXTRA_ARGS, 1 $insert_vals
  ) ON CONFLICT (RecordId) DO UPDATE SET
    Status = excluded.Status,
    LastUpdate = excluded.LastUpdate,
    RunBy = excluded.RunBy,
    TryCount = RunRecord.TryCount + 1
    $update_metrics;"

echo "Executing Atomic Upsert SQL:"
echo "$SQL"

gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="$SQL"

# for test
echo "--- Verification: Current Database State"
DB_STATE_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" --instance="$GCP_INSTANCE_ID" \
  --format=json \
  --sql="SELECT TryCount, Status, LastUpdate, JobReference FROM RunRecord WHERE RecordId=$SQL_RECORD_ID")

echo "$DB_STATE_JSON" | jq -r '.rows[] | "✅ DB Sync Result: [TryCount: \(.[0]), Status: \(.[1]), JobRef: \(.[3]), LastUpdate: \(.[2])]"'

echo "--- Reporting finished"

#for test
# if [ "$FINAL_STATUS" == "FAILED" ]; then
#   echo "🚨 [FAIL] Benchmark metrics were not found or failed. Exiting with error."
#   exit 1
# fi
