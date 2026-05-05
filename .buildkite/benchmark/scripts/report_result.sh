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

set -Eeuo pipefail

# ==============================================================================
# 0. Global Panic Handler (Crash Interceptor)
# ==============================================================================
# shellcheck disable=SC2317
on_crash() {
    local exit_code=$?
    local line_no=$1
    local command="$2"
    
    # Ignore normal exits (Fixed SC2086 by adding double quotes)
    if [ "$exit_code" -eq 0 ]; then
        return
    fi

    echo ""
    echo "================================================================"
    echo "🚨 [FATAL ERROR] Bash Script Crashed Unexpectedly!"
    echo "================================================================"
    echo "File:     $(basename "$0")"
    echo "Line:     $line_no"
    echo "Command:  $command"
    echo "ExitCode: $exit_code"
    echo "================================================================"
    echo ""
}

# Bind the ERR signal: Triggers on_crash immediately if any command fails 
# and is not explicitly caught by an 'if' statement or '||' operator.
trap 'on_crash ${LINENO} "$BASH_COMMAND"' ERR

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

RECORD_ID=$1
RUN_TYPE="${RUN_TYPE:-DAILY}"

# Define Result_file name
RESULT_FILE="${ARTIFACT_FOLDER}/${RECORD_ID}.result"

# Upload logs to GCS if bucket is provided
if [[ -n "${GCS_BUCKET:-}" ]]; then
  # TODO: When switching to Production after validation is complete, 
  # please change to use `$GCS_BUCKET` as the log storage bucket. 
  # For now, it is hardcoded to use the `vllm-bm-bk-storage` bucket.
  # REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"
  REMOTE_LOG_ROOT="gs://vllm-bm-bk-storage/job_logs/$RECORD_ID/"
fi

(
  if [ "${BUILDKITE:-false}" == "true" ]; then
    ENV_CONTEXT="Buildkite environment"
  else
    ENV_CONTEXT="Local environment"
  fi
  printf "[DEBUG] Start scan artifacts folder (Environment: %s)...\n" "$ENV_CONTEXT"
  printf "[INFO] ARTIFACT_FOLDER=\n%s\n" "$ARTIFACT_FOLDER"
  if [ -d "$ARTIFACT_FOLDER" ]; then
    printf "[DEBUG] ls $ARTIFACT_FOLDER=\n%s\n" "$(ls "$ARTIFACT_FOLDER")"
  fi
  printf "[INFO] LOG_FOLDER=\n%s\n" "$LOG_FOLDER"

  # Handle log file
  
  if [[ -n "${GCS_BUCKET:-}" ]]; then
    if command -v gsutil &> /dev/null; then
      echo "gsutil cp $LOG_FOLDER/* $REMOTE_LOG_ROOT"
      gsutil cp -r "$LOG_FOLDER"/* "$REMOTE_LOG_ROOT"
    else
      echo "Warning: gsutil not found. Skipping log upload to GCS."
    fi
  else
    echo "Warning: GCS_BUCKET is not set. Skipping log upload to GCS."
  fi

  # Metric data extraction from log file
  BM_LOG="$LOG_FOLDER/bm_log.txt"

  if [[ "$RUN_TYPE" == *"ACCURACY"* ]]; then
    # Accuracy run logic
    echo "Accuracy run ($RUN_TYPE) detected. Parsing accuracy metrics."
    AccuracyMetricsJSON=$(grep -a "AccuracyMetrics:" "$BM_LOG" | sed 's/AccuracyMetrics: //' || true)
    echo "AccuracyMetricsJSON: $AccuracyMetricsJSON"
    if [ -n "$AccuracyMetricsJSON" ]; then
      echo "AccuracyMetrics=$AccuracyMetricsJSON" > "$RESULT_FILE"
    else
      echo "Error: Accuracy run but no AccuracyMetrics found."
      exit 1
    fi
  else
    # Performance run logic
    throughput=$(grep -i "^Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g' || true)
    echo "throughput: $throughput"

    output_token_throughput=$(grep -i "^Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g' || true)
    total_token_throughput=$(grep -i "^Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g' || true)

    if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Failed to get the throughput and this is not an accuracy run."
      exit 1
    fi

    if [[ -z "$output_token_throughput" || ! "$output_token_throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Failed to get output_token_throughput."
      exit 1
    fi

    if [[ -z "$total_token_throughput" || ! "$total_token_throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Failed to get total_token_throughput."
      exit 1
    fi

    # Compare throughput using awk for float support
    EXPECTED_THROUGHPUT_VAL="${EXPECTED_THROUGHPUT:-0}"
    IS_LOW_THROUGHPUT=$(echo "$throughput $EXPECTED_THROUGHPUT_VAL" | awk '{if ($1 < $2 || $1 == 0) print 1; else print 0}')
    if [ "$IS_LOW_THROUGHPUT" -eq 1 ]; then
      echo "Error: throughput($throughput) is less than expected($EXPECTED_THROUGHPUT_VAL) or is 0"
    fi
    echo "Throughput=$throughput" > "$RESULT_FILE"

    extract_value() {
      local section="$1"
      local label="$2"  # Mean, Median, or P99
      grep "$section (ms):" "$BM_LOG" | \
      awk -v label="$label" '$0 ~ label { print $NF }' || true
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
)

# Database Reporting Logic (ON CONFLICT (RecordId) DO UPDATE SET)
if [[ -n "${GCP_DATABASE_ID:-}" && -n "${GCP_PROJECT_ID:-}" && -n "${GCP_INSTANCE_ID:-}" ]]; then
  BUILDKITE_AGENT_NAME="${BUILDKITE_AGENT_NAME:-local-test}"

  # Parse metric assignments for dynamic columns
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

  # Prepare Base SQL Values
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
  SQL_CONFIG=$(prepare_sql_val "${CASE_CONFIG_JSON:-}" "{}")

  # Construct the atomic Upsert (Insert or Update) SQL statement
  SQL="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, LastUpdate, CreatedBy, JobReference, RunBy,
      Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen,
      ExpectedETEL, NumPrompts, ModelTag, PrefixLen,
      ExtraEnvs, AdditionalConfig, ExtraArgs, TryCount, Config $insert_cols
    ) VALUES (
      $SQL_RECORD_ID, $SQL_STATUS, PENDING_COMMIT_TIMESTAMP(), PENDING_COMMIT_TIMESTAMP(), $SQL_USER, $SQL_JOB_REFERENCE, $SQL_AGENT_NAME,
      $SQL_DEVICE, $SQL_MODEL, $SQL_RUN_TYPE, $SQL_CODE_HASH,
      ${MAX_NUM_SEQS:-NULL}, ${MAX_NUM_BATCHED_TOKENS:-NULL}, ${TENSOR_PARALLEL_SIZE:-NULL}, ${MAX_MODEL_LEN:-NULL},
      $SQL_DATASET, $INPUT_LEN, $OUTPUT_LEN,
      ${EXPECTED_ETEL:-3600000}, ${NUM_PROMPTS:-1000}, $SQL_MODELTAG, ${PREFIX_LEN:-0},
      $SQL_EXTRA_ENVS, $SQL_ADDITIONAL_CONFIG, $SQL_EXTRA_ARGS, 1, JSON r$SQL_CONFIG $insert_vals
    ) ON CONFLICT (RecordId) DO UPDATE SET
      Status = excluded.Status,
      LastUpdate = excluded.LastUpdate,
      RunBy = excluded.RunBy,
      TryCount = RunRecord.TryCount + 1,
      Config = excluded.Config
      $update_metrics;"

  echo "Executing Atomic Upsert SQL:"
  echo "$SQL"

  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="$SQL"
  echo "--- Reporting finished (DB written)"
else
  echo "--- Reporting finished (Local test scenario: GCP variables not set, skipping DB reporting)"
  if [ -f "$RESULT_FILE" ]; then
    echo "--- Final Benchmark Results ($RESULT_FILE) ---"
    cat "$RESULT_FILE"
    echo "------------------------------------------------"
  else
    echo "Warning: $RESULT_FILE not found. No results to display."
  fi
fi
