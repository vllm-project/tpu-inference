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
  printf "[INFO] LOG_FOLDER=%s\n" "$LOG_FOLDER"

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

# Ensure BUILDKITE_AGENT_NAME is set
: "${BUILDKITE_AGENT_NAME:?Need to set BUILDKITE_AGENT_NAME}"

# Case 1: result file does not exist → mark as FAILED
if [ ! -f "$RESULT_FILE" ]; then
  echo "Result file not found: $RESULT_FILE. Marking status as FAILED."

  SQL="UPDATE RunRecord SET Status='FAILED', RunBy='${BUILDKITE_AGENT_NAME}', LastUpdate=CURRENT_TIMESTAMP() WHERE RecordId = '${RECORD_ID}';"

  echo "Executing SQL:"
  echo "$SQL"

  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="$SQL"

  exit 0
fi

# Case 2: result file exists → parse and mark as COMPLETED
assignments=""
while IFS='=' read -r key value; do
  if [[ -n "$key" && -n "$value" ]]; then
    if [[ "$key" == "AccuracyMetrics" ]]; then
      assignments+="${key}=JSON '${value}', "
    elif [[ "$value" =~ ^[0-9.]+$ ]]; then
      assignments+="${key}=${value}, "
    else
      assignments+="${key}='${value}', "
    fi
  fi
done < "$RESULT_FILE"

# Clean up trailing comma+space
assignments="${assignments%, }"

if [ -z "$assignments" ]; then
  echo "Result file was empty. Marking status as FAILED."
  SQL="UPDATE RunRecord SET Status='FAILED', RunBy='${BUILDKITE_AGENT_NAME}', LastUpdate=CURRENT_TIMESTAMP() WHERE RecordId = '${RECORD_ID}';"
else
  SQL="UPDATE RunRecord SET ${assignments}, Status='COMPLETED', RunBy='${BUILDKITE_AGENT_NAME}', LastUpdate=CURRENT_TIMESTAMP() WHERE RecordId = '${RECORD_ID}';"
fi

echo "Executing SQL:"
echo "$SQL"

gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="$SQL"
