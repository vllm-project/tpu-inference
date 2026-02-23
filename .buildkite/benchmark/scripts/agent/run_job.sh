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

# Record the exact start time in a format journalctl understands.
SCRIPT_START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

#
# Check input argument
#
if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
echo "Record ID: $RECORD_ID"

# Note: Removed the failure log upload feature. These logs are now accessible directly via the Buildkite interface.

if [ -z "$RECORD_ID" ] || [ "$RECORD_ID" == "null" ]; then
  echo "Invalid or missing record_id. Skipping message without ack."
  continue
fi

# Query TryCount as JSON
echo "Query $RECORD_ID..."
TRY_COUNT_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --format=json \
  --sql="SELECT TryCount FROM RunRecord WHERE RecordId='$RECORD_ID'")

# Parse TryCount using jq, convert to number, default to 0 if missing
TRY_COUNT=$(echo "$TRY_COUNT_JSON" | jq -r '.rows[0][0] | tonumber // 0')

# Increment TryCount
NEW_TRY_COUNT=$((TRY_COUNT + 1))

# Update record
echo "Updating record $RECORD_ID: Status=RUNNING, TryCount=$NEW_TRY_COUNT, RunBy=$BUILDKITE_AGENT_NAME..."
gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="UPDATE RunRecord SET Status='RUNNING', LastUpdate=CURRENT_TIMESTAMP(), TryCount=$NEW_TRY_COUNT, RunBy='${BUILDKITE_AGENT_NAME}' WHERE RecordId='$RECORD_ID'"


echo "deleting artifacts".
rm -rf artifacts

#
# Create running config
#
echo "Creating running config..."
.buildkite/benchmark/scripts/agent/create_config.sh "$RECORD_ID"
if [ $? -ne 0 ]; then
  echo "Error creating running config."
  exit 1
fi

#
# This makes GCS_BUCKET and other vars available to the whole script.
#
ENV_FILE="artifacts/${RECORD_ID}.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "Error: Config file $ENV_FILE not found after create_config.sh"
  exit 1
fi

#
# Run job in docker
#
BM_JOB_STATUS=$EXIT_SUCCESS
echo "Running job in docker..."
.buildkite/benchmark/scripts/agent/docker_run_bm.sh "artifacts/${RECORD_ID}.env" || {
  echo "Error running benchmark job in docker."
  BM_JOB_STATUS=$EXIT_FAILURE
}
# if [ $? -ne 0 ]; then
#   echo "Error running job in docker."
#   exit 1
# fi

echo "Benchmark script completed."

#
# Report result
#
echo "Reporting result..."
.buildkite/benchmark/scripts/agent/report_result.sh "$RECORD_ID"

echo ".buildkite/benchmark/scripts/cleanup_docker.sh"
.buildkite/benchmark/scripts/cleanup_docker.sh

exit $BM_JOB_STATUS