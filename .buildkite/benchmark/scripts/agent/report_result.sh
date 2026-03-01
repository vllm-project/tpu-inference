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
