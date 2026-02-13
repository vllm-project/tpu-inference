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

# === Usage check ===
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

# === Config ===

RECORD_ID="$1"
mkdir -p ./artifacts

ENV_FILE="artifacts/${RECORD_ID}.env"
ENV_FILE_TMP="$ENV_FILE.tmp"

# The script below generates the result in this format:
# MODEL=meta-llama/Llama-3.1-8B-Instruct
# MAX_NUM_SEQS=128
# MAX_NUM_BATCHED_TOKENS=4096
# TENSOR_PARALLEL_SIZE=1
# MAX_MODEL_LEN=2048
# INPUT_LEN=1800
# OUTPUT_LEN=128
gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --project="$GCP_PROJECT_ID" \
  --sql="SELECT RecordId, Model, RunType, CodeHash, MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen, Dataset, InputLen, OutputLen, NumPrompts, PrefixLen, ExpectedETEL, ExtraEnvs FROM RunRecord WHERE RecordId = '$RECORD_ID';" | \
  gawk 'NR==1 {
    for (i=1; i<=NF; i++) {
      # Insert underscore before uppercase letters preceded by a lowercase letter
      # Then convert to uppercase
      header[i] = toupper(gensub(/([a-z0-9])([A-Z])/, "\\1_\\2", "g", $i))
    }
  }
  NR==2 {
    for (i=1; i<=NF; i++) {
      if ($i != "" && $i != "None") {
        print header[i] "=" $i
      }
    }
  }' > "$ENV_FILE_TMP"

# Process EXTRA_ENVS line
EXTRA_ENVS_LINE=$(grep '^EXTRA_ENVS=' "$ENV_FILE_TMP")
EXTRA_ENVS_VALUE="${EXTRA_ENVS_LINE#EXTRA_ENVS=}"

# Remove the original EXTRA_ENVS line
grep -v '^EXTRA_ENVS=' "$ENV_FILE_TMP" > "${ENV_FILE}"

# Append the split env vars from EXTRA_ENVS
IFS=';' read -ra ENV_PAIRS <<< "$EXTRA_ENVS_VALUE"
for pair in "${ENV_PAIRS[@]}"; do
  echo "$pair" >> "${ENV_FILE}"
done

# Add static fields
cat <<EOF >> "${ENV_FILE}"
TEST_NAME=static
CONTAINER_NAME=vllm-tpu
DOWNLOAD_DIR=/mnt/disks/persist
EOF

