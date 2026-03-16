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

# First, run the query ONCE and get the stable JSON output
JSON_OUTPUT=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --format="json" \
    --sql="SELECT RecordId, Model, RunType, CodeHash, MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen, Dataset, InputLen, OutputLen, NumPrompts, PrefixLen, ExpectedETEL, ExtraEnvs, AdditionalConfig, Device, ExtraArgs FROM RunRecord WHERE RecordId = '$RECORD_ID';")

# Now, use this powerful jq script to parse the JSON and build the .env file.
# This correctly zips headers with values and handles all special characters.
echo "$JSON_OUTPUT" | jq -r '
  # 1. Create an array of SNAKE_CASE header names
  [ .metadata.rowType.fields[].name | gsub("(?<=[a-z0-9])(?=[A-Z])"; "_") | ascii_upcase ] as $headers |

  # 2. Create an array of values from the first (and only) data row
  [ .rows[0][] ] as $values |

  # 3. Loop from 0 to the length of the header array
  range($headers | length) as $i |

  # 4. Get the key and the value for this index
  $headers[$i] as $key |
  $values[$i] as $val |

  # 5. Filter out any null/empty values and print the formatted "KEY=Value" line
  if ($val | type) != "null" and $val != "" then
    "\($key)=\($val | @sh)"
  else
    empty
  end
' > "$ENV_FILE_TMP"

echo "Temp Environment file: "
cat "$ENV_FILE_TMP"

# Process EXTRA_ENVS line
EXTRA_ENVS_LINE=$(grep '^EXTRA_ENVS=' "$ENV_FILE_TMP")
EXTRA_ENVS_VALUE="${EXTRA_ENVS_LINE#EXTRA_ENVS=}"

# Strip leading/trailing single and double quotes from the value
EXTRA_ENVS_VALUE=$(echo "$EXTRA_ENVS_VALUE" | sed -e "s/^'//" -e "s/'$//" -e 's/^"//' -e 's/"$//')

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
LOCAL_HF_HOME="/mnt/disks/persist/models"
DOCKER_HF_HOME="/tmp/hf_home"
EOF
