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

# Exit immediately if a command exits with a non-zero status.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
export LOG_DIR=$1
export MODEL_NAME=$MODEL
export TASK_NAME=mlperf
OUTPUT_PREFIX=${TASK_NAME}_$(echo "$MODEL_NAME" | sed 's/\//-/g')
export OUTPUT_PREFIX

export OUTPUT_BASE_PATH=$LOG_DIR/$OUTPUT_PREFIX.json
export ACCURACY_JSON_PATH=$LOG_DIR/mlperf_accuracy.json

echo "Running lm_eval, output will be timestamped in: $LOG_DIR"

unset MODEL_IMPL_TYPE VLLM_XLA_CHECK_RECOMPILATION

mkdir -p "$LOG_DIR"

CMD=(
    python3
    -m
    scripts.vllm.integration.lm_eval_accuracy
    --model vllm
    --model_args "pretrained=$MODEL_NAME,tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-8},dtype=auto,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=0.98"
    --tasks "$TASK_NAME"
    --include_path "$SCRIPT_DIR"
    --batch_size auto
    --log_samples
    --limit "${NumPrompts:-1000}"
    --output_path "$OUTPUT_BASE_PATH"
    --apply_chat_template
)

eval "CLIENT_CMD_ENVS=(${CLIENT_CMD_ENVS_STR:-})"
echo "[DEBUG] Executing lm_eval_cmd: SKIP_JAX_PRECOMPILE=1 ${CLIENT_CMD_ENVS[*]:-} ${CMD[*]}"
# Execute the command, allowing stderr for error visibility
if ! env SKIP_JAX_PRECOMPILE=1 "${CLIENT_CMD_ENVS[@]}" "${CMD[@]}"; then
    echo "Error: lm_eval command failed. See output above for details."
    exit 1
fi

echo "Finding the latest output file in $LOG_DIR with prefix ${OUTPUT_PREFIX}..."

# Find the most recently modified file in the output directory that starts with the correct prefix
LATEST_FILE=$(find "$LOG_DIR" -type f -name "${OUTPUT_PREFIX}_*.json" -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)

# Check if a file was actually found
if [ -z "$LATEST_FILE" ]; then
    echo "Error: No matching output file found. Exiting."
    exit 1
fi

echo "Found and using file: $LATEST_FILE"

echo "Parsing results and writing to $ACCURACY_JSON_PATH..."
python "$SCRIPT_DIR/parse_lm_eval_mlperf_results.py" "$LATEST_FILE" > "$ACCURACY_JSON_PATH"
