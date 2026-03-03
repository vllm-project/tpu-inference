#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -ex

# Change to the script's directory to ensure relative paths work correctly.
cd "$(dirname "$0")"

# --- Configuration ---
export LOG_DIR=./results
export MODEL_NAME=$MODEL
# Use a specific MMLU subtask if the MMLU_SUBTASK env var is set, otherwise default to the full mmlu group task.
export OUTPUT_PREFIX=${TASK_NAME}_$(echo $MODEL_NAME | sed 's#/#-#g')

export OUTPUT_BASE_PATH=$LOG_DIR/$OUTPUT_PREFIX.json
export ACCURACY_JSON_PATH=/workspace/mmlu_accuracy.json

echo "Running lm_eval for task: $TASK_NAME"
echo "Output will be timestamped in: $LOG_DIR"

mkdir -p "$LOG_DIR"

# Default model arguments
MODEL_ARGS="pretrained=$MODEL_NAME,tensor_parallel_size=${TP_SIZE:-8},dtype=auto,max_model_len=2048,gpu_memory_utilization=0.98"

# Check if running on v7x-8 hardware
if [[ "$DEVICE" == v7x-8 ]]; then
    echo "Running on v7x hardware, adjusting model arguments for DeepSeek-R1."
    MODEL_ARGS="pretrained=$MODEL_NAME,tensor_parallel_size=8,dtype=auto,max_model_len=2048,max_num_seqs=128,max_num_batched_tokens=128,gpu_memory_utilization=0.95"
fi

CMD=(
    lm_eval
    --model vllm
    --model_args "$MODEL_ARGS"
    --tasks mmlu_llama
    --num_fewshot 0
    --apply_chat_template
    --batch_size auto
    --limit 100  # note this is 100 per mmlu sub-task, ~5700 samples overall
    --output_path "$OUTPUT_BASE_PATH"
)

# Execute the command, allowing stderr for error visibility
if ! SKIP_JAX_PRECOMPILE=1 "${CMD[@]}"; then
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
python parse_lm_eval_mmlu_results.py "$LATEST_FILE" > "$ACCURACY_JSON_PATH"
