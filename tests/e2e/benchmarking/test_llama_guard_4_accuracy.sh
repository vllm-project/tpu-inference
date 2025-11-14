#!/bin/bash
set -e

# --- Configuration and Variable Extraction ---
TEST_MODEL=${TEST_MODEL}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
MINIMUM_ACCURACY_THRESHOLD=${MINIMUM_ACCURACY_THRESHOLD} 
EXIT_CODE=0
ACCURACY_LOG_FILE="/tmp/accuracy_output_$$" # Unique temp file for the run

# --- Cleanup Trap (Add the new file to cleanup) ---
cleanup_func() {
    rm -f "$ACCURACY_LOG_FILE"
}
trap cleanup_func EXIT

# --- 2. EXECUTE PYTHON SCRIPT AND STREAM OUTPUT ---
echo "Running Llama Guard 4 Accuracy Check for $TEST_MODEL (TP=$TENSOR_PARALLEL_SIZE)"

# Execute the Python script:
# The 2>&1 redirects stderr (where tqdm writes) to stdout.
# tee prints stdout (including the progress bar) to the terminal AND saves it to the log file.
SKIP_JAX_PRECOMPILE=1 \
python /workspace/tpu_inference/examples/offline_llama_guard_4_inference.py \
  --model="$TEST_MODEL" \
  --tensor-parallel-size="$TENSOR_PARALLEL_SIZE" \
  --max_model_len=2048 \
  --max-num-batched-tokens=4096 \
  --hf_overrides '{"architectures": ["LLaMAForCausalLM"]}' \
  2>&1 | tee "$ACCURACY_LOG_FILE"

PYTHON_EXIT_CODE=$?

# --- 3. EXTRACT ACCURACY FROM LOG FILE ---
# Use grep and awk on the saved log file
ACTUAL_ACCURACY=$(grep "FINAL_ACCURACY:" "$ACCURACY_LOG_FILE" | awk '{print $NF}')

if [[ -z "$ACTUAL_ACCURACY" ]]; then
    echo "Error: Could not extract FINAL_ACCURACY from script output." >&2
    exit 1
fi

echo -e "\n--- ACCURACY CHECK ---"
echo "Target Accuracy: $MINIMUM_ACCURACY_THRESHOLD"
echo "Actual Accuracy: $ACTUAL_ACCURACY"

# --- 4. PERFORM FLOAT COMPARISON ---
if awk -v actual="$ACTUAL_ACCURACY" -v min="$MINIMUM_ACCURACY_THRESHOLD" 'BEGIN { exit !(actual >= min) }'; then
    echo "ACCURACY CHECK PASSED: $ACTUAL_ACCURACY >= $MINIMUM_ACCURACY_THRESHOLD"
    EXIT_CODE=0
else
    echo "ACCURACY CHECK FAILED: $ACTUAL_ACCURACY < $MINIMUM_ACCURACY_THRESHOLD" >&2
    EXIT_CODE=1
fi

exit $EXIT_CODE