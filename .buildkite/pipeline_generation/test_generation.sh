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

# --- Path and Environment Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Detect available Python executable (prefer python3 over python)
if command -v python3 &>/dev/null; then
  PYTHON_EXEC="python3"
elif command -v python &>/dev/null; then
  PYTHON_EXEC="python"
else
  echo "+++ :x: Error: Python not found on this agent!"
  exit 1
fi

# Scripts paths
ADD_MODEL_SCRIPT="${SCRIPT_DIR}/add_model_to_ci.py"
ADD_FEATURE_SCRIPT="${SCRIPT_DIR}/add_feature_to_ci.py"
VALIDATOR="${PROJECT_ROOT}/.buildkite/scripts/validate_buildkite_ymls.sh"

# Define patterns for cleanup and discovery
TEST_PATTERN="test_org_test-*.yml"

# Initialize flags
RUN_MODELS=false
RUN_FEATURES=false
FAILED_CASES=()
VALIDATE_STATUS=0

case "${1:-}" in
  --models)   RUN_MODELS=true ;;
  --features) RUN_FEATURES=true ;;
  --all)      RUN_MODELS=true; RUN_FEATURES=true ;;
  *)          echo "Usage: $0 {--models|--features|--all}"; exit 1 ;;
esac

# Cleanup function triggered on script exit
# shellcheck disable=SC2317
cleanup() {
    # If the job failed (VALIDATE_STATUS != 0), expand the cleanup block for debugging
    if [ "${VALIDATE_STATUS:-0}" -ne 0 ] || [ ${#FAILED_CASES[@]} -ne 0 ]; then
        echo "+++ :wastebasket: Cleaning up (Job failed, preserving logs)"
    else
        echo "--- :wastebasket: Cleaning up generated test files"
    fi
    find "${PROJECT_ROOT}/.buildkite" -name "${TEST_PATTERN}" -type f -delete || true
    echo "✨ Cleanup complete!"
}
trap cleanup EXIT

# --- Model Generation Test ---
if [ "$RUN_MODELS" == "true" ]; then
    echo "--- :python: Starting targeted generation test for Models"
    TYPES=("tpu-optimized" "vllm-native")
    
    for t in "${TYPES[@]}"; do
        MODEL_NAME="test_org/test-model-${t}"
        echo "Generating: $MODEL_NAME"
        
        # Capture error without exiting immediately
        if ! "$PYTHON_EXEC" "$ADD_MODEL_SCRIPT" --model-name "$MODEL_NAME" --type "$t" --host-scale "single"; then
            echo "+++ ❌ Failed to generate Model: $MODEL_NAME"
            FAILED_CASES+=("$MODEL_NAME (Model Gen Failed)")
        fi
    done
fi

# --- Feature Generation Test ---
if [ "$RUN_FEATURES" == "true" ]; then
    echo "--- :python: Starting targeted generation test for Features"
    CATS=("feature support matrix" "kernel support matrix microbenchmarks")
    
    for cat in "${CATS[@]}"; do
        SAFE_CAT=$(echo "$cat" | tr ' ' '_')
        FEATURE_NAME="test_org/test-feature-${SAFE_CAT}"
        echo "Generating: $FEATURE_NAME"

        # Prepare arguments based on category
        ARGS=("--feature-name" "$FEATURE_NAME" "--category" "$cat" "--host-scale" "single")
        [[ "$cat" == *"microbenchmarks"* ]] && ARGS+=("--group" "test_group")

        # Capture error without exiting immediately
        if ! "$PYTHON_EXEC" "$ADD_FEATURE_SCRIPT" "${ARGS[@]}"; then
            echo "+++ ❌ Failed to generate Feature: $FEATURE_NAME"
            FAILED_CASES+=("$FEATURE_NAME (Feature Gen Failed)")
        fi
    done
fi

# --- Final Validation ---

# If generation failed, report it before starting YAML validation
if [ ${#FAILED_CASES[@]} -ne 0 ]; then
    echo "+++ :x: Some generation cases failed!"
    for error in "${FAILED_CASES[@]}"; do
        echo "  - $error"
    done
    exit 1
fi

echo "+++ 🔍 Generation complete. Starting validation..."

# Discover all generated files for the validator
GENERATED_FILES=$(find "${PROJECT_ROOT}/.buildkite" -name "${TEST_PATTERN}" -type f)

if [ -z "$GENERATED_FILES" ]; then
    echo "+++ :warning: No files were generated. If this is unexpected, check script paths."
    exit 0
fi

VALIDATE_STATUS=0
bash "$VALIDATOR" "$GENERATED_FILES" || VALIDATE_STATUS=$?

if [ $VALIDATE_STATUS -eq 0 ]; then
    echo "✅ All generated pipelines passed metadata validation!"
    exit 0
else
    echo "+++ :x: YAML validation failed for generated pipelines!"
    exit $VALIDATE_STATUS
fi
