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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ADD_MODEL_SCRIPT="${SCRIPT_DIR}/add_model_to_ci.py"
OUTPUT_DIR="${PROJECT_ROOT}/models"

# Define the pattern for generated test files to ensure consistency
TEST_FILE_PATTERN="test_org_test-model-*.yml"

# Use trap to ensure cleanup happens on exit, regardless of success or failure
# shellcheck disable=SC2317
cleanup() {
    echo "--- :wastebasket: Cleaning up generated test files"
    # Ensure wildcard expansion by keeping it outside of double quotes
    # shellcheck disable=SC2086
    rm -f "${OUTPUT_DIR}"/${TEST_FILE_PATTERN}
    echo "✨ Cleanup complete!"
}
trap cleanup EXIT

TYPES=("tpu-optimized" "vllm-native")
CATEGORIES=("text-only" "multimodal")
SCALES=("single" "multi")

FAILED_CASES=()

echo "--- :python: Starting generation test for add_model_to_ci.py"

for t in "${TYPES[@]}"; do
    for c in "${CATEGORIES[@]}"; do
        for s in "${SCALES[@]}"; do
            # Note: add_model_to_ci.py replaces '/' with '_' in filenames
            MODEL_NAME="test_org/test-model-${t}-${c}-${s}"
            echo "Generating for: Type=$t, Category=$c, HostScale=$s"
            
            # Run python script and capture error without exiting immediately
            if ! python3 "$ADD_MODEL_SCRIPT" \
                --model-name "$MODEL_NAME" \
                --type "$t" \
                --category "$c" \
                --host-scale "$s"; then
                echo "❌ Failed to generate: $MODEL_NAME"
                FAILED_CASES+=("$MODEL_NAME (Generation Failed)")
            fi
        done
    done
done

# If generation failed, report now
if [ ${#FAILED_CASES[@]} -ne 0 ]; then
    echo "❌ Some generation cases failed!"
    for f in "${FAILED_CASES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo "--- :white_check_mark: Successfully generated all combinations"
# shellcheck disable=SC2086
ls -lh "${OUTPUT_DIR}"/${TEST_FILE_PATTERN}

echo "--- :mag: Validating generated YAML files syntax"

# Use OUTPUT_DIR and strip the repo root prefix to get relative paths starting with .buildkite/
# (required by validate_all_pipelines.sh grep rules)
# shellcheck disable=SC2012,SC2086
GENERATED_FILES=$(ls .buildkite/models/${TEST_FILE_PATTERN} 2>/dev/null || true)

if [ -z "$GENERATED_FILES" ]; then
    echo "❌ No files were generated to validate!"
    exit 1
fi

# Run validation and capture status
VALIDATE_STATUS=0
.buildkite/scripts/validate_all_pipelines.sh "$GENERATED_FILES" || VALIDATE_STATUS=$?

if [ $VALIDATE_STATUS -eq 0 ]; then
    echo "✅ All generated pipelines are syntactically valid!"
    exit $VALIDATE_STATUS
else
    echo "❌ YAML validation failed for generated pipelines!"
    exit $VALIDATE_STATUS
fi
