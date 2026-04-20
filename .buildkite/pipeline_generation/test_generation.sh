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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ADD_MODEL_SCRIPT="${SCRIPT_DIR}/add_model_to_ci.py"
OUTPUT_DIR="${PROJECT_ROOT}/models"

TYPES=("tpu-optimized" "vllm-native")
CATEGORIES=("text-only" "multimodal")
SCALES=("single" "multi")

echo "Starting generation test for add_model_to_ci.py..."

for t in "${TYPES[@]}"; do
    for c in "${CATEGORIES[@]}"; do
        for s in "${SCALES[@]}"; do
            MODEL_NAME="test_org/test-model-${t}-${c}-${s}"
            echo "--------------------------------------------------"
            echo "Generating for: Type=$t, Category=$c, HostScale=$s"
            python3 "$ADD_MODEL_SCRIPT" \
                --model-name "$MODEL_NAME" \
                --type "$t" \
                --category "$c" \
                --host-scale "$s"
        done
    done
done

echo "--------------------------------------------------"
echo "✅ Successfully generated all 8 combinations!"
echo "Checking generated files in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"/test_org_test-model-*

# echo "--------------------------------------------------"
# echo "Cleaning up generated test files..."
# rm "${OUTPUT_DIR}"/test_org_test-model-*.yml
# echo "✨ Cleanup complete!"
