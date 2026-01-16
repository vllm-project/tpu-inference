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


TOTAL=0
FAILED=0

TARGET_DIR="/workspace/tpu_inference/.buildkite"

echo "=== 🔍 Scanning directory: $TARGET_DIR ==="

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

while IFS= read -r file; do
    echo "Validating: $file"
    ((TOTAL++))

    if bk pipeline validate -f "$file"; then
        echo "✅ [PASSED] $file"
    else
        echo "❌ [FAILED] $file"
        ((FAILED++))
    fi
    echo "-------------------------------------------"
done < <(find "$TARGET_DIR" -type f \( -name "*.yml" -o -name "*.yaml" \))

echo "=== 📊 Validation Summary ==="
echo "Total processed: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"

if [ "$FAILED" -ne 0 ]; then
    echo "Result: FAIL (Please fix the errors above)"
    exit 1
else
    echo "Result: SUCCESS"
    exit 0
fi