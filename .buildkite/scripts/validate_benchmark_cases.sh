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


# Exit on unset variable, fail on pipe errors.
set -uo pipefail

# Assign the first argument to a local variable (changed files in PR)
FILES_CHANGED="${1:-}"

# Filter for benchmark JSON case files
JSON_CHANGES=$(echo "$FILES_CHANGED" | grep -E "^\.buildkite/benchmark/cases/.*\.json$" || true)

# Early exit if no benchmark JSON files were modified
if [ -z "$JSON_CHANGES" ]; then
    echo "--- :fast_forward: No benchmark JSON changes found. Skipping validation."
    exit 0
fi

echo "--- 🔍 Validating modified benchmark cases individually"

EXIT_STATUS=0
FAILED_FILES=()

# Iterate through each modified JSON file
while IFS= read -r json_path; do
    [ -z "$json_path" ] && continue
    
    # Check if file exists (handle deleted files in PR)
    if [ ! -f "$json_path" ]; then
        echo "Skipping deleted file: $json_path"
        continue
    fi

    echo "--- Validating: $json_path"
    
    # Create a temporary file to store the generated YAML for validation
    TMP_YAML=$(mktemp)
    trap 'rm -f "$TMP_YAML"' EXIT

    # Generate the YAML and capture it for potential debugging
    # Using '2>&1' to capture stderr from the python script as well
    if ! GENERATED_YAML=$(python3 .buildkite/benchmark/scripts/generate_bk_pipeline.py --input "$json_path" 2>&1); then
        echo "+++ ❌ Validation Script Error for $json_path"
        echo "Error message:"
        echo "---"
        echo "$GENERATED_YAML"
        echo "---"
        EXIT_STATUS=1
        FAILED_FILES+=("$json_path")
        continue
    fi
    
    echo "$GENERATED_YAML" > "$TMP_YAML"

    # Pipe the captured YAML to Buildkite pipeline validation
    if ! bk pipeline validate --file "$TMP_YAML"; then
        echo "+++ ❌ Buildkite Validation Failed for $json_path"
        echo "Generated YAML output for debugging:"
        echo "---"
        echo "$GENERATED_YAML"
        echo "---"
        EXIT_STATUS=1
        FAILED_FILES+=("$json_path")
        continue
    fi

done <<< "$JSON_CHANGES"

if [ $EXIT_STATUS -ne 0 ]; then
    echo "--- ❌ Validation failed for the following files:"
    for f in "${FAILED_FILES[@]}"; do
        echo "  - $f"
    done
    echo "Please fix the errors in the JSON configurations listed above."
    exit 1
fi

echo "✅ Benchmark case validations passed."
