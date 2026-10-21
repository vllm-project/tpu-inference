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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Assign the first argument to a local variable
RAW_FILES_TO_CHECK="${1:-}"

# Define directories relative to the repo root for exact matching
EXCLUDED_FOLDERS=(
    "\.buildkite/kubernetes/"
    "\.buildkite/benchmark/lm_eval/"
)

# Convert the array into a pipe-separated string for regex
EXCLUDE_PATTERN=$(printf "|%s" "${EXCLUDED_FOLDERS[@]}")
EXCLUDE_PATTERN=${EXCLUDE_PATTERN:1}

# Filter: Include YAML files in .buildkite/ while skipping excluded patterns
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | \
    grep -E "^\.buildkite/.*\.ya?ml$" | \
    grep -Ev "^($EXCLUDE_PATTERN)" || true)

# Early exit: If no YAML files were modified, skip validation
if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :fast_forward: No applicable YAML changes found (none detected or all excluded). Skipping validation."
    exit 0
fi

VALIDATE_ARGS=()

echo "--- 📂 Preparing all files for validation"

# Iterate through the list to build the arguments array and check file existence
while IFS= read -r file; do
    # Skip empty lines to prevent errors
    [ -z "$file" ] && continue

    # Check if the file still exists (to handle deleted files in a PR)
    if [ ! -f "$file" ]; then
        echo "Skipping deleted file: $file"
        continue
    fi

    echo "Adding to validation list: $file"
    VALIDATE_ARGS+=("--file" "$file")

done < <(echo "$YAML_FILES_TO_CHECK")

echo "--- 🔍 Executing Buildkite syntax validation..."
if [ ${#VALIDATE_ARGS[@]} -gt 0 ]; then
    # TODO: Currently using standard 'bk pipeline validate' for syntax checking.
    # In the future, this could be extended to include more complex validation logic.
    if ! bk pipeline validate "${VALIDATE_ARGS[@]}"; then
        echo "+++ ❌ Validation Failed"
        echo "Result: FAIL (Please fix the YAML syntax errors above)"
        exit 1
    else
        echo "✅ YAML syntax validation passed."
        echo "Result: SUCCESS"
        exit 0
    fi
else
    echo "--- :v: No existing YAML files found to validate."
    exit 0
fi
