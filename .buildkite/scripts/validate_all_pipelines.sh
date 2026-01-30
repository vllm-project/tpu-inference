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

# Pre-filter: Only keep .yml or .yaml files from the input list
# Using '|| true' to prevent the script from exiting if no matches are found
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | grep -E "\.ya?ml$" || true)

# Early exit: If no YAML files were modified, skip validation
if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :crossed_fingers: No YAML changes detected. Skipping validation."
    exit 0
fi

VALIDATE_ARGS=()

echo "--- 📂 Preparing files for validation"

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

echo "--- 🔍 Validating changed YAML files"
if [ ${#VALIDATE_ARGS[@]} -gt 0 ]; then
    if ! bk pipeline validate "${VALIDATE_ARGS[@]}"; then
        echo "+++ ❌ Validation Failed"
        echo "Result: FAIL (Please fix the YAML syntax errors above)"
        exit 1
    else
        echo "+++ ✅ Validation Successful"
        echo "Result: SUCCESS"
        exit 0
    fi
else
    echo "--- :v: No existing YAML files found to validate."
    exit 0
fi
