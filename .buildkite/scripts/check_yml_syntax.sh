#!/bin/sh
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

# --- Buildkite CLI Diagnostic Section Start ---
echo "=== Buildkite CLI Environment Diagnostics ==="

# 1. Check if 'bk' exists in PATH
if command -v bk >/dev/null 2>&1; then
    echo "[SUCCESS] Buildkite CLI found at $(which bk)"
    bk --version
else
    echo "[ERROR] 'bk' command not found in PATH."
fi

echo "------------------------------------------------"

# 2. Check for API Connectivity
if bk pipeline list --limit 1 >/dev/null 2>&1; then
    echo "[SUCCESS] API Token is valid and functional."
else
    echo "[WARNING] API Token check failed."
    echo "          The 'bk pipeline validate' will run in offline mode (syntax only)."
fi

echo "------------------------------------------------"

# 3. Perform Validation
PIPELINE_FILE=".buildkite/main.yml"
if [ -f "$PIPELINE_FILE" ]; then
    echo "Testing validation for: $PIPELINE_FILE"
    if bk pipeline validate -f "$PIPELINE_FILE"; then
        echo "[SUCCESS] Pipeline configuration is valid."
    else
        echo "[FAILURE] Pipeline validation failed!"
    fi
else
    echo "Notice: $PIPELINE_FILE not found, skipping validation."
fi

echo "=== End of Diagnostics ==="
# --- Buildkite CLI Diagnostic Section End ---