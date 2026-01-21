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
#
# Build vLLM-TPU with tpu-inference from PyPI.
# ./build_vllm_tpu.sh [--local] <tpu-inference-version> <vllm-tpu-version> [vllm-branch-tag-or-commit-hash](default: main)
# Example: ./build_vllm_tpu.sh 0.12.0rc1 0.12.0rc1 releases/v0.12.0
# ---------------------------

set -e

# LOCAL_BUILD: Determine whether to use a local wheel or a PyPI package for tpu-inference.
LOCAL_BUILD=0

# --- Check for local build flag ---
if [[ "$1" == "--local" ]]; then
    LOCAL_BUILD=1
    shift
fi

# --- Script Configuration ---
# TPU_INFERENCE_VERSION: PyPI version or local wheel
# VLLM_TPU_VERSION: Version assigned to the output wheel.
# VLLM_BRANCH: vLLM branch, tag or commit hash to build (default: main).
TPU_INFERENCE_VERSION=$1
VLLM_TPU_VERSION=$2
VLLM_BRANCH=${3:-"main"}
VLLM_REPO="https://github.com/vllm-project/vllm.git"
REPO_DIR="vllm"

# --- Argument Validation ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tpu-inference-version> <vllm-tpu-version> [vllm-branch-tag-or-commit-hash]"
    echo "  [vllm-branch-tag-or-commit-hash] is optional, defaults to 'main'."
    exit 1
fi

echo "--- Starting vLLM-TPU wheel build ---"
echo "TPU Inference Version: ${TPU_INFERENCE_VERSION}"
echo "vLLM-TPU Version: ${VLLM_TPU_VERSION}"
echo "vLLM Branch/Tag/Commit Hash: ${VLLM_BRANCH}"

# --- Step 1: Clone vLLM repository ---
if [ -d "$REPO_DIR" ]; then
    echo "Repository '$REPO_DIR' already exists. Skipping clone."
else
    echo "Cloning vLLM repository..."
    git clone ${VLLM_REPO}
fi
cd ${REPO_DIR}

# --- Step 1.5: Checkout the specified vLLM branch/tag/commit hash ---
echo "Checking out vLLM branch/tag/commit hash: ${VLLM_BRANCH}..."
if ! git checkout "${VLLM_BRANCH}"; then
    echo "ERROR: Failed to checkout branch/tag/commit hash '${VLLM_BRANCH}'. Please check the name of branch/tag/commit hash ."
    exit 1
fi
echo "Successfully checked out ${VLLM_BRANCH}."
git pull || echo "Warning: Failed to pull updates (may be on a tag)."

# --- Step 2: Update tpu-inference version in requirements ---
REQUIRED_LINE="tpu-inference==${TPU_INFERENCE_VERSION}"
REQUIREMENTS_FILE="requirements/tpu.txt"
BACKUP_FILE="${REQUIREMENTS_FILE}.bak"

if [ "$LOCAL_BUILD" == "1" ]; then
    # Determine whether to use a local wheel or a PyPI package for tpu-inference.
    echo "Using local wheel for tpu-inference"
    WHL_PATH="/workspace/tpu_inference/dist/tpu_inference-${TPU_INFERENCE_VERSION}-py3-none-any.whl"
    REQUIRED_LINE="tpu-inference @ file://${WHL_PATH}"

    if [ ! -f "${WHL_PATH}" ]; then
        echo "ERROR: Local wheel not found at ${WHL_PATH} Please ensure you have built tpu-inference first."
        exit 1
    fi
fi

echo "Updating tpu-inference version in $REQUIREMENTS_FILE..."

if [ -f "$REQUIREMENTS_FILE" ]; then
    # Check if the last character is NOT a newline. If not, append one.
    if [ "$(tail -c 1 "$REQUIREMENTS_FILE")" != "" ]; then
        echo "" >> "$REQUIREMENTS_FILE"
        echo "(Action: Added missing newline to the end of $REQUIREMENTS_FILE for safety.)"
    fi
fi

if grep -q "^tpu-inference==" "$REQUIREMENTS_FILE"; then
    # Replace the existing version using sed, which creates the .bak file
    echo "(Action: Existing version found. Replacing.)"
    sed -i.bak "s|^tpu-inference==.*|$REQUIRED_LINE|" "$REQUIREMENTS_FILE"

else
    # Line not found -> Append the new line to the file end, and manually create .bak
    echo "(Action: Line not found. Appending new dependency.)"
    echo "$REQUIRED_LINE" >> "$REQUIREMENTS_FILE"

    # Create an empty .bak file for consistency, so cleanup works later.
    touch "$BACKUP_FILE"
fi

# --- Step 3: Execute the vLLM TPU build script ---
echo "Ensuring 'build' package is installed..."
pip install build
echo "Executing the vLLM TPU build script..."
bash tools/vllm-tpu/build.sh "${VLLM_TPU_VERSION}"

echo "--- Build complete! ---"
echo "The wheel file can be found in the 'vllm/dist' directory."

# --- Step 4: Cleanup and Revert Requirements File ---
echo "--- Cleaning up local changes ---"

if [ -f "$BACKUP_FILE" ]; then
    echo "Reverting $REQUIREMENTS_FILE from backup."
    # Remove the modified file
    rm -f "$REQUIREMENTS_FILE"
    # Rename the backup file back to the original name
    mv "$BACKUP_FILE" "$REQUIREMENTS_FILE"
else
    echo "Warning: Backup file $BACKUP_FILE not found. Skipping revert."
fi

echo "Cleanup complete. Script finished."
