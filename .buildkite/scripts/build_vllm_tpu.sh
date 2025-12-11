#!/bin/bash

set -e

# --- Script Configuration ---
TPU_INFERENCE_VERSION=$1
VLLM_TPU_VERSION=$2
VLLM_BRANCH=${3:-"main"}
VLLM_REPO="https://github.com/vllm-project/vllm.git"
REPO_DIR="vllm"

# --- Argument Validation ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tpu-inference-version> <vllm-tpu-version> [vllm-branch-or-tag]"
    echo "  [vllm-branch-or-tag] is optional, defaults to 'main'."
    exit 1
fi

echo "--- Starting vLLM-TPU wheel build ---"
echo "TPU Inference Version: ${TPU_INFERENCE_VERSION}"
echo "vLLM-TPU Version: ${VLLM_TPU_VERSION}"
echo "vLLM Branch/Tag: ${VLLM_BRANCH}"

# --- Step 1: Clone vLLM repository ---
if [ -d "$REPO_DIR" ]; then
    echo "Repository '$REPO_DIR' already exists. Skipping clone."
else
    echo "Cloning vLLM repository..."
    git clone ${VLLM_REPO}
fi
cd ${REPO_DIR}

# --- Step 1.5: Checkout the specified vLLM branch/tag ---
echo "Checking out vLLM branch/tag: ${VLLM_BRANCH}..."
if ! git checkout "${VLLM_BRANCH}"; then
    echo "ERROR: Failed to checkout branch/tag '${VLLM_BRANCH}'. Please check the branch/tag name."
    exit 1
fi
echo "Successfully checked out ${VLLM_BRANCH}."
git pull || echo "Warning: Failed to pull updates (may be on a tag)."

# --- Step 2: Update tpu-inference version in requirements ---
REQUIRED_LINE="tpu-inference==${TPU_INFERENCE_VERSION}"
REQUIREMENTS_FILE="requirements/tpu.txt"
BACKUP_FILE="${REQUIREMENTS_FILE}.bak"

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
    sed -i.bak "s/^tpu-inference==.*/$REQUIRED_LINE/" "$REQUIREMENTS_FILE"

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
