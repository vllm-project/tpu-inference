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

# set -e

# # --- Buildkite CLI Diagnostic Section Start ---
# echo "=== Buildkite CLI Environment Diagnostics ==="

# # 1. Check if 'bk' exists in PATH
# if command -v bk >/dev/null 2>&1; then
#     echo "[SUCCESS] Buildkite CLI found at $(which bk)"
#     bk --version
# else
#     echo "[ERROR] 'bk' command not found in PATH."
# fi

# echo "------------------------------------------------"

# # 2. Check for API Connectivity
# if bk pipeline list --limit 1 >/dev/null 2>&1; then
#     echo "[SUCCESS] API Token is valid and functional."
# else
#     echo "[WARNING] API Token check failed."
#     echo "          The 'bk pipeline validate' will run in offline mode (syntax only)."
# fi

# echo "------------------------------------------------"

# # 3. Perform Validation
# PIPELINE_FILE=".buildkite/main.yml"
# if [ -f "$PIPELINE_FILE" ]; then
#     echo "Testing validation for: $PIPELINE_FILE"
#     if bk pipeline validate -f "$PIPELINE_FILE"; then
#         echo "[SUCCESS] Pipeline configuration is valid."
#     else
#         echo "[FAILURE] Pipeline validation failed!"
#     fi
# else
#     echo "Notice: $PIPELINE_FILE not found, skipping validation."
# fi

# echo "=== End of Diagnostics ==="
# # --- Buildkite CLI Diagnostic Section End ---

set -e

IMAGE_NAME="validate-yml-test"

# Get the organization slug (If testing locally outside CI, replace with your actual slug)
ORG_SLUG=${BUILDKITE_ORGANIZATION_SLUG:-"dennis-organization-slug"}

cleanup() {
    # Cleanup of existing containers and images.
    echo "Starting cleanup for ${IMAGE_NAME}..."
    # Get all unique image IDs for the repository
    old_images=$(docker images "${IMAGE_NAME}" -q | uniq)
    total_containers=""

    if [ -n "$old_images" ]; then
        echo "Found old ${IMAGE_NAME} images. Checking for dependent containers..."
        # Loop through each image ID and find any containers (running or not) using it.
        for img_id in $old_images;
        do
            total_containers="$total_containers $(docker ps -a -q --filter "ancestor=$img_id")"
        done

        # Remove any found containers
        if [ -n "$total_containers" ]; then
            echo "Removing leftover containers using ${IMAGE_NAME} image(s)..."
            echo "$total_containers" | xargs -n1 | sort -u | xargs -r docker rm -f
        fi

        echo "Removing old ${IMAGE_NAME} image(s)..."
        docker rmi -f "$old_images"
    else
        echo "No ${IMAGE_NAME} images found to clean up."
    fi

    echo "Pruning old Docker build cache..."
    docker builder prune -f

    echo "Cleanup complete."
}

trap cleanup EXIT

echo "--- 1. Building Docker Image"
docker build --no-cache -f docker/Dockerfile.validateyml -t "${IMAGE_NAME}" .

echo "--- 2. Extracting Token from Agent configuration"
# sudo is required because /etc/buildkite-agent/ is usually restricted to root
if [ -f /etc/buildkite-agent/buildkite-agent.cfg ]; then
    ls -l /etc/buildkite-agent/buildkite-agent.cfg
    AGENT_TOKEN=$(grep '^token=' /etc/buildkite-agent/buildkite-agent.cfg | cut -d'"' -f2)
    echo "✅ Token extracted successfully (Prefix: ${AGENT_TOKEN:0:4}...)"
else
    echo "❌ Configuration file not found. Please check the path."
    exit 1
fi



echo "--- 3. Running Container and verifying bk CLI"
# Execute 'bk configure' inside the container followed by a verification command
# docker run --rm \
#   -e BK_TOKEN="$AGENT_TOKEN" \
#   -e BK_ORG="$ORG_SLUG" \
#   "$IMAGE_NAME" \
#   bash -c "
#     echo 'Configuring bk CLI...'
#     bk configure --org \"\$BK_ORG\" --token \"\$BK_TOKEN\"
    
#     echo 'Verifying connection (executing: bk agent list)...'
#     bk agent list --limit 1
#   "

# echo "--- ✅ Test sequence completed"

docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -e BK_TOKEN="$AGENT_TOKEN" \
  -e BK_ORG="$ORG_SLUG" \
  "$IMAGE_NAME" \
  "$@"
