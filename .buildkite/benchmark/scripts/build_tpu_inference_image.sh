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

VLLM_COMMIT_HASH=$1
CODE_HASH=$2

IMAGE_TAG="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPO/vllm-tpu:$CODE_HASH"

gcloud auth configure-docker $GCP_REGION-docker.pkg.dev --quiet
echo "Image tag: $IMAGE_TAG"

# Check if image exists remotely
echo "Checking remote registry..."
if gcloud artifacts docker images describe "$IMAGE_TAG" --format='value(image_summary.digest)' &>/dev/null; then
    echo "Remote image $IMAGE_TAG already exists. Skipping build and push."
    exit 0
fi

cleanup_image() {
    echo "--- Cleanup docker image ---"
    if [[ -n "$IMAGE_TAG" ]]; then
        echo "Removing Docker image: $IMAGE_TAG"
        docker rmi "$IMAGE_TAG" 2>/dev/null || true
    fi
}

VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
--build-arg BASE_IMAGE="python:3.12-slim-bookworm" \
--build-arg VLLM_COMMIT_HASH="$VLLM_COMMIT_HASH" \
--build-arg IS_FOR_V7X=true \
--tag $IMAGE_TAG \
--no-cache -f "docker/Dockerfile" .

trap cleanup_image EXIT

echo "--- Push Docker Image: $IMAGE_TAG ---"
docker push "$IMAGE_TAG"
