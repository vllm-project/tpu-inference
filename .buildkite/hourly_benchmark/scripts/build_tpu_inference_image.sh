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
TPU_INFERENCE_HASH=$2
CODE_HASH="${VLLM_HASH}-${TPU_INFERENCE_HASH}-"

# southamerica-west1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-bm-bk/vllm-tpu:4c4b6f7a9-4e6e6fb4-
IMAGE_TAG="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPO/vllm-tpu:$CODE_HASH"

gcloud auth configure-docker ${region}-docker.pkg.dev --quiet
echo "Image tag: $IMAGE_TAG"

# Check if image exists remotely
echo "Checking remote registry..."
if gcloud artifacts docker images describe "$IMAGE_TAG" --format='value(image_summary.digest)' &>/dev/null; then
    echo "Remote image $IMAGE_TAG already exists. Skipping build and push."
    exit 0
fi

local DOCKERFILE="Dockerfile"

VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
--build-arg BASE_IMAGE="python:3.12-slim-bookworm" \
--build-arg VLLM_COMMIT_HASH="$VLLM_COMMIT_HASH" \
--tag $IMAGE_TAG \
--no-cache -f "docker/${DOCKERFILE}" .

docker push "$IMAGE_TAG"
