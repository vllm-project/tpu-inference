#!/bin/bash
# Copyright 2025 Google LLC
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

setup_environment() {
  local image_name_param=${1:-"vllm-tpu"}
  IMAGE_NAME="$image_name_param"

  local DOCKERFILE_NAME="Dockerfile"

# Determine whether to build from PyPI packages or source.
  if [[ "${RUN_WITH_PYPI:-false}" == "true" ]]; then
    DOCKERFILE_NAME="Dockerfile.pypi"
    echo "Building from PyPI packages. Using docker/${DOCKERFILE_NAME}"
  else
    echo "Building from source. Using docker/${DOCKERFILE_NAME}"
  fi

  if ! grep -q "^HF_TOKEN=" /etc/environment; then
    gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
    sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)"
    echo "Added HF_TOKEN to /etc/environment."
  else
    echo "HF_TOKEN already exists in /etc/environment."
  fi

  # shellcheck disable=1091
  source /etc/environment

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

  # load VLLM_COMMIT_HASH from .env file
  if [ -f .env ]; then
      export "$(grep -v '^#' .env | xargs)"
  fi

  if [ -z "${BUILDKITE:-}" ]; then
      VLLM_COMMIT_HASH=""
      TPU_INFERENCE_HASH=$(git log -n 1 --pretty="%H")
  else
      VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")
      TPU_INFERENCE_HASH="$BUILDKITE_COMMIT"
  fi

  docker build \
      --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
      --build-arg IS_FOR_V7X="${IS_FOR_V7X:-false}" \
      --build-arg IS_TEST="true" \
      --no-cache -f docker/"${DOCKERFILE_NAME}" -t "${IMAGE_NAME}:${TPU_INFERENCE_HASH}" .
}
