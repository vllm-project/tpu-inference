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

cleanup_docker_resource() {
  # Define defaults and get the parameter
  DEFAULT_IMAGES=("vllm-tpu")
  IMAGE_NAME="${1:-}"

  # Combine image sets
  COMBINED_IMAGES=("${DEFAULT_IMAGES[@]}")
  if [[ -n "${IMAGE_NAME}" ]]; then
    COMBINED_IMAGES+=("${IMAGE_NAME}")
  fi

  # Deduplicate arrays and remove empty strings
  mapfile -t TARGET_IMAGES < <(printf "%s\n" "${COMBINED_IMAGES[@]}" | sort -u | grep -v '^$')

  if [[ ${#TARGET_IMAGES[@]} -eq 0 ]]; then
    echo "No target images provided for cleanup."
    exit 0
  fi

  echo "Images to clean up : ${TARGET_IMAGES[*]}"

  # Iterate and cleanup
  for IMG in "${TARGET_IMAGES[@]}"; do
    echo "----------------------------------------"
    echo "Starting cleanup for ${IMG}"

    # Use format to get "Repository ID" and use awk for exact or suffix matching.
    # $1 == img          -> Matches exact local image (e.g., "vllm-tpu")
    # $1 ~ "/"img"$"     -> Matches Google Artifact Registry paths (e.g., ".../.../vllm-tpu")
    OLD_IMAGES=$(docker images --format '{{.Repository}} {{.ID}}' | awk -v img="${IMG}" '$1 == img || $1 ~ "/"img"$" {print $2}' | sort -u)
    
    if [[ -n "$OLD_IMAGES" ]]; then
      echo "Found matching images for ${IMG}. Checking for dependent containers..."
      
      TOTAL_CONTAINERS=""
      for img_id in $OLD_IMAGES; do
        # Find containers using this image ID
        TOTAL_CONTAINERS="$TOTAL_CONTAINERS $(docker ps -a -q --filter "ancestor=$img_id")"
      done
      
      # Format and remove any found containers
      CLEANED_CONTAINERS=$(echo "$TOTAL_CONTAINERS" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
      if [[ -n "$CLEANED_CONTAINERS" ]]; then
        echo "Removing leftover containers using ${IMG} image(s)..."
        echo "$CLEANED_CONTAINERS" | xargs -r docker rm -f
      fi
      
      echo "Removing old ${IMG} image(s) by ID..."
      # Using ID directly ensures all tags of that specific image are untagged and removed
      echo "$OLD_IMAGES" | xargs -r docker rmi -f
    else
      echo "No images matching ${IMG} found to clean up."
    fi
  done

  echo "Pruning old Docker build cache..."
  docker builder prune -f

  echo "Cleanup complete."
}

setup_environment() {
  local image_name_param=${1:-"vllm-tpu"}
  local should_push=${2:-"false"}
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
  cleanup_docker_resource "${IMAGE_NAME}"

  if [ -z "${BUILDKITE:-}" ]; then
      if [ "${USE_VLLM_LKG:-false}" == "true" ] && [ -f ".buildkite/vllm_lkg.version" ]; then
          VLLM_COMMIT_HASH=$(cat .buildkite/vllm_lkg.version)
      else
          VLLM_COMMIT_HASH=""
      fi
      TPU_INFERENCE_HASH=$(git log -n 1 --pretty="%H")
  else
      VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")
      TPU_INFERENCE_HASH="$BUILDKITE_COMMIT"
  fi
 
  # Build with specific hash and 'latest' tag for convenience
  docker build \
      --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
      --build-arg IS_TEST="true" \
      --no-cache -f docker/"${DOCKERFILE_NAME}" \
      -t "${IMAGE_NAME}:${TPU_INFERENCE_HASH}" \
      -t "${IMAGE_NAME}:latest" .

  # Push logic if requested
  if [[ "$should_push" == "true" ]]; then
    echo "--- Pushing Docker image(s) to registry..."
    gcloud auth configure-docker us-central1-docker.pkg.dev
    docker push "${IMAGE_NAME}:${TPU_INFERENCE_HASH}"
    docker push "${IMAGE_NAME}:latest"
  fi
}
