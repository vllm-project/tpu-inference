#!/bin/bash

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

setup_environment() {
  local image_name_param=${1:-"vllm-tpu"}
  IMAGE_NAME="$image_name_param"

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

  echo "Installing Python dependencies"
  python3 -m pip install --progress-bar off buildkite-test-collector==0.1.9
  echo "Python dependencies installed"
  if [ -z "${BUILDKITE:-}" ]; then
      VLLM_COMMIT_HASH=""
      BUILDKITE_COMMIT=$(git log -n 1 --pretty="%H")
  else
      VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")
  fi

  docker build \
      --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
      --no-cache -f docker/Dockerfile -t "${IMAGE_NAME}:${BUILDKITE_COMMIT}" .
}
