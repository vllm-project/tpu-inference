#!/bin/bash
#
# .buildkite/run_in_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
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

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
  echo "ERROR: BUILDKITE_COMMIT environment variable is not set." >&2
  echo "This script expects BUILDKITE_COMMIT to tag the Docker image." >&2
  exit 1
fi

if [ -z "${MODEL_IMPL_TYPE:-}" ]; then
  MODEL_IMPL_TYPE=flax_nnx
fi

# Try to cache HF models
persist_cache_dir="/mnt/disks/persist/models"
home_cache_dir="$HOME/models"

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
elif ( mkdir -p "$home_cache_dir" ); then
  LOCAL_HF_HOME="$home_cache_dir"
else
  echo "Error: Failed to create either $persist_cache_dir or $home_cache_dir"
  exit 1
fi
DOCKER_HF_HOME="/tmp/hf_home"

# (TODO): Consider creating a remote registry to cache and share between agents.
# Subsequent builds on the same host should be cached.

export LOCATION="us-central1"
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev -q

buildkite-agent artifact download image_tag.txt .
export IMAGE_TAG=$(cat image_tag.txt)

if [ -z "$IMAGE_TAG" ]; then
    echo "Error: Can't get Image Tag"
    exit 1
fi

echo "Image to pull: ${IMAGE_TAG}"
docker pull "${IMAGE_TAG}"

echo "--- Running Docker Container ---"

exec docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
  -e HF_HOME="$DOCKER_HF_HOME" \
  -e MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH="$DOCKER_HF_HOME/.cache/jax_cache" \
  -e VLLM_USE_V1=1 \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  ${QUANTIZATION:+-e QUANTIZATION="$QUANTIZATION"} \
  ${NEW_MODEL_DESIGN:+-e NEW_MODEL_DESIGN="$NEW_MODEL_DESIGN"} \
  ${USE_V6E8_QUEUE:+-e USE_V6E8_QUEUE="$USE_V6E8_QUEUE"} \
  "${IMAGE_TAG}" \
  "$@" # Pass all script arguments as the command to run in the container
