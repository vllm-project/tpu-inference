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

cleanup() {
  echo "--- Cleanup docker container and image ---"
  # Get all unique image IDs for the repository 'vllm-tpu'
  old_images=$(docker images "${IMAGE_TAG%:*}" -q | uniq)
  total_containers=""

  if [ -n "$old_images" ]; then
    echo "Found old vllm-tpu images. Checking for dependent containers..."
    # Loop through each image ID and find any containers (running or not) using it.
    for img_id in $old_images; do
      echo "find $img_id..."
      total_containers="$total_containers $(docker ps -a -q --filter "ancestor=$IMAGE_TAG")"
    done

    # Remove any found containers
    if [ -n "$total_containers" ]; then
      echo "Removing leftover containers using vllm-tpu image(s)..."
      for container_id in $total_containers; do
        echo "try removing $container_id..."
        docker rm -f "$container_id" || true
      done
    fi

    echo "Removing old vllm-tpu image..."
    docker rmi -f "$IMAGE_TAG" || true
    echo "Cleanup complete."
  else
    echo "No vllm-tpu images found to clean up."
  fi
}

ENV_VARS=(
  -e TEST_MODEL="${TEST_MODEL:-}"
  -e MINIMUM_ACCURACY_THRESHOLD="${MINIMUM_ACCURACY_THRESHOLD:-}"
  -e MINIMUM_THROUGHPUT_THRESHOLD="${MINIMUM_THROUGHPUT_THRESHOLD:-}"
  -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
  -e INPUT_LEN="${INPUT_LEN:-}"
  -e OUTPUT_LEN="${OUTPUT_LEN:-}"
  -e PREFIX_LEN="${PREFIX_LEN:-}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
)

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

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
else
  echo "Error: Failed to create $persist_cache_dir"
  exit 1
fi
DOCKER_HF_HOME="/tmp/hf_home"

export LOCATION="asia-south1"
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev -q
TAG_FILE_NAME="vllm_image_tag"
buildkite-agent artifact download $TAG_FILE_NAME .
IMAGE_TAG=$(cat $TAG_FILE_NAME)
# Clean tag file
rm $TAG_FILE_NAME
export IMAGE_TAG

if [ -z "$IMAGE_TAG" ]; then
  echo "Error: Can't get Image Tag"
  exit 1
fi
trap cleanup EXIT

echo "Image to pull: ${IMAGE_TAG}"
docker pull "${IMAGE_TAG}"

echo "--- Running Docker Container ---"

# To correctly drive the `trap` command, the `exec` command is not used.
docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
  "${ENV_VARS[@]}" \
  -e HF_HOME="$DOCKER_HF_HOME" \
  -e MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH="$DOCKER_HF_HOME/.cache/jax_cache" \
  -e VLLM_USE_V1=1 \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  ${QUANTIZATION:+-e QUANTIZATION="$QUANTIZATION"} \
  ${NEW_MODEL_DESIGN:+-e NEW_MODEL_DESIGN="$NEW_MODEL_DESIGN"} \
  ${USE_V6E8_QUEUE:+-e USE_V6E8_QUEUE="$USE_V6E8_QUEUE"} \
  ${JAX_RANDOM_WEIGHTS:+-e JAX_RANDOM_WEIGHTS="$JAX_RANDOM_WEIGHTS"} \
  ${SKIP_ACCURACY_TESTS:+-e SKIP_ACCURACY_TESTS="$SKIP_ACCURACY_TESTS"} \
  ${VLLM_MLA_DISABLE:+-e VLLM_MLA_DISABLE="$VLLM_MLA_DISABLE"} \
  "${IMAGE_TAG}" \
  "$@" # Pass all script arguments as the command to run in the container
