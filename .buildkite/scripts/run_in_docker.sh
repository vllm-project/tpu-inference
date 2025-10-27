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

# (TODO): Consider creating a remote registry to cache and share between agents.
# Subsequent builds on the same host should be cached.

# Cleanup of existing containers and images.
echo "Starting cleanup for vllm-tpu..."
# Get all unique image IDs for the repository 'vllm-tpu'
old_images=$(docker images vllm-tpu -q | uniq)
total_containers=""

if [ -n "$old_images" ]; then
    echo "Found old vllm-tpu images. Checking for dependent containers..."
    # Loop through each image ID and find any containers (running or not) using it.
    for img_id in $old_images; do
        total_containers="$total_containers $(docker ps -a -q --filter "ancestor=$img_id")"
    done

    # Remove any found containers
    if [ -n "$total_containers" ]; then
        echo "Removing leftover containers using vllm-tpu image(s)..."
        echo "$total_containers" | xargs -n1 | sort -u | xargs -r docker rm -f
    fi

    echo "Removing old vllm-tpu image(s)..."
    docker rmi -f "$old_images"
else
    echo "No vllm-tpu images found to clean up."
fi

echo "Pruning old Docker build cache..."
docker builder prune -f

echo "Cleanup complete."

IMAGE_NAME="vllm-tpu"
docker build --no-cache -f docker/Dockerfile -t "${IMAGE_NAME}:${BUILDKITE_COMMIT}" .

exec docker run \
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
  "${IMAGE_NAME}:${BUILDKITE_COMMIT}" \
  "$@" # Pass all script arguments as the command to run in the container
