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

# Environment variables for docker run
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

if [ -z "${MODEL_IMPL_TYPE:-}" ]; then
    MODEL_IMPL_TYPE=flax_nnx
fi

IMAGE_NAME='vllm-tpu'
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
setup_environment $IMAGE_NAME

DOCKER_HF_HOME="/tmp/hf_home"

# Try to cache HF models
persist_cache_dir="/mnt/disks/persist/models"

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
else
  echo "Error: Failed to create $persist_cache_dir"
  exit 1
fi

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
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  ${QUANTIZATION:+-e QUANTIZATION="$QUANTIZATION"} \
  ${NEW_MODEL_DESIGN:+-e NEW_MODEL_DESIGN="$NEW_MODEL_DESIGN"} \
  ${USE_V6E8_QUEUE:+-e USE_V6E8_QUEUE="$USE_V6E8_QUEUE"} \
  ${IS_FOR_V7X:+-e IS_FOR_V7X="$IS_FOR_V7X"} \
  ${SKIP_ACCURACY_TESTS:+-e SKIP_ACCURACY_TESTS="$SKIP_ACCURACY_TESTS"} \
  ${VLLM_MLA_DISABLE:+-e VLLM_MLA_DISABLE="$VLLM_MLA_DISABLE"} \
  "${IMAGE_NAME}:${BUILDKITE_COMMIT}" \
  "$@" # Pass all script arguments as the command to run in the container
