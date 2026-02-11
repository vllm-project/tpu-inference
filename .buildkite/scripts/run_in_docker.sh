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

#
# .buildkite/run_in_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
fi

# Default to disabling the embedded vLLM server so tests start clean unless explicitly overridden.
DISABLE_VLLM_SERVER="${DISABLE_VLLM_SERVER:-true}"
export DISABLE_VLLM_SERVER

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
  -e DISABLE_VLLM_SERVER="${DISABLE_VLLM_SERVER:-}"
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

TEST_SUITE_VARS=(
  -e BUILDKITE_ANALYTICS_TOKEN="${BUILDKITE_ANALYTICS_TOKEN:-}"
  -e BUILDKITE_BUILD_ID="${BUILDKITE_BUILD_ID:-}"
  -e BUILDKITE_BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER:-}"
  -e BUILDKITE_JOB_ID="${BUILDKITE_JOB_ID:-}"
  -e BUILDKITE_BRANCH="${BUILDKITE_BRANCH:-}"
  -e BUILDKITE_COMMIT="${BUILDKITE_COMMIT:-}"
  -e BUILDKITE_MESSAGE="${BUILDKITE_MESSAGE:-}"
  -e BUILDKITE_BUILD_URL="${BUILDKITE_BUILD_URL:-}"
)

DOCKER_HF_HOME="/tmp/hf_home"

# Try to cache HF models
persist_cache_dir="/mnt/disks/persist/models"

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
else
  echo "Error: Failed to create $persist_cache_dir"
  exit 1
fi

# Some test scripts set tp=2 on IS_FOR_V7X=true to mitigate test failures.
# TODO (Qiliang Cui) Investigate why tensor-parallel-size=1 breaks in tpu7x.

# Debug: Show processes currently holding TPU devices before starting the container.
echo "=== [DEBUG] TPU device status before docker run ==="
echo "--- Processes using /dev/accel* (TPU devices) ---"
if ls /dev/accel* &>/dev/null; then
  fuser -v /dev/accel* 2>&1 || echo "(no processes using /dev/accel*)"
else
  echo "(no /dev/accel* devices found on host)"
fi
echo "--- Processes using /dev/vfio/* (TPU vfio devices) ---"
if ls /dev/vfio/* &>/dev/null; then
  fuser -v /dev/vfio/* 2>&1 || echo "(no processes using /dev/vfio/*)"
else
  echo "(no /dev/vfio/* devices found on host)"
fi
echo "--- Running docker containers ---"
docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" 2>/dev/null || true
echo "--- Processes matching python/vllm/jax ---"
ps aux | grep -E 'python|vllm|jax' | grep -v grep || echo "(none found)"
echo "=== [DEBUG] End TPU device status ==="

exec docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
  "${ENV_VARS[@]}" \
  "${TEST_SUITE_VARS[@]}" \
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
