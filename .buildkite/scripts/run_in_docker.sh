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

# Centrally default to using prebuilt image when running in Buildkite CI
if [ -n "${BUILDKITE:-}" ]; then
  export USE_PREBUILT_IMAGE="${USE_PREBUILT_IMAGE:-1}"
fi

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
fi

declare -a BENCHMARK_DOCKER_ARGS=()

# Check if the serialized string exists and is not empty.
if [ -n "${BENCHMARK_DOCKER_ARGS_STR:-}" ]; then
  mapfile -t BENCHMARK_DOCKER_ARGS <<< "${BENCHMARK_DOCKER_ARGS_STR}"
fi
printf "[INFO] %s = %s\n" "BENCHMARK_DOCKER_ARGS" "${BENCHMARK_DOCKER_ARGS[*]}"

# TODO(Qiliang Cui): This is temp solution to mitigate the docker image
#     not cleaned issue when migrating benchmark to buildkite.
docker rm -f vllm-tpu || true

# Environment variables for docker run
ENV_VARS=(
  -e TEST_MODEL="${TEST_MODEL:-}"
  -e MINIMUM_ACCURACY_THRESHOLD="${MINIMUM_ACCURACY_THRESHOLD:-}"
  -e MINIMUM_THROUGHPUT_THRESHOLD="${MINIMUM_THROUGHPUT_THRESHOLD:-}"
  -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
  -e TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-}"
  -e INPUT_LEN="${INPUT_LEN:-}"
  -e OUTPUT_LEN="${OUTPUT_LEN:-}"
  -e PREFIX_LEN="${PREFIX_LEN:-}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
  -e USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-}"
  -e BENCH_DATASET="${BENCH_DATASET:-}"
  -e USE_BATCHED_RPA_KERNEL="${USE_BATCHED_RPA_KERNEL:-}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
  # For kernel tuning pipeline env vars
  -e KERNEL_TUNING_CASE_SET_ID="${KERNEL_TUNING_CASE_SET_ID:-}"
  -e KERNEL_TUNING_RUN_ID="${KERNEL_TUNING_RUN_ID:-}"
  -e KERNEL_TUNING_KERNEL_NAME="${KERNEL_TUNING_KERNEL_NAME:-}"
  -e KERNEL_TUNING_CASE_SET_DESC="${KERNEL_TUNING_CASE_SET_DESC:-}"
  -e KERNEL_TUNING_TPU_VERSION="${KERNEL_TUNING_TPU_VERSION:-}"
  -e KERNEL_TUNING_TPU_CORES="${KERNEL_TUNING_TPU_CORES:-}"
  -e KERNEL_TUNING_JOB_PRIORITY="${PRIORITY_KERNEL_TUNING:--10}"
  -e KERNEL_TUNING_MAX_EXECUTION_MINUTES="${KERNEL_TUNING_MAX_EXECUTION_MINUTES:-20}"
  -e HOST_NAME="${HOST_NAME:-}"
  -e GCS_BUCKET="${GCS_BUCKET:-}"
)

if [ -z "${MODEL_IMPL_TYPE:-}" ]; then
    MODEL_IMPL_TYPE=auto
fi

IMAGE_NAME='vllm-tpu'
declare -a DEV_MOUNT=()
if [[ "${DEV_MODE:-false}" == "true" ]]; then
    FULL_IMAGE_TAG="${IMAGE_NAME}:dev"
    DEV_MOUNT+=("-v" "$(pwd):/workspace/tpu_inference")
else
    FULL_IMAGE_TAG="${IMAGE_NAME}:${BUILDKITE_COMMIT}"
fi
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

# ==========================================
# 1. Cache Setup (HF Models & JAX Compilations)
# ==========================================

# Try to cache HF models
persist_cache_dir="/mnt/disks/persist/models"

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
else
  echo "Error: Failed to create $persist_cache_dir"
  exit 1
fi

# Temporary directory for kernel tuning outputs (not persisted)
KERNEL_TUNING_TMP_DIR="/tmp/kernel_tuning"
mkdir -p "$KERNEL_TUNING_TMP_DIR"

# Try to cache the JAX compilations.
GCS_CACHE_BASE="gs://ullm-ci-cache/jax_cache"
echo "[INFO] Probing JAX version from docker image..."
JAX_VERSION=$(docker run --rm "$FULL_IMAGE_TAG" python3 -c "import jax; print(jax.__version__)")
echo "[INFO] Detected JAX Version: ${JAX_VERSION}"

# Centralized GCS cache path
CACHE_NAMESPACE="jax${JAX_VERSION}_tpu${TPU_VERSION:-tpu6e}"
FINAL_CACHE_PATH="${GCS_CACHE_BASE}/${CACHE_NAMESPACE}"

LOCAL_JAX_CACHE_DIR="/mnt/disks/persist/tpu_jax_cache/${CACHE_NAMESPACE}"

if ! mkdir -p "$LOCAL_JAX_CACHE_DIR"; then
  echo "[ERROR] Failed to create $LOCAL_JAX_CACHE_DIR on persistent disk."
  exit 1
fi
echo "[INFO] Pulling JAX Cache from GCS to local directory..."
# Parallel CI builds‘ pushes are safe because JAX's compilation cache 
# entries are content-addressed. Concurrent pushes are thus idempotent;
gsutil -m rsync -d -r "$FINAL_CACHE_PATH" "$LOCAL_JAX_CACHE_DIR" || echo "[WARN] Failed to pull JAX Cache from GCS. Proceeding with cold start."

# ==========================================
# 2. Run Docker Container
# ==========================================
set +e # Temporarily disable exit on error to capture exit code

# Ensure the docker container is killed if the wrapper script exits, fails, or is cancelled.
trap 'docker kill "$IMAGE_NAME" 2>/dev/null || true' EXIT INT TERM

# Some test scripts set tp=2 on TPU_VERSION=tpu7x to mitigate test failures.
# TODO (Qiliang Cui) Investigate why tensor-parallel-size=1 breaks in tpu7x.

# -----------------------------------------------------------------------------
# JAX Cache Env Variables Explanation:
# - VLLM_XLA_CACHE_PATH: Prevents vLLM's CompilationManager from overriding our 
#   path with its default.
# - JAX_COMPILATION_CACHE_DIR: Serves as a global catch-all for unit tests that 
#   initiate models and bypass vLLM's CompilationManager logic entirely.
# -----------------------------------------------------------------------------

docker run \
  --name "$IMAGE_NAME" \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
  -v "$LOCAL_JAX_CACHE_DIR":"$LOCAL_JAX_CACHE_DIR" \
  -v "$KERNEL_TUNING_TMP_DIR":"$KERNEL_TUNING_TMP_DIR" \
  "${DEV_MOUNT[@]}" \
  "${ENV_VARS[@]}" \
  "${TEST_SUITE_VARS[@]}" \
  -e HF_HOME="$DOCKER_HF_HOME" \
  -e MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH="$LOCAL_JAX_CACHE_DIR" \
  -e JAX_COMPILATION_CACHE_DIR="$LOCAL_JAX_CACHE_DIR" \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  ${QUANTIZATION:+-e QUANTIZATION="$QUANTIZATION"} \
  ${NEW_MODEL_DESIGN:+-e NEW_MODEL_DESIGN="$NEW_MODEL_DESIGN"} \
  ${USE_V6E8_QUEUE:+-e USE_V6E8_QUEUE="$USE_V6E8_QUEUE"} \
  ${TPU_VERSION:+-e TPU_VERSION="$TPU_VERSION"} \
  ${SKIP_ACCURACY_TESTS:+-e SKIP_ACCURACY_TESTS="$SKIP_ACCURACY_TESTS"} \
  ${VLLM_MLA_DISABLE:+-e VLLM_MLA_DISABLE="$VLLM_MLA_DISABLE"} \
  ${USE_V7X8_QUEUE:+-e USE_V7X8_QUEUE="$USE_V7X8_QUEUE"} \
  ${MOE_REQUANTIZE_BLOCK_SIZE:+-e MOE_REQUANTIZE_BLOCK_SIZE="$MOE_REQUANTIZE_BLOCK_SIZE"} \
  ${MOE_REQUANTIZE_WEIGHT_DTYPE:+-e MOE_REQUANTIZE_WEIGHT_DTYPE="$MOE_REQUANTIZE_WEIGHT_DTYPE"} \
  -e NUM_PRECOMPILE_WORKERS="${NUM_PRECOMPILE_WORKERS:-1}" \
   "${BENCHMARK_DOCKER_ARGS[@]}" \
  "$FULL_IMAGE_TAG" \
  "$@" # Pass all script arguments as the command to run in the container
DOCKER_EXIT_CODE=$?

set -e

# ==========================================
# 3. Post-Docker Actions
# ==========================================
echo "[INFO] Docker finished with exit code ${DOCKER_EXIT_CODE}."

echo "[INFO] Syncing local JAX Cache back to GCS..."
gsutil -m rsync -r "$LOCAL_JAX_CACHE_DIR" "$FINAL_CACHE_PATH" || echo "[WARN] Failed to sync JAX Cache back to GCS."

exit $DOCKER_EXIT_CODE
