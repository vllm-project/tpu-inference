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

# Bootstrap for the tpu-inference-dev (dev/experimental) pipeline.
# Sets VLLM_COMMIT_HASH from vllm_lkg.version and uploads pipeline_kernel_tuning.yml.
# Configure the Buildkite pipeline's "Steps" to run this script.

set -euo pipefail
# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

# Handles the environment state for different TPU generations.
set_jax_envs() {
    case $1 in
        v6)
            export TPU_VERSION="tpu6e"
            export TPU_QUEUE_SINGLE="tpu_v6e_queue"
            export TPU_QUEUE_MULTI="tpu_v6e_8_queue"
            ;;
        v7)
            export TPU_VERSION="tpu7x"
            export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
            export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
            ;;
        unset)
            unset TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI
            ;;
    esac
}

echo "--- :git: Setting VLLM_COMMIT_HASH from vllm_lkg.version"

LKG_FILE=".buildkite/vllm_lkg.version"

if [ ! -f "${LKG_FILE}" ]; then
    echo "ERROR: ${LKG_FILE} not found. Cannot determine vllm commit hash."
    exit 1
fi

VLLM_COMMIT_HASH="$(cat "${LKG_FILE}")"

if [ -z "${VLLM_COMMIT_HASH}" ]; then
    echo "ERROR: ${LKG_FILE} is empty. Cannot determine vllm commit hash."
    exit 1
fi

buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"

# Check whether the KERNEL_TUNING_KERNEL_NAME flag is set in the environment variables.
# If it is, upload the pipeline_kernel_tuning.yml pipeline, which
# includes steps for generating kernel tuning cases and running kernel tuning jobs. 
# If it is not set, upload the regular pipeline_kernel_tuner.yml pipeline. 
echo "--- :pipeline: Uploading pipeline_kernel_tuning.yml"
echo "Pipeline env vars:"
echo "  KERNEL_TUNING_CASE_SET_ID=${KERNEL_TUNING_CASE_SET_ID:-}"
echo "  KERNEL_TUNING_RUN_ID=${KERNEL_TUNING_RUN_ID:-}"
echo "  KERNEL_TUNING_KERNEL_NAME=${KERNEL_TUNING_KERNEL_NAME:-}"
echo "  KERNEL_TUNING_CASE_SET_DESC=${KERNEL_TUNING_CASE_SET_DESC:-}"
echo "  KERNEL_TUNING_TPU_VERSION=${KERNEL_TUNING_TPU_VERSION:-}"
echo "  HOST_NAME=${HOST_NAME:-}"
set_jax_envs "${KERNEL_TUNING_TPU_VERSION:-v6}"
buildkite-agent pipeline upload .buildkite/pipeline_kernel_tuning.yml

echo "--- Buildkite Kernel Tuning Bootstrap Finished"
