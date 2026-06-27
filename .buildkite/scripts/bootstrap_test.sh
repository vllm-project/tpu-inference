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
    local tpu_version=$1
    local tpu_cores=$2

    echo "Setting JAX environment variables for TPU version: ${tpu_version}, TPU cores: ${tpu_cores}"
    
    # keep in sync with the logic in kernel_tuner_runner.py:get_tpu_queue_by_version_and_cores
    case $tpu_version in
        tpu6e)
            export TPU_VERSION="tpu6e"
            export TPU_QUEUE_SINGLE="tpu_v6e_queue"
            case $tpu_cores in
                1) export TPU_QUEUE_MULTI="tpu_v6e_queue" ;;
                8) export TPU_QUEUE_MULTI="tpu_v6e_8_queue" ;;
                *) echo "ERROR: unsupported tpu_cores=$tpu_cores for tpu6e"; exit 1 ;;
            esac
            echo "Set TPU_VERSION=${TPU_VERSION}, TPU_QUEUE_SINGLE=${TPU_QUEUE_SINGLE}, TPU_QUEUE_MULTI=${TPU_QUEUE_MULTI}"
            ;;
        tpu7x)
            export TPU_VERSION="tpu7x"
            export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
            case $tpu_cores in
                2) export TPU_QUEUE_MULTI="tpu_v7x_2_queue" ;;
                8) export TPU_QUEUE_MULTI="tpu_v7x_8_queue" ;;
                16) export TPU_QUEUE_MULTI="tpu_v7x_16_queue" ;;
                *) echo "ERROR: unsupported tpu_cores=$tpu_cores for tpu7x"; exit 1 ;;
            esac
            echo "Set TPU_VERSION=${TPU_VERSION}, TPU_QUEUE_SINGLE=${TPU_QUEUE_SINGLE}, TPU_QUEUE_MULTI=${TPU_QUEUE_MULTI}"
            ;;
        unset)
            unset TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI
            echo "Unset TPU_VERSION, TPU_QUEUE_SINGLE, TPU_QUEUE_MULTI"
            ;;
        *)
            echo "ERROR: unsupported tpu_version=${tpu_version}"
            exit 1
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


buildkite-agent pipeline upload .buildkite/pipeline_test.yml
