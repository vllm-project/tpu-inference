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

# Bootstrap for the tpu-inference-dev (dev/experimental) pipeline.
# Sets VLLM_COMMIT_HASH from vllm_lkg.version and uploads pipeline_dev.yml.
# Configure the Buildkite pipeline's "Steps" to run this script.

set -euo pipefail

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

upload_with_highest_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=10
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
  { 
    echo "priority: ${JOB_PRIORITY:-PRIORITY_DEFAULT}"; 
    cat "$yaml_file"; 
  } | buildkite-agent pipeline upload
}

set_jax_envs() {
    case $1 in
        v6)
            export TESTS_GROUP_LABEL="[jax] TPU6e Tests Group"
            export TPU_VERSION="tpu6e"
            export TPU_QUEUE_SINGLE="tpu_v6e_queue"
            export TPU_QUEUE_MULTI="tpu_v6e_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=1
            export TENSOR_PARALLEL_SIZE_MULTI=8
            ;;
        v7)
            export TESTS_GROUP_LABEL="[jax] TPU7x Tests Group"
            export TPU_VERSION="tpu7x"
            export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
            export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=2
            export TENSOR_PARALLEL_SIZE_MULTI=8
            export COV_FAIL_UNDER="67"
            ;;
        unset)
            unset TESTS_GROUP_LABEL TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI TENSOR_PARALLEL_SIZE_SINGLE TENSOR_PARALLEL_SIZE_MULTI COV_FAIL_UNDER
            ;;
    esac
}
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"

echo "--- :pipeline: Uploading pipeline_dev.yml"
set_jax_envs v7
upload_with_highest_priority .buildkite/pipeline_jax.yml
set_jax_envs unset

echo "--- Buildkite Dev Bootstrap Finished"
