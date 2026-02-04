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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# load VLLM_COMMIT_HASH from vllm_lkg.version file, if not exists, get the latest commit hash from vllm repo
if [ -f .buildkite/vllm_lkg.version ]; then
    VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version)"
fi
if [ -z "${VLLM_COMMIT_HASH:-}" ]; then
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
fi

VLLM_SHORT_HASH=""

get_vllm_short_hash() {
    local target_dir="vllm_src"
    local repo_url="https://github.com/vllm-project/vllm.git"
    local commit_hash=""

    # Cleaning up target directory: $target_dir
    rm -rf "$target_dir"

    echo "Cloning vLLM from $repo_url ..."
    if ! git clone "$repo_url" "$target_dir"; then
        echo "Error: Failed to clone repository."
        return 1
    fi

    pushd "$target_dir" || return 1

    CURRENT_DIR=$(pwd)
    cleanup() {
        if [[ -d "$CURRENT_DIR" ]]; then
            echo "Cleanup: Removing $CURRENT_DIR..."
            rm -rf "$CURRENT_DIR"
        fi
    }
    trap cleanup EXIT INT TERM

    commit_hash=$(git rev-parse --short HEAD)
    popd
    
    if [[ -z "$commit_hash" ]]; then
        echo "Error: Failed to retrieve commit hash."
        return 1
    fi

    echo "Successfully cloned. Head is at: $commit_hash"
    VLLM_SHORT_HASH="$commit_hash"
}

upload_pipeline() {
    get_vllm_short_hash
    TPU_SHORT_HASH=$(git rev-parse --short HEAD)
    buildkite-agent meta-data set "VLLM_SHORT_HASH" "${VLLM_SHORT_HASH}"
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    buildkite-agent meta-data set "TPU_SHORT_HASH" "${TPU_SHORT_HASH}"
    CODE_HAME="${VLLM_SHORT_HASH}-${TPU_SHORT_HASH}-"
    buildkite-agent meta-data set "CODE_HASH" "${CODE_HAME}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    echo "Using vllm short commit hash: $(buildkite-agent meta-data get "VLLM_SHORT_HASH")"
    echo "Using tpu-inference short commit hash: $(buildkite-agent meta-data get "TPU_SHORT_HASH")"
    echo "Using vllm-tpu commit hash: $(buildkite-agent meta-data get "CODE_HAME")"

    # BUILDKITE_COMMIT is tpu-inference latest commit
    
    # export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
    # export TPU_QUEUE_MULTI="tpu_v7x_8_queue"

    buildkite-agent pipeline upload .buildkite/pipeline_benchmark.yml
}

echo "--- Starting Buildkite Bootstrap"
echo "Running in pipeline: $BUILDKITE_PIPELINE_SLUG"
echo "This is not a Pull Request build. Uploading main pipeline."
upload_pipeline
echo "--- Buildkite Bootstrap Finished"