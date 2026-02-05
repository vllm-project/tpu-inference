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

upload_pipeline() {
    # Prepare commit hash
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    CODE_HAME="${VLLM_COMMIT_HASH}-${$BUILDKITE_COMMIT}-"
    buildkite-agent meta-data set "CODE_HASH" "${CODE_HAME}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    echo "Using vllm-tpu commit hash: $(buildkite-agent meta-data get "CODE_HAME")"

    buildkite-agent pipeline upload .buildkite/pipeline_benchmark.yml
}

echo "--- Starting Buildkite Bootstrap"
echo "Running in pipeline: $BUILDKITE_PIPELINE_SLUG"
echo "This is a Benchmark build. Uploading benchmark pipeline."
upload_pipeline
echo "--- Buildkite Bootstrap Finished"