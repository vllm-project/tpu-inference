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
# Sets VLLM_COMMIT_HASH from pyproject.toml and uploads pipeline_dev.yml.
# Configure the Buildkite pipeline's "Steps" to run this script.

set -euo pipefail

# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file (priority constants +
# upload_with_priority).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

# Resolve and publish Version Metadata for developer pipeline
VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vLLM commit: ${VLLM_COMMIT_HASH}"

# Which pipeline to upload. Defaults to the demo pipeline_dev.yml; set
# DEV_PIPELINE_FILE in the build's env to run a different dev pipeline
# (e.g. .buildkite/xla_autotune/pipeline.yml) without editing this file.
DEV_PIPELINE_FILE="${DEV_PIPELINE_FILE:-.buildkite/pipeline_dev.yml}"

if [[ ! -f "${DEV_PIPELINE_FILE}" ]]; then
  echo "ERROR: dev pipeline file '${DEV_PIPELINE_FILE}' not found on branch" \
       "'${BUILDKITE_BRANCH:-unknown}' at commit ${BUILDKITE_COMMIT:-unknown}." \
       "Create it on your branch, or point DEV_PIPELINE_FILE at an existing file."
  exit 1
fi

echo "--- :pipeline: Uploading ${DEV_PIPELINE_FILE}"
buildkite-agent pipeline upload "${DEV_PIPELINE_FILE}"

echo "--- Buildkite Dev Bootstrap Finished"
