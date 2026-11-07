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

# Builds vllm/vllm-jax from this commit and the vLLM LKG, and pushes it as
# :nightly and as nightly-<date>-<this repo's sha>-<vllm sha>.
set -euo pipefail

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/configs/pipeline_config.sh"

REPO="vllm/vllm-jax"
VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
TAG="nightly-$(date +%Y%m%d)-${BUILDKITE_COMMIT:0:7}-${VLLM_COMMIT_HASH:0:7}"
echo "--- Building ${REPO}:${TAG} (vLLM ${VLLM_COMMIT_HASH})"

# -f rather than piping yes: under pipefail, yes dying of SIGPIPE once prune
# has its answer fails the script (exit 141).
docker system prune -a -f
docker build --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" --no-cache \
  -f docker/Dockerfile -t "${REPO}:nightly" -t "${REPO}:${TAG}" .

echo "--- Pushing"
docker push "${REPO}:nightly"
docker push "${REPO}:${TAG}"
