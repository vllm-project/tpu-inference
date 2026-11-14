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

# Verify the staged release image on one TPU chip: run serve_and_verify.sh with
# the image's own vLLM. verify_wheels_on_tpu.sh runs the same flow with the
# wheels.
#
# A step of tpu-inference-release-verify (verify_pipeline.yml) on the kube
# queue. The launcher (.buildkite/kubernetes/run.sh) starts the image as a TPU
# pod and runs serve_and_verify.sh from the image's checkout of the release
# commit.
#
# Environment:
#   RELEASE_IMAGE_REF        Image to verify, by digest. Default: meta-data
#                            release-image of the triggering release build.
#   VERIFY_MODEL             Model to serve (default:
#                            meta-llama/Llama-3.1-8B-Instruct).
#   VERIFY_NUM_PROMPTS       Requests in the bench (default: 30).
#   VERIFY_SERVER_TIMEOUT_S  How long to wait for /health (default: 3600).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

MODEL="${VERIFY_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_PROMPTS="${VERIFY_NUM_PROMPTS:-30}"
# Covers the model download and the first TPU compile.
SERVER_TIMEOUT_S="${VERIFY_SERVER_TIMEOUT_S:-3600}"

IMAGE="$(release_build_image release-image "${RELEASE_IMAGE_REF:-}")"
VERSION="$(release_build_version)"
export WORKLOAD_IMAGE="${IMAGE}"

echo "--- :kubernetes: Verifying release ${VERSION}: image $(digest_of "${IMAGE}"), ${MODEL}, ${SHAPE:-}"
# An absolute path into the image (docker/Dockerfile copies the checkout to
# /workspace/tpu_inference): the launcher does not guarantee the working
# directory, so CI's kube steps use absolute paths too. TMPDIR=/dev/shm: the
# fleet evicts a pod that writes big files to its boot disk, and the fresh
# compilation cache is one.
exec "${SCRIPT_DIR}/../kubernetes/run.sh" \
  env TMPDIR=/dev/shm TEST_MODEL="${MODEL}" NUM_PROMPTS="${NUM_PROMPTS}" \
  SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S}" \
  bash /workspace/tpu_inference/.buildkite/release/serve_and_verify.sh python3 image \
  "the staged image of release ${VERSION} ($(digest_of "${IMAGE}"))"
