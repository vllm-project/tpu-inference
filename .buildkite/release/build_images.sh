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

# Build vllm/vllm-tpu from the release checkout and stage it in GAR as
# vllm-tpu:release-<version>-build<N>. Nothing goes to Docker Hub here: the
# publish step copies the staged image there by digest, so users get the exact
# bytes this build made.
#
# Built the way the nightly image is, from the release checkout's own
# docker/Dockerfile and vLLM pin (.buildkite/vllm_lkg.version), which belong
# to the code being released.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
TAG="${RELEASE_IMAGE}:${RELEASE_TAG_PREFIX}${VERSION}-build${BUILDKITE_BUILD_NUMBER}"

# A release is built from the pinned vLLM, never from vLLM HEAD.
VLLM_COMMIT_HASH="$(tr -d '[:space:]' < .buildkite/vllm_lkg.version 2>/dev/null || true)"
[[ "${VLLM_COMMIT_HASH}" =~ ^[0-9a-f]{40}$ ]] \
  || die ".buildkite/vllm_lkg.version does not hold a vLLM commit: '${VLLM_COMMIT_HASH}'."

# Leftovers of earlier release builds that failed before their own cleanup, and
# the build cache. Only our release-* tags: CI's vllm-tpu images and the
# agent's other images aren't ours to remove.
echo "--- :broom: Cleaning up earlier release images and the build cache"
docker images --filter "reference=${RELEASE_IMAGE}:${RELEASE_TAG_PREFIX}*" \
  --format '{{.Repository}}:{{.Tag}}' \
  | xargs -r docker rmi -f >/dev/null 2>&1 || true
docker builder prune -f >/dev/null 2>&1 || true

echo "--- :docker: Building ${TAG} (vLLM ${VLLM_COMMIT_HASH})"
# The user image, as the nightly builds it: IS_TEST=false leaves out the CI test
# tooling and BM_INFRA=false the gcloud CLI. Set explicitly, so a change to the
# Dockerfile's defaults can't change what a release ships.
DOCKER_BUILDKIT=1 docker build --no-cache \
  --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
  --build-arg IS_TEST=false \
  --build-arg BM_INFRA=false \
  -f docker/Dockerfile -t "${TAG}" .

echo "--- :docker: Pushing ${TAG}"
gcloud auth configure-docker "${RELEASE_GAR%%/*}" --quiet
docker push "${TAG}"

STAGED="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${TAG}" \
  | grep -m 1 "^${RELEASE_IMAGE}@sha256:")" \
  || die "no ${RELEASE_IMAGE} digest after the push."
buildkite-agent meta-data set release-image "${STAGED}"
# Only this build's image: the agent's other images aren't ours to remove.
docker rmi "${TAG}" >/dev/null 2>&1 || true

buildkite-agent annotate --style info --context release-image <<EOF
Staged image (GAR): \`${TAG}\` = \`$(digest_of "${STAGED}")\` (vLLM \`${VLLM_COMMIT_HASH:0:12}\`)
EOF
