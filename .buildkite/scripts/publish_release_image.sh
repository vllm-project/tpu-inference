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

# Builds the release image (DOCKERHUB_REPO, e.g. vllm/vllm-tpu) from an
# approved build of releases/vX.Y.Z and the vLLM LKG, and pushes it as vX.Y.Z,
# or under RELEASE_VERSION for a pre-release. The approval step in
# pipeline_nightly_image.yml decides whether latest moves (never for a
# pre-release) and whether an existing tag may be overwritten.
set -euo pipefail

REPO="${DOCKERHUB_REPO:?DOCKERHUB_REPO must be set}"
BRANCH="${BUILDKITE_BRANCH:?BUILDKITE_BRANCH must be set}"
if [[ ! "${BRANCH}" =~ ^releases/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: release images are built from releases/vX.Y.Z, not '${BRANCH}'." >&2
  exit 1
fi
# RELEASE_VERSION (e.g. v0.13.0rc1, set on New Build) publishes a pre-release
# under its own tag instead of vX.Y.Z.
TAG="${RELEASE_VERSION:-${BRANCH#releases/}}"
TAG="v${TAG#v}"
if [[ ! "${TAG}" =~ ^v[0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "ERROR: RELEASE_VERSION '${RELEASE_VERSION}' is not usable as an image tag; expected e.g. v0.13.0rc1." >&2
  exit 1
fi

# A release is built from the pinned vLLM, never from vLLM HEAD.
VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version 2>/dev/null || true)"
if [[ ! "${VLLM_COMMIT_HASH}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: .buildkite/vllm_lkg.version does not hold a vLLM commit." >&2
  exit 1
fi

# The approver's choices in the block step of pipeline_nightly_image.yml.
MOVE_LATEST="$(buildkite-agent meta-data get dockerhub_move_latest --default no)"
OVERWRITE="$(buildkite-agent meta-data get dockerhub_overwrite --default no)"
# A pre-release never becomes latest, whatever the approver chose.
if [[ -n "${RELEASE_VERSION:-}" ]]; then
  MOVE_LATEST="no"
fi

# Log in to a private, temporary Docker config that only the Docker Hub
# commands below use: the image build still pulls the base image anonymously,
# the token never reaches the agent's shared config, and the trap logs out
# however the script ends.
{ set +x; } 2>/dev/null # Never trace the token into the build log.
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME must be set}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN must be set}"
DOCKERHUB_CONFIG="$(mktemp -d)"
cleanup_dockerhub() {
  DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker logout >/dev/null 2>&1 || true
  rm -rf "${DOCKERHUB_CONFIG}"
}
trap cleanup_dockerhub EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# printf is a shell builtin, so the token reaches docker only through the
# pipe and never appears on a command line.
printf '%s' "${DOCKERHUB_TOKEN}" \
  | DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker login -u "${DOCKERHUB_USERNAME}" --password-stdin >/dev/null 2>&1 \
  || { echo "ERROR: Docker Hub login failed; check DOCKERHUB_TOKEN and DOCKERHUB_USERNAME." >&2; exit 1; }
# Nothing else needs the token, so no later process inherits it.
unset DOCKERHUB_TOKEN

# Released tags are immutable unless the approver chose to overwrite.
# Checked before the build, so a refused release builds nothing.
if [[ "${OVERWRITE}" != "yes" ]] \
  && DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker manifest inspect "${REPO}:${TAG}" >/dev/null 2>&1; then
  echo "ERROR: ${REPO}:${TAG} already exists; choose Overwrite in the approval step to replace it." >&2
  exit 1
fi

echo "--- Building ${REPO}:${TAG} (vLLM ${VLLM_COMMIT_HASH})"
docker system prune -a -f
docker build --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" --no-cache \
  -f docker/Dockerfile -t "${REPO}:${TAG}" .

# The immutable tag goes first, so latest never points at an unpushed tag.
echo "--- Pushing ${REPO}:${TAG}"
DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker push "${REPO}:${TAG}"
if [[ "${MOVE_LATEST}" == "yes" ]]; then
  echo "--- Moving ${REPO}:latest to ${TAG}"
  docker tag "${REPO}:${TAG}" "${REPO}:latest"
  DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker push "${REPO}:latest"
else
  echo "--- Not moving latest to ${TAG}."
fi
