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

# Copy the staged image to Docker Hub as v<version>, and latest if the approver
# chose to. A copy by digest, never a rebuild: what users pull is what this
# build staged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
STAGED="$(buildkite-agent meta-data get release-image)"
MOVE_LATEST="$(buildkite-agent meta-data get dockerhub_move_latest --default no)"
OVERWRITE="$(buildkite-agent meta-data get dockerhub_overwrite --default no)"
PRERELEASE="$(buildkite-agent meta-data get release-prerelease)"
[[ "${PRERELEASE}" == "yes" ]] && MOVE_LATEST="no"
TAG="v${VERSION}"
DEST="docker.io/${RELEASE_DOCKERHUB_REPO}"

# Keep released tags unless the approver chose to overwrite. Assigned, not
# tested inline, so a failed Docker Hub lookup stops the step (set -e).
if [[ "${OVERWRITE}" != "yes" ]]; then
  EXISTING="$(dockerhub_digest "${TAG}")"
  [[ -z "${EXISTING}" ]] \
    || die "${DEST}:${TAG} is already on Docker Hub (${EXISTING}); approve with overwrite to replace it."
fi

# A throwaway Docker config: GAR read through the agent's gcloud helper, Docker
# Hub write through a login that never touches the agent's own config.
gcloud auth configure-docker "${RELEASE_GAR%%/*}" --quiet
DOCKER_CONFIG="$(mktemp -d)"
export DOCKER_CONFIG
trap 'docker logout >/dev/null 2>&1 || true; rm -rf "${DOCKER_CONFIG}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ -f "${HOME}/.docker/config.json" ]] && cp "${HOME}/.docker/config.json" "${DOCKER_CONFIG}/"
dockerhub_login "${DOCKER_CONFIG}" "${RELEASE_DOCKERHUB_USERNAME}"

TAGS=(--tag "${DEST}:${TAG}")
[[ "${MOVE_LATEST}" == "yes" ]] && TAGS+=(--tag "${DEST}:latest")
echo "--- :docker: Copying ${STAGED} to ${TAGS[*]//--tag /}"
# --prefer-index=false copies the manifest as it is, so the digest on Docker
# Hub is the staged one.
docker buildx imagetools create --prefer-index=false "${TAGS[@]}" "${STAGED}"

echo "--- :mag: Checking the digest on Docker Hub"
got="$(docker buildx imagetools inspect "${DEST}:${TAG}" --format '{{json .Manifest}}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["digest"])')"
[[ "${got}" == "$(digest_of "${STAGED}")" ]] \
  || die "${DEST}:${TAG} is ${got}, not the staged $(digest_of "${STAGED}")."
buildkite-agent annotate --style success --context release-dockerhub \
  "Published \`${DEST}:${TAG}\`$([[ "${MOVE_LATEST}" == "yes" ]] && echo " and \`:latest\`") = \`${got}\`."
