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

# Copy the staged prod image to Docker Hub as v<version>, and latest if the
# approver chose to. A copy by digest, never a rebuild: what users pull is what
# this build staged.
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

VERSION="$(release_version)"
PROD="$(buildkite-agent meta-data get release-image-prod)"
MOVE_LATEST="$(buildkite-agent meta-data get dockerhub_move_latest --default no)"
OVERWRITE="$(buildkite-agent meta-data get dockerhub_overwrite --default no)"
is_prerelease "${VERSION}" && MOVE_LATEST="no"
TAG="v${VERSION}"
DEST="docker.io/${RELEASE_DOCKERHUB_REPO}"

: "${DOCKERHUB_TOKEN_TPU_INFERENCE:?the DOCKERHUB_TOKEN_TPU_INFERENCE secret is not set for this step}"

# A throwaway Docker config: Docker Hub's login never touches the agent's own,
# which keeps the gcloud helper that reads the staged image from GAR.
DOCKER_CONFIG="$(mktemp -d)"
export DOCKER_CONFIG
trap 'docker logout >/dev/null 2>&1 || true; rm -rf "${DOCKER_CONFIG}"' EXIT
[[ -f "${HOME}/.docker/config.json" ]] && cp "${HOME}/.docker/config.json" "${DOCKER_CONFIG}/"
printf '%s' "${DOCKERHUB_TOKEN_TPU_INFERENCE}" \
    | docker login -u "${RELEASE_DOCKERHUB_USERNAME}" --password-stdin >/dev/null \
    || die "Docker Hub login failed for ${RELEASE_DOCKERHUB_USERNAME}."
unset DOCKERHUB_TOKEN_TPU_INFERENCE

if [[ "${OVERWRITE}" != "yes" ]] && docker manifest inspect "${DEST}:${TAG}" >/dev/null 2>&1; then
    die "${DEST}:${TAG} is already on Docker Hub; approve with overwrite to replace it."
fi

TAGS=(--tag "${DEST}:${TAG}")
[[ "${MOVE_LATEST}" == "yes" ]] && TAGS+=(--tag "${DEST}:latest")
echo "--- :docker: Copying ${PROD} to ${TAGS[*]//--tag /}"
# --prefer-index=false copies the manifest as it is, so the digest on Docker
# Hub is the staged one.
docker buildx imagetools create --prefer-index=false "${TAGS[@]}" "${PROD}"

echo "--- :mag: Checking the digest on Docker Hub"
got="$(docker buildx imagetools inspect "${DEST}:${TAG}" --format '{{json .Manifest}}' \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["digest"])')"
[[ "${got}" == "$(digest_of "${PROD}")" ]] \
    || die "${DEST}:${TAG} is ${got}, not the staged $(digest_of "${PROD}")."
buildkite-agent meta-data set release-dockerhub-tags "${TAGS[*]//--tag /}"
buildkite-agent annotate --style success --context release-dockerhub \
    "Published \`${DEST}:${TAG}\`$([[ "${MOVE_LATEST}" == "yes" ]] && echo " and \`:latest\`") = \`${got}\`."
