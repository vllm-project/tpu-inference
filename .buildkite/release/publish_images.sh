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
{ set +x; } 2>/dev/null # Never trace the token into the build log.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
STAGED="$(buildkite-agent meta-data get release-image)"
MOVE_LATEST="$(buildkite-agent meta-data get dockerhub_move_latest --default no)"
OVERWRITE="$(buildkite-agent meta-data get dockerhub_overwrite --default no)"
is_prerelease "${VERSION}" && MOVE_LATEST="no"
TAG="v${VERSION}"
DEST="docker.io/${RELEASE_DOCKERHUB_REPO}"

: "${DOCKERHUB_TOKEN:?the DOCKERHUB_TOKEN secret is not set for this step}"

if [[ "${OVERWRITE}" != "yes" && -n "$(dockerhub_digest "${TAG}")" ]]; then
  die "${DEST}:${TAG} is already on Docker Hub; approve with overwrite to replace it."
fi

# A throwaway Docker config: GAR read through the agent's gcloud helper, Docker
# Hub write through a login that never touches the agent's own config.
gcloud auth configure-docker "${RELEASE_GAR%%/*}" --quiet
DOCKER_CONFIG="$(mktemp -d)"
export DOCKER_CONFIG
trap 'docker logout >/dev/null 2>&1 || true; rm -rf "${DOCKER_CONFIG}"' EXIT
# Only the credential helpers: a credsStore would put the login in the agent's
# keychain instead of this directory.
python3 - "${HOME}/.docker/config.json" "${DOCKER_CONFIG}/config.json" <<'EOF'
import json, os, sys
src, dst = sys.argv[1:]
helpers = json.load(open(src)).get("credHelpers", {}) if os.path.exists(src) else {}
json.dump({"credHelpers": helpers}, open(dst, "w"))
EOF
printf '%s' "${DOCKERHUB_TOKEN}" \
  | docker login -u "${RELEASE_DOCKERHUB_USERNAME}" --password-stdin >/dev/null 2>&1 \
  || die "Docker Hub login failed for ${RELEASE_DOCKERHUB_USERNAME}."
unset DOCKERHUB_TOKEN

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
