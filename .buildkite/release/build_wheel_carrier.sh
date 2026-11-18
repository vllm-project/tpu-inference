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

# Stage the wheels this build made in the wheel carrier, for the wheel check on
# TPU (VLLM_SERVE_TEST=1): vllm-tpu-wheel-carrier:wheels-<version>-build<N> in
# GAR, with its digest in meta-data release-wheel-carrier.
#
# The wheel check (verify_wheels_on_tpu.sh) runs in tpu-inference-release-verify
# on the TPU CI cluster, which cannot read this build's artifacts and only runs
# container images. The carrier takes the wheels there: python:3.12-slim with
# the .whl files in /wheels, not installed, plus the scripts serve_wheels.sh
# runs. It holds no credentials and is never published; the release image is
# build_images.sh's.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
TAG="${RELEASE_WHEEL_CARRIER}:${WHEEL_CARRIER_TAG_PREFIX}${VERSION}-build${BUILDKITE_BUILD_NUMBER}"

CTX="$(mktemp -d "${TMPDIR:-/tmp}/wheel-carrier.XXXXXX")"
trap 'rm -rf "${CTX}"; docker rmi "${TAG}" >/dev/null 2>&1 || true' EXIT

echo "--- :inbox_tray: Wheels built in this build"
(cd "${CTX}" && buildkite-agent artifact download "dist/tpu_inference-*.whl" . \
  && buildkite-agent artifact download "dist/vllm_tpu-*.whl" .)
shopt -s nullglob
CORE_WHEELS=("${CTX}"/dist/tpu_inference-*.whl)
COMBINED_WHEELS=("${CTX}"/dist/vllm_tpu-*.whl)
shopt -u nullglob
[[ "${#CORE_WHEELS[@]}" -eq 1 && "${#COMBINED_WHEELS[@]}" -eq 1 ]] \
  || die "expected one tpu_inference and one vllm_tpu wheel, found" \
    "${#CORE_WHEELS[@]} and ${#COMBINED_WHEELS[@]}."
for wheel in "${CORE_WHEELS[0]}" "${COMBINED_WHEELS[0]}"; do
  [[ "$(basename "${wheel}" | cut -d- -f2)" == "${VERSION}" ]] \
    || die "$(basename "${wheel}") is not version ${VERSION}."
done
ls -l "${CTX}/dist"

# The scripts serve_wheels.sh needs, at their repo paths.
mkdir -p "${CTX}/repo/.buildkite/release" "${CTX}/repo/.buildkite/scripts"
cp "${SCRIPT_DIR}/serve_wheels.sh" "${SCRIPT_DIR}/serve_and_verify.sh" \
  "${SCRIPT_DIR}/lib.sh" "${CTX}/repo/.buildkite/release/"
cp "${REPO_ROOT}/.buildkite/scripts/wheel_utils.py" "${CTX}/repo/.buildkite/scripts/"

cat >"${CTX}/Dockerfile" <<'EOF'
FROM python:3.12-slim-bookworm
# serve_wheels.sh downloads uv and polls the server with curl.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*
COPY dist/ /wheels/
COPY repo/ /workspace/release-verify/
WORKDIR /workspace/release-verify
EOF

echo "--- :docker: Building ${TAG}"
DOCKER_BUILDKIT=1 docker build --no-cache -t "${TAG}" "${CTX}"

echo "--- :docker: Pushing ${TAG}"
gcloud auth configure-docker "${RELEASE_GAR%%/*}" --quiet
docker push "${TAG}"

STAGED="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${TAG}" \
  | grep -m 1 "^${RELEASE_WHEEL_CARRIER}@sha256:")" \
  || die "no ${RELEASE_WHEEL_CARRIER} digest after the push."
buildkite-agent meta-data set release-wheel-carrier "${STAGED}"

buildkite-agent annotate --style info --context release-wheel-carrier <<EOF
Wheel carrier (GAR): \`${TAG}\` = \`$(digest_of "${STAGED}")\`, holding $(basename "${CORE_WHEELS[0]}") and $(basename "${COMBINED_WHEELS[0]}")
EOF
