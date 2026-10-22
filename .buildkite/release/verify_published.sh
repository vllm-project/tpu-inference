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

# After the publish, check it the way a user would, with no credentials:
#   Docker Hub   an anonymous pull of v<version> resolves to the staged digest
#   PyPI         vllm-tpu==<version> resolves from pypi.org alone
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
WANT="$(digest_of "$(buildkite-agent meta-data get release-image)")"
NOTES=()

echo "--- :docker: Anonymous pull of ${RELEASE_DOCKERHUB_REPO}:v${VERSION}"
GOT="$(dockerhub_digest "v${VERSION}")"
[[ -n "${GOT}" ]] || die "${RELEASE_DOCKERHUB_REPO}:v${VERSION} is not on Docker Hub."
[[ "${GOT}" == "${WANT}" ]] \
  || die "Docker Hub serves ${GOT} for v${VERSION}, not the staged ${WANT}."
NOTES+=("Docker Hub: anonymous pull of \`${RELEASE_DOCKERHUB_REPO}:v${VERSION}\` = \`${GOT}\`")

echo "--- :package: PyPI"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
curl -LsSf --retry 3 https://astral.sh/uv/install.sh \
  | env UV_INSTALL_DIR="${WORK}/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
echo "vllm-tpu==${VERSION}" > "${WORK}/req.in"
# publish_wheel.sh already waited for each file, but the simple index can
# still lag a little behind the upload.
for attempt in $(seq 1 10); do
  if "${WORK}/uv/uv" pip compile "${WORK}/req.in" --no-config --quiet --refresh \
    --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
    --index-url https://pypi.org/simple -o "${WORK}/resolved.txt" 2>"${WORK}/err.txt"; then
    break
  fi
  [[ "${attempt}" == 10 ]] && { cat "${WORK}/err.txt" >&2; die "vllm-tpu==${VERSION} does not resolve from pypi.org."; }
  sleep 30
done
NOTES+=("PyPI: \`pip install vllm-tpu==${VERSION}\` resolves from pypi.org ($(grep -c '==' "${WORK}/resolved.txt") packages)")

printf -- '- %s\n' "${NOTES[@]}" | buildkite-agent annotate --style success --context release-verified
printf -- '- %s\n' "${NOTES[@]}"
