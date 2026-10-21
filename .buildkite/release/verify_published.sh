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
#                (it can't while the repository is private, which is a warning)
#   PyPI         vllm-tpu==<version> resolves from pypi.org alone
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

VERSION="$(release_version)"
PROD="$(buildkite-agent meta-data get release-image-prod)"
WANT="$(digest_of "${PROD}")"
NOTES=()

echo "--- :docker: Anonymous pull of ${RELEASE_DOCKERHUB_REPO}:v${VERSION}"
TOKEN="$(curl -fsS --max-time 30 \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${RELEASE_DOCKERHUB_REPO}:pull" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
HEADERS="$(curl -sS --max-time 30 -o /dev/null -D - -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" \
    "https://registry-1.docker.io/v2/${RELEASE_DOCKERHUB_REPO}/manifests/v${VERSION}")"
STATUS="$(head -n 1 <<< "${HEADERS}" | awk '{print $2}')"
case "${STATUS}" in
    200)
        GOT="$(tr -d '\r' <<< "${HEADERS}" | sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: //p')"
        [[ "${GOT}" == "${WANT}" ]] || die "Docker Hub serves ${GOT} for v${VERSION}, not the tested ${WANT}."
        NOTES+=("Docker Hub: anonymous pull of \`${RELEASE_DOCKERHUB_REPO}:v${VERSION}\` = \`${GOT}\`")
        ;;
    401)
        NOTES+=("Docker Hub: \`${RELEASE_DOCKERHUB_REPO}\` is private, so users can't pull it yet")
        buildkite-agent annotate --style warning --context release-dockerhub-private \
            "\`${RELEASE_DOCKERHUB_REPO}\` is private: anonymous pulls get 401."
        ;;
    *) die "Docker Hub answered ${STATUS} for v${VERSION}." ;;
esac

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
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-url https://pypi.org/simple -o "${WORK}/resolved.txt" 2>"${WORK}/err.txt"; then
        break
    fi
    [[ "${attempt}" == 10 ]] && { cat "${WORK}/err.txt" >&2; die "vllm-tpu==${VERSION} does not resolve from pypi.org."; }
    sleep 30
done
NOTES+=("PyPI: \`pip install vllm-tpu==${VERSION}\` resolves from pypi.org ($(grep -c '==' "${WORK}/resolved.txt") packages)")

printf -- '- %s\n' "${NOTES[@]}" | buildkite-agent annotate --style success --context release-verified
printf -- '- %s\n' "${NOTES[@]}"
