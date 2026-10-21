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

# Build the ci, dev and prod images from the release checkout and stage them in
# GAR as <version> and <date>-<version>. Nothing goes to Docker Hub here: the
# publish step copies the staged prod image there by digest, so users get the
# exact bytes this build made.
#
# The images are built with the release checkout's own docker/build_image.sh,
# which belongs to the code being released.
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

VERSION="$(release_version)"
DATE="$(date -u +%Y%m%d)"

echo "--- Cleaning up old Docker images and cache"
bash "${RELEASE_TOOLS}/scripts/cleanup_docker.sh"

ROWS=""
for target in ci dev prod; do
    image="${RELEASE_GAR}/vllm-tpu-${target}"
    echo "--- :docker: Building ${image}:${VERSION}"
    ./docker/build_image.sh --target "${target}" -t "${image}:${VERSION}"
    docker tag "${image}:${VERSION}" "${image}:${DATE}-${VERSION}"

    echo "--- :docker: Pushing ${image}:${VERSION} and :${DATE}-${VERSION}"
    docker push "${image}:${VERSION}"
    docker push "${image}:${DATE}-${VERSION}"

    ref="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${image}:${VERSION}" \
        | grep -m 1 "^${image}@sha256:")" \
        || die "no ${image} digest after the push."
    buildkite-agent meta-data set "release-image-${target}" "${ref}"
    ROWS+="| ${target} | \`${image}:${VERSION}\` | \`$(digest_of "${ref}")\` |"$'\n'

    docker rmi "${image}:${VERSION}" "${image}:${DATE}-${VERSION}" >/dev/null 2>&1 || true
done

buildkite-agent annotate --style info --context release-images <<EOF
Staged images (GAR):

| target | tag | digest |
|---|---|---|
${ROWS}
EOF
