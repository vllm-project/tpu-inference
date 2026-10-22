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
#   PyPI         not handled by this pipeline yet
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

VERSION="$(release_version)"
WANT="$(digest_of "$(buildkite-agent meta-data get release-image)")"

echo "--- :docker: Anonymous pull of ${RELEASE_DOCKERHUB_REPO}:v${VERSION}"
GOT="$(dockerhub_digest "v${VERSION}")"
[[ -n "${GOT}" ]] || die "${RELEASE_DOCKERHUB_REPO}:v${VERSION} is not on Docker Hub."
[[ "${GOT}" == "${WANT}" ]] \
  || die "Docker Hub serves ${GOT} for v${VERSION}, not the staged ${WANT}."

NOTE="Docker Hub: anonymous pull of \`${RELEASE_DOCKERHUB_REPO}:v${VERSION}\` = \`${GOT}\`"
echo "${NOTE}"
buildkite-agent annotate --style success --context release-verified "- ${NOTE}"
