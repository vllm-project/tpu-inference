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

# Resolve the release version and check it can be published, before anything
# is built.
#
#   version   RELEASE_VERSION (a pre-release, e.g. v0.32.0rc1) if set, else the
#             branch: releases/v0.32.0 -> 0.32.0 (lib.sh resolve_release_version).
#             Saved as meta-data with whether it is a pre-release; every later
#             step reads both from there.
#   Docker    warns if v<version> is already on Docker Hub; the approver
#             decides whether to overwrite it.
#   CI        not checked here; the approver confirms tpu-inference-ci passed on
#             this commit (linked in the annotation).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

echo "--- :label: Release version"
VERSION="$(resolve_release_version)"
echo "tpu-inference ${VERSION} from ${BUILDKITE_BRANCH} at ${BUILDKITE_COMMIT}"

echo "--- :docker: ${RELEASE_DOCKERHUB_REPO}:v${VERSION} on Docker Hub"
EXISTING="$(dockerhub_digest "v${VERSION}")"
if [[ -n "${EXISTING}" ]]; then
  DOCKERHUB_LINE="**already exists** (\`${EXISTING}\`): publishing replaces it only if the approver chooses to overwrite"
  buildkite-agent annotate --style warning --context release-dockerhub-exists \
    "\`${RELEASE_DOCKERHUB_REPO}:v${VERSION}\` is already on Docker Hub."
else
  DOCKERHUB_LINE="\`v${VERSION}\` is not on Docker Hub"
fi
echo "${DOCKERHUB_LINE}"

buildkite-agent meta-data set release-version "${VERSION}"
PRE="no"
is_prerelease "${VERSION}" && PRE="yes"
buildkite-agent meta-data set release-prerelease "${PRE}"
buildkite-agent annotate --style info --context release-plan <<EOF
**tpu-inference ${VERSION}** from \`${BUILDKITE_BRANCH}\` at \`${BUILDKITE_COMMIT:0:12}\` (pre-release: ${PRE})

- CI: confirm tpu-inference-ci passed on this commit before approving: https://buildkite.com/vllm/tpu-inference-ci/builds?branch=${BUILDKITE_BRANCH}
- Docker Hub: ${DOCKERHUB_LINE}
EOF
