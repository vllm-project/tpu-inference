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

# Settle the release version and refuse a release that isn't ready, before
# anything is built.
#
#   version     RELEASE_VERSION (a pre-release, e.g. v0.32.0rc1) if set, else
#               the branch: releases/v0.32.0 -> 0.32.0. RELEASE_VERSION must be
#               a version of the branch's X.Y.Z. Every later step reads it from
#               the meta-data set here.
#   Docker Hub  whether v<version> is already there (a warning: the approver
#               decides whether to overwrite it).
#   CI          not checked here yet: the approver checks tpu-inference-ci.
#   PyPI        neither tpu-inference nor vllm-tpu has this version yet.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

echo "--- :label: Release version"
[[ "${BUILDKITE_BRANCH}" =~ ^releases/v([0-9]+\.[0-9]+\.[0-9]+)$ ]] \
  || die "not a release branch: '${BUILDKITE_BRANCH}'."
BRANCH_VERSION="${BASH_REMATCH[1]}"
VERSION="${RELEASE_VERSION:-${BRANCH_VERSION}}"
VERSION="${VERSION#v}"
[[ "${VERSION}" =~ ^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$ ]] \
  || die "version '${VERSION}' is not a public PEP 440 version."
[[ "${VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+) && "${BASH_REMATCH[1]}" == "${BRANCH_VERSION}" ]] \
  || die "version '${VERSION}' is not a version of ${BUILDKITE_BRANCH}."
echo "tpu-inference ${VERSION} from ${BUILDKITE_BRANCH} at ${BUILDKITE_COMMIT}"

echo "--- :docker: ${RELEASE_DOCKERHUB_REPO}:v${VERSION} on Docker Hub"
EXISTING="$(dockerhub_digest "v${VERSION}")"
if [[ -n "${EXISTING}" ]]; then
  DOCKERHUB_LINE="**already exists** (\`${EXISTING}\`): publishing replaces it only if the approver chooses to overwrite"
  buildkite-agent annotate --style warning --context release-dockerhub-exists \
    "\`${RELEASE_DOCKERHUB_REPO}:v${VERSION}\` is already on Docker Hub."
else
  DOCKERHUB_LINE="\`v${VERSION}\` not there yet"
fi
echo "${DOCKERHUB_LINE}"

echo "--- :package: PyPI"
for project in tpu-inference vllm-tpu; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
    "https://pypi.org/pypi/${project}/${VERSION}/json")"
  case "${code}" in
    404) echo "${project} ${VERSION}: not on PyPI yet" ;;
    200) die "${project} ${VERSION} is already on PyPI; a release can't replace it." ;;
    *) die "PyPI answered ${code} for ${project} ${VERSION}." ;;
  esac
done

buildkite-agent meta-data set release-version "${VERSION}"
PRE="no"
is_prerelease "${VERSION}" && PRE="yes"
buildkite-agent annotate --style info --context release-plan <<EOF
**tpu-inference ${VERSION}** from \`${BUILDKITE_BRANCH}\` at \`${BUILDKITE_COMMIT:0:12}\` (pre-release: ${PRE})

- Docker Hub: ${DOCKERHUB_LINE}
- CI: not checked automatically yet. Before approving, check that tpu-inference-ci passed on this commit: https://buildkite.com/vllm/tpu-inference-ci/builds?branch=${BUILDKITE_BRANCH}
- PyPI: \`tpu-inference\` and \`vllm-tpu\` ${VERSION} not there yet
- Builds the image (staged in GAR) and both wheels once, then waits for approval before anything reaches Docker Hub or PyPI.
EOF
