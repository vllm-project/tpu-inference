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
#   version   RELEASE_VERSION (a pre-release, e.g. v0.32.0rc1) if set, else the
#             branch: releases/v0.32.0 -> 0.32.0. RELEASE_VERSION must be a
#             version of the branch's X.Y.Z. Every later step reads it from the
#             meta-data set here.
#   CI        not checked here yet: the approver checks tpu-inference-ci.
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"
BUILDKITE_BRANCH=releases/v0.0.1 # TEST ONLY (jefft/test_pipe)

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

buildkite-agent meta-data set release-version "${VERSION}"
PRE="no"
is_prerelease "${VERSION}" && PRE="yes"
buildkite-agent annotate --style info --context release-plan <<EOF
**tpu-inference ${VERSION}** from \`${BUILDKITE_BRANCH}\` at \`${BUILDKITE_COMMIT:0:12}\` (pre-release: ${PRE})

- CI: not checked automatically yet. Before approving, check that tpu-inference-ci passed on this commit: https://buildkite.com/vllm/tpu-inference-ci/builds?branch=${BUILDKITE_BRANCH}
- Docker image and PyPI: not handled by this pipeline yet.
EOF
