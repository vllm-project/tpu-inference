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
#   version   RELEASE_VERSION (an rc, e.g. v0.30.0rc1) if set, else the branch:
#             releases/v0.30.0 -> 0.30.0. Every later step reads it from the
#             meta-data set here, so the images and wheels can't disagree.
#   CI        tpu-inference-ci passed a full (NIGHTLY=1) run of this commit
#             (ci_gate.py). RELEASE_SKIP_CI_GATE=1 skips it, with a warning.
#   PyPI      neither tpu-inference nor vllm-tpu has this version yet.
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

echo "--- :label: Release version"
VERSION="$(resolve_release_version)"
SOURCE="${RELEASE_VERSION:+RELEASE_VERSION=${RELEASE_VERSION}}"
SOURCE="${SOURCE:-branch ${BUILDKITE_BRANCH}}"
echo "tpu-inference / vllm-tpu ${VERSION} (from ${SOURCE}) at ${BUILDKITE_COMMIT}"

echo "--- :white_check_mark: tpu-inference-ci on ${BUILDKITE_COMMIT:0:12}"
if [[ "${RELEASE_SKIP_CI_GATE:-0}" == "1" ]]; then
    CI_LINE="**skipped** (RELEASE_SKIP_CI_GATE=1)"
    buildkite-agent annotate --style warning --context release-ci-gate \
        "CI gate skipped (RELEASE_SKIP_CI_GATE=1): nothing checked that ${BUILDKITE_COMMIT:0:12} passed CI."
else
    CI_URL="$(python3 "${RELEASE_TOOLS}/release/ci_gate.py" "${BUILDKITE_COMMIT}" | tail -n 1)"
    CI_LINE="passed: ${CI_URL}"
fi

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
buildkite-agent meta-data set release-prerelease "${PRE}"
buildkite-agent annotate --style info --context release-plan <<EOF
**tpu-inference / vllm-tpu ${VERSION}** from \`${BUILDKITE_BRANCH}\` at \`${BUILDKITE_COMMIT:0:12}\` (pre-release: ${PRE})

- tpu-inference-ci: ${CI_LINE}
- Builds the ci, dev and prod images (staged in GAR) and both wheels, then waits for approval before anything reaches Docker Hub or PyPI.
EOF
