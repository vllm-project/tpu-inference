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
# shellcheck shell=bash

# Shared by the tpu-inference-release steps. Sourced, not run.

# Where the staged images go before anyone approves, and where approved images
# are copied to. GAR is internal; Docker Hub is what users pull.
RELEASE_GAR="${RELEASE_GAR:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference}"
RELEASE_DOCKERHUB_REPO="${RELEASE_DOCKERHUB_REPO:-vllm/vllm-tpu}"
RELEASE_DOCKERHUB_USERNAME="${RELEASE_DOCKERHUB_USERNAME:-vllm}"

# Normalized X.Y.Z public PEP 440 version: PyPI rejects local (+...) parts,
# and setuptools normalizes away leading zeros.
_N="(0|[1-9][0-9]*)"
RELEASE_BRANCH_REGEX="^releases/v(${_N}\.${_N}\.${_N})$"
PEP440_PUBLIC_REGEX="^${_N}\.${_N}\.${_N}((a|b|rc)${_N})?(\.post${_N})?(\.dev${_N})?$"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_valid_release_version() {
    [[ "$1" =~ ${PEP440_PUBLIC_REGEX} ]]
}

# A pre-release never moves latest on Docker Hub.
is_prerelease() {
    [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

# Resolve and validate the release version from RELEASE_VERSION (an rc, e.g.
# v0.30.0rc1) if set, else from BUILDKITE_BRANCH (releases/vX.Y.Z -> X.Y.Z).
resolve_release_version() {
    local version source
    if [[ -n "${RELEASE_VERSION:-}" ]]; then
        version="${RELEASE_VERSION#v}"
        source="RELEASE_VERSION=${RELEASE_VERSION}"
    else
        [[ "${BUILDKITE_BRANCH:-}" =~ ${RELEASE_BRANCH_REGEX} ]] \
            || die "not a release branch: '${BUILDKITE_BRANCH:-}'."
        version="${BASH_REMATCH[1]}"
        source="branch ${BUILDKITE_BRANCH}"
    fi
    is_valid_release_version "${version}" \
        || die "version '${version}' from ${source} is not a public PEP 440 version."
    echo "${version}"
}

# The version every step of the build agrees on, set once by preflight.sh.
release_version() {
    buildkite-agent meta-data get release-version
}

# The digest part of <repo>@sha256:<hex>.
digest_of() {
    echo "${1##*@}"
}
