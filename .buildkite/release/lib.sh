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

# Shared by the tpu-inference-release steps. Sourced, not run.

# Release branches and versions: public PEP 440 versions in normalized form
# (no local +... part, no leading zeros).
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

# a, b, rc and dev versions are pre-releases.
is_prerelease() {
  [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

# Resolve and validate the release version: RELEASE_VERSION (a pre-release,
# e.g. v0.32.0rc1) if set, else the branch (releases/vX.Y.Z -> X.Y.Z).
# RELEASE_VERSION must be a version of the branch's X.Y.Z.
resolve_release_version() {
  local branch_version version source
  [[ "${BUILDKITE_BRANCH:-}" =~ ${RELEASE_BRANCH_REGEX} ]] \
    || die "not a release branch: '${BUILDKITE_BRANCH:-}'."
  branch_version="${BASH_REMATCH[1]}"
  if [[ -n "${RELEASE_VERSION:-}" ]]; then
    version="${RELEASE_VERSION#v}"
    source="RELEASE_VERSION=${RELEASE_VERSION}"
  else
    version="${branch_version}"
    source="branch ${BUILDKITE_BRANCH}"
  fi
  is_valid_release_version "${version}" \
    || die "version '${version}' from ${source} is not a public PEP 440 version."
  [[ "${version}" =~ ^(${_N}\.${_N}\.${_N}) && "${BASH_REMATCH[1]}" == "${branch_version}" ]] \
    || die "version '${version}' from ${source} is not a version of ${BUILDKITE_BRANCH}."
  echo "Release version: ${version} (from ${source})" >&2
  echo "${version}"
}

# The version every step of the build agrees on, set once by preflight.sh
# (or resolved on the spot when running a step standalone).
release_version() {
  local rc=0
  buildkite-agent meta-data exists release-version || rc=$?
  case "${rc}" in
    0) buildkite-agent meta-data get release-version ;;
    100) resolve_release_version ;;
    *) die "could not read meta-data release-version (exit ${rc})." ;;
  esac
}

# --- Docker image -------------------------------------------------------------

# Where the release image is staged before anyone approves it: the same image
# as CI's, under release-* tags, which CI's <commit>-<vllm commit>-<tpu> tags
# never collide with. GAR is internal; Docker Hub is what users pull.
RELEASE_GAR="${RELEASE_GAR:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci}"
# shellcheck disable=SC2034 # Read by build_images.sh.
RELEASE_IMAGE="${RELEASE_GAR}/vllm-tpu"
# shellcheck disable=SC2034 # Read by build_images.sh.
RELEASE_TAG_PREFIX="release-"
RELEASE_DOCKERHUB_REPO="${RELEASE_DOCKERHUB_REPO:-vllm/vllm-tpu}"
RELEASE_DOCKERHUB_USERNAME="${RELEASE_DOCKERHUB_USERNAME:-tpuinference}"

# The digest part of <repo>@sha256:<hex>.
digest_of() {
  echo "${1##*@}"
}

# The digest Docker Hub serves for RELEASE_DOCKERHUB_REPO:<tag> to an anonymous
# user, or nothing if the tag doesn't exist. A HEAD request, which Docker Hub
# doesn't count against the pull rate limit.
dockerhub_digest() {
  local tag="$1" token headers status
  token="$(curl -fsS --max-time 30 \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${RELEASE_DOCKERHUB_REPO}:pull" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')" \
    || die "no anonymous Docker Hub token for ${RELEASE_DOCKERHUB_REPO}."
  headers="$(curl -sS --max-time 30 -I -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" \
    "https://registry-1.docker.io/v2/${RELEASE_DOCKERHUB_REPO}/manifests/${tag}" | tr -d '\r')"
  status="$(head -n 1 <<< "${headers}" | awk '{print $2}')"
  case "${status}" in
    200) sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: //p' <<< "${headers}" ;;
    404) ;;
    *) die "Docker Hub answered '${status}' for ${RELEASE_DOCKERHUB_REPO}:${tag}." ;;
  esac
}

# Log in to Docker Hub inside a private <config_dir> and unset the secret token
# so no child process inherits it.
dockerhub_login() {
  local config_dir="$1" username="${2:-${RELEASE_DOCKERHUB_USERNAME}}"
  { set +x; } 2>/dev/null # Never trace the token into the build log.
  # An explicit check rather than ${DOCKERHUB_TOKEN:?}: under an EXIT trap,
  # some bash versions exit 0 after that expansion error.
  [[ -n "${DOCKERHUB_TOKEN:-}" ]] || die "the DOCKERHUB_TOKEN secret is not set for this step."
  printf '%s' "${DOCKERHUB_TOKEN}" \
    | DOCKER_CONFIG="${config_dir}" docker login -u "${username}" --password-stdin >/dev/null 2>&1 \
    || die "Docker Hub login failed for ${username}."
  unset DOCKERHUB_TOKEN
}
