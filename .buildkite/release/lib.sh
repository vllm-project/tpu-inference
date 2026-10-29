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

# Shared by the tpu-inference-release steps and the wheel build and publish
# scripts in .buildkite/scripts. Sourced, not run.

# Newest manylinux tag (i.e. glibc) the release wheels may need: the release
# queue runs on Ubuntu 24.04 (glibc 2.39), the vllm-tpu build fails on a wheel
# tagged above this, and check_wheels.sh resolves for this platform.
WHEEL_MAX_MANYLINUX="2_39"

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
# (or resolved on the spot when a step runs standalone).
release_version() {
  local rc=0
  buildkite-agent meta-data exists release-version || rc=$?
  case "${rc}" in
    0) buildkite-agent meta-data get release-version ;;
    100) resolve_release_version ;;
    *) die "could not read meta-data release-version (exit ${rc})." ;;
  esac
}

# Fail if <project>==<version> is already on PyPI (or <index_base>): a version
# can never be reused.
check_not_published() {
  local project="$1" version="$2" index_base="${3:-https://pypi.org}"
  local url="${index_base%/}/pypi/${project}/${version}/json" code
  code="$(curl -sL -o /dev/null -w '%{http_code}' --max-time 30 "${url}")" \
    || die "could not reach ${url}."
  case "${code}" in
    404) echo "${project} ${version}: not on ${index_base%/} yet" ;;
    200) die "${project} ${version} is already on ${index_base%/}; a release can't replace it." ;;
    *) die "${index_base%/} answered ${code} for ${project} ${version}." ;;
  esac
}

# Install uv into <dir> and print its binary path.
install_uv() {
  local dir="$1"
  curl -LsSf --retry 3 https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${dir}" UV_NO_MODIFY_PATH=1 sh >/dev/null \
    || die "could not install uv into ${dir}."
  [[ -x "${dir}/uv" ]] || die "uv install finished but ${dir}/uv is missing."
  echo "${dir}/uv"
}

# Platform tag of a wheel file, the last -separated part of its name:
# manylinux_2_31_x86_64, or any for a pure-Python wheel.
wheel_platform_tag() {
  local name
  name="$(basename "$1" .whl)"
  echo "${name##*-}"
}

# Fail unless <wheel> is tagged manylinux_X_Y_x86_64 with X.Y no newer than
# WHEEL_MAX_MANYLINUX. The tag is the newest glibc the wheel's binaries need.
check_manylinux_tag() {
  local wheel="$1" tag major minor max_major max_minor
  tag="$(wheel_platform_tag "${wheel}")"
  [[ "${tag}" =~ ^manylinux_([0-9]+)_([0-9]+)_x86_64$ ]] \
    || die "$(basename "${wheel}") is tagged '${tag}'; expected manylinux_X_Y_x86_64."
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  max_major="${WHEEL_MAX_MANYLINUX%_*}"
  max_minor="${WHEEL_MAX_MANYLINUX#*_}"
  if (( major > max_major || (major == max_major && minor > max_minor) )); then
    die "$(basename "${wheel}") needs glibc ${major}.${minor}, newer than the" \
      "${max_major}.${max_minor} the release supports (WHEEL_MAX_MANYLINUX in" \
      ".buildkite/release/lib.sh). Build it on an older OS image."
  fi
}

# Resolve vllm-tpu==<version> for the platform the wheels support (CPython
# 3.12 on x86_64 Linux with the glibc of WHEEL_MAX_MANYLINUX) into <out_file>,
# passing any extra uv flags (e.g. --find-links <dir>).
resolve_vllm_tpu() {
  local uv_bin="$1" version="$2" out_file="$3"
  shift 3
  local req_file
  req_file="$(dirname "${out_file}")/req.in"
  echo "vllm-tpu==${version}" > "${req_file}"
  "${uv_bin}" pip compile "${req_file}" --no-config --quiet \
    --python-version 3.12 --python-platform "x86_64-manylinux_${WHEEL_MAX_MANYLINUX}" \
    --index-url https://pypi.org/simple \
    "$@" -o "${out_file}"
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
