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

# a, b, rc and dev versions are pre-releases.
is_prerelease() {
  [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

# Resolve and validate the release version: RELEASE_VERSION (a pre-release,
# e.g. v0.32.0rc1) if set, else the branch (releases/v0.32.0 -> 0.32.0).
# RELEASE_VERSION must be a version of the branch's X.Y.Z.
resolve_release_version() {
  local branch_version version source
  [[ "${BUILDKITE_BRANCH:-}" =~ ${RELEASE_BRANCH_REGEX} ]] \
    || die "not a release branch: '${BUILDKITE_BRANCH:-}'."
  branch_version="${BASH_REMATCH[1]}"
  version="${branch_version}"
  source="branch ${BUILDKITE_BRANCH}"
  if [[ -n "${RELEASE_VERSION:-}" ]]; then
    version="${RELEASE_VERSION#v}"
    source="RELEASE_VERSION=${RELEASE_VERSION}"
  fi
  is_valid_release_version "${version}" \
    || die "version '${version}' from ${source} is not a public PEP 440 version."
  [[ "${version}" =~ ^(${_N}\.${_N}\.${_N}) && "${BASH_REMATCH[1]}" == "${branch_version}" ]] \
    || die "version '${version}' is not a version of ${BUILDKITE_BRANCH}."
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

# Resolve vllm-tpu==<version> for the platform the wheels support (CPython
# 3.12, x86_64 Linux, glibc >= 2.31) into <out_file>, passing any extra uv
# flags (e.g. --find-links <dir>).
resolve_vllm_tpu() {
  local uv_bin="$1" version="$2" out_file="$3"
  shift 3
  local req_file
  req_file="$(dirname "${out_file}")/req.in"
  echo "vllm-tpu==${version}" > "${req_file}"
  "${uv_bin}" pip compile "${req_file}" --no-config --quiet \
    --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
    --index-url https://pypi.org/simple \
    "$@" -o "${out_file}"
}
