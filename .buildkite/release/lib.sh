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
# shellcheck disable=SC2034 # The settings are read by the scripts that source this.

# Where the release image is staged before anyone approves it. GAR is internal;
# Docker Hub is what users pull.
RELEASE_GAR="${RELEASE_GAR:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci}"
# The same image as CI's, under release-* tags, which CI's
# <commit>-<vllm commit>-<tpu> tags never collide with.
RELEASE_IMAGE="${RELEASE_GAR}/vllm-tpu"
RELEASE_TAG_PREFIX="release-"
RELEASE_DOCKERHUB_REPO="${RELEASE_DOCKERHUB_REPO:-vllm/vllm-tpu}"
RELEASE_DOCKERHUB_USERNAME="${RELEASE_DOCKERHUB_USERNAME:-tpuinference}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# The version every step of the build agrees on, set once by preflight.sh.
release_version() {
  buildkite-agent meta-data get release-version
}

# A pre-release never moves latest on Docker Hub.
is_prerelease() {
  [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

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
