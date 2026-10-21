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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

# The branch decides the tags. main pushes nightly-YYYYMMDD and moves latest.
# releases/vX.Y.Z pushes YYYYMMDD-X.Y.Z and moves X.Y.Z, so a release nightly
# never touches latest.
#
# With PUBLISH_DOCKERHUB=1 (set by .buildkite/release/pipeline.yml) one of the
# images is also pushed to DOCKERHUB_REPO: main pushes dev as
# nightly-YYYYMMDD-<commit>-<vllm> and moves nightly; an approved build of
# releases/vX.Y.Z pushes prod as vX.Y.Z, or under RELEASE_VERSION for a
# pre-release.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
DATE="$(date +%Y%m%d)"
PUBLISH_DOCKERHUB="${PUBLISH_DOCKERHUB:-0}"
case "${BRANCH}" in
  main)
    DATE_TAG="nightly-${DATE}"
    LATEST_TAG="latest"
    ;;
  releases/v*)
    if [[ ! "${BRANCH}" =~ ${RELEASE_BRANCH_REGEX} ]]; then
      die "branch '${BRANCH}' does not name a version usable as an image tag; expected releases/vX.Y.Z."
    fi
    VERSION="${BASH_REMATCH[1]}"
    DATE_TAG="${DATE}-${VERSION}"
    LATEST_TAG="${VERSION}"
    ;;
  *)
    die "nightly images are published from main or releases/vX.Y.Z, not '${BRANCH}'."
    ;;
esac

# Docker Hub gets a release only from a build someone approved. A scheduled
# nightly of a release branch skips the approval step, so it stops at GAR.
if [[ "${BRANCH}" == releases/v* ]] && ! buildkite-agent meta-data exists dockerhub_move_latest; then
  PUBLISH_DOCKERHUB=0
fi
if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
  DOCKERHUB_REPO="${DOCKERHUB_REPO:?DOCKERHUB_REPO must be set when PUBLISH_DOCKERHUB=1}"
  DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME must be set when PUBLISH_DOCKERHUB=1}"
  MOVE_LATEST="yes"
  OVERWRITE="no"
  if [[ "${BRANCH}" == releases/v* ]]; then
    DOCKERHUB_TARGET="prod"
    DOCKERHUB_VERSION="${RELEASE_VERSION:-${VERSION}}"
    DOCKERHUB_VERSION="${DOCKERHUB_VERSION#v}"
    if ! is_valid_release_version "${DOCKERHUB_VERSION}"; then
      die "RELEASE_VERSION '${RELEASE_VERSION}' is not usable as an image tag; expected e.g. v0.29.0rc1."
    fi
    DOCKERHUB_TAG="v${DOCKERHUB_VERSION}"
    DOCKERHUB_LATEST_TAG="latest"
    MOVE_LATEST="$(buildkite-agent meta-data get dockerhub_move_latest --default no)"
    OVERWRITE="$(buildkite-agent meta-data get dockerhub_overwrite --default no)"
    if is_prerelease "${DOCKERHUB_VERSION}"; then
      MOVE_LATEST="no"
    fi
  else
    # The pyproject.toml pin, which build_image.sh builds with.
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/configs/pipeline_config.sh"
    VLLM_COMMIT_HASH="$(get_vllm_commit_hash)"
    DOCKERHUB_TARGET="dev"
    VLLM_REF="${VLLM_COMMIT_HASH}"
    if [[ "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]]; then
      VLLM_REF="${VLLM_REF:0:7}"
    fi
    DOCKERHUB_TAG="nightly-${DATE}-${BUILDKITE_COMMIT:0:7}-${VLLM_REF}"
    DOCKERHUB_LATEST_TAG="nightly"
  fi

  # Log in to a private, temporary Docker config that only the Docker Hub
  # commands below use.
  { set +x; } 2>/dev/null
  : "${DOCKERHUB_TOKEN_TPU_INFERENCE:?DOCKERHUB_TOKEN_TPU_INFERENCE must be set when PUBLISH_DOCKERHUB=1}"
  DOCKERHUB_CONFIG="$(mktemp -d)"
  cleanup_dockerhub() {
    DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker logout >/dev/null 2>&1 || true
    rm -rf "${DOCKERHUB_CONFIG}"
  }
  trap cleanup_dockerhub EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  printf '%s' "${DOCKERHUB_TOKEN_TPU_INFERENCE}" \
    | DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker login -u "${DOCKERHUB_USERNAME}" --password-stdin >/dev/null 2>&1 \
    || { echo "ERROR: Docker Hub login failed; check DOCKERHUB_TOKEN_TPU_INFERENCE and DOCKERHUB_USERNAME." >&2; exit 1; }
  unset DOCKERHUB_TOKEN_TPU_INFERENCE

  if [[ "${BRANCH}" == releases/v* && "${OVERWRITE}" != "yes" ]] \
    && DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker manifest inspect "${DOCKERHUB_REPO}:${DOCKERHUB_TAG}" >/dev/null 2>&1; then
    echo "ERROR: ${DOCKERHUB_REPO}:${DOCKERHUB_TAG} already exists; choose Overwrite in the approval step to replace it." >&2
    exit 1
  fi
fi

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

REGISTRY="${NIGHTLY_REGISTRY:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference}"

cd "${REPO_ROOT}"

# Build and push variants in order of dependency (ci -> dev, prod) so BuildKit layer cache is reused
TARGETS=("ci" "dev" "prod")

for TARGET in "${TARGETS[@]}"; do
  IMAGE_NAME="${REGISTRY}/vllm-tpu-${TARGET}"
  DATE_IMAGE="${IMAGE_NAME}:${DATE_TAG}"
  LATEST_IMAGE="${IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Building Nightly Image (${TARGET}): ${DATE_IMAGE}"
  ./docker/build_image.sh --target "${TARGET}" -t "${DATE_IMAGE}"

  echo "--- :docker: Tagging as ${LATEST_IMAGE}"
  docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

  echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Artifact Registry"
  docker push "${DATE_IMAGE}"
  docker push "${LATEST_IMAGE}"

  # The same image goes to Docker Hub; nothing is rebuilt.
  if [[ "${PUBLISH_DOCKERHUB}" == "1" && "${TARGET}" == "${DOCKERHUB_TARGET}" ]]; then
    DOCKERHUB_IMAGES=("${DOCKERHUB_REPO}:${DOCKERHUB_TAG}")
    if [[ "${MOVE_LATEST}" == "yes" ]]; then
      DOCKERHUB_IMAGES+=("${DOCKERHUB_REPO}:${DOCKERHUB_LATEST_TAG}")
    else
      echo "--- Not moving ${DOCKERHUB_LATEST_TAG} to ${DOCKERHUB_TAG}."
    fi
    for DOCKERHUB_IMAGE in "${DOCKERHUB_IMAGES[@]}"; do
      echo "--- :docker: Pushing ${DOCKERHUB_IMAGE} to Docker Hub"
      docker tag "${DATE_IMAGE}" "${DOCKERHUB_IMAGE}"
      DOCKER_CONFIG="${DOCKERHUB_CONFIG}" docker push "${DOCKERHUB_IMAGE}"
      docker rmi "${DOCKERHUB_IMAGE}" || true
    done
  fi

  echo "--- Cleaning up locally built image tags for ${TARGET}"
  docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" || true
done

echo "--- Successfully built and published all nightly images!"
