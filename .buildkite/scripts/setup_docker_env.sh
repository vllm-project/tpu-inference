#!/bin/bash
# Copyright 2025 Google LLC
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


# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

cleanup_docker_resource() {
  # Define defaults and get the parameter
  DEFAULT_IMAGES=("vllm-tpu")
  IMAGE_NAME="${1:-}"

  # Combine image sets
  COMBINED_IMAGES=("${DEFAULT_IMAGES[@]}")
  if [[ -n "${IMAGE_NAME}" ]]; then
    COMBINED_IMAGES+=("${IMAGE_NAME}")
  fi

  # Deduplicate arrays and remove empty strings
  mapfile -t TARGET_IMAGES < <(printf "%s\n" "${COMBINED_IMAGES[@]}" | sort -u | grep -v '^$')

  if [[ ${#TARGET_IMAGES[@]} -eq 0 ]]; then
    echo "No target images provided for cleanup."
    exit 0
  fi

  echo "Images to clean up : ${TARGET_IMAGES[*]}"

  # Iterate and cleanup
  for IMG in "${TARGET_IMAGES[@]}"; do
    echo "----------------------------------------"
    echo "Starting cleanup for ${IMG}"

    # Use format to get "Repository ID" and use awk for exact or suffix matching.
    # $1 == img          -> Matches exact local image (e.g., "vllm-tpu")
    # $1 ~ "/"img"$"     -> Matches Google Artifact Registry paths (e.g., ".../.../vllm-tpu")
    OLD_IMAGES=$(docker images --format '{{.Repository}} {{.ID}}' | awk -v img="${IMG}" '$1 == img || $1 ~ "/"img"$" {print $2}' | sort -u)
    
    if [[ -n "$OLD_IMAGES" ]]; then
      echo "Found matching images for ${IMG}. Checking for dependent containers..."
      
      TOTAL_CONTAINERS=""
      for img_id in $OLD_IMAGES; do
        # Find containers using this image ID
        TOTAL_CONTAINERS="$TOTAL_CONTAINERS $(docker ps -a -q --filter "ancestor=$img_id")"
      done
      
      # Format and remove any found containers
      CLEANED_CONTAINERS=$(echo "$TOTAL_CONTAINERS" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
      if [[ -n "$CLEANED_CONTAINERS" ]]; then
        echo "Removing leftover containers using ${IMG} image(s)..."
        echo "$CLEANED_CONTAINERS" | xargs -r docker rm -f
      fi
      
      echo "Removing old ${IMG} image(s) by ID..."
      # Using ID directly ensures all tags of that specific image are untagged and removed
      echo "$OLD_IMAGES" | xargs -r docker rmi -f
    else
      echo "No images matching ${IMG} found to clean up."
    fi
  done

  echo "Pruning old Docker build cache..."
  docker builder prune -f

  echo "Cleanup complete."
}

# Verify a built or pulled image actually contains the expected vLLM commit.
# Guards against tag/image mismatches -- e.g. when the CI (LKG vLLM) and
# integration (HEAD vLLM) pipelines build the same tpu-inference commit
# concurrently, they previously shared a vLLM-agnostic tag and could clobber
# each other's image in the registry. No-op when the expected hash is empty
# (local/dev builds that clone vLLM HEAD) or for PyPI builds (no
# /workspace/vllm git checkout to inspect).
verify_image_vllm() {
  local image_ref="$1"
  local expected_vllm="${2:-}"
  if [[ -z "${expected_vllm}" ]]; then
    return 0
  fi
  if [[ "${RUN_WITH_PYPI:-false}" == "true" ]]; then
    echo "[verify-vllm] RUN_WITH_PYPI=true; skipping vLLM commit verification."
    return 0
  fi
  # Read the vLLM commit baked into the image. Capture the exit code separately
  # so an infra failure to *read* the commit (git missing, /workspace/vllm
  # absent, safe.directory "dubious ownership", etc.) is reported as such
  # instead of masquerading as a vLLM mismatch. -c safe.directory pre-empts the
  # most common trip: the container running git as a non-owner user.
  local actual_vllm rc=0 err
  err=$(mktemp)
  actual_vllm=$(docker run --rm --entrypoint git "${image_ref}" \
    -C /workspace/vllm -c safe.directory=/workspace/vllm rev-parse HEAD 2>"${err}") || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[FATAL][verify-vllm] Could not read the vLLM commit from ${image_ref} (git exit ${rc}):" >&2
    cat "${err}" >&2
    rm -f "${err}"
    exit 1
  fi
  rm -f "${err}"
  if [[ "${actual_vllm}" != "${expected_vllm}" ]]; then
    echo "[FATAL][verify-vllm] Image ${image_ref} contains vLLM ${actual_vllm}, but expected ${expected_vllm}."
    echo "[FATAL][verify-vllm] Aborting to avoid testing or promoting the wrong vLLM."
    exit 1
  fi
  echo "[verify-vllm] OK: ${image_ref} contains expected vLLM ${expected_vllm}."
}

setup_environment() {
  local image_name_param=${1:-"vllm-tpu"}
  local should_push=${2:-"false"}
  local push_to_ci_cache=${3:-"false"}
  IMAGE_NAME="$image_name_param"

  # ==========================================
  # Skip Build and Cleanup in DEV_MODE if image already exists
  # ==========================================
  if [[ "${DEV_MODE:-false}" == "true" ]]; then
    if docker image inspect "${IMAGE_NAME}:dev" >/dev/null 2>&1; then
      echo "[DEV_MODE] Base image ${IMAGE_NAME}:dev already exists. Skipping cleanup and build."
      return 0
    fi
  fi

  local CI_IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/${IMAGE_NAME}"
  local LOCAL_TPU_VERSION="${TPU_VERSION:-tpu6e}" 

  local DOCKERFILE_NAME="Dockerfile"

# Determine whether to build from PyPI packages or source.
  if [[ "${RUN_WITH_PYPI:-false}" == "true" ]]; then
    DOCKERFILE_NAME="Dockerfile.pypi"
    echo "Building from PyPI packages. Using docker/${DOCKERFILE_NAME}"
  else
    echo "Building from source. Using docker/${DOCKERFILE_NAME}"
  fi

  if ! grep -q "^HF_TOKEN=" /etc/environment; then
    gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
    sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)"
    echo "Added HF_TOKEN to /etc/environment."
  else
    echo "HF_TOKEN already exists in /etc/environment."
  fi

  # shellcheck disable=1091
  source /etc/environment
  cleanup_docker_resource "${IMAGE_NAME}"

  if [ -z "${BUILDKITE:-}" ]; then
      if [ "${USE_VLLM_LKG:-false}" == "true" ] && [ -f ".buildkite/vllm_lkg.version" ]; then
          VLLM_COMMIT_HASH=$(cat .buildkite/vllm_lkg.version)
      else
          VLLM_COMMIT_HASH=""
      fi
      if [[ "${DEV_MODE:-false}" == "true" ]]; then
          TPU_INFERENCE_HASH="dev"
      else
          TPU_INFERENCE_HASH=$(git log -n 1 --pretty="%H")
      fi
  else
      VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")
      TPU_INFERENCE_HASH="$BUILDKITE_COMMIT"
  fi
 
  # Include the vLLM commit in the cache tag so an image is uniquely identified
  # by BOTH its tpu-inference commit and its vLLM commit. Without this, the CI
  # pipeline (LKG vLLM) and the integration pipeline (HEAD vLLM) produce the same
  # tag for the same tpu-inference commit and can overwrite each other's image in
  # the registry -- a test could then silently run a different vLLM than intended.
  # Exposed as a shell global (not `local`) so callers that source this script --
  # notably run_multihost.sh -- can distribute the exact vLLM-qualified image
  # instead of a tpu-inference-commit-only tag that another build (different
  # vLLM) may have clobbered in the registry.
  CACHE_TAG="${TPU_INFERENCE_HASH}-${LOCAL_TPU_VERSION}"
  if [[ -n "${VLLM_COMMIT_HASH}" ]]; then
    CACHE_TAG="${TPU_INFERENCE_HASH}-${VLLM_COMMIT_HASH}-${LOCAL_TPU_VERSION}"
  elif [[ -n "${BUILDKITE:-}" && "${RUN_WITH_PYPI:-false}" != "true" ]]; then
    # In a real pipeline an empty VLLM_COMMIT_HASH is a bug (bootstrap didn't set
    # the metadata, or the Dockerfile will clone vLLM HEAD non-deterministically).
    # Falling back to the vLLM-agnostic tag here would silently re-introduce the
    # cross-pipeline clobber this change eliminates, so fail loudly instead.
    echo "[FATAL][setup_docker_env] VLLM_COMMIT_HASH is empty in a Buildkite build." >&2
    echo "[FATAL][setup_docker_env] Refusing the vLLM-agnostic tag '${CACHE_TAG}' (can collide" >&2
    echo "[FATAL][setup_docker_env] across CI/integration). Ensure bootstrap set the" >&2
    echo "[FATAL][setup_docker_env] VLLM_COMMIT_HASH metadata (LKG or HEAD)." >&2
    exit 1
  fi

  # ==========================================
  # Pull-Only Mode for TPU execution nodes
  # ==========================================
  if [[ "${USE_PREBUILT_IMAGE:-0}" == "1" ]]; then
    echo "Pulling pre-built Docker image: ${CI_IMAGE_REPO}:${CACHE_TAG} ..."
    docker pull "${CI_IMAGE_REPO}:${CACHE_TAG}"
    verify_image_vllm "${CI_IMAGE_REPO}:${CACHE_TAG}" "${VLLM_COMMIT_HASH}"
    docker tag "${CI_IMAGE_REPO}:${CACHE_TAG}" "${IMAGE_NAME}:${TPU_INFERENCE_HASH}"
    docker tag "${CI_IMAGE_REPO}:${CACHE_TAG}" "${IMAGE_NAME}:latest"
    return 0
  fi

  # Build with specific hash and 'latest' tag for convenience
  docker build \
      --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}" \
      --build-arg IS_TEST="true" \
      --build-arg BM_INFRA="${BM_INFRA:-false}" \
      --no-cache -f docker/"${DOCKERFILE_NAME}" \
      -t "${IMAGE_NAME}:${TPU_INFERENCE_HASH}" \
      -t "${IMAGE_NAME}:latest" \
      -t "${IMAGE_NAME}:${CACHE_TAG}" .

  # Fail fast if the freshly built image does not contain the expected vLLM
  # commit (guards against a mis-set VLLM_COMMIT_HASH build-arg).
  verify_image_vllm "${IMAGE_NAME}:${CACHE_TAG}" "${VLLM_COMMIT_HASH}"

  # ==========================================
  # Push to CI Image Registry (Executed by dedicate CPU builder)
  # ==========================================
  if [[ "$push_to_ci_cache" == "true" ]]; then
    echo "Pushing Docker image to CI Image Registry..."
    docker tag "${IMAGE_NAME}:${CACHE_TAG}" "${CI_IMAGE_REPO}:${CACHE_TAG}"
    docker push "${CI_IMAGE_REPO}:${CACHE_TAG}"
  fi

  # Push logic if requested
  if [[ "$should_push" == "true" ]]; then
    echo "--- Pushing Docker image(s) to registry..."
    gcloud auth configure-docker us-central1-docker.pkg.dev
    docker push "${IMAGE_NAME}:${TPU_INFERENCE_HASH}"
    docker push "${IMAGE_NAME}:latest"
    # Also push the vLLM-qualified tag so consumers (e.g. run_multihost.sh) can
    # pull an image that cannot be clobbered by another build on the same
    # tpu-inference commit built against a different vLLM.
    docker push "${IMAGE_NAME}:${CACHE_TAG}"
  fi

  # Only clean up resources in setup_environment after pushing to a remote registry
  # (e.g., standalone builder jobs in CI). When building locally to run on the same machine
  # (push_to_ci_cache=false and should_push=false), preserve the image so the caller can run it.
  if [[ "$push_to_ci_cache" == "true" || "$should_push" == "true" ]]; then
    echo "--- Cleaning up Docker resources after push..."
    cleanup_docker_resource "${IMAGE_NAME}"
  fi
}
