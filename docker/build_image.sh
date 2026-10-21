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

usage() {
  cat <<USAGE
Usage: ./docker/build_image.sh [options]

Builds the vllm-tpu image from the repository root as the docker context.

Options:
  -t, --image-tag TAG         Tag for the built image (default: vllm-tpu-local)
  -s, --vllm-source DIR       Local vLLM source tree to install in editable mode
  -c, --vllm-commit-hash SHA  vLLM commit SHA or tag override (defaults to pyproject.toml pin)
      --target TARGET         Dockerfile stage to build: prod, dev, ci, or base (default: prod)
      --no-cache              Disable Docker build cache
  -h, --help                  Show this help and exit

Examples:
  ./docker/build_image.sh
  ./docker/build_image.sh --target dev -t vllm-tpu:dev
  ./docker/build_image.sh -s /path/to/vllm/source --target dev
USAGE
}

IMAGE_TAG="vllm-tpu-local"
VLLM_SOURCE=""
VLLM_COMMIT_HASH=""
TARGET="prod"
NO_CACHE="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -s|--vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    -c|--vllm-commit-hash)
      VLLM_COMMIT_HASH="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "===> Building vllm-tpu image..."
echo "Image Tag: $IMAGE_TAG"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"
STAGED_VLLM="${REPO_ROOT}/vllm"

if [ -n "$VLLM_SOURCE" ] && [ ! -d "$VLLM_SOURCE" ]; then
  echo "vllm source directory not found: $VLLM_SOURCE" >&2
  exit 1
fi

if [ -z "$VLLM_COMMIT_HASH" ]; then
  VLLM_COMMIT_HASH="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9._-]+).*/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"

  if [ -n "$VLLM_COMMIT_HASH" ]; then
    echo "-c/--vllm-commit-hash not provided. Defaulting to pyproject.toml pin: $VLLM_COMMIT_HASH"
  else
    echo "ERROR: VLLM_COMMIT_HASH could not be parsed from pyproject.toml." >&2
    exit 1
  fi
fi

echo "vLLM Commit Hash: $VLLM_COMMIT_HASH"

CLEANUP_VLLM=false
cleanup() {
  if [ "$CLEANUP_VLLM" = true ]; then
    echo "===> Cleaning up staged vllm source..."
    rm -rf "${STAGED_VLLM}"
    CLEANUP_VLLM=false
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

is_empty_dir() {
  [ -d "$1" ] && [ -z "$(find "$1" -mindepth 1 -print -quit)" ]
}

if [ -n "$VLLM_SOURCE" ]; then
  if [ "$(cd "$VLLM_SOURCE" && pwd)" = "${STAGED_VLLM}" ]; then
    echo "===> Using in-tree vllm source at ${STAGED_VLLM} in place..."
  elif [ -e "${STAGED_VLLM}" ] && ! is_empty_dir "${STAGED_VLLM}"; then
    echo "Refusing to overwrite existing ${STAGED_VLLM}." >&2
    echo "Move it away, or build from it with: $0 -s ${STAGED_VLLM}" >&2
    exit 1
  else
    echo "===> Copying local vllm source from $VLLM_SOURCE..."
    CLEANUP_VLLM=true
    rm -rf "${STAGED_VLLM}"
    cp -r "$VLLM_SOURCE" "${STAGED_VLLM}"
  fi
fi

echo "Build Target: $TARGET"
DOCKER_ARGS=(
  --progress=plain
  -f "${SCRIPT_DIR}/Dockerfile"
  -t "${IMAGE_TAG}"
  --target "${TARGET}"
  --build-arg VLLM_SOURCE="${VLLM_SOURCE}"
  --build-arg VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH}"
)

if [ "$NO_CACHE" = "true" ]; then
  DOCKER_ARGS+=(--no-cache)
fi

DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" "${REPO_ROOT}"

cleanup
echo "===> Done!"
