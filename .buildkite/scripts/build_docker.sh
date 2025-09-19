#!/bin/bash
#
# .buildkite/build_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

source /etc/environment

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
  echo "ERROR: BUILDKITE_COMMIT environment variable is not set." >&2
  echo "This script expects BUILDKITE_COMMIT to tag the Docker image." >&2
  exit 1
fi

export PROJECT_ID="cienet-cmcs"
export LOCATION="us-central1"
export REPO_NAME="sting-vllm-tpu"
export IMAGE_NAME="vllm-tpu"
export IMAGE_TAG="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${BUILDKITE_BUILD_NUMBER}-${BUILDKITE_COMMIT:0:8}"

export VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')

echo "Image Tag: ${IMAGE_TAG}"
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev -q

echo "--- Build Docker Image ---"
docker build \
    --build-arg VLLM_COMMIT_HASH=${VLLM_COMMIT_HASH} \
    --no-cache -f docker/Dockerfile -t "${IMAGE_TAG}" .

echo "--- Push Docker Image to Artifact Registry ---"
docker push "${IMAGE_TAG}"

# Upload image tag to buildkite artifact
echo "${IMAGE_TAG}" > image_tag.txt
buildkite-agent artifact upload image_tag.txt
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
buildkite-agent meta-data set "TPU_COMMONS_COMMIT_HASH" "${BUILDKITE_COMMIT}"

