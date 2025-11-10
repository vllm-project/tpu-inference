#!/bin/bash
#
# .buildkite/build_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

cleanup() {
  echo "--- Cleanup Docker Image ---"
  docker rmi -f "${IMAGE_TAG}"
}
# Cleanup will be executed regardless of success or failure.
trap cleanup EXIT

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
  echo "ERROR: BUILDKITE_COMMIT environment variable is not set." >&2
  echo "This script expects BUILDKITE_COMMIT to tag the Docker image." >&2
  exit 1
fi

if [ -z "${BUILDKITE_JOB_ID:-}" ]; then
  echo "ERROR: BUILDKITE_JOB_ID environment variable is not set." >&2
  echo "BUILDKITE_JOB_ID is not set. This script requires it for cleanup docker image." >&2
  exit 1
fi

export TAG_FILE_NAME="vllm_image_tag"
export PROJECT_ID="cloud-ullm-inference-ci-cd"
export LOCATION="asia-south1"
export REPO_NAME="tpu-inference-ci-docker"
export IMAGE_NAME="vllm-tpu"
export IMAGE_TAG="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${BUILDKITE_BUILD_NUMBER}-${BUILDKITE_COMMIT:0:8}"

echo "Image Tag: ${IMAGE_TAG}"
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev -q

echo "--- Build Docker Image ---"
docker build \
    --no-cache -f docker/Dockerfile -t "${IMAGE_TAG}" .

echo "--- Push Docker Image to Artifact Registry ---"
docker push "${IMAGE_TAG}"

# Upload image tag to buildkite artifact
echo "${IMAGE_TAG}" > $TAG_FILE_NAME
buildkite-agent artifact upload $TAG_FILE_NAME
# Clean tag file
rm $TAG_FILE_NAME
