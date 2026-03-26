#!/bin/bash
# Build and push custom Pathways server image with custom libtpu.so
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="us-east5-docker.pkg.dev/cloud-tpu-inference-test/wenxindong/pathways_server:latest"
LIBTPU_GCS="https://storage.googleapis.com/wenxindong-vm/libtpu/subhankarshah/libtpu/_libtpu.so"

cd "$SCRIPT_DIR"

# Download custom libtpu.so if not already present
if [ ! -f libtpu.so ]; then
  echo "Downloading custom libtpu.so from GCS..."
  TOKEN=$(gcloud auth print-access-token)
  curl -sf -H "Authorization: Bearer ${TOKEN}" -o libtpu.so "${LIBTPU_GCS}"
  echo "Downloaded libtpu.so ($(ls -lh libtpu.so | awk '{print $5}'), md5: $(md5sum libtpu.so | awk '{print $1}'))"
else
  echo "Using existing libtpu.so ($(ls -lh libtpu.so | awk '{print $5}'), md5: $(md5sum libtpu.so | awk '{print $1}'))"
fi

# Build the image
echo "Building Docker image: ${IMAGE}"
docker build -f Dockerfile.pathways-server -t "${IMAGE}" .

# Push the image
echo "Pushing Docker image: ${IMAGE}"
docker push "${IMAGE}"

# Clean up downloaded libtpu.so
rm -f libtpu.so

echo "Done! Image pushed to ${IMAGE}"
