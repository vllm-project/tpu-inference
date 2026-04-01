#!/bin/bash
# run_export_docker.sh

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Assuming script is in submodules/tpu-inference/tpu_inference/export
TPU_INFERENCE_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

IMAGE_NAME="tpu-inference-export"

echo "Building Docker image $IMAGE_NAME with context $TPU_INFERENCE_DIR..."
# We use the existing Dockerfile in tpu-inference
docker build -t $IMAGE_NAME -f "$TPU_INFERENCE_DIR/docker/Dockerfile" "$TPU_INFERENCE_DIR"

if [ "$#" -gt 0 ]; then
    docker run --rm \
      -v "$TPU_INFERENCE_DIR:/workspace/tpu_inference" \
      -e GOOGLE_EXPORT_MODEL_PATH="$GOOGLE_EXPORT_MODEL_PATH" \
      -e GOOGLE_EXPORT_TOPOLOGY="$GOOGLE_EXPORT_TOPOLOGY" \
      -e VLLM_TPU_VERSION_OVERRIDE="$VLLM_TPU_VERSION_OVERRIDE" \
      $IMAGE_NAME "${@}"
else
    docker run --rm \
      -v "$TPU_INFERENCE_DIR:/workspace/tpu_inference" \
      -e GOOGLE_EXPORT_MODEL_PATH="$GOOGLE_EXPORT_MODEL_PATH" \
      -e GOOGLE_EXPORT_TOPOLOGY="$GOOGLE_EXPORT_TOPOLOGY" \
      -e VLLM_TPU_VERSION_OVERRIDE="$VLLM_TPU_VERSION_OVERRIDE" \
      $IMAGE_NAME python3 -m unittest tpu_inference.export.test_fake_devices
fi

