#!/bin/bash
#
# .buildkite/run_in_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

IMAGE_NAME="vllm-tpu"

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
fi

# For HF_TOKEN.
# shellcheck disable=1091
source /etc/environment

exec docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH= \
  -e VLLM_USE_V1=1 \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  "${IMAGE_NAME}" \
  "$@" # Pass all script arguments as the command to run in the container
