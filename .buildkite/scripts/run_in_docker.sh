#!/bin/bash
#
# .buildkite/run_in_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
fi

# For HF_TOKEN.
# shellcheck disable=1091
source /etc/environment

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
  echo "ERROR: BUILDKITE_COMMIT environment variable is not set." >&2
  echo "This script expects BUILDKITE_COMMIT to tag the Docker image." >&2
  exit 1
fi

if [ -z "${TPU_BACKEND_TYPE:-}" ]; then
  TPU_BACKEND_TYPE=pytorch_xla
fi

# Prune older images on the host to save space.
docker system prune -a -f --filter "until=3h"

# (TODO): Consider creating a remote registry to cache and share between agents.
# Subsequent builds on the same host should be cached.
docker build -f docker/Dockerfile -t "vllm-tpu:${BUILDKITE_COMMIT}" .

exec docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -e TPU_BACKEND_TYPE="$TPU_BACKEND_TYPE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH= \
  -e VLLM_USE_V1=1 \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  "vllm-tpu:${BUILDKITE_COMMIT}" \
  "$@" # Pass all script arguments as the command to run in the container
