#!/bin/bash

set -ex

IMAGE_NAME='vllm-tpu'
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
setup_environment $IMAGE_NAME

SCRIPT_DIR=$SCRIPT_DIR/../../examples/disagg

# call the /examples/disagg/run_disagg_multi_host.sh script
CONTAINER_PREFIX="disagg-node"

CONTAINER_PREFIX=${CONTAINER_PREFIX} \
RUN_IN_BUILDKITE=true \
MODEL=${MODEL:="Qwen/Qwen3-0.6B"} \
DOCKER_IMAGE=${IMAGE_NAME}:${BUILDKITE_COMMIT} \
"$SCRIPT_DIR/run_disagg_multi_host.sh" "$@"

# clear existing containers
CONTAINERS=$(docker ps -a --filter "name=${CONTAINER_PREFIX}*" -q)
if [ -n "$CONTAINERS" ]; then
  # shellcheck disable=SC2086
  docker stop $CONTAINERS
  # shellcheck disable=SC2086
  docker rm -f $CONTAINERS
fi
