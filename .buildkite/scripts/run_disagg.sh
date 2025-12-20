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
