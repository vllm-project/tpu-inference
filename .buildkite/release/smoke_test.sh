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

# Serve the staged release image on one TPU chip and run a short benchmark
# against it, before the publish block. Runs on the kube queue: the launcher
# (.buildkite/kubernetes/run.sh) starts the image as a TPU pod, and the server
# and the client (smoke_bench.sh) both run in it, from the image's checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

MODEL="${SMOKE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_PROMPTS="${SMOKE_NUM_PROMPTS:-30}"
# The model download and the first TPU compile both happen before /health
# answers.
SERVER_TIMEOUT_S="${SMOKE_SERVER_TIMEOUT_S:-3600}"

STAGED="$(buildkite-agent meta-data get release-image)"
[[ "${STAGED}" == "${RELEASE_IMAGE}@sha256:"* ]] \
  || die "release-image is '${STAGED}', not a ${RELEASE_IMAGE} digest."

# The exact image the publish step copies to Docker Hub.
export WORKLOAD_IMAGE="${STAGED}"

serve_cmd="vllm serve ${MODEL} --max-model-len 1024 --tensor-parallel-size 1"
client_cmd="bash .buildkite/release/smoke_bench.sh ${MODEL} ${NUM_PROMPTS}"

echo "--- :kubernetes: Serving ${MODEL} from $(digest_of "${STAGED}") on ${SHAPE:-}"
exec "${SCRIPT_DIR}/../kubernetes/run.sh" \
  env SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S}" \
  bash .buildkite/kubernetes/serve_and_run.sh "${serve_cmd}" "${client_cmd}"
