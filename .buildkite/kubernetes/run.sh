#!/usr/bin/env bash
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

# Hands a step's command to the launcher as a TPU workload.
#
#   .buildkite/kubernetes/run.sh <machine-type>/<topology> <command> [args...]
#
# The launcher owns the pod and its caches, so all this does is resolve the
# image and decide what environment crosses into it.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <machine-type>/<topology> <command> [args...]" >&2
  exit 2
fi

# The launcher's own machine-type and topology names, so a shape the fleet does
# not run is refused before anything is submitted.
shape="$1"
shift
machine_type="${shape%%/*}"
topology="${shape#*/}"
if [[ "$machine_type" == "$shape" || -z "$topology" ]]; then
  echo "$0: expected <machine-type>/<topology>, got '$shape'" >&2
  exit 2
fi

# The build step publishes the exact tag it pushed as `ci-image`.
WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-$(buildkite-agent meta-data get ci-image 2>/dev/null || true)}"
export WORKLOAD_IMAGE

# Names a step sets that no convention would find. A name the step has not set
# is skipped rather than injected empty - tpu_inference rejects
# MODEL_IMPL_TYPE="" where it wants the variable absent.
FORWARD=(
  # HF_TOKEN and BUILDKITE_ANALYTICS_TOKEN are unset here on purpose: naming an
  # unset variable is how the launcher is asked to read it from Secret Manager
  # and put it in the pod, so no value appears in this repo. The Test Engine
  # collector runs inside the workload, so without its token a suite passes and
  # reports nothing.
  HF_TOKEN GITHUB_CI_BOT_TOKEN BUILDKITE_ANALYTICS_TOKEN
  # Model and backend selection
  TPU_VERSION MODEL_IMPL_TYPE TPU_BACKEND_TYPE NEW_MODEL_DESIGN
  QUANTIZATION USE_PREBUILT_IMAGE SKIP_ACCURACY_TESTS BVT_ONLY
  # Not queue selectors here: mlperf.sh reads them to pick the model list and
  # the parallelism.
  USE_V6E8_QUEUE USE_V7X8_QUEUE
  NUM_PRECOMPILE_WORKERS VLLM_LOG_LEVEL VLLM_XLA_CHECK_RECOMPILATION
  # --env wins over the manifest, so a step that sets these replaces the
  # default set below.
  JAX_COMPILATION_CACHE_DIR VLLM_XLA_CACHE_PATH
  # Test parameters
  TEST_MODEL TEST_MODE TEST_LORA_TP TENSOR_PARALLEL_SIZE
  MINIMUM_ACCURACY_THRESHOLD MINIMUM_THROUGHPUT_THRESHOLD
  MODEL INPUT_LEN OUTPUT_LEN PREFIX_LEN MAX_MODEL_LEN
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS NUM_PROMPTS RANDOM_SEED
  MAX_CONCURRENCY REQUEST_RATE TIMEOUT_SECONDS COMPILATION_CONFIG
  USE_CHAT_TEMPLATE BENCH_DATASET USE_BATCHED_RPA_KERNEL
  GPU_MEMORY_UTILIZATION GCS_BUCKET HOST_NAME
)

# Every BUILDKITE_* the agent set is swept through, except four:
#
#   BUILDKITE_COMMAND is this step's own script; forwarding it invites the
#   workload to re-execute the launcher that started it.
#   BUILDKITE_PLUGINS can be large enough to bloat the pod spec.
#   BUILDKITE_AGENT_JOB_API_SOCKET and _TOKEN address a unix socket on the
#   agent host, which a pod in another cluster can never reach; omitted, it
#   fails as "socket empty or undefined" rather than a connection error.
FORWARD_DENY="BUILDKITE_COMMAND BUILDKITE_PLUGINS BUILDKITE_AGENT_JOB_API_SOCKET BUILDKITE_AGENT_JOB_API_TOKEN"

env_args=()
seen=" "
for name in $(compgen -v | grep '^BUILDKITE_' | sort); do
  case " $FORWARD_DENY " in *" $name "*) continue ;; esac
  env_args+=(--env "$name")
  seen="$seen$name "
done
for name in "${FORWARD[@]}"; do
  case "$seen" in *" $name "*) continue ;; esac
  env_args+=(--env "$name")
  seen="$seen$name "
done

# Beyond failing a test that recompiles at runtime, this flag is what makes
# CompilationManager lower jax_persistent_cache_min_compile_time_secs and
# min_entry_size_bytes to -1; without it anything quick to compile is never
# written to the cache.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

# A literal, so bumping JAX means changing it here too: otherwise every step
# points at a namespace nobody writes any more and compiles cold, with no error.
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-jax0.11.0_tputpu6e}"
export JAX_COMPILATION_CACHE_DIR="/cache/jax/${CACHE_NAMESPACE}"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"

# The launcher's built-in Job is one pod holding every chip on one host.
# Anything more - roles that must find each other, a slice across hosts - needs
# a JobSet passed with --manifest.
exec /opt/launcher/launch \
  --machine-type "$machine_type" \
  --topology "$topology" \
  "${env_args[@]}" \
  -- "$@"
