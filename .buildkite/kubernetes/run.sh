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

# Hands a step's command to the launcher as a TPU workload. The launcher owns
# the pod and its caches, so all this does is resolve the image and decide what
# environment crosses into it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: SHAPE=<machine-type>/<topology> $0 <command> [args...]" >&2
  exit 2
fi

# From the step's environment, not an argument: every step already sets SHAPE
# from one of the shape anchors, and passed the same value straight back in.
shape="${SHAPE:-}"
machine_type="${shape%%/*}"
topology="${shape#*/}"
if [[ -z "$shape" || "$machine_type" == "$shape" || -z "$topology" ]]; then
  echo "$0: SHAPE must be <machine-type>/<topology>, got '${shape}'" >&2
  exit 2
fi

# TPU_VERSION only labels and gates steps; the hardware comes from the shape.
# Setting one without the other runs v7x-gated tests on v6e chips and reports
# them as v7x, so refuse rather than produce a mislabelled result.
case "${TPU_VERSION:-tpu6e}:${machine_type}" in
  tpu7x:tpu7x-*|tpu6e:ct6e-*) ;;
  *)
    echo "$0: TPU_VERSION=${TPU_VERSION:-tpu6e} does not match shape ${shape}." >&2
    echo "  A v7x run needs TPU_VERSION=tpu7x with KUBE_SHAPE_SINGLE and" >&2
    echo "  KUBE_SHAPE_MULTI set to tpu7x shapes." >&2
    exit 2
    ;;
esac

WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-$(buildkite-agent meta-data get ci-image 2>/dev/null || true)}"
export WORKLOAD_IMAGE

# A name the step has not set is skipped rather than injected empty -
# tpu_inference rejects MODEL_IMPL_TYPE="" where it wants the variable absent.
FORWARD=(
  # Left unset on purpose: naming an unset variable is how the launcher is asked
  # to read it from Secret Manager, so no value appears in this repo. Without
  # BUILDKITE_ANALYTICS_TOKEN a suite passes and reports nothing to Test Engine.
  # HF_TOKEN is not here: every pod inherits it from the fleet pod defaults.
  BUILDKITE_ANALYTICS_TOKEN
  TPU_VERSION MODEL_IMPL_TYPE TPU_BACKEND_TYPE NEW_MODEL_DESIGN
  QUANTIZATION USE_PREBUILT_IMAGE SKIP_ACCURACY_TESTS BVT_ONLY
  # Not queue selectors: mlperf.sh reads them to pick the model list and the
  # parallelism.
  USE_V6E8_QUEUE USE_V7X8_QUEUE
  NUM_PRECOMPILE_WORKERS VLLM_LOG_LEVEL VLLM_XLA_CHECK_RECOMPILATION
  # --env wins over the manifest, so a step that sets these replaces the
  # default set below.
  JAX_COMPILATION_CACHE_DIR VLLM_XLA_CACHE_PATH
  TEST_MODEL TEST_LORA_TP TENSOR_PARALLEL_SIZE TPU_CORES
  # Without these the MoE weights land in bfloat16 instead of the requantized
  # dtype, which is 300GiB more HBM on DeepSeek-R1 - over the cap on a 4-chip
  # slice, and merely wrong on anything that still fits.
  VLLM_MLA_DISABLE MOE_REQUANTIZE_BLOCK_SIZE MOE_REQUANTIZE_WEIGHT_DTYPE
  MINIMUM_ACCURACY_THRESHOLD MINIMUM_THROUGHPUT_THRESHOLD
  MODEL INPUT_LEN OUTPUT_LEN PREFIX_LEN MAX_MODEL_LEN
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS NUM_PROMPTS RANDOM_SEED
  MAX_CONCURRENCY REQUEST_RATE TIMEOUT_SECONDS COMPILATION_CONFIG
  USE_CHAT_TEMPLATE BENCH_DATASET USE_BATCHED_RPA_KERNEL
  GPU_MEMORY_UTILIZATION GCS_BUCKET HOST_NAME
)

# The BUILDKITE_* the agent set are swept by the launcher itself, so only the
# ones above that it cannot see - an unset name it resolves from Secret
# Manager - still have to be named here.
env_args=()
for name in "${FORWARD[@]}"; do
  env_args+=(--env "$name")
done

# Besides failing a test that recompiles at runtime, this flag is what makes
# CompilationManager lower jax_persistent_cache_min_compile_time_secs and
# min_entry_size_bytes to -1; without it anything quick to compile is never
# cached.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

# Chip generation, then the JAX version that produced the entries: an entry is
# useless to the other generation, and `ls` on the bucket then answers which
# generation is warm. The version comes from the pin the image installs, so a
# JAX bump moves the cache with it. Empty is fatal rather than silently naming
# a namespace nobody has written.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)
jax_version=$(sed -n 's/^jax==\([^ #]*\).*/\1/p' "${repo_root}/requirements.txt" | head -1)
if [[ -z "$jax_version" ]]; then
  echo "$0: no 'jax==' pin in ${repo_root}/requirements.txt" >&2
  exit 2
fi
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-${TPU_VERSION:-tpu6e}/jax-${jax_version}}"
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
