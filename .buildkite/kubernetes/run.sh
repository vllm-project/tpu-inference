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

# The Kubernetes counterpart of scripts/run_in_docker.sh.
#
#   .buildkite/kubernetes/run.sh <machine-type>/<topology> <command> [args...]
#
# run_in_docker.sh picks an image, mounts the caches and forwards a long -e list
# into a container on a long-lived VM. Here the pod is the container, the
# launcher owns the caches, and the shape names the hardware - so what is left
# is resolving the image and deciding what environment crosses into the pod.
# That is the same for every step, which is why this is a script rather than a
# copy of it per step.
#
# It lives here rather than in the launcher because everything in it is this
# repo's: `ci-image` is this repo's metadata key, MODEL_IMPL_TYPE is this repo's
# variable, and the launcher should not have to know either to schedule a pod.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <machine-type>/<topology> <command> [args...]" >&2
  exit 2
fi

# One token rather than two flags, because the pair is what names a shape and
# splitting them across the step invites a step that changes one and not the
# other. The launcher's own names, so there is no table here to fall behind the
# fleet's - a shape it does not run is refused before anything is submitted.
shape="$1"
shift
machine_type="${shape%%/*}"
topology="${shape#*/}"
if [[ "$machine_type" == "$shape" || -z "$topology" ]]; then
  echo "$0: expected <machine-type>/<topology>, got '$shape'" >&2
  exit 2
fi

# The build step publishes the exact tag it pushed. Reading it back beats
# recomputing it from commit hashes in every step, which is how they drift.
WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-$(buildkite-agent meta-data get ci-image 2>/dev/null || true)}"
export WORKLOAD_IMAGE

# Names a step sets that no convention would find. The Hugging Face and Test
# Engine tokens are here too, unset: the launcher holds a grant on both and
# supplies them for a step that asks by name, so the value never appears in a
# pipeline or in this repo.
#
# Named rather than forwarded wholesale so the pod's environment is a decision
# rather than an accident. A name the step has not set is skipped rather than
# injected empty, which ${...} substitution cannot express and which
# tpu_inference depends on: it rejects MODEL_IMPL_TYPE="" where it wants the
# variable absent.
FORWARD=(
  # Secrets and identity. HF_TOKEN and BUILDKITE_ANALYTICS_TOKEN are unset on a
  # kube agent, and naming an unset variable is how the launcher is asked for
  # it: both are in its env_secrets registry, so it reads them from Secret
  # Manager and puts them in the pod. On bare metal an agent env hook exports
  # them instead, which is why run_in_docker.sh can pass them by value.
  #
  # BUILDKITE_ANALYTICS_TOKEN has to reach the pod rather than the agent
  # because the Test Engine collector runs inside the workload. Without it a
  # suite still passes and reports nothing.
  HF_TOKEN GITHUB_CI_BOT_TOKEN BUILDKITE_ANALYTICS_TOKEN
  # Model and backend selection
  TPU_VERSION MODEL_IMPL_TYPE TPU_BACKEND_TYPE NEW_MODEL_DESIGN
  QUANTIZATION USE_PREBUILT_IMAGE SKIP_ACCURACY_TESTS BVT_ONLY
  NUM_PRECOMPILE_WORKERS VLLM_LOG_LEVEL VLLM_XLA_CHECK_RECOMPILATION
  # Where the compilation cache lives. The Job mounts it; a step that sets
  # these replaces the default, since --env wins over the manifest.
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

# Every BUILDKITE_* the agent set, enumerated rather than listed. There are
# around a hundred, most of them build metadata a test may read, and a
# hand-maintained list of them drifts.
#
# What the sweep carries that matters:
#
#   BUILDKITE_AGENT_ACCESS_TOKEN and _ENDPOINT let the workload call
#   buildkite-agent itself - artifact upload, meta-data, annotate, and the OIDC
#   token bktec authenticates with. It is the only thing that can see its own
#   output files: unlike bare metal, where run_in_docker.sh bind-mounts results
#   into the agent's checkout for `artifact_paths` to glob, a pod shares no
#   filesystem with the agent. Not a widening of trust - the step's own
#   commands already run with this token, from the same repo as the test.
#
#   BUILDKITE_PARALLEL_JOB and _COUNT are how bktec knows which shard it is.
#   They are unset on a step without `parallelism:`, and unset names are
#   skipped.
#
# Four are held back:
#
#   BUILDKITE_COMMAND is this step's own script. Forwarding it into the
#   container invites something to re-execute the launcher from inside the
#   workload it launched.
#
#   BUILDKITE_PLUGINS is a JSON blob describing agent-side plugins, which mean
#   nothing in the pod and can be large enough to bloat the pod spec.
#
#   BUILDKITE_AGENT_JOB_API_SOCKET and _TOKEN address a unix socket the agent
#   opens for its own job. The docker plugin can mount that socket into a
#   container because the container shares the host; a workload pod is on
#   another cluster entirely, so the path can never resolve. Forwarding it is
#   worse than omitting it: the CLI fails with "socket empty or undefined",
#   which names the problem, where a path present in the variable but not on
#   disk fails as a connection error.
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

# Persist every compiled module, as bare metal does.
#
# run_in_docker.sh passes -e VLLM_XLA_CHECK_RECOMPILATION=1 on every step. The
# flag's stated job is to fail a test that recompiles at runtime, but it has a
# second effect: CompilationManager only lowers
# jax_persistent_cache_min_compile_time_secs and min_entry_size_bytes to -1
# when it is set. Without it JAX keeps its defaults, and anything small or
# quick to compile is recompiled every run and never written - so the cache
# fills to a fraction of bare metal's and repeating a suite never closes the
# gap.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

# The cache namespace, and the two paths under it. Bare metal builds the same
# string from the versions it has to hand - jax${JAX_VERSION}_tpu${TPU_VERSION}
# - so it follows a JAX bump automatically. This is a literal, which means a
# bump points at a namespace nobody writes any more: a cold cache, no error,
# and two to three times the runtime. Bumping JAX means changing it here too.
#
# Overridable per build, so an experiment can point at an empty namespace and
# measure what a cache built only by Kubernetes costs.
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-jax0.11.0_tputpu6e}"
export JAX_COMPILATION_CACHE_DIR="/cache/jax/${CACHE_NAMESPACE}"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"

# No --manifest: the launcher's built-in Job is one pod holding every chip on
# one host, which is what all of these are. A workload that is more than that -
# roles that have to find each other, a slice across hosts - is a JobSet this
# repo writes and passes with --manifest instead.
exec /opt/launcher/launch \
  --machine-type "$machine_type" \
  --topology "$topology" \
  "${env_args[@]}" \
  -- "$@"
