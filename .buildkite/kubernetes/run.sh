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
#   .buildkite/kubernetes/run.sh <profile> <command> [args...]
#
# run_in_docker.sh picks the image, mounts the caches and forwards a long -e
# list to a container on a long-lived VM. Here the pod is the container and the
# profile decides the hardware, so what is left is resolving the image and the
# token and naming what to forward - which is the same for every step, and the
# reason this is a script rather than 23 copies of it in the pipeline.
#
# All of it lives here rather than in the launcher: `ci-image` is this repo's
# metadata key and MODEL_IMPL_TYPE is this repo's variable. The launcher should
# not have to know either to schedule a pod.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <profile> <command> [args...]" >&2
  exit 2
fi

profile="$1"
shift

# The build step publishes the exact tag it pushed. Reading it back beats
# recomputing it from commit hashes in every step, which is how they drift.
WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-$(buildkite-agent meta-data get ci-image 2>/dev/null || true)}"
export WORKLOAD_IMAGE

# From the Buildkite cluster secret, not a Kubernetes Secret. Fetching it
# through the agent is also what adds it to the log redactor.
HF_TOKEN="${HF_TOKEN:-$(buildkite-agent secret get "${HF_SECRET_KEY:-HF_TOKEN}")}"
export HF_TOKEN

# Buildkite Test Engine, the same way. On bare metal this arrives from the
# agent environment on the VM; the agent-stack pods have no equivalent, so
# every run so far has logged "No BUILDKITE_ANALYTICS_TOKEN environment
# variable present" and Test Engine has received nothing. The name is already
# in FORWARD below - what was missing is the value.
#
# Absent is not fatal: the collector warns and the tests still run. So a
# missing secret must not fail the step, and an empty value must not be
# exported - FORWARD skips names that are unset, but an exported empty string
# is set, and would be injected as empty rather than skipped.
if [ -z "${BUILDKITE_ANALYTICS_TOKEN:-}" ]; then
  # GCP Secret Manager rather than the agent's own secret store, because that
  # is where this token lives. The launcher image ships gcloud and the pod's
  # service account is the identity; the grant is on the single secret rather
  # than the project.
  #
  # Trailing newline stripped: `gcloud secrets versions access` emits the
  # payload verbatim, and a token with a newline on the end is rejected in a
  # way that looks like a bad token rather than a formatting problem.
  _analytics_token="$(gcloud secrets versions access latest \
    --secret="${ANALYTICS_SECRET_ID:-tpu_commons_buildkite_analytics_token}" \
    --project="${ANALYTICS_SECRET_PROJECT:-cloud-tpu-inference-test}" \
    2>/dev/null | tr -d '\r\n' || true)"
  if [ -n "$_analytics_token" ]; then
    export BUILDKITE_ANALYTICS_TOKEN="$_analytics_token"
  else
    echo "run.sh: could not read the Test Engine token from Secret Manager" \
         "(${ANALYTICS_SECRET_PROJECT:-cloud-tpu-inference-test}/" \
         "${ANALYTICS_SECRET_ID:-tpu_commons_buildkite_analytics_token});" \
         "Test Engine will receive no results from this step" >&2
  fi
  unset _analytics_token
fi

# Mirrors the -e list in run_in_docker.sh, so a test sees the same environment
# on either platform.
#
# The BUILDKITE_* names are no longer listed here - they are enumerated from
# the environment below, because this list drifted and lost
# BUILDKITE_ANALYTICS_TOKEN once already. What stays here is everything else:
# names a step sets that no convention would find.
#
# Named rather than forwarded wholesale so the pod's environment is a decision
# rather than an accident. A name the step has not set is skipped rather than
# injected empty - ${...} substitution cannot express that, and tpu_inference
# rejects MODEL_IMPL_TYPE="" where it wants the variable absent.
FORWARD=(
  # Secrets and identity
  HF_TOKEN GITHUB_CI_BOT_TOKEN
  # Model and backend selection
  TPU_VERSION MODEL_IMPL_TYPE TPU_BACKEND_TYPE NEW_MODEL_DESIGN
  QUANTIZATION USE_PREBUILT_IMAGE SKIP_ACCURACY_TESTS BVT_ONLY
  NUM_PRECOMPILE_WORKERS VLLM_LOG_LEVEL VLLM_XLA_CHECK_RECOMPILATION
  # --env wins over the manifest, so a step that sets these replaces the
  # defaults derived below; one that does not is unaffected.
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

# What the BUILDKITE_* sweep above carries, recorded because it is no longer
# visible as a list:
#
#   BUILDKITE_AGENT_ACCESS_TOKEN and _ENDPOINT let the workload call
#   buildkite-agent itself - artifact upload, meta-data, annotate, and the OIDC
#   token bktec authenticates with. It is the only thing that can see its own
#   output files: unlike bare metal, where run_in_docker.sh bind-mounts results
#   into the agent's checkout for `artifact_paths` to glob, a pod shares no
#   filesystem with the agent. Not a widening of trust - the step's own
#   commands already run with this token, from the same repo as the test.
#
#   BUILDKITE_ANALYTICS_TOKEN and BUILDKITE_TEST_ENGINE_* are what Test Engine
#   reports through. BUILDKITE_PARALLEL_JOB and _COUNT are how bktec knows
#   which shard it is; they are unset on a step without `parallelism:`, and
#   unset names are skipped.


# Every BUILDKITE_* the agent sets, plus the named list above.
#
# The list used to carry the BUILDKITE_ names too, and kept losing them:
# BUILDKITE_ANALYTICS_TOKEN went missing once and Test Engine silently received
# nothing for months, and the BUILDKITE_TEST_ENGINE_* names were only there
# because someone added them in advance. Enumerating the environment cannot
# drift; a hand-maintained list of a hundred names will.
#
# Still only names that are actually set, so the "absent rather than empty"
# property the list was written for is unchanged.
#
# Two are held back deliberately:
#
#   BUILDKITE_COMMAND is this step's own script. Forwarding it into the
#   container invites something to re-execute the launcher from inside the
#   workload it launched.
#   BUILDKITE_PLUGINS is a JSON blob describing agent-side plugins, which mean
#   nothing in the pod and can be large enough to bloat the pod spec.
#   BUILDKITE_AGENT_JOB_API_SOCKET and _TOKEN address a unix socket the agent
#   opens for its own job. The docker plugin can mount that socket into a
#   container because the container shares the host; our workload pod is on
#   another cluster entirely, so the path can never resolve. Forwarding it is
#   worse than omitting it: the CLI currently fails with "socket empty or
#   undefined", which names the problem, where a path that exists in the
#   variable but not on disk fails as a connection error.
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

# Persist every compiled module, as bare metal does. run_in_docker.sh passes
# -e VLLM_XLA_CHECK_RECOMPILATION=1 on every step; besides failing a test that
# recompiles at runtime, it is what makes CompilationManager lower JAX's
# persistent-cache thresholds to -1, so small and quick compilations are
# written too. Without it a cache this side builds plateaus at about twice
# the seeded time, however many suites run.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

# The cache namespace, and the two paths derived from it. Bare metal builds the
# same string from the versions it has to hand - jax${JAX_VERSION}_tpu${TPU_VERSION}
# - so it follows a JAX bump automatically. This is a literal, which means a
# bump silently points us at a namespace nobody writes any more: a cold cache,
# no error, 2-3x slower. Deriving it from the image is the real fix.
#
# Overridable per build, which is how a cold cache is measured without
# disturbing the real one.
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-jax0.11.0_tputpu6e}"
export JAX_COMPILATION_CACHE_DIR="/cache/jax/${CACHE_NAMESPACE}"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"

# The workload manifest: one pod on one host with the two cache claims the
# cluster publishes mounted at /cache/jax and the HF hub path. WORKLOAD_MANIFEST
# names a different object for a shape that needs one - a slice across hosts
# is a JobSet, not a Job - while keeping everything else here (image, tokens,
# forwarded env) identical.
manifest="${WORKLOAD_MANIFEST:-.buildkite/kubernetes/manifests/test.yaml}"

exec /opt/launcher/launch \
  --profile "$profile" \
  "${env_args[@]}" \
  --manifest "$manifest" \
  -- "$@"
