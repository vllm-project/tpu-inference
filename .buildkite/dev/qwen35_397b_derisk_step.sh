#!/bin/bash
# Agent-side wrapper for the Qwen3.5-397B GDN de-risk (see
# qwen35_397b_derisk_v7x8.yml). Lives in a committed .sh rather than inline YAML
# so it uses normal single-$ shell syntax -- `buildkite-agent pipeline upload`
# interpolates YAML before the agent sees it, and every $ in an inline command
# would have to be written $$.
set -euo pipefail

ARM="${DERISK_ARM:-attndp2}"
ATTN_DP="${DERISK_ATTN_DP:-2}"

ART="$PWD/dev_artifacts"
mkdir -p "$ART"

echo "=== de-risk arm=${ARM} attn_dp=${ATTN_DP} ==="

# run_in_docker.sh forwards a fixed allowlist into the container
# (run_in_docker.sh:46-68). USE_BATCHED_RPA_KERNEL happens to be on it;
# USE_BATCHED_RPA_SEQ_ON_LANE, ATTN_DP_SIZE, BLOCK_SIZE and the bench knobs are
# not, so inject them here. mapfile splits on newline, so every flag and every
# value needs its own line.
#
# VLLM_XLA_CHECK_RECOMPILATION is hardcoded to 1 at run_in_docker.sh:189, but
# BENCHMARK_DOCKER_ARGS is appended after it and docker takes the last -e, so
# this overrides it. With SKIP_JAX_PRECOMPILE=1 the graph compiles lazily on the
# first request and the recompilation check would abort on an expected recompile.
export BENCHMARK_DOCKER_ARGS_STR="$(printf '%s\n' \
  -v "${ART}:/workspace/artifacts" \
  -e "ART_DIR=/workspace/artifacts" \
  -e "ARM_TAG=${ARM}" \
  -e "ATTN_DP_SIZE=${ATTN_DP}" \
  -e "USE_BATCHED_RPA_SEQ_ON_LANE=1" \
  -e "USE_MOE_EP_KERNEL=0" \
  -e "SKIP_JAX_PRECOMPILE=1" \
  -e "VLLM_XLA_CHECK_RECOMPILATION=0" \
  -e "BLOCK_SIZE=128" \
  -e "KV_CACHE_DTYPE=fp8" \
  -e "LOAD_WAIT_SECONDS=3600" \
  -e "NUM_PROMPTS=64" \
  -e "MAX_CONCURRENCY=16")"

.buildkite/scripts/run_in_docker.sh \
  bash /workspace/tpu_inference/.buildkite/dev/qwen35_397b_derisk.sh

# The container writes as root; make the tree readable by the agent user so the
# artifact upload can pick it up.
sudo -n chown -R "$(id -u):$(id -g)" "$ART" 2>/dev/null \
  || sudo -n chmod -R a+rX "$ART" || true
