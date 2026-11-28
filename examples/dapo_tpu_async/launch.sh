#!/usr/bin/env bash
# Collocated GRPO async-overlap reproduction launcher (TPU v7x, 64 chips).
# The portable part is the env/flag block below; adapt the model path + your
# own job launcher (GKE JobSet, xpk, etc.) around it.
set -euo pipefail

# (1) Skip the redundant prompt-logprobs precompile — see #3455 / PR #3457.
export SKIP_PROMPT_LOGPROBS_PRECOMPILE=1

# (2) Async collectives off the synchronous critical path + SparseCore offload.
#     The first two also clear a 0x176 ICI decode stall on collocated SPMD.
export LIBTPU_INIT_ARGS="\
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_enable_async_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"

# (3) Generous engine timeouts so the (longer) async compile is not killed by a
#     watchdog on a cold jax-cache. Not needed once the cache is warm.
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
export VLLM_RPC_TIMEOUT=1200000

# (4) async_scheduling: true lives in the config (config_template.yaml).
#     On a current vLLM LKG this is honored on the uni path (issue #3456).

echo "Async-overlap reproduction env set. Launch your DAPO/GRPO job with"
echo "config_template.yaml (async_scheduling: true) on a 64-chip v7x slice."
