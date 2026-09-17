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

# One engine of the 1P1D disaggregated benchmark.
#
#   disagg_engine.sh prefill|decode
#
# The two engines agree on every environment variable and all but two serve
# flags, so they share this file and differ only in the case block below.
# The manifest keeps POD_IP, which needs the downward API, and the names the
# TPU webhook has to see already set on the pod spec.
set -euo pipefail

role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 prefill|decode" >&2; exit 2 ;;
esac

# Worker core dumps are several GiB each and overflow the ephemeral-storage
# limit.
ulimit -c 0

# What the KV connector advertises to its peer; pinned so a second interface
# cannot change which IP.
export VLLM_HOST_IP="${POD_IP}"

export JAX_PLATFORMS=tpu,cpu
export MODEL_IMPL_TYPE=vllm
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export USE_BATCHED_RPA_KERNEL=0
export USE_MOE_EP_KERNEL=0
export VLLM_LOGGING_LEVEL=DEBUG

# On the mounted bucket: the default under $HOME does not fit the node's
# boot disk.
export VLLM_CACHE_ROOT=/cache/jax/vllm

# The namespace run.sh uses, so these engines reuse what the single-host steps
# compiled. Spelled from the same two sources, since the lanes only share a
# cache if the path matches exactly. Required, not defaulted: a wrong guess
# compiles cold against a namespace nobody writes, and nothing fails.
: "${TPU_VERSION:?the step must forward TPU_VERSION for the cache namespace}"
jax_version=$(python3 -c 'import jax; print(jax.__version__)')
: "${jax_version:?could not read the installed jax version}"
export JAX_COMPILATION_CACHE_DIR="/cache/jax/${TPU_VERSION}/jax-${jax_version}"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"

# Prefill hands its KV off and needs headroom to do it; decode holds the
# cache it receives. The connector roles are the two halves of one channel.
case "$role" in
  prefill) memory_fraction=0.70; kv_role=kv_producer ;;
  decode)  memory_fraction=0.90; kv_role=kv_consumer ;;
esac

exec vllm serve \
  --seed=42 \
  --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --max-model-len=10240 \
  --max-num-batched-tokens=8192 \
  --max-num-seqs=512 \
  --no-enable-prefix-caching \
  --tensor-parallel-size=8 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization="$memory_fraction" \
  --async-scheduling \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"'"$kv_role"'"}'
