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

# Start the Qwen3.5-397B-A17B-FP8 vLLM-on-TPU server (sharding via SHARDING).
# Usage:
#   ISL=8192 OSL=1024 bash server.sh
#   SHARDING=DP4TP2_EP ISL=8192 OSL=1024 bash server.sh
set -euo pipefail

PORT="${PORT:-8000}"
SHARDING="${SHARDING:-DP8_EP}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
# Headroom over ISL+OSL (matches InferenceX qwen3.5 sglang: ISL+OSL+20). Without it, fixed-length
# (ratio 1.0) requests hit max_model_len exactly and get 400 Bad Request.
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-20}"

case "$SHARDING" in
  DP8_EP)
    DP_SIZE=8
    SHARDING_ARGS=(
      --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'
      --enable-expert-parallel
    ) ;;
  DP4TP2_EP)
    DP_SIZE=4
    SHARDING_ARGS=(
      --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 4}}}'
      --enable-expert-parallel
    ) ;;
  *) echo "ERROR: unknown SHARDING='$SHARDING' (expected DP8_EP|DP4TP2_EP)" >&2; exit 1 ;;
esac

MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))
MAX_NUM_BATCHED_TOKENS=$((ISL / DP_SIZE))

set -x
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d
export NEW_MODEL_DESIGN=1
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false'

args=(
  Qwen/Qwen3.5-397B-A17B-FP8
  --max-model-len="$MAX_MODEL_LEN"
  --max-num-batched-tokens="$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs=64
  --no-enable-prefix-caching
  --gpu-memory-utilization=0.9
  --tensor-parallel-size=8
  --async-scheduling
  --port="$PORT"
  --language-model-only
  --enable-auto-tool-choice
  --tool-call-parser=qwen3_coder
  --reasoning-parser=qwen3
  --limit-mm-per-prompt='{"image": 0, "video": 0}'
  --kv-cache-dtype=fp8
  --block-size=256
  "${SHARDING_ARGS[@]}"
)
vllm serve "${args[@]}"
set +x
