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

#
# GLM-5.1-FP8 on v4-64 with TP=4, EP=8.  Thin wrapper around
# launch_ray_nodocker.sh — exports the env defaults this config needs,
# then delegates to the generic multi-host launcher.
#
# Required env (no defaults are committed to the repo):
#   MODEL_PATH     local path to the GLM-5.1-FP8 checkpoint, reachable
#                  on every host (e.g. a gcsfuse mount)
#   VENV           Python venv path containing vllm + tpu-inference
#   COORDINATOR_IP head-host IP reachable from every worker
#   WORKERS        space-separated IP list of the 7 non-coordinator hosts
#
# Typical usage:
#   MODEL_PATH=/mnt/gcs/GLM-5.1-FP8 VENV=/opt/vllm-env \
#   COORDINATOR_IP=10.0.0.1 WORKERS="10.0.0.2 10.0.0.3 ... 10.0.0.8" \
#     bash scripts/multihost/tpu-v4/launch_glm51_tp4_ep8.sh
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

: "${MODEL_PATH:?set MODEL_PATH to the local path of the GLM-5.1-FP8 checkpoint before running (e.g. a gcsfuse mount)}"
export MODEL_PATH
export TP_SIZE="${TP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-8}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# GLM-5.1 on v4 MUST keep DP=OFF (memory feedback_dp_attention_hbm.md);
# compensated by flash_attn_mla allgather fix on 'expert' axis.
export ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

# ── Precision (default per-axis requant; block-scale path has kernel issues
#    on small K_blocks layers — shared expert down_proj K_blocks=16 < TP×EP=32
#    makes the pallas kernel's accumulator list empty). Revisit after BROKEN is fixed.
# export REQUANTIZE_BLOCK_SIZE=128
# export ENABLE_QUANTIZED_MATMUL_KERNEL=1

# ── MoE direct FP8 path (skip dequant/requant roundtrip, ~30x faster PWAL) ────
export VLLM_MOE_SKIP_REQUANTIZATION="${VLLM_MOE_SKIP_REQUANTIZATION:-1}"

# ── gcsfuse path (kernel VFS page cache; GCS direct cold capped ~200 MB/s) ────
export TPU_GCS_WEIGHT_LOAD="${TPU_GCS_WEIGHT_LOAD:-0}"

# ── Streaming loader (REQUIRED for EP=8 on 744GB FP8 model) ───────────────────
# DefaultLoader mmaps every shard per-host; EP needs all 142 shards (≠ PP),
# so prefetched pages accumulate to ~380GB RSS → Ray OOM-kills at 0.95×400GB.
# tpu_streaming_loader's IncrementalModelLoader runs _init_tpu_ep_weight_filter
# (keyed on sharding_config) so each host only materializes its expert slice,
# freeing CPU pages as soon as a layer is pushed to TPU.
export TPU_TP_SELECTIVE_LOAD="${TPU_TP_SELECTIVE_LOAD:-1}"

exec bash "$SCRIPT_DIR/launch_ray_nodocker.sh" "$@"
