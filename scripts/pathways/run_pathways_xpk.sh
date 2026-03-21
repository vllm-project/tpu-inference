#!/bin/bash
# Entrypoint script for the Pathways xpk workload.
# The COMMAND in deploy_xpk.sh downloads the tpu-inference tar from GCS,
# extracts it to /tmp/tpu-inference, and then runs this script.
set -euo pipefail

echo "XPK Start: $(date)"

# ── Patch local tpu-inference changes ──
echo "Installing tpu-inference patch..."
if [ -d /workspace/tpu_inference ]; then
  cp -a /tmp/tpu-inference/. /workspace/tpu_inference/
fi
pip install --no-deps /tmp/tpu-inference
echo "tpu-inference patch installed."

# ── Environment variables ──
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_TOKEN=${HF_TOKEN:-}
export PHASED_PROFILING_DIR=gs://wenxindong-multipod-dev/profiling/vllm_qwen3_235b_a22b_instruct_2507/dp32/mar20
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1

# ── Launch vllm serve via Pathways ──
JAX_PLATFORMS=proxy,cpu \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
vllm serve gs://wenxindong-multipod-dev/models--Qwen--Qwen3-235B-A22B-Instruct-2507/ \
  --tensor-parallel-size 8 \
  --data-parallel-size 32 \
  --api-server-count 1 \
  --max-model-len 5120 \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --max-num-seqs 256 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --additional_config='{"sharding":{"sharding_strategy": {"enable_dp_attention":1}}}' \
  --load-format runai_streamer

echo "XPK End: $(date)"
