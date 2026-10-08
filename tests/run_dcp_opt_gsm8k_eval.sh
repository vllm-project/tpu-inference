#!/bin/bash
set -euo pipefail

echo "===================================================================="
echo " DCP Optimization with rpa_v3_cp (Write-Only Kernel + 1-Phase Decode)"
echo " Evaluation on Qwen3.5-35B-A3B-FP8 (EP=8, DCP=4) on TPU v7x-8"
echo "===================================================================="
date -u
python3 -c "import jax; print('JAX devices:', jax.devices())"

export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export DCP_DECODE_ONLY_OPT=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export SLICE_ROPE_CACHE=1
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'

export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
mkdir -p "${HF_HUB_CACHE}"

cd /home/wenxindong_google_com/tpu-inference

echo ""
echo "===================================================================="
echo " Step 1: Unit Test write_decode_kv Pallas Kernel"
echo "===================================================================="
python3 tests/kernels/rpa_v3_cp/test_write_kv.py || echo "Warning: test_write_kv exited with code $?"

echo ""
echo "===================================================================="
echo " Step 2: Start vLLM Server (EP=8, DCP=4, rpa_v3_cp + DCP decode opt)"
echo "===================================================================="

MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
LOCAL_MODEL_DIR="/tmp/models/Qwen3.5-35B-A3B-FP8"

SNAPSHOT_DIR=""
for cand in \
    /mnt/models/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/* \
    /mnt/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/* \
    /mnt/models/Qwen/Qwen3.5-35B-A3B-FP8 \
    /mnt/models/Qwen3.5-35B-A3B-FP8 \
    /tmp/models/Qwen3.5-35B-A3B-FP8; do
    if [ -d "$cand" ] && [ -f "$cand/config.json" ]; then
        SNAPSHOT_DIR="$cand"
        break
    fi
done

if [ -n "${SNAPSHOT_DIR}" ]; then
    MODEL_PATH="${SNAPSHOT_DIR}"
    echo "Found cached snapshot: ${MODEL_PATH}"
else
    MODEL_PATH="${MODEL_NAME}"
fi
echo "Using model path: ${MODEL_PATH}"

export VLLM_PORT=8000
vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size=8 \
  --decode-context-parallel-size=4 \
  --enable-expert-parallel \
  --kv-cache-dtype=fp8 \
  --max-model-len=8192 \
  --max-num-seqs=32 \
  --max-num-batched-tokens=512 \
  --gpu-memory-utilization=0.85 \
  --dtype=auto \
  --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  '--limit-mm-per-prompt={"image": 0, "video": 0}' \
  --block-size=128 --enable-chunked-prefill \
  --mamba-cache-mode align \
  --prefix-cache-retention-interval 0 > /tmp/vllm_dcp_opt_serve.log 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

echo "Waiting for vLLM server to be ready on port 8000..."
READY=0
for i in $(seq 1 360); do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is UP and ready after $((i * 5)) seconds!"
        READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server died unexpectedly! Check log:"
        tail -n 100 /tmp/vllm_dcp_opt_serve.log
        exit 1
    fi
    sleep 5
done

if [ "${READY}" -eq 1 ]; then
    echo ""
    echo "===================================================================="
    echo " Step 3: Run GSM8K 100-sample Accuracy Evaluation"
    echo "===================================================================="
    python3 tests/eval_gsm8k.py \
      --base-url "http://localhost:8000/v1/chat/completions" \
      --model "${MODEL_PATH}" \
      --limit 100 \
      --concurrency 4 \
      --output /tmp/dcp_opt_gsm8k_results.json || true

    echo "Shutting down vLLM server..."
    kill "${SERVER_PID}" 2>/dev/null || true
fi

echo "===================================================================="
echo " DCP Optimization GSM8K Evaluation Completed!"
echo "===================================================================="
