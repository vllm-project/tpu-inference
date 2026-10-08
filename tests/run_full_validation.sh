#!/bin/bash
set -euo pipefail

OUTPUTS_DIR="${CDK_OUTPUT_DIR:-/tmp/outputs}"
mkdir -p "${OUTPUTS_DIR}"

echo "===================================================================="
echo " Starting Full DCP Optimization Validation on TPU v7x-8"
echo " Host: $(hostname)"
echo " Date: $(date -u)"
echo "===================================================================="

cd /home/wenxindong_google_com/tpu-inference

if [ "${SKIP_STEP1_2:-0}" = "1" ]; then
    echo "Skipping Step 1 & 2 (already verified)..."
else
    # --------------------------------------------------------------------
    # STEP 1: Unit Tests
    # --------------------------------------------------------------------
    echo ""
    echo "===================================================================="
    echo " STEP 1: Unit Testing write_decode_kv Pallas Kernel"
    echo "===================================================================="
    python3 -m pytest -v -s tests/kernels/rpa_v3_cp/test_write_kv.py 2>&1 | tee "${OUTPUTS_DIR}/step1_unit_tests.log"

    # --------------------------------------------------------------------
    # STEP 2: Attention Layer Performance Benchmark (Qwen3.5-397B Attention Geometry)
    # --------------------------------------------------------------------
    echo ""
    echo "===================================================================="
    echo " STEP 2: Benchmarking Attention Layer Performance (DCP vs TP8)"
    echo " Model Geometry: Qwen3.5-397B (Q_heads=32, KV_heads=2, Head_dim=256)"
    echo "===================================================================="

    export NUM_Q_HEADS=32
    export NUM_KV_HEADS=2
    export HEAD_DIM=256
    export MODEL_SIZE=4
    export DCP_SIZE=2
    export TP_MODEL=8
    export KV_LENS_K="4,32,128,256"
    export BENCH=50
    export WARMUP=5
    export PROFILE_STEPS=0

    python3 tests/layers/common/benchmark_dcp_forward_perf.py 2>&1 | tee "${OUTPUTS_DIR}/step2_attn_layer_benchmark.log"
fi

# --------------------------------------------------------------------
# STEP 3: E2E Benchmark on Qwen3.5-35B-A3B-FP8
# --------------------------------------------------------------------
echo ""
echo "===================================================================="
echo " STEP 3: E2E Benchmark on Qwen3.5-35B-A3B-FP8"
echo "===================================================================="

export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export DCP_DECODE_ONLY_OPT=1
export NUM_PRECOMPILE_WORKERS=8
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export SLICE_ROPE_CACHE=1
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'

export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
mkdir -p "${HF_HUB_CACHE}"

mkdir -p /home/wenxindong_google_com/work/bench_logs/scripts
if [ -f /workspace/gbs1024_trace_file.jsonl ]; then
    cp /workspace/gbs1024_trace_file.jsonl /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl
fi

MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
LOCAL_MODEL_DIR="/tmp/models/Qwen3.5-35B-A3B-FP8"

SNAPSHOT_DIR=""
if [ -d "/mnt/models/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots" ]; then
    for d in /mnt/models/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/*; do
        if [ -d "$d" ] && [ -f "$d/config.json" ]; then
            SNAPSHOT_DIR="$d"
            break
        fi
    done
fi

if [ -n "${SNAPSHOT_DIR}" ]; then
    MODEL_PATH="${SNAPSHOT_DIR}"
    echo "Found cached snapshot in PVC: ${MODEL_PATH}"
elif [ -d "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8" ] && [ -f "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8/config.json" ]; then
    MODEL_PATH="/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8"
    echo "Found cached model in PVC: ${MODEL_PATH}"
elif [ -d "${LOCAL_MODEL_DIR}" ] && [ -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    MODEL_PATH="${LOCAL_MODEL_DIR}"
else
    MODEL_PATH="${MODEL_NAME}"
fi
echo "Using model path: ${MODEL_PATH}"

echo "Starting vLLM server for ${MODEL_PATH} with Native vLLM Model (EP=8, DCP=4, rpa_v3_cp)..."
python3 -u -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 4 \
  --enable-expert-parallel \
  --dtype=auto \
  --kv-cache-dtype=fp8 \
  --max-model-len=8192 --max-num-batched-tokens=512 --max-num-seqs=32 \
  --enable-prefix-caching \
  --gpu-memory-utilization=0.85 --async-scheduling --port=8000 \
  --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  '--limit-mm-per-prompt={"image": 0, "video": 0}' \
  --block-size=128 --enable-chunked-prefill \
  --mamba-cache-mode align \
  --prefix-cache-retention-interval 0 > "${OUTPUTS_DIR}/vllm_serve.log" 2>&1 &

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
        tail -n 100 "${OUTPUTS_DIR}/vllm_serve.log"
        exit 1
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "Waiting for server ($((i * 5))s elapsed)..."
        tail -n 5 "${OUTPUTS_DIR}/vllm_serve.log" 2>/dev/null || true
    fi
    sleep 5
done

if [ "${READY}" -ne 1 ]; then
    echo "Timed out waiting for server to be ready."
    tail -n 100 "${OUTPUTS_DIR}/vllm_serve.log"
    kill "${SERVER_PID}" || true
    exit 1
fi

echo "Running vLLM serving benchmark on ${MODEL_PATH}..."
PYTHONPATH="/home/wenxindong_google_com/tpu-inference/scripts/vllm/benchmarking:${PYTHONPATH:-}" \
python3 /home/wenxindong_google_com/tpu-inference/scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --model "${MODEL_PATH}" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 32 \
  --port 8000 2>&1 | tee "${OUTPUTS_DIR}/step3_e2e_benchmark.log"

# --------------------------------------------------------------------
# STEP 4: lm_eval Correctness Test
# --------------------------------------------------------------------
echo ""
echo "===================================================================="
echo " STEP 4: Model Correctness using lm_eval"
echo "===================================================================="

echo "Running GSM8K accuracy evaluation (limit 100)..."
python3 /home/wenxindong_google_com/tpu-inference/tests/eval_gsm8k.py \
  --base-url "http://localhost:8000/v1/chat/completions" \
  --model "${MODEL_PATH}" \
  --limit 100 \
  --concurrency 32 2>&1 | tee "${OUTPUTS_DIR}/step4_lm_eval.log"

echo "Shutting down vLLM server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "===================================================================="
echo " All Validation Steps Completed!"
echo "===================================================================="
