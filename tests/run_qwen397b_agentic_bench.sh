#!/bin/bash
set -euo pipefail

echo "===================================================================="
echo " Qwen3.5-397B-A17B-FP8 Agentic Benchmark with DCP=4 on TPU v7x-8"
echo "===================================================================="
date -u
python3 -c "import jax; print('JAX devices:', jax.devices())"

export NUM_PRECOMPILE_WORKERS=8
export MODEL_IMPL_TYPE=vllm
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export NEW_MODEL_DESIGN=1
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'

mkdir -p /home/wenxindong_google_com/work/bench_logs/scripts
if [ -f /home/wenxindong_google_com/tpu-inference/gbs1024_trace_file.jsonl ]; then
    cp /home/wenxindong_google_com/tpu-inference/gbs1024_trace_file.jsonl /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl
elif [ -f /workspace/gbs1024_trace_file.jsonl ]; then
    cp /workspace/gbs1024_trace_file.jsonl /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl
fi
echo "Trace file size: $(ls -lh /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl)"

# Ensure HF cache is writable for hub tokenizers / locks
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
mkdir -p "${HF_HUB_CACHE}"

cd /home/wenxindong_google_com/tpu-inference

# Ensure benchmark_agentic.py is present in the current working dir
if [ ! -f benchmark_agentic.py ]; then
    ln -sf scripts/vllm/benchmarking/agentic_benchmark/benchmark_agentic.py benchmark_agentic.py
fi

MODEL_NAME="Qwen/Qwen3.5-397B-A17B-FP8"
SNAPSHOT_DIR=""
for cand in \
    /mnt/models/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/* \
    /mnt/models/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/* \
    /mnt/models/Qwen/Qwen3.5-397B-A17B-FP8 \
    /mnt/models/Qwen3.5-397B-A17B-FP8 \
    /vllm-cache/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/* \
    /vllm-cache/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/* \
    /vllm-cache/Qwen/Qwen3.5-397B-A17B-FP8 \
    /vllm-cache/Qwen3.5-397B-A17B-FP8 \
    /tmp/models/Qwen3.5-397B-A17B-FP8; do
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
    echo "Using model path: ${MODEL_PATH}"
fi

OUTPUTS_DIR="${CDK_OUTPUT_DIR:-/tmp}"
mkdir -p "${OUTPUTS_DIR}"

echo ""
echo "===================================================================="
echo " Starting vLLM Server: Qwen3.5-397B-A17B-FP8 (EP=8, DCP=4, TP=8)"
echo "===================================================================="

vllm serve "${MODEL_PATH}" \
  --max-model-len=65536 --max-num-batched-tokens=2048 --max-num-seqs=16 \
  --enable-prefix-caching \
  --gpu-memory-utilization=0.9 --tensor-parallel-size=8 --async-scheduling --port=8000 \
  --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=bfloat16 \
  --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": false}}, "custom_mamba_cache_multiplier": 16}' \
  --block-size=256 --enable-chunked-prefill \
  --mamba-cache-mode align \
  --prefix-cache-retention-interval 0 \
  --enable-expert-parallel \
  --served-model-name Qwen/Qwen3.5-397B-A17B Qwen/Qwen3.5-397B-A17B-FP8 \
  --decode-context-parallel-size 4 > /tmp/vllm_397b_serve.log 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

echo "Waiting for vLLM server to be ready on port 8000..."
READY=0
for i in $(seq 1 480); do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is UP and ready after $((i * 5)) seconds!"
        READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server died unexpectedly! Check log:"
        tail -n 120 /tmp/vllm_397b_serve.log
        exit 1
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "Waiting for server ($((i * 5))s elapsed)..."
        tail -n 5 /tmp/vllm_397b_serve.log 2>/dev/null || true
    fi
    sleep 5
done

if [ "${READY}" -ne 1 ]; then
    echo "Timed out waiting for server to become ready!"
    tail -n 100 /tmp/vllm_397b_serve.log
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

echo ""
echo "===================================================================="
echo " Benchmark 1: (num-groups 1, concurrency 1)"
echo "===================================================================="
python3 benchmark_agentic.py \
  --model Qwen/Qwen3.5-397B-A17B \
  --model-path-or-id Qwen/Qwen3.5-397B-A17B \
  --trace-file /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl \
  --global-prefix-len 6476 \
  --num-groups 1 \
  --concurrency 1 2>&1 | tee "${OUTPUTS_DIR}/bench1_results.log"

echo ""
echo "===================================================================="
echo " Benchmark 2: (num-groups 2, concurrency 2)"
echo "===================================================================="
python3 benchmark_agentic.py \
  --model Qwen/Qwen3.5-397B-A17B \
  --model-path-or-id Qwen/Qwen3.5-397B-A17B \
  --trace-file /home/wenxindong_google_com/work/bench_logs/scripts/gbs1024_trace_file.jsonl \
  --global-prefix-len 6476 \
  --num-groups 2 \
  --concurrency 2 2>&1 | tee "${OUTPUTS_DIR}/bench2_results.log"

echo ""
echo "Shutting down vLLM server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "===================================================================="
echo " All Benchmarks Completed Successfully!"
echo "===================================================================="
