#!/bin/bash
# Test MMMU-Pro accuracy for a multimodal model.
# Starts vllm serve, waits for health, runs benchmark_serving.py with mmmu_pro.
#
# Required env: TEST_MODEL, TENSOR_PARALLEL_SIZE
# Optional env: MAX_MODEL_LEN (default 4096), MAX_NUM_BATCHED_TOKENS (default 4096),
#               MAX_NUM_SEQS (default 127), NUM_PROMPTS (default 1730),
#               MAX_CONCURRENCY (default 32), MMMU_PRO_OUTPUT_LEN (default 2000),
#               TIMEOUT_SECONDS (default 2400)

set -e

model_name="${TEST_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
tp_size="${TENSOR_PARALLEL_SIZE:-1}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
max_num_seqs="${MAX_NUM_SEQS:-127}"
num_prompts="${NUM_PROMPTS:-1730}"
max_concurrency="${MAX_CONCURRENCY:-32}"
mmmu_pro_output_len="${MMMU_PRO_OUTPUT_LEN:-2000}"
port=8000

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
READY_MESSAGE="Application startup complete."
LOG_FILE="/tmp/vllm_serve.log"
exit_code=0

cleanup() {
    echo "Stopping vllm server..."
    pkill -f "vllm serve" || true
    pgrep -f -i vllm | xargs -r kill -9 2>/dev/null || true
    rm -f "$LOG_FILE"
}
trap cleanup EXIT

echo "=== Starting vllm serve for ${model_name} (tp=${tp_size}) ==="
SKIP_JAX_PRECOMPILE=0 \
MODEL_IMPL_TYPE=flax_nnx \
vllm serve "$model_name" \
    --port "$port" \
    --tensor-parallel-size "$tp_size" \
    --max-model-len "$max_model_len" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --max-num-seqs "$max_num_seqs" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 8, "video": 0}' \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "=== Waiting for server (timeout=${TIMEOUT_SECONDS}s) ==="
start_time=$(date +%s)
while true; do
    elapsed=$(( $(date +%s) - start_time ))
    if [[ "$elapsed" -ge "$TIMEOUT_SECONDS" ]]; then
        echo "TIMEOUT after ${elapsed}s. Server log tail:"
        tail -100 "$LOG_FILE"
        exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died. Log tail:"
        tail -100 "$LOG_FILE"
        exit 1
    fi
    if grep -q "$READY_MESSAGE" "$LOG_FILE" 2>/dev/null; then
        echo "Server ready after ${elapsed}s"
        break
    fi
    sleep 10
done

echo "=== Running MMMU-Pro benchmark ==="
python /workspace/tpu_inference/scripts/vllm/benchmarking/benchmark_serving.py \
    --backend vllm-chat \
    --endpoint /v1/chat/completions \
    --model "$model_name" \
    --dataset-name mmmu_pro \
    --mmmu-pro-subset 'standard (10 options)' \
    --mmmu-pro-output-len "$mmmu_pro_output_len" \
    --run-eval \
    --num-prompts "$num_prompts" \
    --max-concurrency "$max_concurrency"

echo "=== Done ==="
