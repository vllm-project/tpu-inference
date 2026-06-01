#!/bin/bash
# Copyright 2025 Google LLC
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

# Start vllm serve with a Qwen3.5-397B-A17B-FP8 config + custom env vars,
# wait for /health, then drive mmlu_pro through lm_eval.

set -u
set -o pipefail

MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
SERVER_LOG=/tmp/vllm_serve.log

# Custom env vars consumed by vllm + tpu-inference internals.
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d
export NEW_MODEL_DESIGN=1

echo "--- Installing lm-eval (idempotent) ---"
python3 -m pip install --quiet lm-eval || {
  echo "[ERROR] Failed to install lm-eval"
  exit 1
}

echo "--- Starting vllm serve in background ---"
vllm serve "${MODEL}" \
  --max-model-len=9216 \
  --max-num-batched-tokens=1024 \
  --max-num-seqs=64 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size=8 \
  --async-scheduling \
  --port="${PORT}" \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  --limit-mm-per-prompt='{"image": 0, "video": 0}' \
  --kv-cache-dtype=fp8 \
  --enable-expert-parallel \
  --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --mamba-ssm-cache-dtype=bfloat16 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

trap 'echo "--- Tearing down server (pid=${SERVER_PID}) ---"; kill "${SERVER_PID}" 2>/dev/null || true; sleep 5; kill -9 "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

echo "--- Waiting for server /health (server pid=${SERVER_PID}) ---"
# 397B + cold JAX compilation can take 30+ min on first run.
# 180 * 30s = 90 min max wait. Every 10 polls (~5 min) we tail the server
# log into stdout so progress is visible in Buildkite.
HEALTHY=0
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Server healthy after ~$((i * 30))s"
    HEALTHY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Server process died. Last 200 log lines:"
    tail -n 200 "${SERVER_LOG}"
    exit 1
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "--- still waiting (${i}/180, ~$((i * 30))s elapsed); last 30 server log lines ---"
    tail -n 30 "${SERVER_LOG}" 2>/dev/null || echo "(server log not yet readable)"
  fi
  sleep 30
done

if [ "${HEALTHY}" -ne 1 ]; then
  echo "[ERROR] Server did not become healthy in time. Last 200 log lines:"
  tail -n 200 "${SERVER_LOG}"
  exit 1
fi

echo "--- Running lm_eval mmlu_pro --limit 50 ---"
# --output_path is required by lm-eval whenever --log_samples is set.
LM_EVAL_OUTPUT=/tmp/lm_eval_results
mkdir -p "${LM_EVAL_OUTPUT}"
set +e
lm_eval --model local-chat-completions \
  --model_args '{"model": "Qwen/Qwen3.5-397B-A17B-FP8", "base_url": "http://localhost:8000/v1/chat/completions", "enable_thinking": false, "num_concurrent": 500}' \
  --tasks mmlu_pro \
  --apply_chat_template \
  --log_samples \
  --output_path "${LM_EVAL_OUTPUT}" \
  --gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --limit 50 \
  --seed 42
LM_EVAL_EXIT=$?
set -e

echo "--- lm_eval exit code: ${LM_EVAL_EXIT} ---"

if [ "${LM_EVAL_EXIT}" -ne 0 ]; then
  echo "[ERROR] lm_eval failed. Last 200 server log lines:"
  tail -n 200 "${SERVER_LOG}"
fi

exit "${LM_EVAL_EXIT}"
