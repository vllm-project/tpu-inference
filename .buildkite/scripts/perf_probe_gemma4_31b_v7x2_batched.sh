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

# Perf probe: Gemma4-31B 1k/1f512x512/500-GBS256 on v7x-2 (TP=2), batched RPA.
# Same as perf_probe_gemma4_31b_v7x2.sh but sets USE_BATCHED_RPA_KERNEL=1.
set -uo pipefail

RPA_MODE=batched
echo "[probe] RPA_MODE=$RPA_MODE  agent=${BUILDKITE_AGENT_NAME:-unknown}  host=$(hostname 2>/dev/null || echo ?)"

VLLM_LOG=/tmp/vllm_server.log
BM_LOG=/tmp/bm_client.log

echo "[probe] starting vllm serve in background (USE_BATCHED_RPA_KERNEL=1)"
USE_BATCHED_RPA_KERNEL=1 \
VLLM_USE_V1=1 \
MODEL_IMPL_TYPE=flax_nnx \
vllm serve \
  --model google/gemma-4-31B-it \
  --max-num-seqs 256 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --async-scheduling \
  --additional-config "{\"quantization\": { \"qwix\": { \"rules\": [{ \"module_path\": \".*\", \"weight_qtype\": \"float8_e4m3fn\", \"act_qtype\": \"float8_e4m3fn\"}]}}}" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill -TERM -$VLLM_PID 2>/dev/null || kill -TERM $VLLM_PID 2>/dev/null || true' EXIT

echo "[probe] waiting up to 60 min for Application startup complete (pid=$VLLM_PID)"
SERVER_UP=false
for i in $(seq 1 360); do
  if grep -Fq "Application startup complete" "$VLLM_LOG" 2>/dev/null; then
    SERVER_UP=true
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[probe][ERROR] server died early. Last 200 lines of $VLLM_LOG:"
    tail -200 "$VLLM_LOG"
    exit 1
  fi
  if [ $((i % 30)) -eq 0 ]; then
    echo "[probe] still waiting (iter=$i, ~$((i * 10))s elapsed)"
  fi
  sleep 10
done

if [ "$SERVER_UP" != "true" ]; then
  echo "[probe][ERROR] timeout waiting for server. Last 200 lines:"
  tail -200 "$VLLM_LOG"
  exit 1
fi

echo "[probe] server up. Running client benchmark."
vllm bench serve \
  --model google/gemma-4-31B-it \
  --backend openai-chat \
  --request-rate inf \
  --dataset-name random-mm \
  --endpoint /v1/chat/completions \
  --random-mm-bucket-config "{(512, 512, 1): 1.0}" \
  --random-mm-limit-mm-per-prompt "{\"image\": 1}" \
  --random-input-len 1024 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --temperature 0 \
  2>&1 | tee "$BM_LOG"
CLIENT_RC=${PIPESTATUS[0]}

echo "---------- bench result (RPA_MODE=$RPA_MODE) ----------"
echo "[probe-summary] RPA_MODE=$RPA_MODE agent=${BUILDKITE_AGENT_NAME:-unknown}"
grep "Request throughput (req/s):" "$BM_LOG" || echo "[probe] throughput line not found"
grep "Mean TTFT (ms):" "$BM_LOG" || true
grep "Mean TPOT (ms):" "$BM_LOG" || true
grep "Mean ITL (ms):" "$BM_LOG" || true
grep "Output token throughput (tok/s):" "$BM_LOG" || true
echo "---------- end ----------"
exit "$CLIENT_RC"
