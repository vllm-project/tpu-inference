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

# Measures accuracy and performance against an already-running vLLM server.
#
# Meant for CI steps that bring the server up themselves -- e.g. a multi-host
# slice through run_multihost.sh, where loading takes most of an hour and every
# measurement has to share the one server. Three legs, in order:
#
#   smoke  one /v1/completions request, which must return text.
#   gsm8k  lm-evaluation-harness, stock gsm8k (5-shot) over chat completions.
#   bench  `vllm bench serve` on the random dataset.
#
# Results are written to ${OUT_DIR}/metrics.json. No thresholds are applied
# here (see check_eval_thresholds.py), so one leg failing does not throw away
# the others. The exit status is non-zero only if the smoke request fails,
# i.e. the server is not serving. The Python parts live in
# eval_running_server_helpers.py.
#
# Configuration, all through the environment:
#   MODEL                   served model name (required)
#   HOST, PORT              server address (default 127.0.0.1:8000)
#   OUT_DIR                 output directory (default /workspace/artifacts)
#   GSM8K_REASONING_EFFORT  sent as chat_template_kwargs.reasoning_effort; when
#                           empty, the stock gsm8k task runs unchanged
#   GSM8K_MAX_TOKENS        default 1024
#   GSM8K_CONCURRENCY       default 64
#   GSM8K_LIMIT             default: all 1319 test questions
#   GSM8K_TIMEOUT_S         default 3600
#   BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, BENCH_CONCURRENCY   (required)
#   BENCH_NUM_PROMPTS       default BENCH_CONCURRENCY
#   BENCH_NUM_WARMUPS       default 0
#   BENCH_TIMEOUT_S         default 3600
#
# Usage:
#   MODEL=Qwen/Qwen3.8-2.4T-A95B-FP8 GSM8K_REASONING_EFFORT=low \
#   BENCH_INPUT_LEN=8192 BENCH_OUTPUT_LEN=1024 BENCH_CONCURRENCY=368 \
#   BENCH_NUM_WARMUPS=368 bash tests/e2e/benchmarking/eval_running_server.sh
set -uo pipefail

MODEL="${MODEL:?MODEL (the served model name) is required}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:?BENCH_INPUT_LEN is required}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:?BENCH_OUTPUT_LEN is required}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:?BENCH_CONCURRENCY is required}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT_DIR="${OUT_DIR:-/workspace/artifacts}"
BASE_URL="http://${HOST}:${PORT}"
HELPERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval_running_server_helpers.py"

mkdir -p "${OUT_DIR}"
# Per-leg status lines, merged into metrics.json at the end.
STATUS_FILE="${OUT_DIR}/leg_status.tsv"
: > "${STATUS_FILE}"

record_leg() {  # record_leg <leg> <exit code> <seconds>
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${STATUS_FILE}"
}

# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
echo "--- smoke: one completion from ${MODEL}"
start=${SECONDS}
curl -sS --max-time 600 "${BASE_URL}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 "${HELPERS}" smoke-payload "${MODEL}")" \
  -o "${OUT_DIR}/smoke.json"
smoke_rc=$?
if [ "${smoke_rc}" -eq 0 ]; then
  python3 "${HELPERS}" check-smoke "${OUT_DIR}/smoke.json"
  smoke_rc=$?
fi
record_leg smoke "${smoke_rc}" $((SECONDS - start))

# ---------------------------------------------------------------------------
# gsm8k: stock lm_eval gsm8k (5-shot, multi-turn few-shot) over chat
# completions. With GSM8K_REASONING_EFFORT set, the task is overridden only to
# add chat_template_kwargs.
# ---------------------------------------------------------------------------
run_gsm8k() {
  local task="gsm8k" task_dir="${OUT_DIR}/lm_eval_tasks"
  if [ -n "${GSM8K_REASONING_EFFORT:-}" ]; then
    task="gsm8k_chat_reasoning"
    python3 "${HELPERS}" stage-gsm8k-task "${task_dir}" "${task}" \
      "${GSM8K_REASONING_EFFORT}" || return 1
  fi

  local args=(
    --model local-chat-completions
    --model_args "model=${MODEL},base_url=${BASE_URL}/v1/chat/completions,api_key=EMPTY,max_retries=3,timeout=1800,tokenized_requests=False,num_concurrent=${GSM8K_CONCURRENCY:-64},max_length=16384"
    --tasks "${task}"
    --apply_chat_template
    --gen_kwargs "max_tokens=${GSM8K_MAX_TOKENS:-1024},temperature=0,top_p=1.0"
    --batch_size 1
    --log_samples
    --output_path "${OUT_DIR}/gsm8k"
  )
  [ "${task}" != "gsm8k" ] && args+=(--include_path "${task_dir}")
  [ -n "${GSM8K_LIMIT:-}" ] && args+=(--limit "${GSM8K_LIMIT}")
  timeout "${GSM8K_TIMEOUT_S:-3600}" python3 -m lm_eval "${args[@]}" 2>&1 | tee "${OUT_DIR}/gsm8k.log"
}

# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
run_bench() {
  timeout "${BENCH_TIMEOUT_S:-3600}" vllm bench serve \
    --backend vllm \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dataset-name random \
    --random-input-len "${BENCH_INPUT_LEN}" \
    --random-output-len "${BENCH_OUTPUT_LEN}" \
    --num-prompts "${BENCH_NUM_PROMPTS:-${BENCH_CONCURRENCY}}" \
    --max-concurrency "${BENCH_CONCURRENCY}" \
    --num-warmups "${BENCH_NUM_WARMUPS:-0}" \
    --request-rate inf \
    --seed 42 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-dir "${OUT_DIR}" \
    --result-filename bench.json 2>&1 | tee "${OUT_DIR}/bench.log"
}

if [ "${smoke_rc}" -eq 0 ]; then
  echo "--- gsm8k"
  start=${SECONDS}
  run_gsm8k
  record_leg gsm8k $? $((SECONDS - start))

  echo "--- bench: random ${BENCH_INPUT_LEN}/${BENCH_OUTPUT_LEN} at concurrency ${BENCH_CONCURRENCY}"
  start=${SECONDS}
  run_bench
  record_leg bench $? $((SECONDS - start))
else
  echo "[smoke] FAILED (exit ${smoke_rc}); skipping gsm8k and bench"
fi

# ---------------------------------------------------------------------------
# metrics.json
# ---------------------------------------------------------------------------
python3 "${HELPERS}" write-metrics "${OUT_DIR}" "${MODEL}"

exit "${smoke_rc}"
