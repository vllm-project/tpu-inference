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

# Serve a model on a TPU with an installed vLLM and verify the result: vLLM
# selects the TPU platform, the server becomes healthy, it answers a completion
# and a chat request, and it completes every request of a short
# `vllm bench serve`. Both release checks on TPU use this flow:
#   verify image   the staged image's own vLLM (verify_image_on_tpu.sh)
#   verify wheels  the wheels, installed into a fresh venv (serve_wheels.sh)
#
# Usage: serve_and_verify.sh <python> <check> <what>
#   <python>  Python interpreter of the vLLM install under test; the vllm CLI
#             must be in the same directory. A bare name such as python3 is
#             looked up on PATH.
#   <check>   Short name of the check (image or wheels), used as the
#             annotation context.
#   <what>    What is being served, for the log and the annotation.
#
# Needs a TPU host (one v6e chip is enough for the default model) with curl
# and network access to huggingface.co. The default model is gated: HF_TOKEN
# (or a `huggingface-cli login`) must give access to it.
#
# Environment:
#   TEST_MODEL        Model to serve and bench (default:
#                     meta-llama/Llama-3.1-8B-Instruct).
#   MAX_MODEL_LEN     --max-model-len (default: 1024).
#   NUM_PROMPTS       Requests in the bench (default: 30).
#   SERVER_TIMEOUT_S  How long to wait for /health (default: 1800); covers the
#                     model download and the first XLA compile.
#   TMPDIR            Where the scratch goes (the compilation cache, the logs);
#                     the release checks on kube set /dev/shm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"
# The checks below run with the Python of the vLLM install under test.
WHEEL_UTILS="${SCRIPT_DIR}/../scripts/wheel_utils.py"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <python> <check> <what>" >&2
  exit 2
fi
PY="$1"
CHECK="$2"
WHAT="$3"
[[ "${PY}" == */* ]] || PY="$(command -v "${PY}")" || die "$1 not found on PATH."
VLLM="$(dirname "${PY}")/vllm"
[[ -x "${PY}" && -x "${VLLM}" ]] || die "expected a Python and the vllm CLI at ${PY} and ${VLLM}."

TEST_MODEL="${TEST_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/serve-and-verify.XXXXXX")"
SERVER_PID=""
TAIL_PID=""

# vllm serve runs its engine, which holds the TPU, in child processes: killing
# only the server can leave them behind (what `fuser /dev/vfio/*` and `kill -9`
# clean up by hand). Started with setsid, the server leads its own process
# group, so signalling the group reaches them all. <grace> is how long it gets
# to shut down.
stop_server() {
  local grace="$1"
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "--- :octagonal_sign: Stopping the server"
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    local deadline=$((SECONDS + grace))
    while kill -0 "${SERVER_PID}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null
  SERVER_PID=""
}

cleanup() {
  local rc=$?
  set +e
  # Expand the section the step failed in, before stop_server opens another.
  (( rc == 0 )) || echo "^^^ +++"
  # Being in its own process group, the server never sees the agent's signal
  # when the job is cancelled or times out, and the agent kills this script
  # after its grace period (10s by default), so stop it fast in that case.
  if (( rc > 128 )); then stop_server 5; else stop_server 60; fi
  [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null
  rm -rf "${WORK}"
}
trap cleanup EXIT
# Stop on signals instead of continuing; the EXIT trap then cleans up once.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Run from ${WORK}, not from a checkout: vLLM starts some helpers with
# `python -m`, which puts the working directory first on sys.path, so a
# checkout's tpu_inference/ would shadow the install under test.
cd "${WORK}"

# --- The platform -----------------------------------------------------------
echo "--- :mag: Checking that vLLM picks the TPU platform"
"${PY}" "${WHEEL_UTILS}" verify_tpu_platform

# --- Serve ------------------------------------------------------------------
echo "--- :test_tube: Serving ${TEST_MODEL} from ${WHAT}"
# A fresh compilation cache, as on a user's first run, so a cached compile
# cannot hide one that fails. The weights are not under test: they come from
# the usual Hugging Face cache, on the CI fleet its shared model cache, which
# also keeps a 16 GB download off the pod's boot disk.
if [[ -z "${HF_HUB_CACHE:-}" && -z "${HF_HOME:-}" && -d /cache/hf/hub ]]; then
  export HF_HUB_CACHE=/cache/hf/hub
fi
export JAX_COMPILATION_CACHE_DIR="${WORK}/xla-cache"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"
PORT="$("${PY}" "${WHEEL_UTILS}" free_port)"
BASE_URL="http://127.0.0.1:${PORT}"
SERVER_LOG="${WORK}/server.log"
SERVE_CMD=("${VLLM}" serve "${TEST_MODEL}"
  --host 127.0.0.1 --port "${PORT}"
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}")
echo "${SERVE_CMD[*]}"
# Created up front: the server's redirect happens in the forked child, which
# could lose the race with tail below and leave it nothing to open.
: >"${SERVER_LOG}"
SERVE_START=${SECONDS}
if command -v setsid >/dev/null 2>&1; then
  setsid "${SERVE_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
else
  "${SERVE_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!
# Stream the server log into the job log until the server stops.
tail -n +1 -f "${SERVER_LOG}" &
TAIL_PID=$!

DEADLINE=$((SECONDS + SERVER_TIMEOUT_S))
until curl -sf -o /dev/null --max-time 5 "${BASE_URL}/health"; do
  kill -0 "${SERVER_PID}" 2>/dev/null \
    || die "vllm serve exited before it became healthy; its log is above."
  (( SECONDS < DEADLINE )) \
    || die "vllm serve was not healthy within ${SERVER_TIMEOUT_S}s."
  sleep 5
done
echo "--- :white_check_mark: Healthy after $((SECONDS - SERVE_START))s"

# --- Ask it -----------------------------------------------------------------
echo "--- :speech_balloon: Asking ${TEST_MODEL}"
"${PY}" "${WHEEL_UTILS}" verify_server_answers \
  "${BASE_URL}" "${TEST_MODEL}" "${WORK}/answers.txt"

# --- Bench it ---------------------------------------------------------------
echo "--- :stopwatch: vllm bench serve"
BENCH_ARGS=(--model "${TEST_MODEL}" --host 127.0.0.1 --port "${PORT}"
  --backend vllm --request-rate inf
  --dataset-name random --num-prompts "${NUM_PROMPTS}"
  --random-input-len 128 --random-output-len 128
  --percentile-metrics "ttft,tpot,itl,e2el"
  --ignore-eos --skip-chat-template
  --save-result --result-dir "${WORK}" --result-filename bench.json)
echo "vllm bench serve ${BENCH_ARGS[*]}"
"${VLLM}" bench serve "${BENCH_ARGS[@]}"
# vllm bench serve exits 0 even when requests fail, so check its result.
"${PY}" "${WHEEL_UTILS}" verify_bench_result \
  "${WORK}/bench.json" "${WORK}/bench.txt"

echo "+++ :white_check_mark: ${WHAT} serves ${TEST_MODEL} on a TPU"
if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_JOB_ID:-}" ]]; then
  buildkite-agent annotate --style success --context "release-serve-${CHECK}" <<EOF || echo "WARNING: could not annotate the build." >&2
Served and benched \`${TEST_MODEL}\` on a TPU from ${WHAT}:

\`\`\`
$(cat "${WORK}/answers.txt" "${WORK}/bench.txt")
\`\`\`
EOF
fi
