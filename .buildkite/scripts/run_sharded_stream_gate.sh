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

# Single-host merge gate for K3_SHARDED_EXPERT_STREAMING.
#
# Serves MODEL twice on one host with --load-format=runai_streamer and fixed
# greedy prompts:
#   round 1: K3_SHARDED_EXPERT_STREAMING=0 (stock full read)
#   round 2: K3_SHARDED_EXPERT_STREAMING=1 (filtered read)
# On a single host every device is addressable by the one process, so the
# needed-expert set is ALL experts and the filter must be a byte-level no-op.
# The gate therefore asserts, and fails loudly otherwise:
#   * round 2's serve log carries the "[sharded-stream]" line with
#     kept == total checkpoint tensors and 0.0 GiB skipped;
#   * round 1's serve log does NOT carry that line (flag off = stock path);
#   * the greedy completions of the two rounds are byte-identical.
#
# Serve/health plumbing mirrors run_serving_probe.sh (same env knobs, same
# wait loop, serve log echoed on every exit path) so this run reads like the
# other single-host K3 gates.
#
# RUN THIS INSIDE THE CONTAINER:
#   .buildkite/scripts/run_in_docker.sh \
#     bash /workspace/tpu_inference/.buildkite/scripts/run_sharded_stream_gate.sh
#
# Configuration by environment variable (defaults match run_serving_probe.sh):
#   MODEL                    model id or gs:// snapshot to serve (required)
#   EXTRA_SERVE_FLAGS        extra flags appended to `vllm serve` verbatim
#   PORT                     default 8000
#   TENSOR_PARALLEL_SIZE     default 8
#   MAX_MODEL_LEN            default 2048
#   MAX_NUM_SEQS             default 16
#   GPU_MEMORY_UTILIZATION   default 0.8
#   STARTUP_TIMEOUT_SECONDS  per round, default 3600
#   GATE_MAX_TOKENS          greedy tokens per prompt, default 32

set -uo pipefail

MODEL=${MODEL:?MODEL must be set to the model id or gs:// path to serve}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-3600}
GATE_MAX_TOKENS=${GATE_MAX_TOKENS:-32}
OUT_DIR=${OUT_DIR:-/tmp/sharded_stream_gate}

# Fixed prompts, greedy decode. The depth-sliced K3 is not a language model,
# so the TEXT is meaningless by construction -- what the gate checks is that
# both rounds produce the SAME bytes, which greedy decoding on identical
# weights must.
PROMPTS=(
  "The capital of France is"
  "1 2 3 4 5 6 7 8"
  "def fibonacci(n):"
  "Water is composed of"
)

mkdir -p "${OUT_DIR}"

echo "[shard-gate] model=${MODEL}"
echo "[shard-gate] tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN}" \
     "max_num_seqs=${MAX_NUM_SEQS} util=${GPU_MEMORY_UTILIZATION}"
echo "[shard-gate] serve extra flags: ${EXTRA_SERVE_FLAGS:-<none>}"
echo "[shard-gate] MODEL_IMPL_TYPE=${MODEL_IMPL_TYPE:-<unset>}" \
     "NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN:-<unset>}"

SERVER_PID=""
CURRENT_LOG=""

# Reached only through the EXIT trap installed below.
# shellcheck disable=SC2317
dump_serve_log() {
  if [ -n "${CURRENT_LOG}" ]; then
    echo "--- [shard-gate] vllm serve log (${CURRENT_LOG}) ---"
    cat "${CURRENT_LOG}" 2>/dev/null || echo "[shard-gate] no serve log written"
    echo "--- [shard-gate] end of serve log ---"
  fi
  if [ -n "${SERVER_PID}" ]; then
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap dump_serve_log EXIT

start_server() {
  # $1 = 0|1 for K3_SHARDED_EXPERT_STREAMING, $2 = serve log path.
  CURRENT_LOG=$2
  echo "--- [shard-gate] starting vllm serve (K3_SHARDED_EXPERT_STREAMING=$1) ---"
  # shellcheck disable=SC2086
  K3_SHARDED_EXPERT_STREAMING=$1 vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    ${EXTRA_SERVE_FLAGS:-} \
    > "$2" 2>&1 &
  SERVER_PID=$!
}

wait_health() {
  echo "--- [shard-gate] waiting up to ${STARTUP_TIMEOUT_SECONDS}s for /health ---"
  local start elapsed
  start=$(date +%s)
  while true; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}"
      local serve_rc=$?
      echo "[shard-gate] FAILED: vllm serve exited with ${serve_rc} after" \
           "$(( $(date +%s) - start ))s without becoming healthy"
      return 1
    fi
    elapsed=$(( $(date +%s) - start ))
    if [ "${elapsed}" -ge "${STARTUP_TIMEOUT_SECONDS}" ]; then
      echo "[shard-gate] FAILED: /health never came up within" \
           "${STARTUP_TIMEOUT_SECONDS}s (server still running)"
      return 1
    fi
    if [ $(( elapsed % 60 )) -lt 10 ]; then
      echo "[shard-gate] still loading, ${elapsed}s elapsed; last serve line:" \
           "$(tail -n 1 "${CURRENT_LOG}" 2>/dev/null | tr -d '\r' | tail -c 300)"
    fi
    sleep 10
  done
  echo "[shard-gate] healthy after $(( $(date +%s) - start ))s"
}

stop_server() {
  # The TPU is exclusive: round 2 cannot start until round 1's server and its
  # engine processes are fully gone.
  echo "--- [shard-gate] stopping vllm serve (pid ${SERVER_PID}) ---"
  kill "${SERVER_PID}" 2>/dev/null || true
  local waited=0
  while kill -0 "${SERVER_PID}" 2>/dev/null && [ "${waited}" -lt 120 ]; do
    sleep 5
    waited=$(( waited + 5 ))
  done
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[shard-gate] server ignored SIGTERM for ${waited}s, sending SIGKILL"
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  pkill -9 -f "vllm serve" 2>/dev/null || true
  SERVER_PID=""
  # Give the TPU driver a moment to release the devices.
  sleep 20
}

capture_completions() {
  # $1 = output json path. Greedy /v1/completions for every fixed prompt.
  python3 - "${PORT}" "${MODEL}" "${GATE_MAX_TOKENS}" "$1" "${PROMPTS[@]}" << 'PY'
import json
import sys
import urllib.request

port, model, max_tokens, out_path = sys.argv[1:5]
prompts = sys.argv[5:]
results = []
for prompt in prompts:
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    text = body["choices"][0]["text"]
    print(f"[shard-gate] prompt {prompt!r} -> {text!r}")
    results.append({"prompt": prompt, "text": text})
with open(out_path, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
PY
}

run_round() {
  # $1 = 0|1 flag value, $2 = serve log, $3 = completions json.
  start_server "$1" "$2" || return 1
  wait_health || return 1
  capture_completions "$3" || {
    echo "[shard-gate] FAILED: completion capture failed (flag=$1)"
    return 1
  }
  stop_server
  CURRENT_LOG=""
  return 0
}

LOG_OFF="${OUT_DIR}/serve_flag_off.log"
LOG_ON="${OUT_DIR}/serve_flag_on.log"
OUT_OFF="${OUT_DIR}/completions_flag_off.json"
OUT_ON="${OUT_DIR}/completions_flag_on.json"

run_round 0 "${LOG_OFF}" "${OUT_OFF}" || exit 1
run_round 1 "${LOG_ON}" "${OUT_ON}" || exit 1

echo "--- [shard-gate] checking the [sharded-stream] no-op invariants ---"
python3 - "${LOG_OFF}" "${LOG_ON}" << 'PY'
import re
import sys

log_off = open(sys.argv[1], errors="replace").read()
log_on = open(sys.argv[2], errors="replace").read()

pattern = re.compile(
    r"\[sharded-stream\] process \d+ streams (\d+)/(\d+) routed experts: "
    r"keeping (\d+)/(\d+) checkpoint tensors \(([\d.]+) GiB\), "
    r"skipping ([\d.]+) GiB", re.S)

if pattern.search(log_off):
    sys.exit("[shard-gate] FAILED: flag-OFF round logged a [sharded-stream] "
             "filter line; the flag did not gate the feature off.")

match = pattern.search(log_on)
if not match:
    sys.exit("[shard-gate] FAILED: flag-ON round has no [sharded-stream] "
             "filter line. Either the filtered iterator never ran (check for "
             "a [sharded-stream] fallback warning in the serve log above) or "
             "the log format changed.")

experts_kept, experts_total, kept, total, kept_gib, skipped_gib = match.groups()
print(f"[shard-gate] flag-ON: {experts_kept}/{experts_total} experts, "
      f"{kept}/{total} tensors, {kept_gib} GiB kept, {skipped_gib} GiB skipped")
if kept != total:
    sys.exit(f"[shard-gate] FAILED: single host must keep every tensor, got "
             f"{kept}/{total}.")
if experts_kept != experts_total:
    sys.exit(f"[shard-gate] FAILED: single host must need every expert, got "
             f"{experts_kept}/{experts_total}.")
if float(skipped_gib) != 0.0:
    sys.exit(f"[shard-gate] FAILED: single host must skip 0 bytes, got "
             f"{skipped_gib} GiB.")
print("[shard-gate] no-op invariants hold")
PY
CHECK_RC=$?
if [ "${CHECK_RC}" -ne 0 ]; then
  echo "--- [shard-gate] flag-ON serve log for the failure above ---"
  tail -n 200 "${LOG_ON}" 2>/dev/null
  exit "${CHECK_RC}"
fi

echo "--- [shard-gate] comparing greedy outputs byte-for-byte ---"
if ! cmp "${OUT_OFF}" "${OUT_ON}"; then
  echo "[shard-gate] FAILED: greedy completions differ between the stock read"
  echo "and the filtered read. Flag OFF:"
  cat "${OUT_OFF}"
  echo "Flag ON:"
  cat "${OUT_ON}"
  exit 1
fi
echo "[shard-gate] PASS: filter was a no-op and outputs are byte-identical"
exit 0
