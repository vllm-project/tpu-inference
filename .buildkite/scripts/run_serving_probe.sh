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

# Serve one model on a single host and run tests/e2e/multihost/serving_probe.py
# against it. The multi-host rig (run_multihost.sh) already pairs a serve
# command with a client command; this is the same pairing for the jobs that fit
# on one host, so a single-host and a multi-host run are checked by identical
# invariants.
#
# RUN THIS INSIDE THE CONTAINER, e.g.
#   .buildkite/scripts/run_in_docker.sh bash /workspace/tpu_inference/.buildkite/scripts/run_serving_probe.sh
#
# Configuration is by environment variable:
#   MODEL                 model id or gs:// snapshot to serve (required;
#                         TEST_MODEL is accepted as a fallback name)
#   EXTRA_SERVE_FLAGS     extra flags appended to `vllm serve` verbatim
#   PROBE_EXTRA_FLAGS     extra flags appended to the probe verbatim
#   PORT                  default 8000
#   TENSOR_PARALLEL_SIZE  default 8
#   MAX_MODEL_LEN         default 2048
#   MAX_NUM_SEQS          default 16
#   GPU_MEMORY_UTILIZATION default 0.8
#   STARTUP_TIMEOUT_SECONDS how long to wait for /health, default 3600
#                         (SERVE_STARTUP_TIMEOUT_SECONDS, the run_multihost.sh
#                         name, is honored as a fallback). Weight
#                         streaming from GCS dominates it, so scale it with the
#                         checkpoint size, not with the model's compute.
#   PROBE_CONCURRENCY     default 8
#   PROBE_LONG_TOKENS     default 512
#
# The serve log is echoed on every exit path, including success, because the
# load-path detail this script exists to expose (which tensor failed, which
# shard was being read) is only in that log.

set -uo pipefail

# TEST_MODEL is the name the nightly model steps already export; accept it so
# the same step env drives either harness.
MODEL="${MODEL:-${TEST_MODEL:-}}"
MODEL=${MODEL:?MODEL must be set to the model id or gs:// path to serve}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
# SERVE_STARTUP_TIMEOUT_SECONDS is the run_multihost.sh name for the same
# knob; honor either so one env block works on both rigs.
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-${SERVE_STARTUP_TIMEOUT_SECONDS:-3600}}"
PROBE_CONCURRENCY=${PROBE_CONCURRENCY:-8}
PROBE_LONG_TOKENS=${PROBE_LONG_TOKENS:-512}
SERVE_LOG=${SERVE_LOG:-/tmp/vllm_serve.log}
PROBE=${PROBE:-/workspace/tpu_inference/tests/e2e/multihost/serving_probe.py}

echo "[serve-probe] model=${MODEL}"
echo "[serve-probe] tp=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN}" \
     "max_num_seqs=${MAX_NUM_SEQS} util=${GPU_MEMORY_UTILIZATION}"
echo "[serve-probe] serve extra flags: ${EXTRA_SERVE_FLAGS:-<none>}"
echo "[serve-probe] probe extra flags: ${PROBE_EXTRA_FLAGS:-<none>}"
echo "[serve-probe] MODEL_IMPL_TYPE=${MODEL_IMPL_TYPE:-<unset>}" \
     "NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN:-<unset>}"

SERVER_PID=""
# Reached only through the EXIT trap installed below.
# shellcheck disable=SC2317
dump_serve_log() {
  echo "--- [serve-probe] vllm serve log (${SERVE_LOG}) ---"
  cat "${SERVE_LOG}" 2>/dev/null || echo "[serve-probe] no serve log written"
  echo "--- [serve-probe] end of serve log ---"
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap dump_serve_log EXIT

echo "--- [serve-probe] starting vllm serve ---"
# shellcheck disable=SC2086
vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  ${EXTRA_SERVE_FLAGS:-} \
  > "${SERVE_LOG}" 2>&1 &
SERVER_PID=$!

echo "--- [serve-probe] waiting up to ${STARTUP_TIMEOUT_SECONDS}s for /health ---"
START=$(date +%s)
READY=0
while true; do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}"
    SERVE_RC=$?
    echo "[serve-probe] FAILED: vllm serve exited with ${SERVE_RC} after" \
         "$(( $(date +%s) - START ))s without becoming healthy"
    exit 1
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "${ELAPSED}" -ge "${STARTUP_TIMEOUT_SECONDS}" ]; then
    echo "[serve-probe] FAILED: /health never came up within" \
         "${STARTUP_TIMEOUT_SECONDS}s (server still running)"
    exit 1
  fi
  if [ $(( ELAPSED % 60 )) -lt 10 ]; then
    # Echo where the server has got to. Streaming a large checkpoint takes
    # tens of minutes, and without this the job looks identical to a hang
    # until the serve log is dumped at exit.
    echo "[serve-probe] still loading, ${ELAPSED}s elapsed; last serve line:" \
         "$(tail -n 1 "${SERVE_LOG}" 2>/dev/null | tr -d '\r' | tail -c 300)"
  fi
  sleep 10
done
echo "[serve-probe] healthy after $(( $(date +%s) - START ))s"

echo "--- [serve-probe] running serving probes ---"
# shellcheck disable=SC2086
python3 "${PROBE}" \
  --base-url "http://localhost:${PORT}" \
  --concurrency "${PROBE_CONCURRENCY}" \
  --long-tokens "${PROBE_LONG_TOKENS}" \
  ${PROBE_EXTRA_FLAGS:-}
PROBE_RC=$?
echo "[serve-probe] probes exited ${PROBE_RC} (ready=${READY})"
exit "${PROBE_RC}"
