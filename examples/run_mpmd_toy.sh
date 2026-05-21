#!/usr/bin/env bash
# Launch two independent MPMD workers on a single 4-chip v7x host: each
# worker is its own standalone single-process JAX cluster pinned to 2
# physical chips, with no collectives between them.
#
#   ./examples/run_mpmd_toy.sh
#
# Override how long each worker runs (default 30s):
#   MPMD_DURATION_S=60 ./examples/run_mpmd_toy.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/mpmd_toy.py"
DURATION_S="${MPMD_DURATION_S:-30}"

# Worker 0 -> physical chips 0,1 on libtpu port 8476.
echo "[launcher] starting worker 0 on chips 0,1"
MPMD_TASK_ID=0 \
MPMD_DURATION_S="${DURATION_S}" \
TPU_VISIBLE_CHIPS=0,1 \
TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 \
TPU_PROCESS_BOUNDS=1,1,1 \
TPU_PROCESS_PORT=8476 \
TPU_PROCESS_ADDRESSES=localhost:8476 \
CLOUD_TPU_TASK_ID=0 \
  python "${WORKER}" &
PID0=$!

# Worker 1 -> physical chips 2,3 on libtpu port 8477.
echo "[launcher] starting worker 1 on chips 2,3"
MPMD_TASK_ID=1 \
MPMD_DURATION_S="${DURATION_S}" \
TPU_VISIBLE_CHIPS=2,3 \
TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 \
TPU_PROCESS_BOUNDS=1,1,1 \
TPU_PROCESS_PORT=8477 \
TPU_PROCESS_ADDRESSES=localhost:8477 \
CLOUD_TPU_TASK_ID=0 \
  python "${WORKER}" &
PID1=$!

RC=0
wait "${PID0}" || RC=$?
wait "${PID1}" || RC=$?
echo "[launcher] both workers exited (last non-zero rc=${RC})"
exit "${RC}"
