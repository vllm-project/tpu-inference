#!/usr/bin/env bash
# Sweep TP4-CP2 / TP2-CP4 / TP1-CP8 (with TP8 as built-in baseline) across batch sizes.
# KV lengths (4K / 32K / 128K / 256K) are swept inside the Python script each run.
#
# Usage:
#   bash run_dcp_sweep.sh [output_log]
#
# Env-var overrides:
#   BATCH_SIZES="1 8 32 64"  NUM_Q_HEADS=128  NUM_KV_HEADS=16  BENCH=30  bash run_dcp_sweep.sh

set -euo pipefail

LOG="${1:-dcp_sweep_$(date +%Y%m%d_%H%M%S).log}"
SCRIPT="$(dirname "$0")/benchmark_dcp_forward_perf.py"

BATCH_SIZES="${BATCH_SIZES:-1 8 32 64}"

# Model config
NUM_Q_HEADS="${NUM_Q_HEADS:-128}"
NUM_KV_HEADS="${NUM_KV_HEADS:-16}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
WARMUP="${WARMUP:-3}"
BENCH="${BENCH:-50}"

# Each entry: "MODEL_SIZE DCP_SIZE LABEL"
# The Python script always runs TP8 as baseline + one DCP config per invocation.
DCP_CONFIGS=(
  "4 2 TP4-CP2"
  "2 4 TP2-CP4"
  "1 8 TP1-CP8"
)

export NUM_Q_HEADS NUM_KV_HEADS HEAD_DIM PAGE_SIZE MAX_NUM_SEQS WARMUP BENCH

run() {
  echo ""
  echo "── batch=$1  model=$2  dcp=$3  ($4) ──────────────────────"
  NUM_SEQS_REAL="$1" MODEL_SIZE="$2" DCP_SIZE="$3" \
    python "$SCRIPT" 2>&1 | grep -E "^(={2}|  |──|>>|\s*[0-9])" | sed "s/^/  /"
}

{
  echo "============================================================"
  echo " DCP decode sweep — $(date)"
  echo " q_heads=$NUM_Q_HEADS  kv_heads=$NUM_KV_HEADS  head_dim=$HEAD_DIM"
  echo " page_size=$PAGE_SIZE  max_seqs=$MAX_NUM_SEQS"
  echo " batch_sizes: $BATCH_SIZES"
  echo " KV lengths:  swept inside script (4K/32K/128K/256K)"
  echo "============================================================"

  for BS in $BATCH_SIZES; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " batch_size = $BS"
    echo "════════════════════════════════════════════════════════════"

    # TP8-only baseline (DCP_SIZE=1 makes the DCP section also TP8, ignore that output)
    # Actually TP8 is always printed inside every run; no need for a separate pass.

    for cfg in "${DCP_CONFIGS[@]}"; do
      read -r MODEL DCP LABEL <<< "$cfg"
      run "$BS" "$MODEL" "$DCP" "$LABEL"
    done
  done

  echo ""
  echo "Done — $(date)"
} 2>&1 | tee "$LOG"

echo ""
echo "Results saved to: $LOG"
