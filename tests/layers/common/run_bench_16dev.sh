#!/bin/bash
# NUM_DEVICE=16, Q_HEAD=128, KV_HEAD=16
#
# Setup 1 - TP16:       model=16, dcp=1   → each device Q_head=8,   KV_head=1
# Setup 2 - TP16 DCP2:  model=8,  dcp=2   → each device Q_head=16,  KV_head=2
# Setup 3 - TP16 DCP8:  model=2,  dcp=8   → each device Q_head=64,  KV_head=8
# Setup 4 - TP16 DCP16: model=1,  dcp=16  → each device Q_head=128, KV_head=16
#
# Batch sizes: 32, 64, 128
# Context lengths: 4K, 32K, 128K, 256K, 512K

set -e

SCRIPT="python3 tests/layers/common/benchmark_dcp_forward_perf.py"
BASE="USE_BATCHED_RPA_KERNEL=1 PAGE_SIZE=256 NUM_Q_HEADS=128 NUM_KV_HEADS=16 TP_MODEL=16 KV_LENS_K=4,32,64,128"

for BS in 32 64 128; do
  echo ""
  echo "######################################################################"
  echo "# BATCH_SIZE = $BS"
  echo "######################################################################"

  echo "--- Setup 1: TP16 (model=16, dcp=1) ---"
  env $BASE NUM_SEQS_REAL=$BS MAX_NUM_SEQS=$BS \
      MODEL_SIZE=16 DCP_SIZE=1 \
      OUTPUT_DIR=profiles/16dev/bs${BS}/setup1_tp16 \
      $SCRIPT

  echo "--- Setup 2: TP16 DCP2 (model=8, dcp=2) ---"
  env $BASE NUM_SEQS_REAL=$BS MAX_NUM_SEQS=$BS \
      MODEL_SIZE=8 DCP_SIZE=2 \
      OUTPUT_DIR=profiles/16dev/bs${BS}/setup2_dcp2 \
      $SCRIPT

  echo "--- Setup 3: TP16 DCP8 (model=2, dcp=8) ---"
  env $BASE NUM_SEQS_REAL=$BS MAX_NUM_SEQS=$BS \
      MODEL_SIZE=2 DCP_SIZE=8 \
      OUTPUT_DIR=profiles/16dev/bs${BS}/setup3_dcp8 \
      $SCRIPT

  echo "--- Setup 4: TP16 DCP16 (model=1, dcp=16) ---"
  env $BASE NUM_SEQS_REAL=$BS MAX_NUM_SEQS=$BS \
      MODEL_SIZE=1 DCP_SIZE=16 \
      OUTPUT_DIR=profiles/16dev/bs${BS}/setup4_dcp16 \
      $SCRIPT
done
