#!/bin/bash
set -e

BASEDIR="/home/wyzhang_google_com/mnt/ullm"
VENV="${BASEDIR}/venv/vllm/bin/activate"

source $VENV

cd ${BASEDIR}/tpu-inference/wyzhang

echo "Running gather baseline..."
MOE_GATHER_MODE="gather" ./bench_serve.sh --iter=3 --input_len=1024 --output_len=8192 --target_dir=gather-gather

echo "Running onehot..."
MOE_GATHER_MODE="onehot" ./bench_serve.sh --iter=3 --input_len=1024 --output_len=8192 --target_dir=onehot-onehot

echo "Running fence..."
MOE_GATHER_MODE="fence" ./bench_serve.sh --iter=3 --input_len=1024 --output_len=8192 --target_dir=gather-fence

echo "All experiments finished."
