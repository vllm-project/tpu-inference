#!/bin/bash
set -euo pipefail

echo "=== Environment Info ==="
date -u
python3 -c "import jax; print('JAX devices:', jax.devices())"

cd /home/wenxindong_google_com/tpu-inference

echo "=== 1. Running Unit Tests for write_decode_kv ==="
pytest -v -s tests/kernels/rpa_v3_cp/test_write_kv.py

echo "=== 2. Running Attention Forward Benchmark (DCP vs TP8) ==="
export KV_LENS_K="4,32,128,256"
export BENCH=50
python3 tests/layers/common/benchmark_dcp_forward_perf.py

echo "=== DCP Unit Tests and Layer Benchmark Complete ==="
