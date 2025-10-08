#!/bin/bash

# Install dependencies
pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

TPU_INFERENCE_DIR="/workspace/tpu_inference/"

# RPA v3 test files - add new tests here
RPA_V3_TESTS=(
    "tests/kernels/ragged_paged_attention_kernel_v3_test.py"
    "tests/layers/attention/test_deepseek_v3_attention.py"
)

# Convert array to space-separated string for pytest
FULL_PATHS=()
for test in "${RPA_V3_TESTS[@]}"; do
    FULL_PATHS+=("$TPU_INFERENCE_DIR/$test")
done

# Run all tests in a single pytest command
pytest "${FULL_PATHS[@]}"
