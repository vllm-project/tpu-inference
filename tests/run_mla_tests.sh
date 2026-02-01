#!/bin/bash
# Run MLA comparison tests locally
# Usage: ./tests/run_mla_tests.sh
#
# Requirements:
#   pip install numpy torch jax jaxlib
# Or with uv:
#   uv pip install numpy torch jax jaxlib

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "MLA Test Suite"
echo "========================================"
echo ""

# Test 1: JAX vs PyTorch comparison
echo "Test 1: JAX vs PyTorch MLA Implementation Comparison"
echo "------------------------------------------------------"
python tests/layers/vllm/mla/mla_compare_jax_torch.py
echo ""

# Test 2: Projection matrix tests
echo "Test 2: W_K/W_V Projection Matrix Tests"
echo "------------------------------------------------------"
python tests/layers/vllm/mla/mla_test_projections.py
echo ""

# Test 3: KV cache and metadata tests
echo "Test 3: KV Cache and Metadata Tests"
echo "------------------------------------------------------"
python tests/layers/vllm/mla/mla_test_cache_and_metadata.py
echo ""

# Test 4: Kernel tests (optional - needs tpu_inference module)
echo "Test 4: MLA Kernel Tests"
echo "------------------------------------------------------"
python tests/layers/vllm/mla/mla_test_kernel.py || echo "Note: Kernel tests may fail if tpu_inference is not installed"
echo ""

# Test 5: Full layer tests (optional - needs tpu_inference module)
echo "Test 5: Full MLA Layer Tests"
echo "------------------------------------------------------"
python tests/layers/vllm/mla/mla_test_full_layer.py || echo "Note: Full layer tests may fail if tpu_inference is not installed"
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
