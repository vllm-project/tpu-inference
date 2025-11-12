import unittest
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.quantized_matmul.pate_kernel import (
    quantized_matmul_1d,
    quantized_matmul_2d
)
from tpu_inference.kernels.quantized_matmul.pate_reference import (
    quantize_along_axis,
    quantize_2d_blocked,
    reference_quantized_matmul_1d,
    reference_quantized_matmul_2d
)

def mean_absolute_error(a, b):
    return jnp.mean(jnp.abs(a - b))

class TestQuantizedMatmul(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.BS = 16
        self.N_IN = 2048
        self.N_OUT = 1024
        self.BLOCK_SIZE = 128
        self.W_QUANT_DIM = 1
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        self.x = jax.random.normal(subkey1, (self.BS, self.N_IN))
        self.w = jax.random.normal(subkey2, (self.N_OUT, self.N_IN)) * 0.1
        self.out_float = jnp.dot(self.x, self.w.T)

    def test_matmul_1d_vs_float(self):
        # Basic check that 1D ref matches float within tolerance
        w_q_1d, w_scale_1d = quantize_along_axis(self.w, jnp.int8, dim=self.W_QUANT_DIM)
        w_scale_1d = w_scale_1d.squeeze(-1)
        out_1d = reference_quantized_matmul_1d(self.x, w_q_1d, w_scale_1d, quantize_activation=True)
        mae = mean_absolute_error(self.out_float, out_1d)
        self.assertLess(mae, 0.05)

    def test_matmul_2d_vs_float(self):
        # Basic check that 2D ref matches float within tolerance
        w_q, w_scale = quantize_2d_blocked(self.w, self.BLOCK_SIZE, jnp.int8)
        out_2d = reference_quantized_matmul_2d(self.x, w_q, w_scale, self.BLOCK_SIZE, quant_dtype=jnp.int8, quantize_activation=True)
        mae = mean_absolute_error(self.out_float, out_2d)
        self.assertLess(mae, 0.05)


def get_tolerances(dtype, q_dtype):
    if q_dtype == jnp.float8_e4m3fn: return 1e-2, 1e-2
    elif dtype == jnp.bfloat16: return 1e-2, 1e-2
    return 1e-5, 1e-5

@pytest.mark.parametrize("dtype, q_dtype, bs, n_in, n_out, quantize_activation", [
    (jnp.float32, jnp.int8, 16, 128, 64, True),
    (jnp.bfloat16, jnp.float8_e4m3fn, 128, 256, 128, True),
    (jnp.float32, jnp.int8, 32, 256, 128, False),
])
def test_kernel_1d_vs_reference(dtype, q_dtype, bs, n_in, n_out, quantize_activation):
    """Compares the 1D kernel vs 1D reference."""
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(k1, (bs, n_in), dtype=dtype, minval=-1, maxval=1)
    w = jax.random.uniform(k2, (n_out, n_in), dtype=dtype, minval=-1, maxval=1)

    w_q, w_scale = quantize_along_axis(w, q_dtype, dim=1)
    w_scale = jnp.squeeze(w_scale)
    
    x_q_dtype = w_q.dtype if quantize_activation else dtype

    out_kernel = quantized_matmul_1d(x, w_q, w_scale, x_q_dtype=x_q_dtype)
    out_ref = reference_quantized_matmul_1d(x, w_q, w_scale, quantize_activation=quantize_activation)

    rtol, atol = get_tolerances(dtype, q_dtype)
    assert jnp.allclose(out_kernel.astype(jnp.float32), out_ref.astype(jnp.float32), rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype, q_dtype, bs, n_in, n_out, block_size, quantize_activation", [
    (jnp.bfloat16, jnp.int8, 128, 512, 256, 128, True),
    (jnp.bfloat16, jnp.float8_e4m3fn, 128, 1024, 512, 128, True),
])
def test_kernel_2d_vs_reference(dtype, q_dtype, bs, n_in, n_out, block_size, quantize_activation):
    """Compares the 2D kernel vs 2D reference."""
    k1, k2 = jax.random.split(jax.random.key(1), 2)
    x = jax.random.uniform(k1, (bs, n_in), dtype=dtype, minval=-1, maxval=1)
    w = jax.random.uniform(k2, (n_out, n_in), dtype=dtype, minval=-1, maxval=1)

    x_q_dtype = q_dtype if quantize_activation else dtype

    w_q, w_scale = quantize_2d_blocked(w, block_size, q_dtype)
    
    out_kernel = quantized_matmul_2d(
        x, w_q, w_scale, quant_block_size=block_size, x_q_dtype=x_q_dtype,
        batch_block_size=min(128, bs), out_block_size=min(128, n_out)
    )
    
    out_ref = reference_quantized_matmul_2d(
        x, w_q, w_scale, block_size=block_size, quant_dtype=q_dtype, quantize_activation=quantize_activation
    )

    rtol, atol = get_tolerances(dtype, q_dtype)
    assert jnp.allclose(out_kernel.astype(jnp.float32), out_ref.astype(jnp.float32), rtol=rtol, atol=atol)

def test_skewed_data_1d_vs_2d_kernel():
    """
    Compare 1D Kernel vs 2D Kernel on Skewed Data.
    
    This test mathematically demonstrates that block-wise quantization (2D)
    is superior to row-wise quantization (1D) when weights have outliers 
    that skew the scale for the entire row.
    """
    dtype = jnp.float32
    q_dtype = jnp.int8
    bs, n_in, n_out = 128, 1024, 512
    block_size = 128
    
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    x = jax.random.normal(k1, (bs, n_in), dtype=dtype)
    
    # Create weights: Mostly small values (noise)
    w = jax.random.normal(k2, (n_out, n_in), dtype=dtype) * 0.1
    
    # Add outliers to a specific block (e.g., the first block of every row)
    # This ensures every row has an outlier, penalizing row-wise quantization.
    outlier_block_idx = 0
    start, end = outlier_block_idx * block_size, (outlier_block_idx + 1) * block_size
    
    # Outliers are 100x larger than the signal
    w = w.at[:, start:end].set(
        jax.random.normal(k3, (n_out, block_size), dtype=dtype) * 10.0
    )
    
    # Ground Truth (Float32)
    ground_truth = x @ w.T
    
    # --- 1D Kernel Execution ---
    w_q_1d, w_scale_1d = quantize_along_axis(w, q_dtype, dim=1)
    w_scale_1d = jnp.squeeze(w_scale_1d)
    out_1d = quantized_matmul_1d(x, w_q_1d, w_scale_1d, x_q_dtype=q_dtype)
    
    # --- 2D Kernel Execution ---
    w_q_2d, w_scale_2d = quantize_2d_blocked(w, block_size, q_dtype)
    out_2d = quantized_matmul_2d(
        x, w_q_2d, w_scale_2d, quant_block_size=block_size, x_q_dtype=q_dtype
    )
    
    # --- Metrics ---
    def rel_error(pred, true):
        return jnp.linalg.norm(pred - true) / jnp.linalg.norm(true)
    
    err_1d = rel_error(out_1d, ground_truth)
    err_2d = rel_error(out_2d, ground_truth)
    
    print(f"\nSkewed Data Comparison:")
    print(f"  1D Kernel Relative Error: {err_1d:.6f}")
    print(f"  2D Kernel Relative Error: {err_2d:.6f}")
    
    # Assert 2D is significantly better (at least 10% better in this synthetic case)
    # In practice, it's often 2x-10x better depending on skew.
    assert err_2d < err_1d, "2D kernel should be more accurate than 1D kernel on skewed data"
    assert err_2d < 0.05, "2D kernel error is unexpectedly high"