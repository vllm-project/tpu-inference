# SPDX-License-Identifier: Apache-2.0
"""Tests for 2D quantized matmul kernel (Reference & High-Perf V7)."""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from absl.testing import absltest, parameterized

# ==============================================================================
# Import Target Kernels
# ==============================================================================
from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d, dispatch_w8a8_v7

jax.config.parse_flags_with_absl()

# Constants
F8_E4M3FN_MAX = 448.0 
EPS = 1e-6 

# ==============================================================================
# Helpers & Adapters
# ==============================================================================

def adapter_v7_kernel(
    activations, 
    weights_quantized, 
    weight_scales, 
    quant_block_size, 
    x_q_dtype, 
    batch_block_size=None, 
    out_block_size=128
):
    """Adapter to unify V7 signature with Reference signature."""
    return dispatch_w8a8_v7(
        activations, 
        weights_quantized, 
        weight_scales, 
        out_block_size=out_block_size, 
        quant_dtype=x_q_dtype
    )

def simple_quantize_block(x_block, dtype):
    """
    Python-side Ground Truth Quantization.
    Matches the logic inside the kernel (Reference & V7).
    """
    abs_max = jnp.max(jnp.abs(x_block))
    abs_max = jnp.maximum(abs_max, EPS)
    
    if dtype == jnp.int8:
        # Int8: Round Half Up strategy
        max_val = 127.0
        scale = abs_max / max_val
        x_scaled = x_block / scale
        x_q = jnp.floor(x_scaled + 0.5)
        x_q = jnp.clip(x_q, -128, 127).astype(jnp.int8)
        
    elif dtype == jnp.float8_e4m3fn:
        # FP8: Clip range, then cast (rounding handled by cast)
        max_val = 448.0
        scale = abs_max / max_val
        x_scaled = x_block / scale
        x_clipped = jnp.clip(x_scaled, -max_val, max_val)
        x_q = x_clipped.astype(jnp.float8_e4m3fn)
    else:
        scale = jnp.array(1.0, dtype=jnp.float32)
        x_q = x_block
    
    return x_q, scale.astype(jnp.float32)

def quantize_entire_tensor(data, block_size, dtype):
    """Helper to quantize a whole tensor block-wise."""
    shape = data.shape
    # Reshape to [..., Blocks, BlockSize]
    n_blocks = shape[-1] // block_size
    data_reshaped = data.reshape(*shape[:-1], n_blocks, block_size)
    
    # Flatten to list of blocks for processing
    flat_blocks = data_reshaped.reshape(-1, block_size)
    qs_list = []
    scales_list = []
    
    for i in range(flat_blocks.shape[0]):
        q, s = simple_quantize_block(flat_blocks[i], dtype)
        qs_list.append(q)
        scales_list.append(s)
        
    # Reassemble
    # qs: [..., Blocks, BlockSize]
    qs = jnp.stack(qs_list).reshape(data_reshaped.shape)
    
    # scales: [..., Blocks]
    # Note: No trailing singleton dimension here.
    scales = jnp.stack(scales_list).reshape(*data_reshaped.shape[:-1])
    
    # Reconstructed float values for Ground Truth comparison
    # recon = q * scale (broadcasting scale across block dim)
    recon = qs.astype(jnp.float32) * scales[..., None]
    recon = recon.reshape(shape)
    
    # Quantized components for Kernel input
    # data_q: [..., In]
    data_q = qs.reshape(shape)
    
    return recon, data_q, scales

def get_tolerances(dtype, q_dtype):
    """Returns (rtol, atol) tuned for quantization noise."""
    if q_dtype == jnp.float8_e4m3fn: 
        return 0.10, 0.20
    elif dtype == jnp.bfloat16: 
        return 0.05, 0.35 
    return 0.05, 0.25 


# ==============================================================================
# Ground Truth Implementation
# ==============================================================================

def reference_matmul_loop(x_q, x_scales, w_q, w_scales, block_size, out_dtype):
    """Mathematically correct loop-based accumulation using quantized inputs."""
    bs, n_in = x_q.shape
    n_out = w_q.shape[0]
    n_blocks = n_in // block_size
    
    out_accum = jnp.zeros((bs, n_out), dtype=jnp.float32)
    
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size
        
        # Load Slices (cast to float32 for compute)
        x_slice = x_q[:, start:end].astype(jnp.float32)
        w_slice = w_q[:, start:end].astype(jnp.float32)
        
        # Load Scales
        sx = x_scales[:, b][:, None] # [Batch, 1]
        sw = w_scales[:, b][None, :] # [1, Out]
        
        # Dot Product
        dot = jnp.dot(x_slice, w_slice.T) 
        
        # Dequantize & Accumulate
        out_accum += dot * sx * sw
        
    return out_accum.astype(out_dtype)


# ==============================================================================
# Test Suite
# ==============================================================================

@jtu.with_config(jax_numpy_dtype_promotion="standard")
class TestQuantizedMatmulUnified(jtu.JaxTestCase):
    
    @parameterized.named_parameters(
        ("ref_bf16_int8", quantized_matmul_2d, jnp.bfloat16, jnp.int8),
        ("v7_bf16_int8", adapter_v7_kernel, jnp.bfloat16, jnp.int8),
        ("ref_fp32_int8", quantized_matmul_2d, jnp.float32, jnp.int8),
        ("v7_fp32_int8", adapter_v7_kernel, jnp.float32, jnp.int8),
        ("ref_bf16_fp8", quantized_matmul_2d, jnp.bfloat16, jnp.float8_e4m3fn),
        ("v7_bf16_fp8", adapter_v7_kernel, jnp.bfloat16, jnp.float8_e4m3fn),
    )
    def test_kernel_correctness(self, kernel_fn: Callable, dtype, q_dtype):
        """Standard random data correctness test."""
        bs, n_in, n_out = 32, 256, 128
        block_size = 128
        
        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)

        # Generate inputs
        data_x = jax.random.uniform(k1, (bs, n_in), dtype=dtype, minval=-1.0, maxval=1.0)
        data_w = jax.random.uniform(k2, (n_out, n_in), dtype=dtype, minval=-1.0, maxval=1.0)
        
        # Pre-quantize inputs for Ground Truth
        # (Kernel quantizes X internally, so we simulate that for GT)
        _, x_q, x_scales = quantize_entire_tensor(data_x, block_size, q_dtype)
        _, w_q, w_scales = quantize_entire_tensor(data_w, block_size, q_dtype)

        # 1. Ground Truth
        out_ref = reference_matmul_loop(x_q, x_scales, w_q, w_scales, block_size, dtype)

        # 2. Kernel Execution
        # Pass RAW x (kernel quantizes it). Pass Quantized W (standard API).
        out_kernel = kernel_fn(
            data_x, w_q, w_scales, quant_block_size=block_size, x_q_dtype=q_dtype,
            batch_block_size=128, out_block_size=128
        )

        rtol, atol = get_tolerances(dtype, q_dtype)
        self.assertAllClose(out_kernel, out_ref, rtol=rtol, atol=atol)

    @parameterized.named_parameters(
        ("ref", quantized_matmul_2d),
        ("v7", adapter_v7_kernel),
    )
    def test_skewed_data_robustness(self, kernel_fn: Callable):
        """
        Demonstrates 2D Quantization superiority using Sparse Outliers.
        We inject single massive values into specific blocks.
        """
        dtype = jnp.float32
        q_dtype = jnp.int8
        bs, n_in, n_out = 64, 1024, 64
        block_size = 128
        
        k1, k2, k3 = jax.random.split(jax.random.key(100), 3)
        
        x = jax.random.normal(k1, (bs, n_in), dtype=dtype)
        w = jax.random.normal(k2, (n_out, n_in), dtype=dtype)
        
        # Inject Sparse Outliers (one per row, in the second block)
        outlier_val = 500.0
        row_indices = jnp.arange(n_out)
        col_index = 150 # Inside block 1 (128-255)
        w = w.at[row_indices, col_index].set(outlier_val)

        # --- Metric 1: Weight Reconstruction Error (Frobenius Norm) ---
        # This is purely algorithmic: how well does the quantization represent W?
        
        # 1D Quantization (Row-wise Baseline)
        w_max_1d = jnp.max(jnp.abs(w), axis=1, keepdims=True) / 127.0
        w_q_1d = jnp.clip(jnp.round(w / w_max_1d), -128, 127)
        w_recon_1d = w_q_1d * w_max_1d
        
        # 2D Quantization (Block-wise Target)
        w_recon_2d, w_q_2d, w_scales_2d = quantize_entire_tensor(w, block_size, q_dtype)

        err_norm_1d = jnp.linalg.norm(w - w_recon_1d)
        err_norm_2d = jnp.linalg.norm(w - w_recon_2d)
        
        print(f"\n[{kernel_fn.__name__}] Weight Reconstruction Error:")
        print(f"  1D (Row-wise): {err_norm_1d:.4f}")
        print(f"  2D (Block-wise): {err_norm_2d:.4f}")
        
        # 2D error should be significantly lower (outlier only ruins 1/8th of the blocks)
        self.assertLess(err_norm_2d, err_norm_1d * 0.6)

        # --- Metric 2: Kernel Output Consistency ---
        # Run Kernel and verify it matches the 2D Math logic
        
        # Prepare inputs
        # w_scales_2d is (Out, Blocks). Do NOT squeeze.
        w_scale_input = w_scales_2d 
        w_q_flat = w_q_2d.astype(jnp.int8).reshape(n_out, n_in)
        
        # Ground Truth for Kernel (using quantized X and W)
        x_recon_2d, _, _ = quantize_entire_tensor(x, block_size, q_dtype)
        out_math_2d = jnp.dot(x_recon_2d, w_recon_2d.T)
        
        out_kernel = kernel_fn(
            x, w_q_flat, w_scale_input, quant_block_size=block_size, x_q_dtype=q_dtype
        )
        
        # Verify correctness
        self.assertAllClose(out_kernel, out_math_2d, rtol=0.05, atol=0.5)

    @parameterized.named_parameters(
        ("ref_prime", quantized_matmul_2d, 7, 137, 263),
        ("v7_prime", adapter_v7_kernel, 7, 137, 263),
    )
    def test_padding_shapes(self, kernel_fn: Callable, bs, n_in, n_out):
        """Tests handling of awkward shapes requiring padding."""
        dtype = jnp.float32
        q_dtype = jnp.int8
        block_size = 128
        k1, k2 = jax.random.split(jax.random.key(99))
        x = jax.random.uniform(k1, (bs, n_in), dtype=dtype)
        
        # Generate Padded Weights (Standard practice: W is stored padded)
        padded_k = ((n_in + block_size - 1) // block_size) * block_size
        _, w_q_pad, w_scales_pad = quantize_entire_tensor(
            jax.random.uniform(k2, (n_out, padded_k), dtype=dtype),
            block_size, q_dtype
        )
        
        # Trim input to simulate API usage
        w_q_input = w_q_pad[:, :n_in]
        
        out = kernel_fn(
            x, w_q_input, w_scales_pad, quant_block_size=block_size, x_q_dtype=q_dtype
        )
        self.assertEqual(out.shape, (bs, n_out))

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())