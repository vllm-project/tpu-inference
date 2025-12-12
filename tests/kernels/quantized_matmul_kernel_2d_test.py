# SPDX-License-Identifier: Apache-2.0
"""Tests for 2D Block-wise Quantized Matmul Kernel."""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from absl.testing import absltest, parameterized

# ==============================================================================
# Import Target Kernel
# ==============================================================================
from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d
from tpu_inference.kernels.quantized_matmul.util import next_multiple

jax.config.parse_flags_with_absl()

# Constants
EPS = 1e-6 

# ==============================================================================
# Python Ground Truth (The "Math" Logic)
# ==============================================================================

def simple_quantize_block(x_block, dtype):
    """
    Python-side Ground Truth Quantization.
    Quantizes a single vector (representing one Quant Group for one Row).
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
        # Fallback/Passthrough
        scale = jnp.array(1.0, dtype=jnp.float32)
        x_q = x_block
    
    return x_q, scale.astype(jnp.float32)

def quantize_entire_tensor(data, quant_group_size, dtype):
    """
    Helper to quantize a whole tensor group-wise.
    Handles shapes that are not perfect multiples of quant_group_size by padding.
    """
    original_shape = data.shape
    n_in = original_shape[-1]
    
    # 1. Pad to nearest multiple of quant_group_size
    padded_n_in = next_multiple(n_in, quant_group_size)
    if padded_n_in > n_in:
        padding = padded_n_in - n_in
        # Pad the last dimension
        pad_width = [(0, 0)] * (len(original_shape) - 1) + [(0, padding)]
        data_padded = jnp.pad(data, pad_width)
    else:
        data_padded = data

    # 2. Reshape to [..., Groups, GroupSize]
    padded_shape = data_padded.shape
    n_groups = padded_shape[-1] // quant_group_size
    data_reshaped = data_padded.reshape(*padded_shape[:-1], n_groups, quant_group_size)
    
    # 3. Quantize Blocks
    flat_blocks = data_reshaped.reshape(-1, quant_group_size)
    qs_list = []
    scales_list = []
    
    for i in range(flat_blocks.shape[0]):
        q, s = simple_quantize_block(flat_blocks[i], dtype)
        qs_list.append(q)
        scales_list.append(s)
        
    # 4. Reassemble
    # qs: [..., Groups, GroupSize]
    qs = jnp.stack(qs_list).reshape(data_reshaped.shape)
    
    # scales: [..., Groups]
    # Explicitly reshape to include n_groups to avoid broadcasting issues
    scales = jnp.stack(scales_list).reshape(*padded_shape[:-1], n_groups)
    
    # 5. Reconstruct Ground Truth (Float)
    # recon = q * scale
    recon_padded = qs.astype(jnp.float32) * scales[..., None]
    recon_padded = recon_padded.reshape(padded_shape)
    
    # Slice back to original shape for correctness comparison
    recon = recon_padded[..., :n_in]
    
    # 6. Prepare Kernel Inputs
    # data_q needs to match original shape (Kernel expects BxIn)
    data_q_padded = qs.reshape(padded_shape)
    data_q = data_q_padded[..., :n_in]
    
    # scales needs to keep the padding! 
    return recon, data_q, scales

def reference_matmul_loop(x_q, x_scales, w_q, w_scales, quant_group_size, out_dtype):
    """
    Mathematically correct loop-based accumulation using quantized inputs.
    """
    bs, n_in = x_q.shape
    n_out = w_q.shape[0]
    
    # We must pad locally to perform the block-wise math
    padded_n_in = next_multiple(n_in, quant_group_size)
    n_groups = padded_n_in // quant_group_size
    
    # Pad inputs for the loop
    if padded_n_in > n_in:
        pad_len = padded_n_in - n_in
        x_q_pad = jnp.pad(x_q, ((0,0), (0, pad_len)))
        w_q_pad = jnp.pad(w_q, ((0,0), (0, pad_len)))
    else:
        x_q_pad = x_q
        w_q_pad = w_q
        
    out_accum = jnp.zeros((bs, n_out), dtype=jnp.float32)
    
    for g in range(n_groups):
        start = g * quant_group_size
        end = (g + 1) * quant_group_size
        
        # Load Slices (cast to float32 for compute)
        x_slice = x_q_pad[:, start:end].astype(jnp.float32)
        w_slice = w_q_pad[:, start:end].astype(jnp.float32)
        
        # Load Scales
        sx = x_scales[:, g][:, None] # [Batch, 1]
        sw = w_scales[:, g][None, :] # [1, Out]
        
        # Dot Product
        dot = jnp.dot(x_slice, w_slice.T) 
        
        # Dequantize & Accumulate
        out_accum += dot * sx * sw
        
    return out_accum.astype(out_dtype)

def get_tolerances(dtype, q_dtype):
    """Returns (rtol, atol) tuned for quantization noise."""
    if q_dtype == jnp.float8_e4m3fn: 
        return 0.10, 0.20
    elif dtype == jnp.bfloat16: 
        return 0.05, 0.35 
    return 0.05, 0.25 


# ==============================================================================
# Test Suite
# ==============================================================================

@jtu.with_config(jax_numpy_dtype_promotion="standard")
class TestQuantizedMatmul2D(jtu.JaxTestCase):
    
    @parameterized.named_parameters(
        ("bf16_int8", jnp.bfloat16, jnp.int8),
        ("fp32_int8", jnp.float32, jnp.int8),
        ("bf16_fp8", jnp.bfloat16, jnp.float8_e4m3fn),
    )
    def test_kernel_correctness(self, dtype, q_dtype):
        """Standard random data correctness test."""
        bs, n_in, n_out = 32, 256, 128
        quant_group_size = 128
        
        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)

        data_x = jax.random.uniform(k1, (bs, n_in), dtype=dtype, minval=-1.0, maxval=1.0)
        data_w = jax.random.uniform(k2, (n_out, n_in), dtype=dtype, minval=-1.0, maxval=1.0)
        
        _, x_q, x_scales = quantize_entire_tensor(data_x, quant_group_size, q_dtype)
        _, w_q, w_scales = quantize_entire_tensor(data_w, quant_group_size, q_dtype)

        # 1. Ground Truth
        out_ref = reference_matmul_loop(x_q, x_scales, w_q, w_scales, quant_group_size, dtype)

        # 2. Kernel
        out_kernel = quantized_matmul_2d(
            data_x, w_q, w_scales, output_load_size=256, quant_dtype=q_dtype
        )

        rtol, atol = get_tolerances(dtype, q_dtype)
        self.assertAllClose(out_kernel, out_ref, rtol=rtol, atol=atol)

    def test_skewed_data_robustness(self):
        """Verifies 2D Quantization handles outliers correctly."""
        dtype = jnp.float32
        q_dtype = jnp.int8
        bs, n_in, n_out = 64, 1024, 64
        quant_group_size = 128
        
        k1, k2 = jax.random.split(jax.random.key(100))
        
        x = jax.random.normal(k1, (bs, n_in), dtype=dtype)
        w = jax.random.normal(k2, (n_out, n_in), dtype=dtype)
        
        # Inject Sparse Outliers
        outlier_val = 500.0
        row_indices = jnp.arange(n_out)
        col_index = 150 # Inside group 1 (128-255)
        w = w.at[row_indices, col_index].set(outlier_val)

        w_recon_2d, w_q_2d, w_scales_2d = quantize_entire_tensor(w, quant_group_size, q_dtype)

        # Inputs
        w_scale_input = w_scales_2d 
        w_q_flat = w_q_2d.astype(jnp.int8).reshape(n_out, n_in)
        
        # Ground Truth
        x_recon_2d, _, _ = quantize_entire_tensor(x, quant_group_size, q_dtype)
        out_math_2d = jnp.dot(x_recon_2d, w_recon_2d.T)
        
        # Kernel
        out_kernel = quantized_matmul_2d(
            x, w_q_flat, w_scale_input, output_load_size=256, quant_dtype=q_dtype
        )
        
        self.assertAllClose(out_kernel, out_math_2d, rtol=0.05, atol=0.5)

    @parameterized.named_parameters(
        ("prime_shapes", 7, 137, 263),
        ("small_shapes", 1, 128, 128),
        ("large_uneven", 129, 2050, 257),
    )
    def test_padding_shapes(self, bs, n_in, n_out):
        """Tests handling of awkward shapes requiring padding."""
        dtype = jnp.float32
        q_dtype = jnp.int8
        quant_group_size = 128
        
        k1, k2 = jax.random.split(jax.random.key(99))
        x = jax.random.uniform(k1, (bs, n_in), dtype=dtype)
        w_raw = jax.random.uniform(k2, (n_out, n_in), dtype=dtype)
        
        # Quantize W (Helper now handles padding internally)
        _, w_q, w_scales = quantize_entire_tensor(w_raw, quant_group_size, q_dtype)
        
        # Ensure test inputs match kernel expectations
        self.assertEqual(w_q.shape, (n_out, n_in))
        
        out = quantized_matmul_2d(
            x, w_q, w_scales, output_load_size=256, quant_dtype=q_dtype
        )
        
        self.assertEqual(out.shape, (bs, n_out))

    @parameterized.named_parameters(
        ("batch_8_boundary", 8),
        ("batch_9_boundary", 9),   # Triggers padding to 16
        ("batch_32_boundary", 32),
        ("batch_33_boundary", 33), # Triggers padding to 64
        ("batch_128_boundary", 128),
        ("batch_129_boundary", 129), # Triggers padding to 256
    )
    def test_dispatcher_heuristics(self, bs):
        """
        Verifies that the heuristic logic for selecting Batch Tile Size 
        works correctly at the boundaries.
        """
        n_in, n_out = 1024, 1024
        dtype = jnp.float32
        q_dtype = jnp.int8
        
        k1, k2 = jax.random.split(jax.random.key(0))
        x = jax.random.normal(k1, (bs, n_in), dtype=dtype)
        w = jax.random.normal(k2, (n_out, n_in), dtype=dtype)
        
        _, w_q, w_scales = quantize_entire_tensor(w, 128, q_dtype)
        
        out = quantized_matmul_2d(
            x, w_q, w_scales, output_load_size=256, quant_dtype=q_dtype
        )
        self.assertEqual(out.shape, (bs, n_out))

    def test_zero_and_nan_inputs(self):
        """
        Sanity check: Zeros should produce Zeros. NaNs should not hang the TPU.
        """
        bs, n_in, n_out = 16, 256, 128
        dtype = jnp.bfloat16
        q_dtype = jnp.int8
        
        # Case 1: All Zeros
        x_zeros = jnp.zeros((bs, n_in), dtype=dtype)
        w_zeros = jnp.zeros((n_out, n_in), dtype=dtype)
        _, w_q, w_s = quantize_entire_tensor(w_zeros, 128, q_dtype)
        
        out = quantized_matmul_2d(x_zeros, w_q, w_s, 128, q_dtype)
        self.assertAllClose(out, jnp.zeros_like(out), atol=1e-6)
        
        # Case 2: NaNs in Input (Should propagate or result in 0 depending on quantization)
        x_nan = x_zeros.at[0, 0].set(jnp.nan)
        out_nan = quantized_matmul_2d(x_nan, w_q, w_s, 128, q_dtype)
        
        # We just check it returns successfully. 
        # Quantization max(abs(nan)) behavior can vary, but usually results in NaN or garbage.
        # The key is that the kernel finishes.
        out_nan.block_until_ready() 

    def test_accumulation_depth(self):
        """
        Stress Test: Large K dimension to verify accumulator stability.
        If the accumulator is too small, this sum will overflow or lose precision.
        """
        bs, n_in, n_out = 4, 16384, 128 # Large In dimension
        dtype = jnp.float32
        q_dtype = jnp.int8
        
        # Create inputs that sum to a known large value
        # Activations = 1.0, Weights = 1.0 -> Sum should be roughly n_in
        x = jnp.ones((bs, n_in), dtype=dtype)
        w = jnp.ones((n_out, n_in), dtype=dtype)
        
        _, w_q, w_s = quantize_entire_tensor(w, 128, q_dtype)
        
        out = quantized_matmul_2d(x, w_q, w_s, 128, q_dtype)
        
        # Expected value is roughly n_in (allow for some quantization noise)
        expected = jnp.full((bs, n_out), n_in, dtype=dtype)
        self.assertAllClose(out, expected, rtol=0.05)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())