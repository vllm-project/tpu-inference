# SPDX-License-Identifier: Apache-2.0
"""Tests for 2D Block-wise Quantized Matmul Kernel."""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from absl.testing import absltest, parameterized

from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d, quantize_weights_2d
from tpu_inference.kernels.quantized_matmul.util import next_multiple

jax.config.parse_flags_with_absl()

EPS = 1e-6 
INT8_MIN, INT8_MAX = -128.0, 127.0
FP8_MIN, FP8_MAX = -448.0, 448.0

def simple_quantize_block(input_block, dtype):
    """
    Python-side Ground Truth Quantization.
    Quantizes a single vector (representing one Quant Group for one Row).
    """
    abs_max = jnp.max(jnp.abs(input_block))
    abs_max = jnp.maximum(abs_max, EPS)
    
    if dtype == jnp.int8:
        scale = abs_max / INT8_MAX 
        block_scaled = input_block / scale
        block_quant_f32 = jnp.floor(block_scaled + 0.5)
        block_quantized = jnp.clip(block_quant_f32, INT8_MIN, INT8_MAX).astype(jnp.int8)
    elif dtype == jnp.float8_e4m3fn:
        scale = abs_max / FP8_MAX
        block_scaled = input_block / scale
        block_clipped = jnp.clip(block_scaled, FP8_MIN, FP8_MAX)
        block_quantized = block_clipped.astype(jnp.float8_e4m3fn)
    else:
        scale = jnp.array(1.0, dtype=jnp.float32)
        block_quantized = input_block
        
    return block_quantized, scale.astype(jnp.float32)

def quantize_entire_tensor(input_tensor, quant_group_size, dtype):
    """
    Helper to quantize a whole tensor group-wise using independent logic.
    """
    original_shape = input_tensor.shape
    n_input_features = original_shape[-1]
    
    # 1. Pad to nearest multiple of quant_group_size
    padded_n_input = next_multiple(n_input_features, quant_group_size)
    if padded_n_input > n_input_features:
        padding = padded_n_input - n_input_features
        pad_width = [(0, 0)] * (len(original_shape) - 1) + [(0, padding)]
        tensor_padded = jnp.pad(input_tensor, pad_width)
    else:
        tensor_padded = input_tensor

    # 2. Reshape to [..., Groups, GroupSize]
    padded_shape = tensor_padded.shape
    n_groups = padded_shape[-1] // quant_group_size
    tensor_reshaped = tensor_padded.reshape(*padded_shape[:-1], n_groups, quant_group_size)
    
    # 3. Quantize individual blocks
    flat_blocks = tensor_reshaped.reshape(-1, quant_group_size)
    quantized_list = []
    scales_list = []
    
    for i in range(flat_blocks.shape[0]):
        q, s = simple_quantize_block(flat_blocks[i], dtype)
        quantized_list.append(q)
        scales_list.append(s)
        
    # 4. Reassemble
    quantized_tensor = jnp.stack(quantized_list).reshape(tensor_reshaped.shape)
    scales_tensor = jnp.stack(scales_list).reshape(*padded_shape[:-1], n_groups)
    
    # 5. Reconstruct floating point ground truth
    reconstruction_padded = quantized_tensor.astype(jnp.float32) * scales_tensor[..., None]
    reconstruction_padded = reconstruction_padded.reshape(padded_shape)
    reconstruction = reconstruction_padded[..., :n_input_features]
    
    # 6. Prepare kernel inputs
    quantized_data_padded = quantized_tensor.reshape(padded_shape)
    quantized_data_sliced = quantized_data_padded[..., :n_input_features]
    
    return reconstruction, quantized_data_sliced, scales_tensor

def reference_matmul_loop(
    activations_quantized, activation_scales, 
    weights_quantized, weight_scales, 
    quant_group_size, out_dtype
):
    """
    Mathematically correct loop-based accumulation using quantized inputs.
    Simulates the math: sum( (act_q * weight_q) * (act_scale * weight_scale) )
    """
    batch_size, n_input_features = activations_quantized.shape
    n_output_features = weights_quantized.shape[0]
    
    padded_n_input = next_multiple(n_input_features, quant_group_size)
    n_groups = padded_n_input // quant_group_size
    
    if padded_n_input > n_input_features:
        pad_len = padded_n_input - n_input_features
        act_q_pad = jnp.pad(activations_quantized, ((0,0), (0, pad_len)), constant_values=0)
        w_q_pad = jnp.pad(weights_quantized, ((0,0), (0, pad_len)), constant_values=0)
    else:
        act_q_pad = activations_quantized
        w_q_pad = weights_quantized
        
    output_accum = jnp.zeros((batch_size, n_output_features), dtype=jnp.float32)
    
    for g in range(n_groups):
        start = g * quant_group_size
        end = (g + 1) * quant_group_size
        
        act_slice = act_q_pad[:, start:end].astype(jnp.float32)
        weight_slice = w_q_pad[:, start:end].astype(jnp.float32)
        
        scale_act = activation_scales[:, g][:, None]   # [Batch, 1]
        scale_weight = weight_scales[:, g][None, :]    # [1, Out]
        
        dot = jnp.dot(act_slice, weight_slice.T) 
        
        output_accum += dot * scale_act * scale_weight
        
    return output_accum.astype(out_dtype)

def get_tolerances(dtype, q_dtype):
    if q_dtype == jnp.float8_e4m3fn: return 0.10, 0.20
    elif dtype == jnp.bfloat16: return 0.05, 0.35 
    return 0.05, 0.25 


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class TestQuantizedMatmul2D(jtu.JaxTestCase):
    @parameterized.named_parameters(
        ("bf16_int8", jnp.bfloat16, jnp.int8),
        ("bf16_fp8", jnp.bfloat16, jnp.float8_e4m3fn),
    )
    def test_kernel_correctness(self, dtype, q_dtype):
        """
        Basic correctness test
        """
        batch_size, n_input_features, n_output_features = 32, 256, 128
        quant_group_size = 128
        
        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)

        activations = jax.random.uniform(k1, (batch_size, n_input_features), dtype=dtype, minval=-1.0, maxval=1.0)
        weights = jax.random.uniform(k2, (n_output_features, n_input_features), dtype=dtype, minval=-1.0, maxval=1.0)
        
        # 1. Get ground Truth
        _, acts_q, acts_scales = quantize_entire_tensor(activations, quant_group_size, q_dtype)
        _, weights_q, weights_scales = quantize_entire_tensor(weights, quant_group_size, q_dtype)
        output_ref = reference_matmul_loop(
            acts_q, acts_scales, weights_q, weights_scales, quant_group_size, dtype
        )

        # 2. Get kernel results
        output_kernel = quantized_matmul_2d(
            activations, weights_q, weights_scales, output_load_size=128, quant_dtype=q_dtype
        )

        rtol, atol = get_tolerances(dtype, q_dtype)
        self.assertAllClose(output_kernel, output_ref, rtol=rtol, atol=atol)

    def test_quantize_weights_2d(self):
        """
        Ensures that `quantize_weights_2d` produces identical outputs to the ground truth
        """
        n_output_features, n_input_features = 128, 256
        quant_group_size = 128
        dtype = jnp.int8
        
        key = jax.random.key(0)
        weights = jax.random.normal(key, (n_output_features, n_input_features))
        
        _, truth_q, truth_s = quantize_entire_tensor(weights, quant_group_size, dtype)
        
        test_q, test_s = quantize_weights_2d(weights, quant_group_size, dtype)
        
        self.assertAllClose(truth_q, test_q)
        self.assertAllClose(truth_s, test_s)

    def test_skewed_data_robustness(self):
        """
        2D Block-wise Effectiveness Test.
        Demonstrates that 2D (Block-wise) quantization maintains significantly 
        higher accuracy than 1D (Row-wise) quantization when outliers are present.
        """
        dtype = jnp.float32
        q_dtype = jnp.int8
        batch_size, n_input_features, n_output_features = 64, 1024, 64
        quant_group_size = 128
        
        k1, k2 = jax.random.split(jax.random.key(100))
        activations = jax.random.normal(k1, (batch_size, n_input_features), dtype=dtype)
        weights = jax.random.normal(k2, (n_output_features, n_input_features), dtype=dtype)
        
        # 1. Inject a single outlier per row
        outlier_val = 500.0
        row_indices = jnp.arange(n_output_features)
        col_index = 150 # Inside group 1 (128-255)
        weights = weights.at[row_indices, col_index].set(outlier_val)

        # 2. Get ground truth
        output_true = jnp.dot(activations, weights.T)

        # 3. 1D Quantization Baseline (Row-wise)
        weights_max_1d = jnp.max(jnp.abs(weights), axis=1, keepdims=True)
        scale_1d = 127.0 / weights_max_1d
        weights_q_1d = jnp.clip(jnp.round(weights * scale_1d), -128, 127)
        weights_recon_1d = weights_q_1d * (1.0 / scale_1d)
        output_1d = jnp.dot(activations, weights_recon_1d.T)

        # 4. 2D Kernel Result
        _, weights_q_2d, weights_scales_2d = quantize_entire_tensor(weights, quant_group_size, q_dtype)
        output_kernel = quantized_matmul_2d(
            activations, weights_q_2d, weights_scales_2d, output_load_size=128, quant_dtype=q_dtype
        )
        
        # 5. Calculate error
        err_1d = jnp.linalg.norm(output_1d - output_true)
        err_2d = jnp.linalg.norm(output_kernel - output_true)
        norm_true = jnp.linalg.norm(output_true)
        rel_err_1d = err_1d / norm_true
        rel_err_2d = err_2d / norm_true
        
        print(f"\nSkewed Data Test:")
        print(f"  1D (Row-wise) Error: {err_1d:.4f} (Rel: {rel_err_1d:.2%})")
        print(f"  2D (Block-wise) Error: {err_2d:.4f} (Rel: {rel_err_2d:.2%})")
        
        # Assert 2D is significantly better (less than half the error)
        self.assertLess(err_2d, err_1d * 0.5)

    @parameterized.named_parameters(
        ("prime_shapes", 7, 137, 263),
        ("small_shapes", 1, 128, 128),
        ("large_uneven", 129, 2050, 257),
    )
    def test_padding_shapes(self, batch_size, n_input_features, n_output_features):
        """
        Padding Logic Test.
        Verifies correct handling of "awkward" shapes (primes, odd numbers).
        """
        dtype = jnp.float32
        q_dtype = jnp.int8
        quant_group_size = 128
        
        k1, k2 = jax.random.split(jax.random.key(99))
        activations = jax.random.uniform(k1, (batch_size, n_input_features), dtype=dtype)
        weights_raw = jax.random.uniform(k2, (n_output_features, n_input_features), dtype=dtype)
        
        _, weights_q, weights_scales = quantize_entire_tensor(weights_raw, quant_group_size, q_dtype)
        self.assertEqual(weights_q.shape, (n_output_features, n_input_features))
        
        output_kernel = quantized_matmul_2d(
            activations, weights_q, weights_scales, output_load_size=128, quant_dtype=q_dtype
        )
        self.assertEqual(output_kernel.shape, (batch_size, n_output_features))

    @parameterized.named_parameters(
        ("batch_8_boundary", 8),
        ("batch_9_boundary", 9),
        ("batch_32_boundary", 32),
        ("batch_33_boundary", 33),
        ("batch_128_boundary", 128),
        ("batch_129_boundary", 129),
    )
    def test_batch_boundaries(self, batch_size):
        """
        Verifies the logic for selecting `BatchLoadSize` works at boundaries.
        """
        n_input_features, n_output_features = 1024, 1024
        dtype = jnp.float32
        q_dtype = jnp.int8
        quant_group_size = 128
        output_load_size = 128
        
        k1, k2 = jax.random.split(jax.random.key(0))
        activations = jax.random.normal(k1, (batch_size, n_input_features), dtype=dtype)
        weights = jax.random.normal(k2, (n_output_features, n_input_features), dtype=dtype)
        
        _, weights_q, weights_scales = quantize_entire_tensor(weights, quant_group_size, q_dtype)
        
        output_kernel = quantized_matmul_2d(
            activations, weights_q, weights_scales, 
            output_load_size=output_load_size, quant_dtype=q_dtype
        )
        self.assertEqual(output_kernel.shape, (batch_size, n_output_features))

    def test_zero_and_nan_inputs(self):
        """
        Sanity check for edge case inputs.
        """
        batch_size, n_input_features, n_output_features = 16, 256, 128
        dtype = jnp.bfloat16
        q_dtype = jnp.int8
        quant_group_size = 128
        output_load_size = 128
        
        # Case 1: All Zeros
        activations_zeros = jnp.zeros((batch_size, n_input_features), dtype=dtype)
        weights_zeros = jnp.zeros((n_output_features, n_input_features), dtype=dtype)
        _, weights_q, weights_s = quantize_entire_tensor(weights_zeros, quant_group_size, q_dtype)
        
        output_zeros = quantized_matmul_2d(
            activations_zeros, weights_q, weights_s, 
            output_load_size=output_load_size, quant_dtype=q_dtype
        )
        self.assertAllClose(output_zeros, jnp.zeros_like(output_zeros), atol=1e-6)
        
        # Case 2: NaNs in Input
        activations_nan = activations_zeros.at[0, 0].set(jnp.nan)
        output_nan = quantized_matmul_2d(
            activations_nan, weights_q, weights_s, 
            output_load_size=output_load_size, quant_dtype=q_dtype
        )
        output_nan.block_until_ready() 

    def test_accumulation_stability(self):
        """
        Tests large reduction dimension to verify accumulator stability
        """
        batch_size, n_input_features, n_output_features = 4, 16384, 128 
        dtype = jnp.float32
        q_dtype = jnp.int8
        quant_group_size = 128
        output_load_size = 128
        
        activations = jnp.ones((batch_size, n_input_features), dtype=dtype)
        weights = jnp.ones((n_output_features, n_input_features), dtype=dtype)
        
        _, weights_q, weights_s = quantize_entire_tensor(weights, quant_group_size, q_dtype)
        
        output_kernel = quantized_matmul_2d(
            activations, weights_q, weights_s, 
            output_load_size=output_load_size, quant_dtype=q_dtype
        )
        
        expected = jnp.full((batch_size, n_output_features), n_input_features, dtype=dtype)
        self.assertAllClose(output_kernel, expected, rtol=0.05)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())