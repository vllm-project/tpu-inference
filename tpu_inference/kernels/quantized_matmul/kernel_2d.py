# SPDX-License-Identifier: Apache-2.0
"""
2D (Block-wise) Quantized Matrix Multiplication Kernel.
Optimized for 256x256 MXU architectures using Software Pipelining.
"""

import functools
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.util import next_multiple
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

# ==============================================================================
# Constants & Configuration
# ==============================================================================

QUANT_GROUP_SIZE = 128
PIPELINE_STEPS = 16
CACHE_LOAD_SIZE = QUANT_GROUP_SIZE * PIPELINE_STEPS

# ==============================================================================
# Helper Logic (VPU)
# ==============================================================================

def quantize_and_scale_group(
    act_group: jax.Array, 
    weight_scale: jax.Array, 
    quant_max: float, 
    dtype: jnp.dtype
) -> Tuple[jax.Array, jax.Array]:
    """
    Quantizes a specific group of input features and calculates the fused scale.
    """
    # 1. Cast to FP32.
    act_f32 = act_group.astype(jnp.float32)
    
    # 2. Compute Max per Row (Batch Item).
    act_max = jnp.max(jnp.abs(act_f32), axis=1, keepdims=True)
    act_max = jnp.maximum(act_max, 1e-6)

    # 3. Quantize Activations.
    scale_to_int = quant_max / act_max
    act_quant_f32 = jnp.floor((act_f32 * scale_to_int) + 0.5)

    if dtype == jnp.int8:
        act_quant = jnp.clip(act_quant_f32, -128.0, 127.0).astype(jnp.int8)
    else:
        act_quant = jnp.clip(act_quant_f32, -448.0, 448.0).astype(dtype)

    # 4. Compute Combined Scale.
    combined_scale = (act_max / quant_max) * weight_scale
    
    return act_quant, combined_scale


# ==============================================================================
# Optimized Pipeline Kernel
# ==============================================================================

def _fused_matmul_kernel(
    # --- Global Memory References ---
    activations_ref,    # [Batch, In]
    weights_ref,        # [Out, In]
    scales_ref,         # [Groups, Out] (Transposed)
    output_ref,         # [Batch, Out]
    
    # --- VMEM Scratchpad ---
    accumulator,        # [BatchTile, OutTile]
    
    # --- Static Constants ---
    *,
    batch_tile_size: int, 
    out_tile_size: int, 
    quant_dtype: jnp.dtype
):
    # 1. Initialize Accumulator
    accumulator[...] = jnp.zeros(
        (batch_tile_size, out_tile_size), dtype=jnp.float32
    )

    # 2. Config
    if quant_dtype == jnp.int8:
        quant_max = 127.0
        dot_dtype = jnp.int32 
    else:
        quant_max = 448.0
        dot_dtype = jnp.float32 

    # 3. Pipeline Stage
    def _process_cache_load(
        act_cache: jax.Array,    # [BatchTile, CacheLoadSize]
        weight_cache: jax.Array, # [OutTile, CacheLoadSize]
        scales_cache: jax.Array  # [PipelineSteps, OutTile]
    ):
        # --- PROLOGUE ---
        a_0 = act_cache[:, 0:QUANT_GROUP_SIZE] 
        w_0 = weight_cache[:, 0:QUANT_GROUP_SIZE]
        s_0 = scales_cache[0, :][None, :]

        # VPU: Fused Quantization
        a_q_curr, scale_curr = quantize_and_scale_group(
            a_0, s_0, quant_max, quant_dtype
        )
        
        # MXU
        dot_curr = jax.lax.dot_general(
            a_q_curr, w_0, (((1,), (1,)), ((), ())), preferred_element_type=dot_dtype
        )
        
        dot_prev, scale_prev = dot_curr, scale_curr

        # --- LOOP ---
        for step in range(1, PIPELINE_STEPS):
            start = step * QUANT_GROUP_SIZE
            end   = (step + 1) * QUANT_GROUP_SIZE
            
            # 1. Load Next
            a_next = act_cache[:, start:end]
            w_next = weight_cache[:, start:end]
            s_next = scales_cache[step, :][None, :]

            # 2. VPU: Quantize Next
            a_q_next, scale_next = quantize_and_scale_group(
                a_next, s_next, quant_max, quant_dtype
            )
            
            # 3. MXU
            dot_next = jax.lax.dot_general(
                a_q_next, w_next, (((1,), (1,)), ((), ())), preferred_element_type=dot_dtype
            )

            # 4. Accumulate Previous
            accumulator[...] += dot_prev.astype(jnp.float32) * scale_prev
            
            # 5. Shift
            dot_prev, scale_prev = dot_next, scale_next

        # --- EPILOGUE ---
        accumulator[...] += dot_prev.astype(jnp.float32) * scale_prev

    # 4. Emit Pipeline
    total_in = weights_ref.shape[1]
    n_cache_loads = total_in // CACHE_LOAD_SIZE

    pltpu.emit_pipeline(
        _process_cache_load,
        grid=(n_cache_loads,),
        in_specs=[
            # Act: [BatchTile, CacheLoad]
            pl.BlockSpec((batch_tile_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            # Weight: [OutTile, CacheLoad]
            pl.BlockSpec((out_tile_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            # Scales: [PipelineSteps, OutTile]
            pl.BlockSpec((PIPELINE_STEPS, out_tile_size), lambda i: (i, 0))
        ]
    )(activations_ref, weights_ref, scales_ref)

    # 5. Store
    output_ref[...] = accumulator[...].astype(output_ref.dtype)


# ==============================================================================
# Kernel Launch & Implementation
# ==============================================================================

@functools.partial(jax.jit, static_argnames=['out_tile_size', 'quant_dtype', 'batch_tile_size'])
def _quantized_matmul_impl(
    activations, weights_quantized, weight_scales, 
    out_tile_size, quant_dtype, batch_tile_size
):
    """
    Internal helper: Handles padding/alignment and launches the Pallas kernel.
    """
    bs, n_in = activations.shape
    n_out = weights_quantized.shape[0]
    
    # 1. Padding
    padded_bs = next_multiple(bs, batch_tile_size)
    padded_out = next_multiple(n_out, out_tile_size)
    padded_in = next_multiple(n_in, CACHE_LOAD_SIZE)
    
    # Pad Activations
    if padded_bs > bs or padded_in > n_in:
        pad_b = padded_bs - bs
        pad_in = padded_in - n_in
        activations = jnp.pad(activations, ((0, pad_b), (0, pad_in)))
        
    # Pad Weights and Scales
    if padded_out > n_out or padded_in > n_in:
        pad_out = padded_out - n_out
        pad_in = padded_in - n_in
        weights_quantized = jnp.pad(weights_quantized, ((0, pad_out), (0, pad_in)))
        
        # Scales Padding Correction:
        # We must calculate target groups based on PADDED input size.
        target_groups = padded_in // QUANT_GROUP_SIZE
        n_groups = weight_scales.shape[1]
        
        if target_groups > n_groups:
            # Note: We do not pad 'pad_out' to scales here if it was handled above or logic differs.
            # Usually weight_scales matches weights in dim 0.
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, target_groups - n_groups)), constant_values=1.0)
        elif pad_out > 0:
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, 0)), constant_values=1.0)

    # 2. Transpose Scales
    w_scale_t = weight_scales.T

    # 3. Grid Definition
    n_batch_tiles = padded_bs // batch_tile_size
    n_out_tiles = padded_out // out_tile_size
    grid = (n_batch_tiles, n_out_tiles)
    
    # 4. Block Specifications
    in_specs = [
        # Act: Load [BatchTile, AllFeatures]
        pl.BlockSpec((batch_tile_size, padded_in), lambda b, o: (b, 0)),
        # Wgt: Load [OutTile, AllFeatures]
        pl.BlockSpec((out_tile_size, padded_in), lambda b, o: (o, 0)),
        # Scl: Load [AllGroups, OutTile]
        pl.BlockSpec((w_scale_t.shape[0], out_tile_size), lambda b, o: (0, o)),
    ]

    out_spec = pl.BlockSpec((batch_tile_size, out_tile_size), lambda b, o: (b, o))

    # 5. Launch
    kernel = pl.pallas_call(
        functools.partial(
            _fused_matmul_kernel, 
            batch_tile_size=batch_tile_size, 
            out_tile_size=out_tile_size, 
            quant_dtype=quant_dtype
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_spec,
            grid=grid,
            scratch_shapes=[pltpu.VMEM((batch_tile_size, out_tile_size), jnp.float32)]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), activations.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_device_vmem_limit())
    )
    
    out = kernel(activations, weights_quantized, w_scale_t)
    return out[:bs, :n_out]


def quantized_matmul_2d(
    activations: jax.Array, 
    weights_quantized: jax.Array, 
    weight_scales: jax.Array, 
    out_block_size: int, 
    quant_dtype: Any
) -> jax.Array:
    """
    Main Entry Point for 2D Block-wise Quantized Matmul.
    
    Automatically adapts the Batch Tile Size based on the input batch size
    to maximize hardware occupancy.

    Args:
        activations: Input tensor [Batch, InputFeatures].
        weights_quantized: Weight tensor [OutputFeatures, InputFeatures].
        weight_scales: Scale tensor [OutputFeatures, InputFeatures // 128].
        out_block_size: Tiling size for output features (recommend 128 or 256).
        quant_dtype: Target quantization format (jnp.int8 or jnp.float8_e4m3fn).
    
    Returns:
        Result tensor [Batch, OutputFeatures] in original activation dtype.
    """
    batch_size = activations.shape[0]

    if batch_size <= 8:   tile_b = 8
    elif batch_size <= 16: tile_b = 16
    elif batch_size <= 32: tile_b = 32
    elif batch_size <= 64: tile_b = 64
    elif batch_size <= 128: tile_b = 128
    else: tile_b = 256

    return _quantized_matmul_impl(
        activations, 
        weights_quantized, 
        weight_scales, 
        out_tile_size=out_block_size, 
        quant_dtype=quant_dtype, 
        batch_tile_size=tile_b
    )