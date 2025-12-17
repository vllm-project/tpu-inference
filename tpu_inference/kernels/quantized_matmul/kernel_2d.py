# SPDX-License-Identifier: Apache-2.0
"""
2D (Block-wise) Quantized Matrix Multiplication Kernel.

Architecture:
  - Read-Quantize-Compute loop.
  - Weights are pre-quantized (offline).
  - Activations are quantized dynamically (online) in the VPU.
  - Accumulation happens in FP32.

TPU Hardware Mapping:
  - HBM: Stores inputs (BF16) and compressed weights (Int8/FP8).
  - VMEM: Caches blocks of data (CACHE_LOAD_SIZE).
  - VPU: Performs dynamic quantization (Cast -> Max -> Scale -> Clip).
  - MXU: Performs dot_product.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.util import next_multiple
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

# The number of quantized weights that share one scale
QUANT_GROUP_SIZE = 128

# Number of quantization groups for a single fetch from HBM.
GROUPS_PER_LOAD = 64

# The total amount of data loaded into VMEM at once
CACHE_LOAD_SIZE = QUANT_GROUP_SIZE * GROUPS_PER_LOAD

# Max and min of fp8 and int8 types for scaling
INT8_MIN, INT8_MAX = -128.0, 127.0
FP8_MIN, FP8_MAX = -448.0, 448.0


def quantize_weights_2d(
    weights: jax.Array,
    quant_group_size: int = QUANT_GROUP_SIZE,
    quant_dtype: jnp.dtype = jnp.int8
) -> Tuple[jax.Array, jax.Array]:
    """
    Offline utility to quantize weights in the format expected by the kernel.
    
    Args:
        weights: Input weight matrix [OutputFeatures, InputFeatures].
        quant_group_size: Size of quantization block (default 128).
        quant_dtype: Target dtype (jnp.int8 or jnp.float8_e4m3fn).

    Returns:
        weights_quantized: [OutputFeatures, InputFeatures] in target dtype.
        weight_scales: [OutputFeatures, Ceil(InputFeatures / GroupSize)] in float32.
    """
    n_out, n_in = weights.shape
    
    # 1. Pad Input Features to multiple of group size
    padded_in = next_multiple(n_in, quant_group_size)
    if padded_in > n_in:
        weights_padded = jnp.pad(weights, ((0, 0), (0, padded_in - n_in)))
    else:
        weights_padded = weights

    # 2. Reshape to [Out, Groups, GroupSize]
    n_groups = padded_in // quant_group_size
    weights_blocked = weights_padded.reshape(n_out, n_groups, quant_group_size)
    
    # 3. Compute scale
    abs_max = jnp.max(jnp.abs(weights_blocked), axis=-1, keepdims=True)
    abs_max = jnp.maximum(abs_max, 1e-6) 
    
    if quant_dtype == jnp.int8:
        scale = abs_max / INT8_MAX
        w_scaled = weights_blocked / scale
        w_quant = jnp.floor(w_scaled + 0.5)
        w_quant = jnp.clip(w_quant, INT8_MIN, INT8_MAX).astype(jnp.int8)
    else:
        scale = abs_max / FP8_MAX
        w_scaled = weights_blocked / scale
        w_quant = jnp.clip(w_scaled, FP8_MIN, FP8_MAX).astype(quant_dtype)

    # 4. Flatten Weights back to [Out, PaddedIn]
    weights_quant_padded = w_quant.reshape(n_out, padded_in)
    
    # 5. Flatten Scales to [Out, Groups]
    scales = scale.squeeze(-1).astype(jnp.float32)
    
    # 6. Slice weights back to original shape [Out, In]
    weights_quantized = weights_quant_padded[:, :n_in]
    
    return weights_quantized, scales


def quantize_and_scale_group(
    activation_group: jax.Array, 
    weight_scale: jax.Array, 
    quant_max: float, 
    dtype: jnp.dtype
) -> Tuple[jax.Array, jax.Array]:
    """
    Quantizes a specific group of input features and calculates the scale.
    """
    # 1. Cast to FP32.
    activation_f32 = activation_group.astype(jnp.float32)
    
    # 2. Compute max of each row of the group 
    activation_max = jnp.max(jnp.abs(activation_f32), axis=1, keepdims=True)
    activation_max = jnp.maximum(activation_max, 1e-6)

    # 3. Quantize Activations
    scale_to_int = quant_max / activation_max
    
    activation_quant_f32 = jnp.floor((activation_f32 * scale_to_int) + 0.5)
    
    # Note: Using float literals (constants defined above) ensures safe type promotion
    if dtype == jnp.int8:
        activation_quantized = jnp.clip(activation_quant_f32, INT8_MIN, INT8_MAX).astype(dtype)
    else:
        activation_quantized = jnp.clip(activation_quant_f32, FP8_MIN, FP8_MAX).astype(dtype)

    # 4. Compute combined scale for accumulator
    combined_scale = (activation_max / quant_max) * weight_scale
    return activation_quantized, combined_scale


def _fused_matmul_kernel(
    # HBM References
    activations_ref,    # [Batch, TotalInputFeatures]
    weights_ref,        # [TotalOutputFeatures, TotalInputFeatures]
    scales_ref,         # [TotalQuantGroups, TotalOutputFeatures] (Transposed)
    output_ref,         # [Batch, TotalOutputFeatures]
    # VMEM Scratchpad
    accumulator,        # [BatchLoad, OutputLoad]
    *,
    batch_load_size: int, 
    output_load_size: int, 
    quant_dtype: jnp.dtype
):
    """
    The Pallas Kernel executing on the TPU.
    """
    # 1. Initialize Accumulator (FP32)
    accumulator[...] = jnp.zeros(
        (batch_load_size, output_load_size), dtype=jnp.float32
    )

    #  2. Config based on dtype
    if quant_dtype == jnp.int8:
        quant_max = INT8_MAX
        dot_dtype = jnp.int32 
    else:
        quant_max = FP8_MAX
        dot_dtype = jnp.float32 

    # 3. Defines the work done on a single cache load (QUANT_GROUP_SIZE * GROUPS_PER_LOAD)
    def _process_cache_load(
        activation_cache: jax.Array,    # [BatchLoad, CacheLoadSize]
        weight_cache: jax.Array,        # [OutputLoad, CacheLoadSize]
        scales_cache: jax.Array         # [GroupsPerLoad, OutputLoad]
    ):
        # A. Get 0th group
        activation_group = activation_cache[:, 0:QUANT_GROUP_SIZE] 
        weight_group = weight_cache[:, 0:QUANT_GROUP_SIZE]
        
        # B. Scales are loaded as [GroupsPerLoad, OutputLoad]. Take the 0-th row and broadcast.
        scale_group = scales_cache[0, :][None, :]

        # C. Quantize group
        activation_quantized, accumulator_scale = quantize_and_scale_group(
            activation_group, scale_group, quant_max, quant_dtype
        )
        
        # D. Matrix multiplication
        dot_product = jax.lax.dot_general(
            activation_quantized, weight_group, 
            (((1,), (1,)), ((), ())), 
            preferred_element_type=dot_dtype
        )
        
        # E. Hold the result of the previous step while the next step runs
        dot_product_prev = dot_product
        accumulator_scale_prev = accumulator_scale

        # F. Iterate through the remaining groups in this cache load.
        for group_idx in range(1, GROUPS_PER_LOAD):
            start_feat = group_idx * QUANT_GROUP_SIZE
            end_feat   = (group_idx + 1) * QUANT_GROUP_SIZE
            
            # F.1 Load next group data
            activation_group_next = activation_cache[:, start_feat:end_feat]
            weight_group_next = weight_cache[:, start_feat:end_feat]
            scale_group_next = scales_cache[group_idx, :][None, :]

            # F.2 Quantize next group
            activation_quantized_next, accumulator_scale_next = quantize_and_scale_group(
                activation_group_next, scale_group_next, quant_max, quant_dtype
            )
            
            # F.3 Multiply the next group
            dot_product_next = jax.lax.dot_general(
                activation_quantized_next, weight_group_next, 
                (((1,), (1,)), ((), ())), 
                preferred_element_type=dot_dtype
            )

            # F.4 Add previous result to accumulator while performing dot product
            accumulator[...] += dot_product_prev.astype(jnp.float32) * accumulator_scale_prev
            
            # F.5 Shift data to prepare for next step
            dot_product_prev = dot_product_next
            accumulator_scale_prev = accumulator_scale_next

        # G. Add the last result to the accumulator
        accumulator[...] += dot_product_prev.astype(jnp.float32) * accumulator_scale_prev

    # 4. Emit the pallas Pipeline
    total_features = weights_ref.shape[1]
    n_cache_loads = total_features // CACHE_LOAD_SIZE
    pltpu.emit_pipeline(
        _process_cache_load,
        grid=(n_cache_loads,),
        in_specs=[
            # Activation: [BatchLoad, CacheLoadSize]
            pl.BlockSpec((batch_load_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            # Weight: [OutputLoad, CacheLoadSize]
            pl.BlockSpec((output_load_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            # Scales: [GroupsPerLoad, OutputLoad] 
            pl.BlockSpec((GROUPS_PER_LOAD, output_load_size), lambda i: (i, 0))
        ]
    )(activations_ref, weights_ref, scales_ref)

    # 5. Store result to HBM
    output_ref[...] = accumulator[...].astype(output_ref.dtype)


@functools.partial(jax.jit, static_argnames=['output_load_size', 'quant_dtype', 'batch_load_size'])
def _quantized_matmul_impl(
    activations, 
    weights_quantized, 
    weight_scales, 
    output_load_size, 
    quant_dtype, 
    batch_load_size
):
    """
    Handles padding/alignment and launches the Pallas kernel.
    """
    bs, n_in = activations.shape
    n_out = weights_quantized.shape[0]
    
    # 1. Padding dims to multiples of their load sizes
    padded_bs = next_multiple(bs, batch_load_size)
    padded_out = next_multiple(n_out, output_load_size)
    padded_in = next_multiple(n_in, CACHE_LOAD_SIZE)
    
    # 2. Pad Activations
    if padded_bs > bs or padded_in > n_in:
        pad_b = padded_bs - bs
        pad_in = padded_in - n_in
        activations = jnp.pad(activations, ((0, pad_b), (0, pad_in)))
        
    # 3. Pad Weights and Scales
    if padded_out > n_out or padded_in > n_in:
        pad_out = padded_out - n_out
        pad_in = padded_in - n_in
        weights_quantized = jnp.pad(weights_quantized, ((0, pad_out), (0, pad_in)), constant_values=0)
        
        # Scales are indexed by QuantGroup, so we pad based on the padded input size
        target_groups = padded_in // QUANT_GROUP_SIZE
        n_groups = weight_scales.shape[1]
        
        if target_groups > n_groups:
            # Pad the groups dimension
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, target_groups - n_groups)), constant_values=1.0)
        elif pad_out > 0:
            # Only pad the output dimension if groups matched (unlikely)
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, 0)), constant_values=1.0)

    # 4. Transpose Scales
    # [Output, TotalGroups] -> [TotalGroups, Output]
    w_scale_t = weight_scales.T

    # 5. Create the grid
    n_batch_loads = padded_bs // batch_load_size
    n_output_loads = padded_out // output_load_size
    grid = (n_batch_loads, n_output_loads)
    
    # 6. Define the in/out specs for the kernel
    in_specs = [
        # Activation: Load [BatchLoad, AllFeatures] (Kernel handles slicing)
        pl.BlockSpec((batch_load_size, padded_in), lambda b, o: (b, 0)),
        # Weight: Load [OutputLoad, AllFeatures]
        pl.BlockSpec((output_load_size, padded_in), lambda b, o: (o, 0)),
        # Scale: Load [AllGroups, OutputLoad]
        pl.BlockSpec((w_scale_t.shape[0], output_load_size), lambda b, o: (0, o)),
    ]
    out_spec = pl.BlockSpec((batch_load_size, output_load_size), lambda b, o: (b, o))

    # 7. Execute the kernel
    kernel = pl.pallas_call(
        functools.partial(
            _fused_matmul_kernel, 
            batch_load_size=batch_load_size, 
            output_load_size=output_load_size, 
            quant_dtype=quant_dtype
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_spec,
            grid=grid,
            scratch_shapes=[pltpu.VMEM((batch_load_size, output_load_size), jnp.float32)]
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
    output_load_size: int, 
    quant_dtype: jnp.dtype
) -> jax.Array:
    """
    Entry point for 2D block-wise quantized matmul.

    Args:
        activations: Input tensor [Batch, InputFeatures].
        weights_quantized: Weight tensor [OutputFeatures, InputFeatures].
        weight_scales: Scale tensor [OutputFeatures, InputFeatures // 128].
        output_load_size: Tiling size for output features (recommend 128 or 256).
        quant_dtype: Target quantization format (jnp.int8 or jnp.float8_e4m3fn).
    
    Returns:
        Result tensor [Batch, OutputFeatures] in original activation dtype.
    """
    batch_size = activations.shape[0]

    # Improves overall performance at different sizes
    if batch_size <= 8:   batch_load = 8
    elif batch_size <= 16: batch_load = 16
    elif batch_size <= 32: batch_load = 32
    elif batch_size <= 64: batch_load = 64
    elif batch_size <= 128: batch_load = 128
    else: batch_load = 256

    return _quantized_matmul_impl(
        activations, 
        weights_quantized, 
        weight_scales, 
        output_load_size=output_load_size, 
        quant_dtype=quant_dtype, 
        batch_load_size=batch_load
    )


def _fused_matmul_kernel_w8a16(
    # HBM References
    activations_ref,    # [Batch, TotalInputFeatures] (BF16)
    weights_ref,        # [TotalOutputFeatures, TotalInputFeatures] (Int8/FP8)
    scales_ref,         # [TotalQuantGroups, TotalOutputFeatures] (Transposed)
    output_ref,         # [Batch, TotalOutputFeatures]
    # VMEM Scratchpad
    accumulator,        # [BatchLoad, OutputLoad]
    *,
    batch_load_size: int, 
    output_load_size: int, 
    quant_dtype: jnp.dtype
):
    """
    W8A16 Pallas Kernel:
    - Activations stay in BF16 (no dynamic quantization overhead).
    - Weights are cast Int8/FP8 -> BF16.
    - Dot Product performed in BF16/FP32.
    - Scales applied after dot product: Sum(A * W_int) * Scale.
    """
    # 1. Initialize Accumulator
    accumulator[...] = jnp.zeros((batch_load_size, output_load_size), dtype=jnp.float32)

    def _process_cache_load(
        activation_cache: jax.Array,    # [BatchLoad, CacheLoadSize]
        weight_cache: jax.Array,        # [OutputLoad, CacheLoadSize]
        scales_cache: jax.Array         # [GroupsPerLoad, OutputLoad]
    ):
        # A. Pipeline setup: Process 0-th group
        activation_group = activation_cache[:, 0:QUANT_GROUP_SIZE] 
        weight_group = weight_cache[:, 0:QUANT_GROUP_SIZE]
        
        # Scales are [Groups, Output]. Get 0th row -> [1, Output]
        scale_group = scales_cache[0, :][None, :]

        # Cast weights (De-quantize step 1)
        weight_group_bf16 = weight_group.astype(jnp.bfloat16)
        
        # Dot Product (BF16 x BF16)
        # We delay scaling until after the dot product to reduce VPU scalar ops
        dot_product = jax.lax.dot_general(
            activation_group, weight_group_bf16, 
            (((1,), (1,)), ((), ())), 
            preferred_element_type=jnp.float32
        )
        
        # Pipeline registers
        dot_product_prev = dot_product
        scale_prev = scale_group

        # B. Loop through remaining groups
        for group_idx in range(1, GROUPS_PER_LOAD):
            start_feat = group_idx * QUANT_GROUP_SIZE
            end_feat   = (group_idx + 1) * QUANT_GROUP_SIZE
            
            # Load next
            activation_group_next = activation_cache[:, start_feat:end_feat]
            weight_group_next = weight_cache[:, start_feat:end_feat]
            scale_group_next = scales_cache[group_idx, :][None, :]

            # Compute next
            weight_group_next_bf16 = weight_group_next.astype(jnp.bfloat16)
            dot_product_next = jax.lax.dot_general(
                activation_group_next, weight_group_next_bf16, 
                (((1,), (1,)), ((), ())), 
                preferred_element_type=jnp.float32
            )

            # Accumulate previous: Result = Sum(Act * W_int) * Scale
            accumulator[...] += dot_product_prev * scale_prev.astype(jnp.float32)
            
            # Shift
            dot_product_prev = dot_product_next
            scale_prev = scale_group_next

        # C. Accumulate last result
        accumulator[...] += dot_product_prev * scale_prev.astype(jnp.float32)

    # 2. Emit Pipeline
    total_features = weights_ref.shape[1]
    n_cache_loads = total_features // CACHE_LOAD_SIZE
    
    pltpu.emit_pipeline(
        _process_cache_load,
        grid=(n_cache_loads,),
        in_specs=[
            pl.BlockSpec((batch_load_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            pl.BlockSpec((output_load_size, CACHE_LOAD_SIZE), lambda i: (0, i)),
            pl.BlockSpec((GROUPS_PER_LOAD, output_load_size), lambda i: (i, 0))
        ]
    )(activations_ref, weights_ref, scales_ref)

    output_ref[...] = accumulator[...].astype(output_ref.dtype)


@functools.partial(jax.jit, static_argnames=['output_load_size', 'quant_dtype', 'batch_load_size'])
def _quantized_matmul_w8a16_impl(
    activations, 
    weights_quantized, 
    weight_scales, 
    output_load_size, 
    quant_dtype, 
    batch_load_size
):
    # Reusing the padding logic structure
    bs, n_in = activations.shape
    n_out = weights_quantized.shape[0]
    
    padded_bs = next_multiple(bs, batch_load_size)
    padded_out = next_multiple(n_out, output_load_size)
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
        weights_quantized = jnp.pad(weights_quantized, ((0, pad_out), (0, pad_in)), constant_values=0)
        
        target_groups = padded_in // QUANT_GROUP_SIZE
        n_groups = weight_scales.shape[1]
        
        if target_groups > n_groups:
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, target_groups - n_groups)), constant_values=1.0)
        elif pad_out > 0:
            weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, 0)), constant_values=1.0)

    w_scale_t = weight_scales.T

    grid = (padded_bs // batch_load_size, padded_out // output_load_size)
    
    # Kernel Call
    kernel = pl.pallas_call(
        functools.partial(
            _fused_matmul_kernel_w8a16, 
            batch_load_size=batch_load_size, 
            output_load_size=output_load_size, 
            quant_dtype=quant_dtype
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_load_size, padded_in), lambda b, o: (b, 0)),
                pl.BlockSpec((output_load_size, padded_in), lambda b, o: (o, 0)),
                pl.BlockSpec((w_scale_t.shape[0], output_load_size), lambda b, o: (0, o)),
            ],
            out_specs=pl.BlockSpec((batch_load_size, output_load_size), lambda b, o: (b, o)),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((batch_load_size, output_load_size), jnp.float32)]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), activations.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_device_vmem_limit())
    )
    
    out = kernel(activations, weights_quantized, w_scale_t)
    return out[:bs, :n_out]


def quantized_matmul_w8a16_2d(
    activations: jax.Array, 
    weights_quantized: jax.Array, 
    weight_scales: jax.Array, 
    output_load_size: int, 
    quant_dtype: jnp.dtype
) -> jax.Array:
    """Entry point for W8A16 Kernel"""
    batch_size = activations.shape[0]

    if batch_size <= 8:   batch_load = 8
    elif batch_size <= 16: batch_load = 16
    elif batch_size <= 32: batch_load = 32
    elif batch_size <= 64: batch_load = 64
    elif batch_size <= 128: batch_load = 128
    else: batch_load = 256

    return _quantized_matmul_w8a16_impl(
        activations, 
        weights_quantized, 
        weight_scales, 
        output_load_size=output_load_size, 
        quant_dtype=quant_dtype, 
        batch_load_size=batch_load
    )
