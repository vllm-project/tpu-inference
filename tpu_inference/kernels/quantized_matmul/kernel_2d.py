# SPDX-License-Identifier: Apache-2.0
"""
2D (Block-wise) Quantized Matrix Multiplication Kernel for TPU.
"""

import functools
from typing import Any, Tuple, Optional, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.util import next_multiple, unfold_args
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

DType = Any
Shape = Tuple[int, ...]

# BLOCK_K: The fundamental tile size for the inner dimension. Quantized model blocks.
BLOCK_K = 128

# SUPER_CHUNK: The number of BLOCK_K tiles processed in one pipeline stage.
# A larger super chunk amortizes the overhead of the pipeline control logic.
# 16 blocks * 128 size = 2048 elements loaded per pipeline step.
SUPER_CHUNK = 16
TILE_K_SIZE = BLOCK_K * SUPER_CHUNK


# 1. Reference Implementation
def _quantize_block_ref(
    input_slice: jax.Array,
    abs_max_val: jax.Array,
    target_dtype: jnp.dtype
) -> Tuple[jax.Array, jax.Array]:
    """
    Quantizes a block of floating point values to the target integer/fp8 type.
    
    Math:
        scale = abs_max_val / max_representable_value
        x_quant = clamp(round(x / scale))
    """
    is_floating_point_target = jnp.issubdtype(target_dtype, jnp.floating)
    
    # Determine the dynamic range of the target format
    if is_floating_point_target:
        dtype_info = jnp.finfo(target_dtype)
        type_max = float(dtype_info.max)
        type_min = float(dtype_info.min)
        
        # Specific clamp for FP8 E4M3FN to ensure numerical stability on hardware
        if target_dtype == jnp.float8_e4m3fn:
            type_max = 448.0
    else:
        # Integer quantization (int8)
        dtype_info = jnp.iinfo(target_dtype)
        type_max = float(dtype_info.max)
        type_min = float(dtype_info.min)

    # Calculate Scale.
    # We use jnp.tiny to prevent division by zero if a block is all zeros.
    scale_basis = jnp.maximum(abs_max_val, jnp.finfo(jnp.float32).tiny)
    scale = scale_basis / type_max
    
    # Perform Quantization
    x_scaled = input_slice.astype(jnp.float32) / scale

    if not is_floating_point_target:
        # Integer: Round nearest even + Clip
        x_scaled = jnp.floor(x_scaled + 0.5)
        quantized_data = jnp.clip(x_scaled, type_min, type_max).astype(target_dtype)
    else:
        # Float: Cast (rounding handles implicitly)
        quantized_data = x_scaled.astype(target_dtype)
        
    return quantized_data, scale.astype(jnp.float32)


def _reference_kernel_body(
    # References to Global Memory (HBM)
    activations_ref,         # [Batch, In]
    weights_quantized_ref,   # [Out, In]
    weight_scales_ref,       # [In_Blocks, Out] (Transposed for access)
    act_abs_max_ref,         # [In_Blocks, Batch] (Transposed for access)
    output_ref,              # [Batch, Out]
    # VMEM Scratchpads
    accumulator_scratch,     # [Batch_Block, Out_Block]
    act_quant_scratch,       # Temporary buffer for quantized activations
    act_scale_scratch,       # Temporary buffer for activation scales
    # Static Arguments (Compile-time constants)
    *, 
    quant_dtype: DType, 
    accumulation_dtype: DType, 
    use_accumulator: bool, 
    cache_quantized_inputs: bool
):
    """
    The inner Pallas kernel for the reference implementation.
    Executed on the grid: (Batch Tiles, Out Tiles, In Blocks).
    """
    # Grid Indices
    out_idx = pl.program_id(1)
    in_block_idx = pl.program_id(2)
    total_in_blocks = pl.num_programs(2)
    
    needs_quantization = quant_dtype != activations_ref.dtype
    
    # Logic: Should we perform quantization this step?
    # If caching is enabled (save_x_q), we only quantize on the first output tile 
    # and reuse for subsequent output tiles to save VPU cycles.
    if cache_quantized_inputs: 
        perform_quant = (out_idx == 0)
    else: 
        perform_quant = needs_quantization

    # Logic: Accumulation handling
    # If we have multiple input blocks, we must accumulate results in VMEM.
    if use_accumulator:
        is_first_step = (in_block_idx == 0)
        is_last_step = (in_block_idx == (total_in_blocks - 1))
    else:
        # Single block case: It is both the first and last step.
        is_first_step, is_last_step = True, True

    def _matmul_step(do_quant: bool, is_first: bool, is_last: bool):
        # 1. Load Activation Metadata
        # Get the max value for the current block to compute quantization scale
        act_max_row = act_abs_max_ref[in_block_idx]
        act_max_current = act_max_row[:, None]

        # 2. Quantize Activations (VPU Bound)
        # If inputs are float, we quantize them to int8/fp8 here.
        if needs_quantization:
            if do_quant:
                 x_q_local, x_scale_local = _quantize_block_ref(
                     activations_ref[...], act_max_current, quant_dtype
                 )
                 # Cache if requested (though this ref kernel disables caching by default)
                 if cache_quantized_inputs: 
                     act_quant_scratch[...] = x_q_local
                     act_scale_scratch[...] = x_scale_local
            else:
                 # Load from scratchpad
                 x_q_local = act_quant_scratch[...]
                 x_scale_local = act_scale_scratch[...]
        else:
            # Pass-through (already quantized or keeping as float)
            x_q_local = activations_ref[...]
            x_scale_local = 1.0 

        # 3. Matrix Multiplication (MXU Bound)
        # dot_general maps to the TPU Matrix Multiply Unit.
        # Dimensions: [Batch, Block_K] @ [Out, Block_K].T -> [Batch, Out]
        dot_product = jax.lax.dot_general(
            x_q_local, 
            weights_quantized_ref[...], 
            (((1,), (1,)), ((), ())), 
            preferred_element_type=accumulation_dtype,
        )
        
        # 4. De-quantization and Scaling (VPU Bound)
        # Result = Dot * Act_Scale * Weight_Scale
        w_scale_current = weight_scales_ref[in_block_idx][None, :]
        
        acc_float = dot_product.astype(jnp.float32)
        if needs_quantization: 
            acc_float *= x_scale_local
        acc_float *= w_scale_current

        # 5. Accumulation (VPU Bound)
        if not is_first:
            acc_float += accumulator_scratch[...]
            
        if is_last:
            output_ref[...] = acc_float.astype(activations_ref.dtype)
        elif use_accumulator:
            accumulator_scratch[...] = acc_float

    # `unfold_args` unrolls the static booleans, generating specialized code paths 
    # to avoid branching inside the kernel.
    unfold_args((perform_quant, is_first_step, is_last_step), (), _matmul_step)


@functools.partial(jax.jit, static_argnames=['x_q_dtype', 'quant_block_size', 'batch_block_size', 'out_block_size'])
def quantized_matmul_2d(
    activations: jax.Array, 
    weights_quantized: jax.Array, 
    weight_scales: jax.Array, 
    quant_block_size: int, 
    x_q_dtype: Optional[DType] = None, 
    *, 
    batch_block_size: int = 128, 
    out_block_size: int = 128
) -> jax.Array:
    """
    Reference Frontend: Prepares data and launches the reference Pallas kernel.
    
    Args:
        activations: Input [Batch, In_Dim].
        weights_quantized: Weights [Out_Dim, In_Dim].
        weight_scales: Scales [Out_Dim, In_Blocks].
        quant_block_size: Size of the quantization block (K dimension).
        x_q_dtype: Target quantization type for activations.
        batch_block_size: Tiling size for batch dimension.
        out_block_size: Tiling size for output dimension.
    """
    if x_q_dtype is None: x_q_dtype = activations.dtype
    
    batch_size, n_in = activations.shape
    n_out, _ = weights_quantized.shape

    # --- Padding Step 1: Input Dimension (K) ---
    # The input dimension must be divisible by the quantization block size.
    padded_n_in = next_multiple(n_in, quant_block_size)
    if n_in < padded_n_in:
        padding = padded_n_in - n_in
        activations = jnp.pad(activations, ((0, 0), (0, padding)))
        weights_quantized = jnp.pad(weights_quantized, ((0, 0), (0, padding)))
    
    n_in_blocks = padded_n_in // quant_block_size

    # --- Pre-computation: Activation Max ---
    # We calculate the max absolute value per block ahead of time (Standard approach).
    # Reshape: [Batch, In] -> [Batch, Blocks, Block_Size]
    x_blocked = activations.reshape(batch_size, n_in_blocks, quant_block_size)
    x_abs_max = jnp.max(jnp.abs(x_blocked), axis=-1).astype(jnp.float32)

    # --- Padding Step 2: Tiling Dimensions (Batch, Out) ---
    # TPU kernels require grid sizes to match memory layouts perfectly.
    padded_batch = next_multiple(batch_size, batch_block_size)
    padded_out = next_multiple(n_out, out_block_size)
    
    if batch_size < padded_batch:
        activations = jnp.pad(activations, ((0, padded_batch - batch_size), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, padded_batch - batch_size), (0, 0)))
    if n_out < padded_out:
        weights_quantized = jnp.pad(weights_quantized, ((0, padded_out - n_out), (0, 0)))
        weight_scales = jnp.pad(weight_scales, ((0, padded_out - n_out), (0, 0)))

    # --- Transposition ---
    # Pallas block specs often prefer [Block_Index, ...]. 
    # We transpose these to optimize the memory access pattern inside the kernel.
    x_abs_max_t = jnp.transpose(x_abs_max) # [Blocks, Batch]
    w_scale_t = jnp.transpose(weight_scales.astype(jnp.float32)) # [Blocks, Out]

    # Grid Dimensions
    n_batch_tiles = padded_batch // batch_block_size
    n_out_tiles = padded_out // out_block_size
    
    use_accumulator = n_in_blocks > 1
    
    # Heuristic for dot precision: Use int32 accumulation if inputs are int8/weights int8.
    dot_dtype = jnp.int32 if (x_q_dtype != activations.dtype and jnp.issubdtype(weights_quantized.dtype, jnp.integer)) else jnp.float32

    kernel = pl.pallas_call(
        functools.partial(
            _reference_kernel_body, 
            quant_dtype=x_q_dtype, 
            accumulation_dtype=dot_dtype, 
            use_accumulator=use_accumulator, 
            cache_quantized_inputs=False
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, quant_block_size), lambda b, o, i: (b, i)),
                pl.BlockSpec((out_block_size, quant_block_size), lambda b, o, i: (o, i)),
                pl.BlockSpec((n_in_blocks, out_block_size), lambda b, o, i: (0, o)),
                pl.BlockSpec((n_in_blocks, batch_block_size), lambda b, o, i: (0, b))
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.float32) if use_accumulator else None, 
                None, None
            ],
            grid=(n_batch_tiles, n_out_tiles, n_in_blocks),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_batch, padded_out), activations.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary', 'arbitrary'), 
            vmem_limit_bytes=get_device_vmem_limit()
        ),
    )
    result = kernel(activations, weights_quantized, w_scale_t, x_abs_max_t)
    return result[:batch_size, :n_out]


# ==============================================================================
# 2. Optimized V7 Implementation (Pipeline Optimized)
# ==============================================================================

def _compute_quant_params(
    activation_slice: jax.Array, 
    weight_scale_row: jax.Array, 
    quant_max_val: float, 
    dtype: DType
) -> Tuple[jax.Array, jax.Array]:
    """
    Optimized VPU routine for quantization.
    
    Optimizations:
    1. FP32 Math: Keeps the VPU running in high-throughput mode.
    2. Reciprocal Math: Divisions are expensive on TPU. We calculate 
       `scale = max_val / x_max` and then multiply.
    """
    # Cast to FP32 for calculation
    x_f32 = activation_slice.astype(jnp.float32)
    x_abs = jnp.abs(x_f32)
    
    # Find absolute max per row (Batch)
    x_max = jnp.max(x_abs, axis=1, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6) # Prevent div/0

    # Calculate scale factor to map float range to int range
    # e.g., if max is 2.0 and we want int8 (127), scale is 63.5
    scale_to_int = quant_max_val / x_max
    
    # Quantize
    val = x_f32 * scale_to_int
    val_rounded = jnp.floor(val + 0.5)

    if dtype == jnp.int8:
        val_clipped = jnp.clip(val_rounded, -128.0, 127.0)
        x_q = val_clipped.astype(jnp.int8)
    else:
        val_clipped = jnp.clip(val_rounded, -448.0, 448.0)
        x_q = val_clipped.astype(dtype)

    # Calculate the combined dequantization scale for the accumulator:
    # final_scale = (x_max / quant_max_val) * weight_scale
    combined_scale_factor = (x_max / quant_max_val) * weight_scale_row
    
    return x_q, combined_scale_factor

def _v7_pipelined_matmul_kernel(
    activations_ref, 
    weights_ref, 
    scales_ref, 
    output_ref, 
    accumulator_vmem, 
    *,
    batch_block_size: int, 
    out_block_size: int, 
    quant_dtype: DType
):
    """
    V7 Kernel using Software Pipelining (`emit_pipeline`).
    
    The goal is to hide the latency of VPU operations (Quantization + Scaling)
    behind the execution of the MXU (Matmul).
    
    Pipeline Stages (Conceptual):
      Step N:
        - VPU: Quantize Block (N+1) [Prepare Next]
        - MXU: Matmul Block (N)     [Compute Current]
        - VPU: Accumulate Block (N-1) [Finalize Prev]
    """
    # Clear accumulator (f32 for precision)
    accumulator_vmem[...] = jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32)

    # Config based on dtype
    if quant_dtype == jnp.int8:
        max_val, dot_preferred = 127.0, jnp.int32
    else:
        max_val, dot_preferred = 448.0, jnp.float32

    def _pipeline_stage(
        act_chunk: jax.Array,    # [Batch, TILE_K_SIZE]
        weight_chunk: jax.Array, # [Out, TILE_K_SIZE]
        scales_chunk: jax.Array  # [SUPER_CHUNK, Out] (Transposed slice)
    ):
        """
        Processes one SUPER_CHUNK (16 * 128 = 2048 K-elements).
        Inside here, we manually unroll loops to interleave VPU/MXU instructions.
        """
        
        # --- PROLOGUE (Stage 0) ---
        # We must start the first sub-block (k=0) to prime the pipeline registers.
        
        # Slicing: Get the 0-th sub-block of size BLOCK_K
        x_0 = act_chunk[:, 0:BLOCK_K] 
        w_0 = weight_chunk[:, 0:BLOCK_K]
        s_0 = scales_chunk[0, :][None, :]

        # VPU: Quantize
        x_q_curr, scale_curr = _compute_quant_params(x_0, s_0, max_val, quant_dtype)
        
        # MXU: Matmul
        dot_curr = jax.lax.dot_general(
            x_q_curr, w_0, 
            (((1,), (1,)), ((), ())), 
            preferred_element_type=dot_preferred
        )
        
        # Init Registers
        dot_prev, scale_prev = dot_curr, scale_curr

        # --- LOOP (Stages 1 to N-1) ---
        # Iterate through the rest of the SUPER_CHUNK
        for j in range(1, SUPER_CHUNK):
            k_start, k_end = j * BLOCK_K, (j + 1) * BLOCK_K
            
            # Load pointers for NEXT
            x_next = act_chunk[:, k_start:k_end]
            w_next = weight_chunk[:, k_start:k_end]
            s_next = scales_chunk[j, :][None, :]

            # VPU: Quantize NEXT
            x_q_next, scale_next = _compute_quant_params(x_next, s_next, max_val, quant_dtype)
            
            # MXU: Matmul NEXT
            dot_next = jax.lax.dot_general(
                x_q_next, w_next, 
                (((1,), (1,)), ((), ())), 
                preferred_element_type=dot_preferred
            )

            # VPU: Accumulate PREV (Happens in parallel with Matmul NEXT)
            # We cast to float32 immediately to ensure scaling is high precision
            accumulator_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev
            
            # Shift registers
            dot_prev, scale_prev = dot_next, scale_next

        # --- EPILOGUE (Stage N) ---
        # Accumulate the very last sub-block result
        accumulator_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev

    # Calculate number of super chunks
    total_k = weights_ref.shape[1]
    n_super_chunks = total_k // TILE_K_SIZE

    # Emit the Pallas Pipeline
    # This generates the DMA async copy loops and sync barriers automatically.
    pltpu.emit_pipeline(
        _pipeline_stage,
        grid=(n_super_chunks,),
        in_specs=[
            pl.BlockSpec((batch_block_size, TILE_K_SIZE), lambda i: (0, i)),
            pl.BlockSpec((out_block_size, TILE_K_SIZE), lambda i: (0, i)),
            pl.BlockSpec((SUPER_CHUNK, out_block_size), lambda i: (i, 0)) # Note: Pre-transposed scales
        ]
    )(activations_ref, weights_ref, scales_ref)

    # Write final result to Global Memory
    output_ref[...] = accumulator_vmem[...].astype(output_ref.dtype)


@functools.partial(jax.jit, static_argnames=['out_block_size', 'quant_dtype', 'override_block_b'])
def _dispatch_impl(
    activations, 
    weights_quantized, 
    weight_scales, 
    out_block_size, 
    quant_dtype, 
    override_block_b
):
    """
    Internal Helper: Handles Padding and Grid Setup for the V7 Kernel.
    """
    bs, n_in = activations.shape
    n_out, _ = weights_quantized.shape
    CURRENT_BLOCK_B = override_block_b
    
    # Align dimensions to Block/Tile sizes
    padded_bs = next_multiple(bs, CURRENT_BLOCK_B)
    padded_out = next_multiple(n_out, out_block_size)
    padded_in = next_multiple(n_in, TILE_K_SIZE)
    
    # Pad Activations
    if padded_bs > bs: 
        activations = jnp.pad(activations, ((0, padded_bs - bs), (0, 0)))
    if padded_in > n_in: 
        activations = jnp.pad(activations, ((0, 0), (0, padded_in - n_in)))
        
    # Pad Weights and Scales (K dimension)
    if padded_in > n_in:
        pad_k = padded_in - n_in
        weights_quantized = jnp.pad(weights_quantized, ((0, 0), (0, pad_k)))
        
        # Scales must be padded by number of BLOCKS, not raw elements
        pad_blocks = pad_k // BLOCK_K
        weight_scales = jnp.pad(weight_scales, ((0, 0), (0, pad_blocks)), constant_values=1.0)
        
    # Pad Weights and Scales (Output dimension)
    if padded_out > n_out:
        pad_out = padded_out - n_out
        weights_quantized = jnp.pad(weights_quantized, ((0, pad_out), (0, 0)))
        weight_scales = jnp.pad(weight_scales, ((0, pad_out), (0, 0)))

    # Transpose Scales: [Out, Blocks] -> [Blocks, Out]
    # This allows the kernel to load a slice [SUPER_CHUNK, Out_Tile] contiguously.
    w_scale_t = weight_scales.T

    # Define Grid
    n_batch_blocks = padded_bs // CURRENT_BLOCK_B
    n_out_blocks = padded_out // out_block_size
    grid = (n_batch_blocks, n_out_blocks)
    
    # Define Block Specifications
    # These lambda functions map (batch_idx, out_idx) -> (start_row, start_col) in memory.
    in_specs = [
        # Activations: Load [Batch_Tile, All_K] (Kernel handles K tiling internally via pipeline)
        pl.BlockSpec((CURRENT_BLOCK_B, padded_in), lambda b, o: (b, 0)),
        # Weights: Load [Out_Tile, All_K]
        pl.BlockSpec((out_block_size, padded_in), lambda b, o: (o, 0)),
        # Scales: Load [All_Blocks, Out_Tile]
        pl.BlockSpec((w_scale_t.shape[0], out_block_size), lambda b, o: (0, o)),
    ]

    kernel = pl.pallas_call(
        functools.partial(
            _v7_pipelined_matmul_kernel, 
            batch_block_size=CURRENT_BLOCK_B, 
            out_block_size=out_block_size, 
            quant_dtype=quant_dtype
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((CURRENT_BLOCK_B, out_block_size), lambda b, o: (b, o)),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((CURRENT_BLOCK_B, out_block_size), jnp.float32)]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), activations.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_device_vmem_limit())
    )
    out = kernel(activations, weights_quantized, w_scale_t)
    
    # Slice off padding
    return out[:bs, :n_out]


def dispatch_w8a8_v7(
    activations: jax.Array, 
    weights_quantized: jax.Array, 
    weight_scales: jax.Array, 
    out_block_size: int, 
    quant_dtype: DType
) -> jax.Array:
    """
    Adaptive Dispatcher for V7 Kernel.
    
    This function selects the optimal Block Batch Size (`override_block_b`) based
    on the runtime batch size. This tuning is critical for TPU occupancy.
    
    Tuning Logic:
    - Small Batch (<=48): Use block=32. Reduces latency for single requests.
      (~1.4x speedup vs large blocks).
    - Medium Batch (<=96): Use block=64.
    - Large Batch (>192): Use block=256. This is compute-bound and maximizes
      MXU utilization.
    
    Args:
        activations: Input tensor.
        weights_quantized: Quantized Weight tensor.
        weight_scales: Weight scales.
        out_block_size: Tiling size for output features.
        quant_dtype: Target quantization format (e.g., int8).
    
    Returns:
        Matmul result.
    """
    batch_size = activations.shape[0]

    if batch_size <= 8:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=8)
    elif batch_size <= 16:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=16)
    elif batch_size <= 32:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=32)
    elif batch_size <= 64:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=64)
    elif batch_size <= 128:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=128)
    else:
        return _dispatch_impl(activations, weights_quantized, weight_scales, out_block_size, quant_dtype, override_block_b=256)