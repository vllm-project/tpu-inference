# SPDX-License-Identifier: Apache-2.0
"""2D (Block-wise) Quantized matmul kernel.

This module implements "Block-wise" or "Sub-channel" quantization.
It contains:
1. Standard 2D Quantized Matmul (quantized_matmul_2d)
2. High-performance V7 Kernel (dispatch_real_v7) [Updated for standard layout]
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.logger import init_logger
from tpu_inference.kernels.quantized_matmul.util import (
    unfold_args,
    next_multiple,
)
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    get_device_vmem_limit,
)

logger = init_logger(__name__)

BLOCK_K = 128
BLOCK_B = 64
SUPER_CHUNK = 16
TILE_K_SIZE = BLOCK_K * SUPER_CHUNK

# ... (Keep _quantize_array_2d, _validate_inputs_2d, _quantized_matmul_kernel_2d, quantized_matmul_2d as is) ...

def _quantize_array_2d(
    x: jax.Array,
    x_abs_max: jax.Array,
    quant_dtype: jnp.dtype,
):
    """Quantizes an input array using block-wise scaling.
    
    This function is called INSIDE the Pallas kernel for activation quantization.
    
    Args:
        x: Input array block [batch_block_size, quant_block_size]
        x_abs_max: Max values for this specific block [batch_block_size, 1]
        quant_dtype: Target dtype (int8, float8_e4m3fn, etc.)
    """
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    
    if is_float:
        dtype_info = jnp.finfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        if quant_dtype == jnp.float8_e4m3fn:
            dtype_max = 448.0
        dtype_min = float(dtype_info.min)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        dtype_min = float(dtype_info.min)

    # Prevent division by zero
    scale_basis = jnp.maximum(x_abs_max, jnp.finfo(jnp.float32).tiny)
    scale = scale_basis / dtype_max

    x_scaled = x.astype(jnp.float32) / scale

    if not is_float:
        # CHANGED: Match V7 Kernel behavior (Round Half Up)
        # Previous: x_scaled = jnp.round(x_scaled) (Round Half To Even)
        x_scaled = jnp.floor(x_scaled + 0.5)
        
        quantized_array = jnp.clip(x_scaled, dtype_min, dtype_max).astype(quant_dtype)
    else:
        quantized_array = x_scaled.astype(quant_dtype)

    return quantized_array, scale.astype(jnp.float32)


def _validate_inputs_2d(x, w_q, w_scale, x_abs_max, x_q_dtype, batch_block_size, out_block_size, quant_block_size):
    """Validates input shapes and dtypes for 2D quantized matmul."""
    if x.dtype != x_q_dtype:
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(w_q.dtype, jnp.integer):
            raise ValueError(f'{x_q_dtype=} and {w_q.dtype=} must be same int/float type.')
    
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f'{x.shape[1]=} must be equal to {w_q.shape[1]=}')
    
    n_in_blocks = x.shape[1] // quant_block_size

    if w_q.shape[0] != w_scale.shape[0]:
        raise ValueError(f'{w_q.shape[0]=} must be equal to {w_scale.shape[0]=}')
    if n_in_blocks != w_scale.shape[1]:
        raise ValueError(f'{n_in_blocks=} must be equal to {w_scale.shape[1]=}')

    if x_abs_max.shape != (x.shape[0], n_in_blocks):
        raise ValueError(f'{x_abs_max.shape=} must be equal to ({x.shape[0]}, {n_in_blocks})')


def _quantized_matmul_kernel_2d(
    x_ref, w_q_ref, w_scale_ref, x_abs_max_ref, out_ref,
    acc_scratch, x_q_scratch, x_scale_scratch,
    *, x_q_dtype, dot_dtype, save_acc, save_x_q
):
    """Pallas kernel for 2D (block-wise) quantized matrix multiplication.
    
    This kernel executes a "Mini-Matmul" for one block of the reduction dimension.
    """
    # Grid Dimensions:
    # 0: Batch Dimension (Parallelized)
    # 1: Output Dimension (Parallelized)
    # 2: Reduction/Input Dimension (Serialized Loop)
    out_idx, in_block_idx = pl.program_id(1), pl.program_id(2)
    n_in_blocks = pl.num_programs(2)
    
    quantize_activation = x_q_dtype != x_ref.dtype
    
    # Logic for scratchpad reuse
    if save_x_q:
        quant = out_idx == 0
    else:
        quant = quantize_activation

    # Logic for accumulator initialization/finalization
    if save_acc:
        is_first_step = (in_block_idx == 0)
        is_last_step = (in_block_idx == (n_in_blocks - 1))
    else:
        is_first_step, is_last_step = True, True

    def matmul_body(quant, is_first_step, is_last_step):
        # 1. INPUT QUANTIZATION (Optional)
        x_abs_max_row = x_abs_max_ref[in_block_idx]
        x_abs_max_current = x_abs_max_row[:, None]

        if quantize_activation:
            if quant:
                 # Quantize raw input (float) -> quantized input (int8/float8)
                 x_q_tmp, x_scale_tmp = _quantize_array_2d(
                     x_ref[...], x_abs_max_current, x_q_dtype
                 )
                 # Cache the quantized values if reusing across output tiles
                 if save_x_q:
                     x_q_scratch[...] = x_q_tmp
                     x_scale_scratch[...] = x_scale_tmp
            else:
                 # Load pre-quantized values from scratchpad
                 x_q_tmp = x_q_scratch[...]
                 x_scale_tmp = x_scale_scratch[...]
        else:
            # W8A16 Mode: Activations stay in High Precision (BF16/F32)
            # We skip quantization and set scale to 1.0
            x_q_tmp = x_ref[...]
            x_scale_tmp = 1.0 

        # 2. MATRIX MULTIPLICATION
        # Perform the dot product for this specific block (e.g., 128 elements).
        acc = jax.lax.dot_general(
            x_q_tmp, w_q_ref[...], (((1,), (1,)), ((), ())),
            preferred_element_type=dot_dtype,
        )

        # 3. IMMEDIATE DEQUANTIZATION
        # Fetch the scale for this specific block of weights
        w_scale_current = w_scale_ref[in_block_idx][None, :]
        
        # Cast to Float32 for accumulation
        acc = acc.astype(jnp.float32)
        
        # Apply scales
        if quantize_activation:
            acc *= x_scale_tmp
            
        acc *= w_scale_current

        # 4. ACCUMULATION
        # Add this block's contribution to the running total in VMEM.
        if not is_first_step:
            acc += acc_scratch[...]
            
        if is_last_step:
            # Final write to HBM
            out_ref[...] = acc.astype(x_ref.dtype)
        elif save_acc:
            # Store partial sum in VMEM
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@functools.partial(jax.jit, static_argnames=[
    'x_q_dtype', 'quant_block_size', 'batch_block_size', 'out_block_size'
])
def quantized_matmul_2d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quant_block_size: int,
    x_q_dtype=None,
    *,
    batch_block_size: int = 128,
    out_block_size: int = 128
) -> jax.Array:
    """Performs 2D (block-wise) quantized matrix multiplication."""
    if x_q_dtype is None:
        x_q_dtype = x.dtype
    
    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    # 1. PADDING THE REDUCTION DIMENSION
    # We must ensure the input feature dimension (n_in) is a multiple of quant_block_size
    padded_n_in = next_multiple(orig_n_in, quant_block_size)
    
    if orig_n_in < padded_n_in:
        # Pad with zeros. 
        # For w_q, 0 * x = 0, so it does not affect the accumulation sum.
        padding_diff = padded_n_in - orig_n_in
        x = jnp.pad(x, ((0, 0), (0, padding_diff)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padding_diff)))

    n_in_blocks = padded_n_in // quant_block_size

    # 2. COMPUTE BLOCK SCALES
    # Reshape input into [Batch, Num_Blocks, Block_Size] to find per-block max.
    x_blocked = x.reshape(orig_n_batch, n_in_blocks, quant_block_size)
    x_abs_max = jnp.max(jnp.abs(x_blocked), axis=-1)
    x_abs_max = x_abs_max.astype(jnp.float32)

    # 3. PADDING THE BATCH AND OUTPUT DIMENSIONS
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, padded_n_batch - orig_n_batch), (0, 0)))

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, padded_n_out - orig_n_out), (0, 0)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    # x_abs_max: [batch, blocks] -> [blocks, batch]
    x_abs_max_t = jnp.transpose(x_abs_max)
    # w_scale: [out, blocks] -> [blocks, out]
    w_scale_t = jnp.transpose(w_scale)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    
    # Enable accumulator scratchpad if we have >1 block to sum over
    save_acc = n_in_blocks > 1
    
    # Disable x_q saving for now
    save_x_q = False

    dot_dtype = jnp.float32
    acc_dtype = jnp.float32
    
    # Use Int32 accumulation for the dot product if both inputs are quantized
    if x_q_dtype != x.dtype and jnp.issubdtype(w_q.dtype, jnp.integer):
        dot_dtype = jnp.int32
    
    vmem_limit_bytes = get_device_vmem_limit()

    kernel = pl.pallas_call(
        functools.partial(
            _quantized_matmul_kernel_2d,
            x_q_dtype=x_q_dtype,
            dot_dtype=dot_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        # The grid is 3D: (n_batch, n_out, n_in_blocks)
        # - Dimension 0 (n_batch): Parallelized across cores
        # - Dimension 1 (n_out):   Parallelized across cores
        # - Dimension 2 (n_in_blocks): The REDUCTION loop. Runs sequentially.
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                # X: Load [Batch_Block, Quant_Block] using grid dims (b, i)
                pl.BlockSpec((batch_block_size, quant_block_size), lambda b, o, i: (b, i)),
                
                # W_Q: Load [Out_Block, Quant_Block] using grid dims (o, i)
                pl.BlockSpec((out_block_size, quant_block_size), lambda b, o, i: (o, i)),
                
                # W_Scale & X_Max: Load the entire row of blocks
                # We fix the row index (0) and load the slice corresponding to output (o) or batch (b).
                # Inside the kernel, we index into this using `in_block_idx`.
                pl.BlockSpec((n_in_blocks, out_block_size), lambda b, o, i: (0, o)),
                pl.BlockSpec((n_in_blocks, batch_block_size), lambda b, o, i: (0, b)),
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                # Accumulator for partial floating point sums
                pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None,
                # Scratchpad for quantized activations
                pltpu.VMEM((batch_block_size, quant_block_size), x_q_dtype) if save_x_q else None,
                pltpu.VMEM((batch_block_size, 1), jnp.float32) if save_x_q else None,
            ],
            grid=(n_batch, n_out, n_in_blocks),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    _validate_inputs_2d(x, w_q, w_scale, x_abs_max, x_q_dtype, 
                        batch_block_size, out_block_size, quant_block_size)

    out = kernel(x, w_q, w_scale_t, x_abs_max_t)
    return out[:orig_n_batch, :orig_n_out]


# ==============================================================================
# V7 Kernel Implementation
# ==============================================================================

class QuantizationResult(NamedTuple):
    q_data: jax.Array
    scales: jax.Array

@functools.partial(jax.jit, static_argnames=['out_block_size', 'dtype'])
def quantize_offline_tiled_v7(w: jax.Array, out_block_size: int, dtype: jnp.dtype = jnp.int8) -> QuantizationResult:
    n_out, n_in = w.shape
    padded_in = next_multiple(n_in, TILE_K_SIZE)
    padded_out = next_multiple(n_out, out_block_size)

    if padded_in > n_in or padded_out > n_out:
        w = jnp.pad(w, ((0, padded_out - n_out), (0, padded_in - n_in)))

    n_out_outer = padded_out // out_block_size
    n_in_outer = padded_in // TILE_K_SIZE

    w_reshaped = w.reshape(n_out_outer, out_block_size, n_in_outer, TILE_K_SIZE)
    w_tiled = w_reshaped.transpose(0, 2, 3, 1) # (N_o, K_o, K_t, N_t)

    w_blocked = w_tiled.reshape(n_out_outer, n_in_outer, SUPER_CHUNK, BLOCK_K, out_block_size)
    w_max = jnp.max(jnp.abs(w_blocked), axis=3)
    w_max = jnp.maximum(w_max, 1e-6)

    if dtype == jnp.int8:
        max_val = 127.0
        scales = w_max / max_val
        val = w_blocked / scales[:, :, :, None, :]
        w_q = jnp.clip(jnp.floor(val + 0.5), -128, 127).astype(jnp.int8)
    elif dtype == jnp.float8_e4m3fn:
        max_val = 448.0
        scales = w_max / max_val
        val = w_blocked / scales[:, :, :, None, :]
        w_q = val.astype(jnp.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    w_q_tiled = w_q.reshape(n_out_outer, n_in_outer, TILE_K_SIZE, out_block_size)
    # NOTE: We do NOT return 'dtype' here because JIT functions cannot return Python types.
    return QuantizationResult(w_q_tiled, scales.astype(jnp.float32))

def _compute_quant_params(x_slice, w_scale_row, max_val, dtype):
    """
    On-the-fly quantization of activation chunk inside the kernel.
    """
    # FIX: Cast input to float32 to ensure high precision during 
    # scaling and rounding, matching _quantize_array_2d behavior.
    x_f32 = x_slice.astype(jnp.float32)
    
    x_max = jnp.max(jnp.abs(x_f32), axis=1, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6)
    scale_inv = max_val / x_max
    
    # Perform scaling in float32
    val = x_f32 * scale_inv

    if dtype == jnp.int8:
        x_q = jnp.clip(jnp.floor(val + 0.5), -128, 127).astype(jnp.int8)
    else:
        # float8: just cast
        x_q = val.astype(dtype)

    # Combined scale: (x_max / max_val) * w_scale
    scale_factor = (x_max / max_val) * w_scale_row
    return x_q, scale_factor

def _kernel_real_tiled_entry(x_ref, w_ref, scales_ref, out_ref, acc_vmem, *,
                             batch_block_size, out_block_size, dtype):
    """
    Kernel entry for V7.
    
    Args:
        x_ref: [Batch_Block, N_In]
        w_ref: [Out_Block, N_In] (Standard Layout)
        scales_ref: [SUPER_CHUNK, Out_Block] (Transposed Layout for alignment)
        out_ref: [Batch_Block, Out_Block]
    """
    acc_vmem[...] = jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32)

    # Determine constants based on dtype
    if dtype == jnp.int8:
        max_val = 127.0
        dot_preferred = jnp.int32
    else:
        max_val = 448.0
        dot_preferred = jnp.float32

    def pipeline_step(x_chunk, w_chunk_std, s_chunk_transposed):
        # FIX: Transpose issue resolved by passing transposed scales from Python.
        
        # --- Stage 0 ---
        x_0 = x_chunk[:, 0:BLOCK_K] # [B, K]
        w_0 = w_chunk_std[:, 0:BLOCK_K] # [Out, K]
        
        # s_chunk_transposed is [SUPER_CHUNK, Out_Block]
        # We take row 0. Shape [Out_Block]
        s_0 = s_chunk_transposed[0, :][None, :] # Broadcast to [1, Out]

        x_q_curr, scale_curr = _compute_quant_params(x_0, s_0, max_val, dtype)
        
        # DOT PRODUCT:
        # x_q_curr: [B, K]
        # w_0:      [Out, K]
        # Contract dim 1 of X and dim 1 of W.
        # Output: [B, Out]
        dot_curr = jax.lax.dot_general(
            x_q_curr, w_0, 
            (((1,), (1,)), ((), ())), # Contraction on Dim 1 for both
            preferred_element_type=dot_preferred
        )
        dot_prev = dot_curr
        scale_prev = scale_curr

        # --- Steady State ---
        for j in range(1, SUPER_CHUNK):
            k_start = j * BLOCK_K
            k_end = (j + 1) * BLOCK_K

            x_next = x_chunk[:, k_start:k_end]
            w_next = w_chunk_std[:, k_start:k_end] # Slice columns
            s_next = s_chunk_transposed[j, :][None, :] # Row slice (aligned)

            x_q_next, scale_next = _compute_quant_params(x_next, s_next, max_val, dtype)
            dot_next = jax.lax.dot_general(
                x_q_next, w_next, 
                (((1,), (1,)), ((), ())), 
                preferred_element_type=dot_preferred
            )

            acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev
            dot_prev = dot_next
            scale_prev = scale_next

        # --- Epilogue ---
        acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev

    # Total K size
    total_k = w_ref.shape[1]
    n_k_chunks = total_k // TILE_K_SIZE

    pltpu.emit_pipeline(
        pipeline_step,
        grid=(n_k_chunks,),
        in_specs=[
            # X: Slice [Batch, TILE_K]
            pl.BlockSpec((batch_block_size, TILE_K_SIZE), lambda i: (0, i)),
            # W: Slice [Out_Block, TILE_K] (Standard Layout)
            pl.BlockSpec((out_block_size, TILE_K_SIZE), lambda i: (0, i)),
            # Scales: Slice [SUPER_CHUNK, Out_Block] (Transposed)
            # Access pattern: We want rows i*SUPER_CHUNK to (i+1)*SUPER_CHUNK
            pl.BlockSpec((SUPER_CHUNK, out_block_size), lambda i: (i, 0))
        ]
    )(x_ref, w_ref, scales_ref)

    out_ref[...] = acc_vmem[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=['out_block_size', 'quant_dtype'])
def dispatch_real_v7(
    x: jax.Array, 
    w_q: jax.Array, 
    w_scale: jax.Array,
    out_block_size: int, 
    quant_dtype: jnp.dtype
):
    """
    Main V7 Dispatcher. Accepts Standard Layout Weights.
    
    Args:
        x: [Batch, N_In]
        w_q: [N_Out, N_In]
        w_scale: [N_Out, N_Blocks]
    """
    bs, n_in = x.shape
    n_out, _ = w_q.shape
    
    # 1. Padding
    padded_bs = next_multiple(bs, BLOCK_B)
    padded_out = next_multiple(n_out, out_block_size)
    padded_in = next_multiple(n_in, TILE_K_SIZE)
    
    if padded_bs > bs: 
        x = jnp.pad(x, ((0, padded_bs - bs), (0, 0)))
        
    if padded_in > n_in:
        padding_k = padded_in - n_in
        x = jnp.pad(x, ((0, 0), (0, padding_k)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padding_k)))
        # Pad scales. 1 scale per 128 (BLOCK_K) elements.
        pad_blocks = padding_k // BLOCK_K
        w_scale = jnp.pad(w_scale, ((0, 0), (0, pad_blocks)), constant_values=1.0)

    if padded_out > n_out:
        padding_out = padded_out - n_out
        w_q = jnp.pad(w_q, ((0, padding_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, padding_out), (0, 0)))

    # 2. Transpose Scales on Host/HBM
    # Scales are small, so this is cheap. 
    # [N_Out, N_Blocks] -> [N_Blocks, N_Out]
    w_scale_t = w_scale.T

    # 3. Grid Setup
    n_batch_blocks = padded_bs // BLOCK_B
    n_out_blocks = padded_out // out_block_size
    grid = (n_batch_blocks, n_out_blocks)
    
    total_in_supported = padded_in
    n_blocks_total = w_scale_t.shape[0] # Transposed shape

    # 4. BlockSpecs
    in_specs = [
        # X: [Batch_Block, N_In]
        pl.BlockSpec((BLOCK_B, total_in_supported), lambda b, o: (b, 0)),
        
        # W_Q: [Out_Block, N_In]
        pl.BlockSpec((out_block_size, total_in_supported), lambda b, o: (o, 0)),
        
        # W_Scale (Transposed): [N_Blocks, Out_Block]
        pl.BlockSpec((n_blocks_total, out_block_size), lambda b, o: (0, o)),
    ]

    kernel = pl.pallas_call(
        functools.partial(_kernel_real_tiled_entry,
                          batch_block_size=BLOCK_B,
                          out_block_size=out_block_size,
                          dtype=quant_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((BLOCK_B, out_block_size), lambda b, o: (b, o)),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((BLOCK_B, out_block_size), jnp.float32)]
        ),
        # FIX: Use x.dtype instead of hardcoded jnp.bfloat16
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), x.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_device_vmem_limit())
    )

    out = kernel(x, w_q, w_scale_t)
    return out[:bs, :n_out]