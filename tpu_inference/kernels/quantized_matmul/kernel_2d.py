# SPDX-License-Identifier: Apache-2.0
"""2D (Block-wise) Quantized matmul kernel.

This module implements "Block-wise" or "Sub-channel" quantization.
Unlike standard 1D (per-channel) quantization, where a single scale factor applies 
to an entire row/column, 2D quantization breaks the reduction dimension into 
smaller blocks (e.g., 128 elements).
"""

import functools

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
        x_scaled = jnp.round(x_scaled)
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