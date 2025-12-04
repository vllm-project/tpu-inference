# SPDX-License-Identifier: Apache-2.0
"""Utility functions for 2D quantized matmul."""

import jax
import jax.numpy as jnp

F8_E4M3FN_MAX = 448.0
EPS = jnp.finfo(jnp.float16).tiny

def quantize_2d_blocked(x: jax.Array, block_size: int, dtype: jnp.dtype = jnp.int8):
    """Quantizes a 2D tensor using block-wise quantization.
    
    Args:
        x: Input array of shape [n_rows, n_cols]
        block_size: Size of each quantization block
        dtype: Target quantization dtype (int8, float8_e4m3fn, etc.)
        
    Returns:
        x_q: Quantized array of shape [n_rows, n_cols] in dtype
        dequant_scale: Float32 scales of shape [n_rows, n_col_blocks]
    """
    n_rows, n_cols = x.shape
    if n_cols % block_size != 0:
        raise ValueError(
            f"Number of columns {n_cols} must be divisible by block_size {block_size}"
        )
    n_col_blocks = n_cols // block_size
    x_blocked = x.reshape(n_rows, n_col_blocks, block_size)

    abs_max = jnp.max(jnp.abs(x_blocked), axis=-1)
    abs_max = jnp.maximum(abs_max, EPS)

    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = float(dtype_info.max)
        
        dequant_scale = abs_max / max_val
        scale_expanded = jnp.expand_dims(dequant_scale, axis=-1)
        
        x_scaled = x_blocked / scale_expanded
        x_scaled = jnp.round(x_scaled)
        
        x_q_blocked = jnp.clip(x_scaled, dtype_info.min, dtype_info.max).astype(dtype)
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        return x_q, dequant_scale.astype(jnp.float32)

    elif dtype == jnp.float8_e4m3fn:
        dequant_scale = abs_max / F8_E4M3FN_MAX
        scale_expanded = jnp.expand_dims(dequant_scale, axis=-1)
        
        x_scaled = x_blocked / scale_expanded
        x_q_blocked = x_scaled.astype(dtype)
        
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        return x_q, dequant_scale.astype(jnp.float32)
    else:
        raise TypeError(f"Unsupported dtype for quantization: {dtype}")