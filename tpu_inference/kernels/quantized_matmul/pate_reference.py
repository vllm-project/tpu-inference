import functools
import jax
import jax.numpy as jnp

F8_E4M3FN_MAX = 448.0

def quantize_along_axis(x: jax.Array, dtype: jnp.dtype, dim: int = -1):
    """Quantizes a tensor along a specified dimension (1D quantization).
    
    Args:
        x: Input array to quantize
        dtype: Target quantization dtype (int8, float8_e4m3fn, etc.)
        dim: Dimension along which to compute scales (default: -1 for rows)
        
    Returns:
        x_q: Quantized array in dtype
        dequant_scale: Float32 scale factors for dequantization
    """
    x_abs_max = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    x_abs_max = jnp.maximum(x_abs_max, jnp.finfo(jnp.float16).tiny)

    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = float(dtype_info.max)
        
        dequant_scale = x_abs_max / max_val
        
        x_scaled = x / dequant_scale
        x_scaled = jnp.round(x_scaled)
        
        x_q = jnp.clip(x_scaled, dtype_info.min, dtype_info.max).astype(dtype)
        return x_q, dequant_scale.astype(jnp.float32)

    elif dtype == jnp.float8_e4m3fn:
        dequant_scale = x_abs_max / F8_E4M3FN_MAX
        x_scaled = x / dequant_scale
        x_q = x_scaled.astype(dtype)
        return x_q, dequant_scale.astype(jnp.float32)
    else:
        raise TypeError(f"Unsupported dtype for quantization: {dtype}")


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
    abs_max = jnp.maximum(abs_max, jnp.finfo(jnp.float16).tiny)

    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = float(dtype_info.max)
        
        dequant_scale = abs_max / max_val
        scale_expanded = jnp.expand_dims(dequant_scale, axis=-1)
        
        x_scaled = x_blocked / scale_expanded
        x_scaled = jnp.round(x_scaled)
        
        # Clip before casting to prevent overflow
        x_q_blocked = jnp.clip(x_scaled, dtype_info.min, dtype_info.max).astype(dtype)
        
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        return x_q, dequant_scale.astype(jnp.float32)

    elif dtype == jnp.float8_e4m3fn:
        dequant_scale = abs_max / F8_E4M3FN_MAX
        scale_expanded = jnp.expand_dims(dequant_scale, axis=-1)
        
        x_scaled = x_blocked / scale_expanded
        
        # No rounding for float8
        x_q_blocked = x_scaled.astype(dtype)
        
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        return x_q, dequant_scale.astype(jnp.float32)
    else:
        raise TypeError(f"Unsupported dtype for quantization: {dtype}")


@functools.partial(jax.jit, static_argnames=["quantize_activation", "act_quant_dim"])
def reference_quantized_matmul_1d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool = True,
    act_quant_dim: int = -1
):
    """Reference implementation of 1D (per-row) quantized matrix multiplication.
    
    Args:
        x: Input activations [batch_size, in_features]
        w_q: Pre-quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features] or [1, out_features]
        quantize_activation: If True, quantize x on-the-fly
        act_quant_dim: Dimension along which to quantize activations
        
    Returns:
        Output activations [batch_size, out_features] in x.dtype
    """
    quant_dtype = w_q.dtype
    if quantize_activation:
        acc_dtype = jnp.float32
        if jnp.issubdtype(quant_dtype, jnp.integer):
            acc_dtype = jnp.int32
            
        x_q, x_scale = quantize_along_axis(x, quant_dtype, dim=act_quant_dim)

        out = jax.lax.dot_general(
            x_q.astype(jnp.float32) if acc_dtype == jnp.float32 else x_q,
            w_q.astype(jnp.float32) if acc_dtype == jnp.float32 else w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x, w_q.astype(jnp.float32),
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    if w_scale.ndim == 1:
        w_scale = jnp.expand_dims(w_scale, 0)
    elif w_scale.shape[1] == 1:
        w_scale = jnp.transpose(w_scale)
    
    out *= w_scale
    return out.astype(x.dtype)


@functools.partial(jax.jit, static_argnames=[
    "quantize_activation", "block_size", "quant_dtype"
])
def reference_quantized_matmul_2d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    block_size: int,
    quant_dtype: jnp.dtype = jnp.int8,
    quantize_activation: bool = True
) -> jax.Array:
    """Reference implementation of 2D (block-wise) quantized matrix multiplication.

    Args:
        x: Input activations [batch_size, in_features]
        w_q: Pre-quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features, n_in_blocks]
        block_size: Size of each quantization block
        quant_dtype: Quantization dtype for activations
        quantize_activation: If True, quantize x on-the-fly block-wise
        
    Returns:
        Output activations [batch_size, out_features] in x.dtype
    """
    bs, n_in = x.shape
    n_out, _ = w_q.shape
    n_in_blocks = n_in // block_size

    acc_dtype = jnp.float32
    if quantize_activation:
        if jnp.issubdtype(quant_dtype, jnp.integer):
            acc_dtype = jnp.int32
        x_q, x_scale = quantize_2d_blocked(x, block_size, quant_dtype)
    else:
        x_q = x
        x_scale = jnp.ones((bs, n_in_blocks), dtype=jnp.float32)

    out = jnp.zeros((bs, n_out), dtype=jnp.float32)
    
    # Iterate over blocks and accumulate dequantized results
    for block_idx in range(n_in_blocks):
        block_start = block_idx * block_size
        block_end = (block_idx + 1) * block_size
        
        x_q_block = x_q[:, block_start:block_end]
        w_q_block = w_q[:, block_start:block_end]
        
        x_s_block = x_scale[:, block_idx]
        w_s_block = w_scale[:, block_idx]

        block_out = jax.lax.dot_general(
            x_q_block, w_q_block,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        
        # Dequantize immediately (scale per block)
        block_out *= jnp.expand_dims(x_s_block, 1)
        block_out *= jnp.expand_dims(w_s_block, 0)
        
        out += block_out
        
    return out.astype(x.dtype)