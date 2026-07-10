# SPDX-License-Identifier: Apache-2.0
"""Utility functions for quantized matmul."""
import jax
import jax.numpy as jnp


def quantize_tensor(x: jax.Array,
                    dtype: jnp.dtype,
                    dim: int = -1,
                    block_size: int | None = None):
    if block_size is not None:
        # Flatten all leading dims into a single batch dim for block
        # quantization, then restore the original shape.
        orig_shape = x.shape
        k_dim = orig_shape[-1]
        x_flat = x.reshape(-1, k_dim)
        n_dim = x_flat.shape[0]
        x_reshaped = x_flat.reshape(n_dim, -1, block_size)
        x_q, scale = quantize_block(x_reshaped, axis=-1, target_dtype=dtype)

        x_q = x_q.reshape(orig_shape)

        return x_q, scale.transpose(1, 2, 0).astype(jnp.float32)
    data_q, scale = quantize_block(x, axis=dim, target_dtype=dtype)
    return data_q, scale.astype(jnp.float32)


def xla_quantized_batched_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    dimension_numbers: tuple,
    quantize_activation: bool = True,
) -> jax.Array:
    """Quantized matmul with batch dimensions via dot_general.

    Generalizes xla_quantized_matmul to support batch dimensions (axes
    shared between both operands that appear in the output). Uses
    jax.lax.dot_general with configurable dimension_numbers.

    Args:
        x: Activation tensor (e.g. [T, N, H]).
        w_q: Quantized weight (e.g. [A, N, H]).
        w_scale: Per-output-channel weight scale (e.g. [A]).
        dimension_numbers: ``((contracting_x, contracting_w),
            (batch_x, batch_w))`` for dot_general.
        quantize_activation: Whether to dynamically quantize activations.

    Returns:
        Output of the batched quantized matmul. Shape is determined by
        dot_general: batch dims first, then lhs free dims, then rhs free dims.
    """
    contract_dims, batch_dims = dimension_numbers

    if quantize_activation:
        acc_dtype = jnp.float32
        if jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        # Permute x_scale to match dot_general output ordering.
        # dot_general output: batch dims, lhs free dims, rhs free dims.
        # x_scale has same shape as x but with contracting dim(s) as 1.
        contract_set = set(contract_dims[0])
        batch_set = set(batch_dims[0])
        lhs_free = [
            i for i in range(x.ndim)
            if i not in contract_set and i not in batch_set
        ]
        perm = list(batch_dims[0]) + lhs_free + list(contract_dims[0])
        if perm != list(range(x.ndim)):
            x_scale = jnp.transpose(x_scale, perm)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=jnp.float32,
        )

    # Broadcast w_scale to match the output shape.
    # dot_general output order: batch dims, lhs free dims, rhs free dims.
    # w_scale is per-output-channel (rhs free dims), so expand leading dims.
    n_leading = out.ndim - w_scale.ndim
    for _ in range(n_leading):
        w_scale = jnp.expand_dims(w_scale, 0)
    out *= w_scale
    return out.astype(x.dtype)


def get_max_min(target_dtype):
    if jnp.issubdtype(target_dtype, jnp.floating):
        return jnp.finfo(target_dtype).max.astype(
            jnp.float32), jnp.finfo(target_dtype).min.astype(jnp.float32)
    else:
        return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min


def quantize_block(data, axis, target_dtype):
    """Calculates scale and quantizes a block of data."""
    abs_max = jnp.max(
        jnp.abs(data),
        axis=axis,
        keepdims=True,
    )
    dtype_max, dtype_min = get_max_min(target_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)

    if jnp.issubdtype(target_dtype, jnp.floating):
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    else:
        data_q = jnp.round(data / scale).astype(target_dtype)
    return data_q, scale
