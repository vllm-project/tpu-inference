# SPDX-License-Identifier: Apache-2.0
"""Utility functions for quantized matmul kernel."""
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax._src import dtypes

from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import TunedValue


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
    """Minimize run-time branching of fn by converting jnp.bool to python bool."""
    if conditions:
        arg = conditions[0]
        if isinstance(arg, bool):
            unfold_args(conditions[1:], fn_conditions + (arg, ), fn)
        else:
            assert arg.dtype == jnp.bool and arg.size == 1
            jax.lax.cond(
                arg,
                lambda: unfold_args(conditions[1:], fn_conditions +
                                    (True, ), fn),
                lambda: unfold_args(conditions[1:], fn_conditions +
                                    (False, ), fn),
            )
    else:
        fn(*fn_conditions)


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


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(tuned_value: TunedValue):
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    return (
        f"quantized_matmul_kernel_{batch_block_size}_{out_block_size}_{in_block_size}"
    )


def xla_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation=True,
) -> jax.Array:
    """
    Reference (pure JAX) implementation of the quantized matmul kernel below.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_output_features, n_input_features]
        w_s: Weight quantization scale. [n_output_features]
        mesh: Mesh to shard on.
        weight_sharding: PartitionSpec for the weight tensor.

    Returns:
        Output of the quantized matmul.
    """
    if quantize_activation:
        acc_dtype = jnp.float32
        if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1, ), (1, )), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32,
        )
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(x.dtype)


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


def quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max: jax.Array,  # [1, bs_block_size]
    quant_dtype: jnp.dtype,
):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)

    # TODO(kyuyeunk): Investigate performance gain from non xlu transpose.
    scale = jnp.transpose(x_abs_max / dtype_max)
    scale_inv = jnp.nan_to_num(1 / scale, dtype_max)
    return (x * scale_inv).astype(quant_dtype), scale.astype(jnp.float32)


def get_vmem_limit(
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int,
):
    """Calculate VMEM limit for the kernel."""

    # Calculate in/out VMEM size.
    x_size = (batch_block_size * in_block_size * dtypes.itemsize_bits(x_dtype))
    x_abs_max_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)
    w_q_size = (out_block_size * in_block_size *
                dtypes.itemsize_bits(w_q_dtype))
    w_scale_size = out_block_size * dtypes.itemsize_bits(scale_dtype)
    out_size = (batch_block_size * out_block_size *
                dtypes.itemsize_bits(out_dtype))

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2  # Account for compute and vreg spills.

    # Account for double buffering.
    # Double buffering is used only if there are multiple blocks per in/out.
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if (n_out > 1) else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    # Calculate scratch VMEM size.
    acc_size = (batch_block_size * out_block_size *
                dtypes.itemsize_bits(acc_dtype))
    x_q_size = (batch_block_size * in_block_size *
                dtypes.itemsize_bits(x_q_dtype))
    x_scale_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2  # Account for compute and vreg spills.

    # Add in/out and scratch VMEM size.
    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8  # Convert bits to bytes.
    # Specify upper limit. Defaults to 96MB.
    vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

    return vmem_limit_bytes


def validate_inputs(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    x_abs_max: jax.Array,
    x_q_dtype: jnp.dtype,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
    """Verify inputs invoking the kernel."""

    if x.dtype != x_q_dtype:
        # If the input is quantized, then it should be the same subdtype as w_q
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(
                w_q.dtype, jnp.integer):
            raise ValueError(
                f"{x_q_dtype=} and {w_q.dtype=} must be the same int or float type."
            )

    # Verify input shapes.
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f'{x.shape[1]=} must be equal to {w_q.shape[1]=}')
    if w_q.shape[0] != w_scale.shape[1] and (w_scale.ndim == 3 and w_q.shape[0]
                                             != w_scale.shape[2]):
        raise ValueError(
            f"{w_q.shape[0]=} must be equal to {w_scale.shape[1]=}")
    if x_abs_max is not None and x_abs_max.shape != (1, x.shape[0]):
        raise ValueError(
            f"{x_abs_max.shape=} must be equal to (1, {x.shape[0]=})")
    if x.shape[0] % batch_block_size != 0:
        raise ValueError(
            f"{x.shape[0]=} must be a multiple of {batch_block_size=}")
    if w_q.shape[0] % out_block_size != 0:
        raise ValueError(
            f"{w_q.shape[0]=} must be a multiple of {out_block_size=}")
    if x.shape[1] % in_block_size != 0:
        raise ValueError(
            f"{x.shape[1]=} must be a multiple of {in_block_size=}")


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
