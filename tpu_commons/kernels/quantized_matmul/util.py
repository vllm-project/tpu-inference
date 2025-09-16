# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import jax
import jax.numpy as jnp


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
    """Minimize run-time branching by converting jnp.bool to python bool."""
    if len(conditions) == 0:
        fn(*fn_conditions)
    else:
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


def quantize_tensor(x: jax.Array, dtype: jnp.dtype, dim: int = -1):
    if jnp.issubdtype(dtype, jnp.floating):
        dtype_info = jnp.finfo(dtype)
    else:
        dtype_info = jnp.iinfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    max_val = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    scale = max_val / dtype_max
    x_q = jnp.clip(x / scale, dtype_min, dtype_max).astype(dtype)
    return x_q, scale.astype(jnp.float32)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(bs_block_size, out_block_size, in_block_size):
    kernel_id = f'{bs_block_size}_{out_block_size}_{in_block_size}'
    return f'quantized_matmul_kernel_{kernel_id}'
