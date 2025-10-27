# SPDX-License-Identifier: Apache-2.0
"""Utility functions for quantized matmul kernel."""
from typing import Any, Callable

import jax
import jax.numpy as jnp

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


def quantize_tensor(x: jax.Array, dtype: jnp.dtype, dim: int = -1):
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    x_abs_max = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    scale = x_abs_max / max_val
    x_q = jnp.clip(x / scale, min_val, max_val).astype(dtype)
    return x_q, scale.astype(jnp.float32)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(tuned_value: TunedValue):
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    return f'quantized_matmul_kernel_{batch_block_size}_{out_block_size}_{in_block_size}'
