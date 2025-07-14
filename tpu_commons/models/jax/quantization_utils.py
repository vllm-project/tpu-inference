# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp

MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)


def quantize(x: jax.Array, quant_dtype: jnp.dtype):
    """Quantizes uses a per-tensor approach.
      TODO (jacobplatin): support a per-token approach

    Args:
        x: the value to quantize
        quant_dtype: the dtype to quantize to

    Returns:
         x (jax.Array): the quantized value
         scale (jax.Array): the scale factor (of shape (1,))
          NOTE: this should really be a float, but static types don't play
          nicely with JAX tracing
    """
    # Would be nicer to do this as a dictionary, but indexing with
    # a jnp.dtype didn't work for some reason
    if quant_dtype == jnp.int8:
        dtype_max = MAX_INT8
    elif quant_dtype == jnp.int4:
        dtype_max = MAX_INT4
    elif quant_dtype == jnp.float8_e4m3fn:
        dtype_max = E4M3_MAX
    else:
        raise ValueError(f"Unsupported quant dtype: {quant_dtype}")

    scale = jnp.max(jnp.abs(x)) / dtype_max

    # Ensure scales are not zero to avoid division by zero errors.
    scale = jnp.maximum(scale, 1e-6)

    x = (x / scale).astype(quant_dtype)

    # Upcast to float32 to avoid a SMEM Mosaic error with bfloat16
    # NOTE: the scales are really floats but static types don't play
    # nicely with JAX tracing
    scale = scale.reshape(-1).astype(jnp.float32)

    return x, scale
