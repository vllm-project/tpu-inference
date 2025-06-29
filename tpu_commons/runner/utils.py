# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
from typing import Optional

import jax.numpy as jnp


def determine_do_sampling(top_k: int, temperature: float) -> bool:
    """
  Determine whether sampling should be done for the next tokens in the model forward pass.

  Args:
    top_k: The top_k value (from SamplingParams).
    temperature: The temperature value (from SamplingParams).

  Returns:
    True if sampling should be done, False otherwise.
  """
    return top_k != 1 and temperature != 0.0


def get_jnp_dtype_from_str(dtype_str: str) -> type:
    """
    Gets the JAX numpy dtype from a string.

    Args:
      dtype_str: The string representation of the dtype.

    Returns:
      The JAX numpy dtype.
    """
    str_to_dtype_dict = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "int8": jnp.int8,
        "fp8": jnp.
        float8_e4m3fn  # TODO (jacobplatin): is this the correct float8 dtype?
    }

    if dtype_str not in str_to_dtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return str_to_dtype_dict[dtype_str]


def pad_to_multiple(x: int,
                    multiple: int = 8,
                    max_limit: Optional[int] = None,
                    keep_one: bool = False) -> int:
    assert x > 0
    if keep_one and x == 1:
        return x
    x = x + (-x % multiple)
    if max_limit is not None:
        x = min(x, max_limit)
    return x
