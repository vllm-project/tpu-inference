# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
import functools
import time
from typing import Optional

import jax.numpy as jnp
from jax._src.interpreters import pxla
from vllm.logger import init_logger

MIN_NUM_SEQS = 8

logger = init_logger(__name__)


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


def get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


class LatencyTracker:

    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        logger.info(f"Latency for '{self.name}': {elapsed_time:.3f} seconds")


class ForbidCompile:
    """
    A context manager to forbid JAX compilation in a specific block of code.

    It works by temporarily wrapping the internal JAX caching function
    `_cached_lowering_to_hlo`. If a call within the `with` block results
    in a cache miss (i.e., triggers a new compilation), it raises a
    RuntimeError.

    Usage:
        # This will raise an error because it's the first compilation.
        with ForbidCompile():
            jitted_func(x)

        # "Warm up" the cache first.
        jitted_func(x)
        # This will now succeed without error.
        with ForbidCompile():
            jitted_func(x)
    """

    def __init__(
            self,
            message="JAX compilation occurred but was forbidden in this context."
    ):
        self.message = message
        self._original_func = None

    def __enter__(self):
        # Store the original function
        self._original_func = pxla._cached_lowering_to_hlo
        original_cached_func = self._original_func

        # Create a wrapper
        @functools.wraps(original_cached_func)
        def wrapper(*args, **kwargs):
            # Get cache statistics before the call
            info_before = original_cached_func.cache_info()
            misses_before = info_before.misses

            # Execute the original cached function
            result = original_cached_func(*args, **kwargs)

            # Get cache statistics after the call
            info_after = original_cached_func.cache_info()
            misses_after = info_after.misses

            # Check if a cache miss occurred
            if misses_after > misses_before:
                raise RuntimeError(self.message)

            return result

        # Monkey-patch the function with our wrapper
        pxla._cached_lowering_to_hlo = wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original function
        if self._original_func:
            pxla._cached_lowering_to_hlo = self._original_func
        # Don't suppress any exceptions that occurred inside the 'with' block
        return False


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
