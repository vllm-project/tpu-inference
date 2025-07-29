# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
import bisect
import functools
import time
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
from jax._src.interpreters import pxla
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons import utils
from tpu_commons.logger import init_logger

DEFAULT_KV_CACHE_DTYPE = jnp.bfloat16
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


def get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        num = get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    logger.info(f"Prepared request paddings: {paddings}")
    return paddings


def get_token_paddings(min_token_size: int, max_token_size: int,
                       padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        while True:
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        while num <= padding_gap:
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            paddings.append(num)
    logger.info(f"Prepared token paddings: {paddings}")
    return paddings


def get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x.
    """
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


class LatencyTracker:

    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        logger.debug(f"Latency for '{self.name}': {elapsed_time:.3f} seconds")


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


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    layer_names: List[str],
    devices: List[Any],
) -> List[jax.Array]:
    """
    Creates the KV caches, one per each decoder layer in the model, where the shape of each cache is
    (num_blocks, block_size, num_kv_heads * 2, head_size).

    Args:
        num_blocks: The number of blocks in the KV cache.
        block_size: The size of each block in the KV cache.
        num_kv_heads: The number of KV heads in the KV cache.
        head_size: The size of each head in the KV cache.
        mesh: The mesh to shard the KV caches across.
        layer_names: The names of the decoder layers in the model.
        devices: The devices to shard the KV caches across.

    Returns:
        A list of KV caches, one per each decoder layer in the model.

    """
    # TODO (jacobplatz): update this for quantized KV cache
    cache_dtype = DEFAULT_KV_CACHE_DTYPE
    # TODO(xiang): fix this together with get_kv_cache_spec
    # cache_dtype = kv_cache_spec.dtype

    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads * 2,
        head_size,
    )

    # Shard the num_kv_heads dim along the 'model' axis.
    sharding = NamedSharding(mesh, PartitionSpec("data", None, "model"))

    def _allocate() -> jax.Array:
        return jnp.empty(
            shape=cache_shape,
            dtype=cache_dtype,
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    kv_caches = []
    for _ in layer_names:
        kv_caches.append(sharded_allocate())
    logger.info(f"Init kv-cache | "
                f"shape={len(layer_names)} * {cache_shape} | "
                f"sharding={sharding} | "
                f"dtype={cache_dtype} | "
                f"hbm={utils.hbm_usage_gb(devices)}Gb")
    return kv_caches
