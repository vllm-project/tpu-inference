# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

import functools
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Tuple

import jax
from vllm.config import get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory

from tpu_inference.kernels.dma.host_dma import d2h_dma, h2d_dma
from tpu_inference.logger import init_logger

# Corresponds to the initial hash value
NONE_HASH = 0

logger = init_logger(__name__)

CPU_OFFLOADING_SWAP_OP_TYPE = Literal["jax", "pallas"]


@dataclass(order=True)
class CacheKey:
    """
    A key for the cache engine.
    """
    model_name: str
    chunk_hash: int

    def __hash__(self):
        return hash((
            self.model_name,
            self.chunk_hash,
        ))

    def __eq__(self, other):
        if type(self) is type(other):
            return (self.model_name == other.model_name
                    and self.chunk_hash == other.chunk_hash)
        return False


class TokenProcessor:

    def __init__(self, model_name: str, chunk_size: int = 16):
        self.model_name = model_name
        self.chunk_size = chunk_size
        logger.info(f"TokenProcessor initialized with chunk_size={chunk_size}")

    def _hash_tokens(
        self,
        tokens: List[int],
        prefix_hash: Optional[int] = None,
    ) -> int:
        hasher = hashlib.sha256()
        hasher.update(str(prefix_hash).encode('utf-8'))
        hasher.update(str(tuple(tokens)).encode('utf-8'))
        return int(hasher.hexdigest(), 16)

    def process_tokens(
        self,
        tokens: Optional[List[int]] = None,
    ) -> Iterable[Tuple[int, int, CacheKey]]:
        """Process the tokens and return the corresponding cache keys."""
        if not tokens:
            return

        total_len = len(tokens)
        prefix_hash = NONE_HASH

        for i in range(0, total_len, self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            prefix_hash = self._hash_tokens(chunk, prefix_hash)
            start_idx = i
            end_idx = min(start_idx + self.chunk_size, total_len)
            logger.info(
                f"Processing chunk: start={start_idx}, end={end_idx}, hash={prefix_hash}"
            )
            yield (
                start_idx,
                end_idx,
                CacheKey(model_name=self.model_name, chunk_hash=prefix_hash),
            )


def get_kv_connector_cache_layout():
    """
    Retrieve the required kv cache layout for the configured kv connector
    Return: None, when no kv_transfer_config is found; otherwise, the layout str
    """
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = \
            connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD.")
    return None


SwapFn = Callable[
    [
        List[jax.Array],  # src_kv_caches
        jax.sharding.NamedSharding,  # src_sharding
        jax.sharding.NamedSharding,  # dst_sharding
        Literal["h2d", "d2h"],  # direction
    ],
    List[jax.Array],  # return value
]

JittedKVCacheSwapFn = Callable[[List[jax.Array]], List[jax.Array]]


# NOTE(jcgu): keep the same interface as the pallas one
def jax_swap_kv_caches(
    src_kv_caches: List[jax.Array],
    src_sharding: jax.sharding.NamedSharding,
    dst_sharding: jax.sharding.NamedSharding,
    direction: Literal["h2d", "d2h"],
) -> List[jax.Array]:
    """Swap in / out multi-layer kv_cache using jax device_put

    Args:
        src_kv_caches: [kv_cache of each layer]
        src_sharding: kv_caches' original sharding
        dst_sharding: kv_caches' target sharding (different memory_kind)
        direction: h2d -> swap_in, d2h -> swap_out
    Returns:
        a list of jax.Array objects with the dst_sharding
    """

    def _jax_device_put(input_array):
        return jax.device_put(input_array, dst_sharding)

    return jax.tree.map(_jax_device_put, src_kv_caches)


def pallas_swap_kv_caches(
    src_kv_caches: List[jax.Array],
    src_sharding: jax.sharding.NamedSharding,
    dst_sharding: jax.sharding.NamedSharding,
    direction: Literal["h2d", "d2h"],
) -> List[jax.Array]:
    """Swap in / out multi-layer kv_cache using pallas dma kernel

    Args:
        src_kv_caches: [kv_cache of each layer]
        src_sharding: kv_caches' original sharding
        dst_sharding: kv_caches' target sharding (different memory_kind)
        direction: h2d -> swap_in, d2h -> swap_out
    Returns:
        a list of jax.Array objects with the dst_sharding
    """

    def swap_in_fn(inputs, input_sharding, out_sharding):

        def _swap_in(hbm_sharded_array):
            return h2d_dma(hbm_sharded_array, input_sharding, out_sharding)

        return jax.tree.map(_swap_in, inputs)

    def swap_out_fn(inputs, input_sharding, out_sharding):

        def _swap_out(hbm_sharded_array):
            return d2h_dma(hbm_sharded_array, input_sharding, out_sharding)

        return jax.tree.map(_swap_out, inputs)

    if direction == "d2h":
        return swap_out_fn(src_kv_caches, src_sharding, dst_sharding)
    elif direction == "h2d":
        return swap_in_fn(src_kv_caches, src_sharding, dst_sharding)


def get_jitted_kv_cache_swap_fn(
    swap_op_type: CPU_OFFLOADING_SWAP_OP_TYPE,
    host_sharding: jax.sharding.NamedSharding,
    device_sharding: jax.sharding.NamedSharding
) -> Tuple[JittedKVCacheSwapFn, JittedKVCacheSwapFn]:
    """jit compile the swap_in and swap_out functions

    Args:
        swap_op_type : (str) pallas or jax
        host_sharding:
        device_sharding:

    Returns:
        A tuple containing the jitted swap-in and swap-out functions.
    """
    _swap_fn: SwapFn = pallas_swap_kv_caches if swap_op_type == "pallas" else jax_swap_kv_caches
    # swap_in (host_sharding), swap_out (device_sharding)
    swap_in_fn = functools.partial(jax.jit(
        _swap_fn,
        static_argnames=["src_sharding", "dst_sharding", "direction"],
        out_shardings=device_sharding),
                                   src_sharding=host_sharding,
                                   dst_sharding=device_sharding,
                                   direction="h2d")
    swap_out_fn = functools.partial(jax.jit(
        _swap_fn,
        static_argnames=["src_sharding", "dst_sharding", "direction"],
        out_shardings=host_sharding),
                                    src_sharding=device_sharding,
                                    dst_sharding=host_sharding,
                                    direction="d2h")
    return swap_in_fn, swap_out_fn
