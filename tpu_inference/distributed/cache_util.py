# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

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


def swap_ops(
    src_kv_cache: jax.Array,
    out_sharding: Optional[jax.sharding.NamedSharding],
    direction: Literal["h2d", "d2h"],
    op_type: CPU_OFFLOADING_SWAP_OP_TYPE,
) -> jax.Array:
    if op_type == "jax":
        return jax_swap_kv_cache(src_kv_cache, out_sharding, direction)
    return dma_kv_cache(src_kv_cache, out_sharding, direction)


def jax_swap_kv_cache(
    src_kv_cache: jax.Array,
    out_sharding: Optional[jax.sharding.NamedSharding],
    direction: Literal["h2d", "d2h"],
) -> jax.Array:
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(src_kv_cache,
                          cpu_device if direction == "d2h" else out_sharding)


def dma_kv_cache(
    src_kv_cache: jax.Array,
    out_sharding: jax.sharding.NamedSharding,
    direction: CPU_OFFLOADING_SWAP_OP_TYPE,
) -> jax.Array:
    dma_fn = d2h_dma if direction == "d2h" else h2d_dma
    return dma_fn(src_kv_cache, out_sharding)
