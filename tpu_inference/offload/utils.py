# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from vllm.config import get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory

from tpu_inference.logger import init_logger

ReqId = str

CpuChunkId = int

# Corresponds to the initial hash value
NONE_HASH = 0

logger = init_logger(__name__)

CPU_OFFLOADING_SWAP_OP_TYPE = Literal["jax", "pallas", "jax_copy"]


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


# TODO(jcgu): cleanup
@functools.partial(
    jax.jit,
    static_argnames=("block_size"),
    donate_argnames=(
        "kv_caches",
        "kv_cache_slices",
    ),
)
def jitted_insert_kv_cache_slices(
    block_size,
    kv_caches: List[jax.Array],
    kv_cache_slices: List[List[jax.Array]],
    block_numbers: jax.Array,
) -> List[jax.Array]:
    """
    JIT-compiled function to insert KV cache slices into the physical
    cache for all layers at once. This fuses reshape, and scatter
    operations into a single efficient kernel.
    """

    def _update_layer(cache, slices):
        """The function to apply to each layer's cache and slices."""
        # new_shape = (1, block_size, *slices[0].shape[1:])
        for (i, block_idx) in enumerate(block_numbers):
            # reshaped_block = slices[i].reshape(new_shape)
            reshaped_block = jax.lax.expand_dims(slices[i], dimensions=(0, ))
            cache = jax.lax.dynamic_update_slice_in_dim(cache,
                                                        reshaped_block,
                                                        block_idx,
                                                        axis=0)
        return cache

    return jax.tree.map(_update_layer, kv_caches, kv_cache_slices)


@functools.partial(jax.jit, static_argnames=['num_blocks'])
def stack_kv_cache_cross_layers(kv_caches: List[jax.Array],
                                block_ids: jax.Array,
                                num_blocks: int) -> List[jax.Array]:
    """
    This uses jax.tree.map to apply the operation across all layers.
    """

    def _gather_blocks(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    gathered_kv_layers = jax.tree.map(_gather_blocks, kv_caches)
    stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)

    # Split the stacked_blocks along axis=0 into individual blocks
    # NOTE(jcgu): num_blocks == len(block_ids)
    split_blocks = jnp.split(stacked_blocks,
                             indices_or_sections=num_blocks,
                             axis=0)
    # split_blocks = jnp.array_split(stacked_blocks, num_blocks, axis=0)

    return split_blocks


@functools.partial(
    jax.jit,
    static_argnames=("dim_nums", ),
    donate_argnames=(
        "kv_caches",
        "stacked_blocks",
    ),
)
def update_kv_caches(
        kv_caches: List[jax.Array], stacked_blocks: List[jax.Array],
        block_indices: jax.Array,
        dim_nums: jax.lax.ScatterDimensionNumbers) -> List[jax.Array]:
    """
    Updates KV caches by unstacking gathered blocks and inserting slices using scatter.

    Args:
      kv_caches: List of original KV caches for each layer.
      stacked_blocks: List of gathered blocks, each with shape (1, num_layers, ...).
      block_indices: Array of block indices to update.

    Returns:
      List of updated KV caches for each layer.
    """
    concatenated_blocks = jnp.concatenate(stacked_blocks, axis=0)
    layer_slices_tuple = jnp.unstack(concatenated_blocks, axis=1)
    layer_slices_list = list(layer_slices_tuple)

    def _update_layer(cache_layer, slices):
        # NOTE(jcgu): make it as a static arg
        # dnums = jax.lax.ScatterDimensionNumbers(
        #     update_window_dims=tuple(range(1, len(slices.shape))),
        #     inserted_window_dims=(0,),
        #     scatter_dims_to_operand_dims=(0,)
        # )
        return jax.lax.scatter(cache_layer, block_indices[:, None], slices,
                               dim_nums)

    return jax.tree.map(_update_layer, kv_caches, layer_slices_list)
