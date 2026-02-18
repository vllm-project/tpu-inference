# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from vllm.config import get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory

from tpu_inference.kernels.dma.host_dma import d2h_dma, h2d_dma
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


SwapFn = Callable[
    [
        List[jax.Array],  # src_kv_caches
        jax.sharding.NamedSharding,  # src_sharding
        jax.sharding.NamedSharding,  # dst_sharding
        Literal["h2d", "d2h"],  # direction
    ],
    List[jax.Array],  # return value
]

KVCacheSwapFn = Callable[[List[jax.Array]], List[jax.Array]]


def _split_per_shard_to_chunks(
    per_shard_data: List[List[jax.Array]],
    split_size_list: List[int],
) -> List[List[List[jax.Array]]]:
    """Split per-shard layer data into per-block chunks.

    Args:
        per_shard_data: List[List[jax.Array]] — layers × device shards,
            where each shard has all blocks concatenated along axis 0.
        split_size_list: sizes to split along axis 0 (one per block).

    Returns:
        List[List[List[jax.Array]]] — layers × blocks × device shards.
    """
    chunks_on_cpu = []
    for layer_shards in per_shard_data:
        split_per_shard = [
            jax.lax.split(shard, split_size_list, axis=0)
            for shard in layer_shards
        ]
        num_blocks = len(split_per_shard[0])
        layer_chunks = [[
            split_per_shard[s][b] for s in range(len(layer_shards))
        ] for b in range(num_blocks)]
        chunks_on_cpu.append(layer_chunks)
    return chunks_on_cpu


def jax_copy_swap_kv_caches(
    src_kv_caches,
    src_sharding: jax.sharding.NamedSharding,
    dst_sharding: jax.sharding.NamedSharding,
    direction: Literal["h2d", "d2h"],
):
    """Swap in / out multi-layer kv_cache by copying individual device shards.

    For d2h: accesses each device shard separately, copies it, and returns
        a list of list of single-device arrays (one inner list per layer).
    For h2d: accepts a list of lists. Each inner element can be either:
        - jax.Array (old format): transferred via device_put with dst_sharding
        - List[jax.Array] (new per-shard format): each shard is device_put
          individually and reassembled via make_array_from_single_device_arrays

    Args:
        src_kv_caches: For d2h: List[jax.Array] (one per layer).
                       For h2d: List[List[jax.Array | List[jax.Array]]]
                           (layers × blocks, each block in old or new format).
        src_sharding: kv_caches' original sharding
        dst_sharding: kv_caches' target sharding (different memory_kind)
        direction: h2d -> swap_in, d2h -> swap_out
    Returns:
        d2h: List[List[jax.Array]] — per-layer lists of copied shards.
        h2d: List[List[jax.Array]] — per-layer lists of blocks on device.
    """
    if direction == "d2h":
        # Copy each device shard individually and return as list of lists.
        dst_memory_kind = dst_sharding.memory_kind
        result = []
        for kv_cache in src_kv_caches:
            shards = [
                jax.device_put(
                    shard.data,
                    jax.sharding.SingleDeviceSharding(
                        shard.device, memory_kind=dst_memory_kind))
                for shard in kv_cache.addressable_shards
            ]
            result.append(shards)
        return result
    else:  # h2d
        dst_memory_kind = dst_sharding.memory_kind
        devices = list(dst_sharding.mesh.devices.flat)
        result = []
        for layer_blocks in src_kv_caches:
            layer_result = []
            for block in layer_blocks:
                if isinstance(block, list):
                    # New format: block is a list of per-device shards.
                    copied_shards = [
                        jax.device_put(
                            shard,
                            jax.sharding.SingleDeviceSharding(
                                devices[i], memory_kind=dst_memory_kind))
                        for i, shard in enumerate(block)
                    ]
                    shard_shape = copied_shards[0].shape
                    spec = dst_sharding.spec
                    global_shape = tuple(
                        shard_shape[i] *
                        (dst_sharding.mesh.shape[spec[i]]
                         if i < len(spec) and spec[i] is not None else 1)
                        for i in range(len(shard_shape)))
                    arr = jax.make_array_from_single_device_arrays(
                        shape=global_shape,
                        sharding=dst_sharding,
                        arrays=copied_shards,
                    )
                    layer_result.append(arr)
                else:
                    # Old format: block is a single jax.Array.
                    layer_result.append(jax.device_put(block, dst_sharding))
            result.append(layer_result)
        return result


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

        def _swap_in(host_sharded_array):
            return h2d_dma(host_sharded_array, input_sharding, out_sharding)

        return jax.tree.map(_swap_in, inputs)

    def swap_out_fn(inputs, input_sharding, out_sharding):

        def _swap_out(hbm_sharded_array):
            return d2h_dma(hbm_sharded_array, input_sharding, out_sharding)

        return jax.tree.map(_swap_out, inputs)

    if direction == "d2h":
        return swap_out_fn(src_kv_caches, src_sharding, dst_sharding)
    elif direction == "h2d":
        return swap_in_fn(src_kv_caches, src_sharding, dst_sharding)


def get_kv_cache_swap_fn(
    swap_op_type: CPU_OFFLOADING_SWAP_OP_TYPE,
    host_sharding: jax.sharding.NamedSharding,
    device_sharding: jax.sharding.NamedSharding,
    jitted: bool = True,
) -> Tuple[KVCacheSwapFn, KVCacheSwapFn]:
    """get the right swap_in and swap_out functions

    Args:
        swap_op_type : (str) pallas or jax
        host_sharding:
        device_sharding:

    Returns:
        A tuple containing the jitted swap-in and swap-out functions.
    """
    _swap_fn: SwapFn = (pallas_swap_kv_caches if swap_op_type == "pallas" else
                        jax_copy_swap_kv_caches
                        if swap_op_type == "jax_copy" else jax_swap_kv_caches)
    if jitted and swap_op_type != "jax_copy":
        _swap_in_fn = jax.jit(
            _swap_fn,
            static_argnames=["src_sharding", "dst_sharding", "direction"],
            out_shardings=device_sharding)
        _swap_out_fn = jax.jit(
            _swap_fn,
            static_argnames=["src_sharding", "dst_sharding", "direction"],
            out_shardings=host_sharding)
    else:
        _swap_in_fn = _swap_fn
        _swap_out_fn = _swap_fn

    # swap_in (h2d)
    swap_in_fn = functools.partial(_swap_in_fn,
                                   src_sharding=host_sharding,
                                   dst_sharding=device_sharding,
                                   direction="h2d")
    # swap_out (d2h)
    swap_out_fn = functools.partial(_swap_out_fn,
                                    src_sharding=device_sharding,
                                    dst_sharding=host_sharding,
                                    direction="d2h")
    return swap_in_fn, swap_out_fn


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


@functools.partial(jax.jit)
def jitted_gather_kv_cache(kv_caches: List[jax.Array],
                            block_ids: jax.Array) -> List[jax.Array]:
    """
    JIT-compiled function to gather KV cache slices for all layers at once.
    This uses jax.tree.map to apply the operation across all layers.
    """

    def gather_and_reshape(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    return jax.tree.map(gather_and_reshape, kv_caches)


# @functools.partial(jax.jit)
# def jitted_stack_kv_cache_cross_layers(kv_caches: List[jax.Array],
#                             block_ids: jax.Array) -> List[jax.Array]:
#     """
#     JIT-compiled function to gather KV cache slices for all layers at once.
#     This uses jax.tree.map to apply the operation across all layers.
#     """

#     def gather_and_reshape(layer_kv_cache):
#         return layer_kv_cache.at[block_ids].get()

#     gathered_kv_layers = jax.tree.map(gather_and_reshape, kv_caches)
#     stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)

#     # Split the stacked_blocks along axis=0 into individual blocks
#     split_blocks = jnp.split(stacked_blocks, indices_or_sections=len(block_ids), axis=0)

#     # Squeeze the first dimension from each of the split arrays
#     # squeezed_blocks = [jnp.squeeze(c, axis=0) for c in split_blocks]

#     return split_blocks

@functools.partial(jax.jit, static_argnames=['num_blocks'])
def jitted_stack_kv_cache_cross_layers(kv_caches: List[jax.Array],
                            block_ids: jax.Array, num_blocks: int) -> List[jax.Array]:
    """
    This uses jax.tree.map to apply the operation across all layers.
    """

    def _gather_blocks(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    gathered_kv_layers = jax.tree.map(_gather_blocks, kv_caches)
    stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)

    # Split the stacked_blocks along axis=0 into individual blocks
    # split_blocks = jnp.split(stacked_blocks, indices_or_sections=len(block_ids), axis=0)
    split_blocks = jnp.split(stacked_blocks, indices_or_sections=num_blocks, axis=0)
    # split_blocks = jnp.array_split(stacked_blocks, num_blocks, axis=0)
    
    # Squeeze the first dimension from each of the split arrays
    # squeezed_blocks = [jnp.squeeze(c, axis=0) for c in split_blocks]

    # return squeezed_blocks
    return split_blocks


@functools.partial(jax.jit, static_argnames=())
def insert_slices_with_scatter(cache_layer: jax.Array, slices: jax.Array, block_indices: jax.Array) -> jax.Array:
  """
  Inserts slices into a cache layer at specified block indices using jax.lax.scatter.

  Args:
    cache_layer: The layer of the KV cache to update.
                 Shape (num_blocks, block_size, ...)
    slices: The slices to insert. Shape (num_blocks_to_insert, block_size, ...)
    block_indices: The indices in the cache layer to insert the slices.
                 Shape (num_blocks_to_insert,)

  Returns:
    The updated cache layer.
  """
  dnums = jax.lax.ScatterDimensionNumbers(
      update_window_dims=tuple(range(1, len(slices.shape))),
      inserted_window_dims=(0,),
      scatter_dims_to_operand_dims=(0,)
  )
  return jax.lax.scatter(
      cache_layer,
      block_indices[:, None],
      slices,
      dnums
  )


@functools.partial(
    jax.jit,
    static_argnames=("dim_nums",),
    donate_argnames=(
        "kv_caches",
        "stacked_blocks",
    ),
)
def update_kv_caches(kv_caches: List[jax.Array], stacked_blocks: List[jax.Array], block_indices: jax.Array, dim_nums: jax.lax.ScatterDimensionNumbers) -> List[jax.Array]:
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
        # dnums = jax.lax.ScatterDimensionNumbers(
        #     update_window_dims=tuple(range(1, len(slices.shape))),
        #     inserted_window_dims=(0,),
        #     scatter_dims_to_operand_dims=(0,)
        # )
        return jax.lax.scatter(
            cache_layer,
            block_indices[:, None],
            slices,
            dim_nums
        )

    return jax.tree.map(_update_layer, kv_caches, layer_slices_list)
