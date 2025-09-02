from typing import List

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import tpu_commons.kernels.ragged_paged_attention.v3.kernel as rpa
from tpu_commons import utils
from tpu_commons.logger import init_logger

logger = init_logger(__name__)

DEFAULT_KV_CACHE_DTYPE = jnp.bfloat16


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    layer_names: List[str],
) -> List[jax.Array]:
    """
    Creates the KV caches, a list of arrays, each array is for one attention layer.

    As required by RPA-v3, the shape of the array needs to be per shard.
    (num_blocks, block_size, cdiv(num_kv_heads_per_shard * 2, packing), packing, head_size).
    num_kv_heads_per_shard = num_kv_heads // shard_size
    packing =  (32 // dtype bits)

    Args:
        num_blocks: The number of blocks in the KV cache.
        block_size: The size of each block in the KV cache.
        num_kv_heads: The number of KV heads in the KV cache.
        head_size: The size of each head in the KV cache.
        mesh: The mesh to shard the KV caches across.
        layer_names: The names of the decoder layers in the model.

    Returns:
        A list of KV caches, one per each decoder layer in the model.

    """
    # TODO (jacobplatin): update this for quantized KV cache
    cache_dtype = DEFAULT_KV_CACHE_DTYPE
    # TODO(xiang): fix this together with get_kv_cache_spec
    # cache_dtype = kv_cache_spec.dtype

    # NOTE(jevinjiang): Instead of sharding automatically, we manually calculate
    # the kv cache for each shard because the padding logic for RPA's KV cache
    # needs to know the exact head number on each shard. In other words, we can
    # not determine the padding logics for kv cache globally.
    shard_cnt = mesh.shape["model"]
    assert num_kv_heads % shard_cnt == 0
    cache_shape_per_shard = rpa.get_kv_cache_shape(num_blocks, block_size,
                                                   num_kv_heads // shard_cnt,
                                                   head_size, cache_dtype)
    # Intended to be replicated.
    sharding = NamedSharding(mesh, PartitionSpec())

    def _allocate() -> jax.Array:
        return jnp.empty(
            shape=cache_shape_per_shard,
            dtype=cache_dtype,
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    kv_caches = []
    for _ in layer_names:
        kv_caches.append(sharded_allocate())
    logger.info(
        f"Init kv-cache | "
        f"shape={len(layer_names)} * {shard_cnt} * {cache_shape_per_shard} | "
        f"sharding={sharding} | "
        f"dtype={cache_dtype} | "
        f"hbm={utils.hbm_usage_gb(mesh.devices.flatten())}Gb")
    return kv_caches
