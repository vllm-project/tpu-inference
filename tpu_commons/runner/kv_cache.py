from typing import List

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import tpu_commons.kernels.ragged_paged_attention.v3.kernel as rpa
from tpu_commons.logger import init_logger

logger = init_logger(__name__)


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    layer_names: List[str],
    cache_dtype: jnp.dtype = jnp.bfloat16,
) -> List[jax.Array]:
    """
    Creates the KV caches, a list of arrays, each array is for one attention layer.

    The shape of the KV cache per layer is:
    (num_blocks, block_size, cdiv(num_kv_heads * 2, packing), packing, head_size).
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
    # TODO(xiang): fix this together with get_kv_cache_spec
    # cache_dtype = kv_cache_spec.dtype

    cache_shape = rpa.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                                         head_size, cache_dtype)

    sharding = NamedSharding(mesh, PartitionSpec(None, None, "model"))

    def _allocate() -> jax.Array:
        return jnp.empty(
            shape=cache_shape,
            dtype=cache_dtype,
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    kv_caches = []
    for _ in layer_names:
        kv_caches.append(sharded_allocate())
    return kv_caches
