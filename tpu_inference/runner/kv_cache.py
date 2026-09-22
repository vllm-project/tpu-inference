# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from functools import cache, partial
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import tpu_inference.envs as envs
import tpu_inference.kernels.mla.v2.kernel as mla
import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64

if envs.USE_BATCHED_RPA_KERNEL:
    import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa
else:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa

from tpu_inference import utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

logger = init_logger(__name__)

DEFAULT_KV_CACHE_DTYPE = jnp.bfloat16


@dataclass
class KVCacheMetadata:
    """
    Used to store metadata about the KV cache for logging in the KV cache manager.
    Specifcally, with Hybrid KV cache, we can have multiple KV cache types
    so we need to store the metadata for each KV cache type separately
    """
    count: int = 0
    shape: tuple = None
    dtype: jnp.dtype = None
    sharding: NamedSharding = None


def get_kv_cache_shape_with_mesh(mesh: Mesh,
                                 total_num_pages: int,
                                 block_size: int,
                                 actual_num_kv_heads: int,
                                 actual_head_dim: int,
                                 kv_dtype: any,
                                 use_mla: bool = False):
    """Gets the KV cache shape based on the mesh configuration.

    This function scales block_size by the CONTEXT (DCP, PCP) axis and num_heads by duplicate kv heads.

    """

    model_cnt = utils.get_mesh_shape_product(mesh, ShardingAxisName.KV_HEAD)
    context_cnt = utils.get_mesh_shape_product(mesh,
                                               ShardingAxisName.KV_CONTEXT)
    physical_block_size = block_size * context_cnt

    # NOTE(chengjiyao): Currently, the attention kernel is tailored to the
    # specific model, rather than being determined by the head_dim. If new
    # models are introduced with a head_dim of 64, this will require additional
    # model-specific adjustments.
    if use_mla:
        # No assertion needed: MLA compresses all KV into a single latent vector,
        # so actual_num_kv_heads is never used in mla.get_kv_cache_shape().
        get_kv_cache_shape_fn = mla.get_kv_cache_shape
        shape = list(
            get_kv_cache_shape_fn(
                total_num_pages,
                physical_block_size,
                actual_head_dim,
                kv_dtype,
                envs.MLA_KV_PACKING_SIZE,
                transpose_kv_cache=envs.MLA_TRANSPOSE_KV_CACHE))
    else:
        assert actual_num_kv_heads % model_cnt == 0
        get_kv_cache_shape_fn = (
            rpa_hd64.get_kv_cache_shape if actual_head_dim == 64 \
                else rpa.get_kv_cache_shape
        )
        shape = list(
            get_kv_cache_shape_fn(total_num_pages, physical_block_size,
                                  actual_num_kv_heads // model_cnt,
                                  actual_head_dim, kv_dtype))
        shape[2] *= model_cnt
    return tuple(shape)


@cache
def _get_kv_cache_allocator(
        cache_shape: tuple, cache_dtype: jnp.dtype,
        sharding: NamedSharding) -> Callable[[], jax.Array]:

    @partial(jax.jit, out_shardings=sharding)
    def _allocate() -> jax.Array:
        return jnp.zeros(
            shape=cache_shape,
            dtype=cache_dtype,
        )

    return _allocate


@cache
def _get_mamba_cache_allocator(
        cache_shape: tuple, cache_dtype: jnp.dtype,
        sharding: NamedSharding) -> Callable[[], jax.Array]:

    # `MAMBA_CACHE_POISON` exists for the repro in
    # tools/mamba_prefix_cache_repro: `jnp.empty` hands back recycled HBM, so
    # on a fresh boot it is usually clean and the bug hides. Filling the pool
    # with NaN instead makes any read of an unwritten slot show up
    # immediately, as `!`.
    poison = envs.MAMBA_CACHE_POISON

    @partial(jax.jit, out_shardings=sharding)
    def _allocate() -> jax.Array:
        if poison == "nan":
            return jnp.full(cache_shape, jnp.nan, dtype=cache_dtype)
        if poison == "zeros":
            return jnp.zeros(shape=cache_shape, dtype=cache_dtype)
        return jnp.empty(
            shape=cache_shape,
            dtype=cache_dtype,
        )

    return _allocate


def create_mamba_cache(cache_shape: tuple, cache_dtype: jnp.dtype,
                       sharding: NamedSharding) -> jax.Array:
    """Creates a fresh Mamba state array with a cached allocator."""
    return _get_mamba_cache_allocator(cache_shape, cache_dtype, sharding)()


# A step allocates a handful of mamba slots, but the exact count varies. Round
# the scatter's index array up to one of these sizes so the count does not
# retrigger compilation every step.
_ZERO_ROWS_BUCKETS = (8, 32, 128, 512)


def _bucket_num_rows(num_rows: int) -> int:
    for bucket in _ZERO_ROWS_BUCKETS:
        if num_rows <= bucket:
            return bucket
    last = _ZERO_ROWS_BUCKETS[-1]
    return -(-num_rows // last) * last


@cache
def _get_mamba_block_zeroer(
    sharding: NamedSharding,
) -> Callable[[jax.Array, jax.Array], jax.Array]:

    @partial(jax.jit, donate_argnums=(0, ), out_shardings=sharding)
    def _zero(state: jax.Array, rows: jax.Array) -> jax.Array:
        return state.at[rows].set(0)

    return _zero


def zero_mamba_blocks(state: jax.Array, rows: np.ndarray) -> jax.Array:
    """Zeros whole rows of a mamba state array.

    The GDN kernel writes one checkpoint per forward pass, at
    ``(seq_len - 1) // mamba_block_size``, so every other slot a request owns
    still holds whatever its previous owner left there. Zeroing a slot when it
    is handed out means a resume that lands on a slot no pass ever
    checkpointed reads a fresh-sequence state instead of unrelated state.

    ``state`` is donated. ``rows`` is padded up to a bucket size by repeating
    its first entry; the scatter then writes the same zeros twice, which is
    harmless and keeps the shape off the recompilation path.
    """
    if rows.size == 0:
        return state
    padded = np.full(_bucket_num_rows(rows.size), rows[0], dtype=np.int32)
    padded[:rows.size] = rows
    return _get_mamba_block_zeroer(state.sharding)(state, jnp.asarray(padded))


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    layer_names: List[str],
    cache_dtype: jnp.dtype = DEFAULT_KV_CACHE_DTYPE,
    use_mla: bool = False,
) -> List[jax.Array]:
    """
    Creates a list of KV cache where each array mapps to single attention layer.

    The shape of the KV cache per layer is:
    (num_blocks, block_size, cdiv(num_kv_heads * 2, packing), packing, head_dim)
    where packing = (32 // dtype bits)

    Args:
        num_blocks: The number of blocks in the KV cache.
        block_size: The size of each block in the KV cache.
        num_kv_heads: The number of KV heads in the KV cache.
        head_size: The size of each head in the KV cache.
        mesh: The mesh to shard the KV caches across.
        layer_names: The names of the decoder layers in the model.
        cache_dtype: The datatype of KV cache.

    Returns:
        A list of KV caches, one per each decoder layer in the model.

    """
    # TODO(xiang): fix this together with get_kv_cache_spec
    # cache_dtype = kv_cache_spec.dtype

    cache_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                               num_kv_heads, head_size,
                                               cache_dtype, use_mla)

    # num_blocks --> shard by data batch
    # block_size --> shard by context
    # head       --> shard by heads
    if use_mla:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT))
    else:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                          ShardingAxisName.KV_HEAD))

    sharded_allocate = _get_kv_cache_allocator(cache_shape, cache_dtype,
                                               sharding)
    return [sharded_allocate() for _ in layer_names]


def create_kv_cache_of_shape(
    cache_shape: tuple,
    mesh: Mesh,
    cache_dtype: jnp.dtype = DEFAULT_KV_CACHE_DTYPE,
) -> jax.Array:
    """Creates a single zero-filled KV cache array with an explicit shape.
    """
    sharding = NamedSharding(
        mesh, PartitionSpec(ShardingAxisName.BATCH,
                            ShardingAxisName.KV_CONTEXT))
    return _get_kv_cache_allocator(tuple(cache_shape), cache_dtype, sharding)()


def get_attention_page_size_bytes(mesh, block_size, num_kv_heads, head_size,
                                  dtype, use_mla) -> int:
    jax_dtype = to_jax_dtype(dtype)
    bits = dtypes.itemsize_bits(jax_dtype)
    kv_cache_shape = get_kv_cache_shape_with_mesh(
        mesh=mesh,
        total_num_pages=1,
        block_size=block_size,
        actual_num_kv_heads=num_kv_heads,
        actual_head_dim=head_size,
        kv_dtype=jax_dtype,
        use_mla=use_mla,
    )
    return int(bits * np.prod(kv_cache_shape)) // 8
