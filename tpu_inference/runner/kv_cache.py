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
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import tpu_inference.envs as envs
import tpu_inference.kernels.mla.v2.kernel as mla
import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference import utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

logger = init_logger(__name__)

DEFAULT_KV_CACHE_DTYPE = jnp.bfloat16

# Architectures whose forward pass can consume the unified block-major KV
# cache (detect the single-element `kv_caches` list and thread `layer_idx`
# into `attention()`). All other architectures index `kv_caches` per layer
# and must keep the per-layer allocation path.
UNIFIED_KV_CACHE_SUPPORTED_ARCHS = (
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
)


def model_uses_unified_kv_cache(vllm_config) -> bool:
    """Whether this config takes the unified block-major KV cache path.

    Shared by the allocation gate (kv_cache_manager) and the model_loader's
    kv cache out_shardings pin so the pin always matches the allocated
    layout. Speculative decoding keeps the per-layer path (draft model
    forwards index `kv_caches` per layer).
    """
    if vllm_config.speculative_config is not None:
        return False
    return any(arch in UNIFIED_KV_CACHE_SUPPORTED_ARCHS
               for arch in vllm_config.model_config.architectures)


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

    This function scales block_size by the CONTEXT (DCP) axis and num_heads by duplicate kv heads.

    """

    model_cnt = utils.get_mesh_shape_product(mesh,
                                             ShardingAxisName.KV_CACHE_HEAD)

    context_cnt = utils.get_mesh_shape_product(mesh, ShardingAxisName.CONTEXT)
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
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT))
    else:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT,
                          ShardingAxisName.KV_CACHE_HEAD))

    def _allocate() -> jax.Array:
        return jnp.zeros(
            shape=cache_shape,
            dtype=cache_dtype,
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    kv_caches = []
    for _ in layer_names:
        kv_caches.append(sharded_allocate())
    return kv_caches


def create_unified_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    num_layers: int,
    cache_dtype: jnp.dtype = DEFAULT_KV_CACHE_DTYPE,
) -> jax.Array:
    """Creates a single unified KV cache array shared by all layers.

    The array is block-major (num_blocks outermost, num_layers second), so a
    block's all-layer data is contiguous and transfers with a single DMA:
    dim0-slice b == one whole logical block.

    The RPA kernel still consumes a 5D (num_pages, ...) cache; the fold of
    (num_blocks, num_layers) into the flat page dim happens per-device inside
    sharded_ragged_paged_attention's shard_map, where it is a bitcast. Do NOT
    reshape this array in GSPMD-traced code: that results in an extra
    all-to-all causing OOM.

    Shape: (num_blocks, num_layers, block_size,
    cdiv(num_kv_heads * 2, packing), packing, head_dim),
    where packing = 32 // dtype bits.
    """
    single_layer_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks,
                                                      block_size, num_kv_heads,
                                                      head_size, cache_dtype)
    unified_shape = (single_layer_shape[0], num_layers,
                     *single_layer_shape[1:])

    # PartitionSpec right-pads with None, so spell out enough dims to land
    # ATTN_HEAD on dim3 (kv heads).
    sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None, None,
                      ShardingAxisName.ATTN_HEAD, None, None),
    )

    def _allocate() -> jax.Array:
        return jnp.empty(
            shape=unified_shape,
            dtype=cache_dtype,
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    return sharded_allocate()


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
