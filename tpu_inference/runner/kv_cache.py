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

from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.ops.mappings import t2j_dtype

import tpu_inference.kernels.mla.v1.kernel as mla
import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

DEFAULT_KV_CACHE_DTYPE = jnp.bfloat16


def get_kv_cache_shape_with_mesh(mesh: Mesh,
                                 total_num_pages: int,
                                 page_size: int,
                                 actual_num_kv_heads: int,
                                 actual_head_dim: int,
                                 kv_dtype: any,
                                 use_mla: bool = False):
    """Gets the KV cache shape based on the mesh configuration."""

    model_cnt = mesh.shape["model"]
    assert actual_num_kv_heads % model_cnt == 0
    # NOTE(chengjiyao): Currently, the attention kernel is tailored to the
    # specific model, rather than being determined by the head_dim. If new
    # models are introduced with a head_dim of 64, this will require additional
    # model-specific adjustments.
    if use_mla:
        get_kv_cache_shape_fn = mla.get_kv_cache_shape
        shape = list(
            get_kv_cache_shape_fn(total_num_pages, page_size, actual_head_dim,
                                  kv_dtype))
    else:
        get_kv_cache_shape_fn = (
            rpa_hd64.get_kv_cache_shape if actual_head_dim == 64 \
                else rpa.get_kv_cache_shape
        )
        shape = list(
            get_kv_cache_shape_fn(total_num_pages, page_size,
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

    if use_mla:
        sharding = NamedSharding(mesh,
                                 PartitionSpec(ShardingAxisName.MLP_TENSOR))
    else:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.ATTN_DATA, None,
                          ShardingAxisName.ATTN_HEAD))

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


def get_attention_page_size_bytes(mesh: Mesh,
                                  kv_cache_specs: dict[str, Any]) -> int:
    """
    Calculate KV cache page size of RPA kernel.

    Args:
        mesh: The mesh to shard the KV caches across.
        kv_cache_specs: Dictionary of KV cache specs.

    Returns:
        KV cache page size in bytes.
    """

    # Import it here to avoid circular import.
    from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

    page_size_bytes_set = set()
    for kv_cache_spec in kv_cache_specs.values():
        assert isinstance(kv_cache_spec, AttentionSpec)

        dtype = t2j_dtype(kv_cache_spec.dtype)
        bits = (dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else
                dtypes.itemsize_bits(dtype))
        use_mla = isinstance(kv_cache_spec, MLAAttentionSpec)

        kv_cache_shape = get_kv_cache_shape_with_mesh(
            mesh=mesh,
            total_num_pages=1,  # Pass 1 to get shape of a single page.
            page_size=kv_cache_spec.block_size,
            actual_num_kv_heads=kv_cache_spec.num_kv_heads,
            actual_head_dim=kv_cache_spec.head_size,
            kv_dtype=dtype,
            use_mla=use_mla,
        )
        page_size_bytes = (bits * np.prod(kv_cache_shape)) // 8
        page_size_bytes_set.add(page_size_bytes)

    # Ensure that page size is the same for all kv caches.
    assert len(page_size_bytes_set) == 1
    return page_size_bytes_set.pop()
