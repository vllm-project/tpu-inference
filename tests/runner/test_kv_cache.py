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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner.kv_cache import (create_kv_caches,
                                           get_attention_page_size_bytes,
                                           get_kv_cache_shape_with_mesh)
from tpu_inference.utils import get_dtype_packing


@pytest.fixture
def mesh():
    devices = np.array(jax.local_devices()[:1])
    devices = devices.reshape((1, 1, -1))
    return Mesh(devices, axis_names=("data", "attn_dp", "model"))


def test_create_kv_caches(mesh: Mesh):
    """
    Tests that `create_kv_caches` correctly allocates and shards the KV caches
    for all specified layers.
    """
    num_blocks = 64
    block_size = 16
    num_kv_heads = 8
    head_size = 128
    layer_names = ["decoder.0", "decoder.1", "decoder.2"]  # Test with 3 layers

    expected_sharding = NamedSharding(
        mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None, "model"))
    expected_dtype = jnp.bfloat16
    expected_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  num_kv_heads, head_size,
                                                  expected_dtype)

    with patch("tpu_inference.logger.init_logger",
               return_value=MagicMock()), patch(
                   "tpu_inference.utils.hbm_usage_gb",
                   return_value=[(0.0, 0.0), (0.0, 0.0)]):
        kv_caches = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            mesh=mesh,
            layer_names=layer_names,
        )

        assert isinstance(kv_caches, list)
        assert len(kv_caches) == len(layer_names)

        for cache_array in kv_caches:
            assert isinstance(cache_array, jax.Array)
            assert cache_array.shape == expected_shape
            assert cache_array.dtype == expected_dtype
            assert cache_array.sharding == expected_sharding

        # Ensure that separate array objects were created for each layer
        assert kv_caches[0] is not kv_caches[1]


def test_create_kv_caches_mla(mesh: Mesh):
    """
    Tests that `create_kv_caches` correctly allocates and shards the KV caches
    for all specified layers when `use_mla` is True.
    """
    num_blocks = 64
    block_size = 16
    num_kv_heads = 1  # Not used for MLA shape calculation
    head_size = 512 + 64  # Combined dimension for MLA
    layer_names = ["decoder.0", "decoder.1"]

    # For MLA, sharding is by the 'model' axis on the token dimension.
    expected_sharding = NamedSharding(
        mesh, PartitionSpec(ShardingAxisName.MLP_TENSOR))
    expected_dtype = jnp.bfloat16
    expected_shape = get_kv_cache_shape_with_mesh(
        mesh,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        expected_dtype,
        use_mla=True,
    )

    with patch("tpu_inference.logger.init_logger",
               return_value=MagicMock()), patch(
                   "tpu_inference.utils.hbm_usage_gb",
                   return_value=[(0.0, 0.0), (0.0, 0.0)]):
        kv_caches = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            mesh=mesh,
            layer_names=layer_names,
            use_mla=True,
        )

        assert isinstance(kv_caches, list)
        assert len(kv_caches) == len(layer_names)

        for cache_array in kv_caches:
            assert isinstance(cache_array, jax.Array)
            assert cache_array.shape == expected_shape
            assert cache_array.dtype == expected_dtype
            assert cache_array.sharding == expected_sharding


def test_get_kv_cache_shape_with_mesh_mla(mesh: Mesh):
    """
    Tests `get_kv_cache_shape_with_mesh` with `use_mla=True`.
    """
    total_num_pages = 64
    page_size = 16
    actual_num_kv_heads = 1  # Not used for MLA
    actual_head_dim = 512 + 128  # lkv_dim + r_dim
    kv_dtype = jnp.bfloat16

    # Expected shape calculation for MLA:
    # kv_packing = 2 (for bfloat16)
    # shape[0] = total_num_pages = 64
    # shape[1] = align_to(page_size, 2) // 2 = 16 // 2 = 8
    # shape[2] = 2
    # shape[3] = align_to(actual_head_dim, 128) = align_to(640, 128) = 640
    expected_shape = (64, 8, 2, 640)

    shape = get_kv_cache_shape_with_mesh(
        mesh,
        total_num_pages,
        page_size,
        actual_num_kv_heads,
        actual_head_dim,
        kv_dtype,
        use_mla=True,
    )

    assert shape == expected_shape


def test_get_attention_page_size_bytes(mesh: Mesh):
    """
    Tests `get_attention_page_size_bytes`.
    """
    block_size = 16
    num_kv_heads = 8
    head_size = 128
    dtype = torch.bfloat16

    full_attn_spec = FullAttentionSpec(block_size=block_size,
                                       num_kv_heads=num_kv_heads,
                                       head_size=head_size,
                                       dtype=dtype)

    kv_cache_specs = {"layer.0": full_attn_spec}

    page_size_bytes = get_attention_page_size_bytes(mesh, kv_cache_specs)

    shape = get_kv_cache_shape_with_mesh(mesh, 1, block_size, num_kv_heads,
                                         head_size, jnp.bfloat16)
    expected_page_size = (
        (32 // get_dtype_packing(jnp.bfloat16)) * np.prod(shape)) // 8

    assert page_size_bytes == expected_page_size


def test_get_attention_page_size_bytes_mla(mesh: Mesh):
    """
    Tests `get_attention_page_size_bytes` for MLA.
    """
    block_size = 16
    num_kv_heads = 1
    head_size = 512 + 128  # lkv_dim + r_dim
    dtype = torch.bfloat16

    mla_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        cache_dtype_str="bfloat16",
    )

    kv_cache_specs = {"layer.0": mla_spec}

    page_size_bytes = get_attention_page_size_bytes(mesh, kv_cache_specs)

    shape = get_kv_cache_shape_with_mesh(mesh,
                                         1,
                                         block_size,
                                         num_kv_heads,
                                         head_size,
                                         jnp.bfloat16,
                                         use_mla=True)
    expected_page_size = (
        (32 // get_dtype_packing(jnp.bfloat16)) * np.prod(shape)) // 8

    assert page_size_bytes == expected_page_size
