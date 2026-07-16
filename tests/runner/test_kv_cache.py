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

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner.kv_cache import (create_kv_caches,
                                           create_unified_kv_cache,
                                           get_attention_page_size_bytes,
                                           get_kv_cache_shape_with_mesh,
                                           model_uses_unified_kv_cache)
from tpu_inference.utils import get_dtype_packing


@pytest.fixture
def mesh():
    devices = np.array(jax.local_devices()[:1])
    devices = devices.reshape((1, 1, 1, 1, 1, -1))
    return Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model", "dcp"))


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

    expected_dtype = jnp.bfloat16

    with patch("tpu_inference.logger.init_logger",
               return_value=MagicMock()), patch(
                   "tpu_inference.utils.hbm_usage_gb",
                   return_value=[
                       (0.0, 0.0), (0.0, 0.0)
                   ]), patch("tpu_inference.envs.NEW_MODEL_DESIGN", True):
        expected_sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT,
                          ShardingAxisName.KV_CACHE_HEAD))
        expected_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks,
                                                      block_size, num_kv_heads,
                                                      head_size,
                                                      expected_dtype)
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
        mesh, PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT))
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
                   return_value=[
                       (0.0, 0.0), (0.0, 0.0)
                   ]), patch("tpu_inference.envs.NEW_MODEL_DESIGN", True):
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
    page_size = 128
    actual_num_kv_heads = 1  # Not used for MLA
    actual_head_dim = 512 + 128  # lkv_dim + r_dim
    kv_dtype = jnp.bfloat16

    # Expected shape calculation for MLA:
    # kv_packing = 32 (envs.MLA_KV_PACKING_SIZE)

    # shape[0] = total_num_pages = 64
    # shape[1] = align_to(page_size, kv_packing) // kv_packing = 4
    # shape[2] = kv_packing = 32
    # shape[3] = align_to(actual_head_dim, 128) = align_to(640, 128) = 640
    expected_shape = (64, 4, 32, 640)

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

    page_size_bytes = get_attention_page_size_bytes(mesh, block_size,
                                                    num_kv_heads, head_size,
                                                    dtype, False)

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

    page_size_bytes = get_attention_page_size_bytes(mesh, block_size,
                                                    num_kv_heads, head_size,
                                                    dtype, True)

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


def test_create_kv_caches_batch_equivalence(mesh: Mesh):
    """
    Tests that calling create_kv_caches once with N layer names is equivalent to
    calling it N times with a single layer name and aggregating the results.
    """
    num_blocks = 16
    block_size = 8
    num_kv_heads = 4
    head_size = 64
    layer_names = ["layer.0", "layer.1", "layer.2"]

    with patch("tpu_inference.logger.init_logger", return_value=MagicMock()), \
         patch("tpu_inference.utils.hbm_usage_gb", return_value=[(0.0, 0.0)]):

        # Single batch call
        kv_caches_batch = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            mesh=mesh,
            layer_names=layer_names,
        )

        # Multiple iterative calls
        kv_caches_iterative = []
        for name in layer_names:
            layer_cache = create_kv_caches(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                mesh=mesh,
                layer_names=[name],
            )
            kv_caches_iterative.extend(layer_cache)

        assert len(kv_caches_batch) == len(kv_caches_iterative)
        for b_cache, i_cache in zip(kv_caches_batch, kv_caches_iterative):
            assert b_cache.shape == i_cache.shape
            assert b_cache.dtype == i_cache.dtype
            assert b_cache.sharding == i_cache.sharding
            # Note: Content is empty/uninitialized, so we don't compare values,
            # just the metadata and allocation properties.


# ---- Tests for the unified block-major KV cache path ----


def _make_vllm_config(architectures, speculative_config=None):
    """Minimal config mock exposing exactly what the gate reads."""
    config = MagicMock()
    config.speculative_config = speculative_config
    config.model_config.architectures = architectures
    return config


@pytest.mark.parametrize("architectures,speculative_config,expected", [
    (["Qwen2ForCausalLM"], None, True),
    (["Qwen3ForCausalLM"], None, True),
    (["SomeVisionEncoder", "Qwen2ForCausalLM"], None, True),
    (["LlamaForCausalLM"], None, False),
    (["Qwen2_5_VLForConditionalGeneration"], None, False),
    (["Qwen3MoeForCausalLM"], None, False),
    ([], None, False),
    (["Qwen3ForCausalLM"], "draft", False),
])
def test_model_uses_unified_kv_cache(architectures, speculative_config,
                                     expected):
    """Allowlisted archs (anywhere in the list) take the unified path;
    other archs and speculative decoding keep the per-layer path."""
    config = _make_vllm_config(architectures,
                               speculative_config=speculative_config)
    assert model_uses_unified_kv_cache(config) is expected


@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("cache_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_create_unified_kv_cache(mesh: Mesh, num_layers, cache_dtype):
    """6D block-major pool: (num_blocks, num_layers, *per_layer_dims),
    sharded rank-6 with ATTN_HEAD on dim3 (kv heads)."""
    num_blocks = 16
    block_size = 16
    num_kv_heads = 8
    head_size = 128

    single_layer_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks,
                                                      block_size, num_kv_heads,
                                                      head_size, cache_dtype)

    cache = create_unified_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        mesh=mesh,
        num_layers=num_layers,
        cache_dtype=cache_dtype,
    )

    assert isinstance(cache, jax.Array)
    assert cache.ndim == 6
    assert cache.shape == (single_layer_shape[0], num_layers,
                           *single_layer_shape[1:])
    assert cache.dtype == cache_dtype

    expected_sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None, None,
                      ShardingAxisName.ATTN_HEAD, None, None),
    )
    assert cache.sharding == expected_sharding


def test_create_unified_kv_cache_matches_model_loader_pin(mesh: Mesh):
    """model_loader's out_shardings pin must be layout-equivalent to the
    allocated sharding, or every run_model call would reshard the pool."""
    cache = create_unified_kv_cache(
        num_blocks=8,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        mesh=mesh,
        num_layers=2,
    )
    loader_pin = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None, None,
                      ShardingAxisName.ATTN_HEAD),
    )
    assert loader_pin.is_equivalent_to(cache.sharding, cache.ndim)


def test_create_unified_kv_cache_tp_shards_kv_head_dim():
    """At TP>1 the pool is sharded on dim3 (kv heads); the legacy rank-3
    spec would land ATTN_HEAD on dim2 and must NOT be equivalent."""
    num_model_devices = 2
    if jax.device_count() < num_model_devices:
        pytest.skip(f"requires >= {num_model_devices} devices")

    devices = np.array(jax.devices()[:num_model_devices]).reshape(
        (1, 1, 1, 1, num_model_devices, 1))
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model", "dcp"))

    num_blocks = 8
    num_layers = 3
    cache = create_unified_kv_cache(
        num_blocks=num_blocks,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        mesh=mesh,
        num_layers=num_layers,
    )

    shard_shape = cache.addressable_shards[0].data.shape
    # dim0 (blocks) and dim1 (layers) are replicated across the model axis;
    # dim3 (kv heads) is split.
    assert shard_shape[0] == cache.shape[0]
    assert shard_shape[1] == num_layers
    assert shard_shape[2] == cache.shape[2]
    assert shard_shape[3] == cache.shape[3] // num_model_devices

    legacy_rank3_pin = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD),
    )
    assert not legacy_rank3_pin.is_equivalent_to(cache.sharding, cache.ndim)
