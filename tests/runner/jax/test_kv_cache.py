from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons.models.jax.attention import get_kv_cache_shape_with_mesh
from tpu_commons.runner.jax.kv_cache import create_kv_caches


@pytest.fixture
def mesh():
    devices = jax.devices()
    return Mesh(devices, axis_names=("model", ))


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

    expected_sharding = NamedSharding(mesh, PartitionSpec())
    expected_dtype = jnp.bfloat16
    expected_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  num_kv_heads, head_size,
                                                  expected_dtype)

    with patch("tpu_commons.logger.init_logger",
               return_value=MagicMock()), patch(
                   "tpu_commons.utils.hbm_usage_gb",
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
