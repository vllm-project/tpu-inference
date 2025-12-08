# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

# Import the functions to be tested
from tpu_inference.utils import (GBYTES, enable_megacore,
                                 get_jax_dtype_from_str_dtype, get_megacore,
                                 get_padded_head_dim, hbm_usage_bytes,
                                 hbm_usage_gb, quantize_kv)


def test_enable_and_get_megacore():
    """Tests the enable_megacore and get_megacore functions."""
    assert not get_megacore()
    enable_megacore()
    assert get_megacore()


@patch.dict(os.environ, {"TPU_MULTIHOST_BACKEND": "ray"})
def test_hbm_usage_bytes_ray_backend():
    """Tests hbm_usage_bytes when TPU_MULTIHOST_BACKEND is ray."""
    mock_device1 = MagicMock()
    mock_device1.memory_stats.return_value = {
        "bytes_in_use": 100 * GBYTES,
        "bytes_limit": 128 * GBYTES
    }
    mock_device2 = MagicMock()
    mock_device2.memory_stats.side_effect = Exception("Memory stats failed")

    devices = [mock_device1, mock_device2]
    usage = hbm_usage_bytes(devices)

    expected_usage = [(100 * GBYTES, 128 * GBYTES),
                      (100 * GBYTES, 128 * GBYTES)]
    assert usage == expected_usage


@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_hbm_usage_bytes_pathways_disabled():
    """Tests hbm_usage_bytes when VLLM_TPU_USING_PATHWAYS is False."""
    mock_device1 = MagicMock()
    mock_device1.memory_stats.return_value = {
        "bytes_in_use": 100 * GBYTES,
        "bytes_limit": 128 * GBYTES
    }
    mock_device2 = MagicMock()
    mock_device2.memory_stats.return_value = {
        "bytes_in_use": 50 * GBYTES,
        "bytes_limit": 128 * GBYTES
    }

    devices = [mock_device1, mock_device2]
    usage = hbm_usage_bytes(devices)

    expected_usage = [(100 * GBYTES, 128 * GBYTES),
                      (50 * GBYTES, 128 * GBYTES)]
    assert usage == expected_usage


@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", True)
@patch("jax.live_arrays")
@patch("jax.devices")
def test_hbm_usage_bytes_pathways_enabled(mock_devices, mock_live_arrays):
    """Tests hbm_usage_bytes when VLLM_TPU_USING_PATHWAYS is True."""
    # Mock TPU v5p devices
    mock_jax_device = MagicMock()
    mock_jax_device.device_kind = "TPU v5p"
    mock_devices.return_value = [mock_jax_device]

    # Create mock devices
    mock_device1 = MagicMock()
    mock_device2 = MagicMock()
    devices = [mock_device1, mock_device2]

    # Create mock addressable shards with data property
    mock_data1_dev1 = MagicMock()
    mock_data1_dev1.device = mock_device1
    mock_data1_dev1.nbytes = 2000  # 2000 bytes on device1

    mock_data1_dev2 = MagicMock()
    mock_data1_dev2.device = mock_device2
    mock_data1_dev2.nbytes = 2000  # 2000 bytes on device2

    mock_data2_dev1 = MagicMock()
    mock_data2_dev1.device = mock_device1
    mock_data2_dev1.nbytes = 1000  # 1000 bytes on device1

    mock_shard1_dev1 = MagicMock()
    mock_shard1_dev1.data = mock_data1_dev1

    mock_shard1_dev2 = MagicMock()
    mock_shard1_dev2.data = mock_data1_dev2

    mock_shard2_dev1 = MagicMock()
    mock_shard2_dev1.data = mock_data2_dev1

    # Create mock arrays with addressable_shards
    mock_array1 = MagicMock()
    mock_array1.addressable_shards = [mock_shard1_dev1, mock_shard1_dev2]

    mock_array2 = MagicMock()
    mock_array2.addressable_shards = [mock_shard2_dev1]

    mock_live_arrays.return_value = [mock_array1, mock_array2]

    usage = hbm_usage_bytes(devices)

    # Expected calculations:
    # Array1: 2000 bytes on device1, 2000 bytes on device2
    # Array2: 1000 bytes on device1
    # Device1 total: 2000 + 1000 = 3000 bytes
    # Device2 total: 2000 + 0 = 2000 bytes
    # hbm_limit = 95 * GBYTES for TPU v5p
    expected_usage = [(3000, 95 * GBYTES), (2000, 95 * GBYTES)]
    assert usage == expected_usage


@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_hbm_usage_gb_pathways_disabled():
    """Tests hbm_usage_gb when VLLM_TPU_USING_PATHWAYS is False."""
    mock_device1 = MagicMock()
    mock_device1.memory_stats.return_value = {
        "bytes_in_use": 100 * GBYTES,
        "bytes_limit": 128 * GBYTES
    }
    mock_device2 = MagicMock()
    mock_device2.memory_stats.return_value = {
        "bytes_in_use": 50.5 * GBYTES,
        "bytes_limit": 128.0 * GBYTES
    }

    devices = [mock_device1, mock_device2]
    usage = hbm_usage_gb(devices)

    expected_usage = [(100.0, 128.0), (50.5, 128.0)]
    assert usage == expected_usage


@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", True)
@patch("jax.live_arrays")
@patch("jax.devices")
def test_hbm_usage_bytes_pathways_no_arrays(mock_devices, mock_live_arrays):
    """Tests hbm_usage_bytes when VLLM_TPU_USING_PATHWAYS is True but no live arrays."""
    # Mock TPU v6e devices
    mock_jax_device = MagicMock()
    mock_jax_device.device_kind = "TPU v6e"
    mock_devices.return_value = [mock_jax_device]

    mock_device1 = MagicMock()
    mock_device2 = MagicMock()
    devices = [mock_device1, mock_device2]

    # No live arrays
    mock_live_arrays.return_value = []

    usage = hbm_usage_bytes(devices)

    # No arrays means no memory usage, defaultdict returns 0 for missing keys
    # HBM limit for TPU v6e is 32 GB
    expected_usage = [(0, 32 * GBYTES), (0, 32 * GBYTES)]
    assert usage == expected_usage


@pytest.mark.parametrize(
    "head_dim, expected_padded_head_dim",
    [
        (1, 128),
        (64, 64),
        (127, 128),
        (128, 128),
        (129, 256),
        (255, 256),
        (256, 256),
        (0, 0),  # Although head_dim is usually positive, testing boundary
    ],
)
def test_get_padded_head_dim(head_dim, expected_padded_head_dim):
    """Tests the get_padded_head_dim function."""
    assert get_padded_head_dim(head_dim) == expected_padded_head_dim


def test_quantize_kv_float8_e4m3fn():
    """Tests the quantize_kv function with float8_e4m3fn dtype."""
    key = jnp.array([-1.0, 0.5, 1.0, 1.5])
    value = jnp.array([2.0, 0.0, -2.0, -3.0])
    kv_cache_quantized_dtype = jnp.float8_e4m3fn
    k_scale = 0.1
    v_scale = 0.2

    quantized_key, quantized_value = quantize_kv(key, value,
                                                 kv_cache_quantized_dtype,
                                                 k_scale, v_scale)

    # Expected key: key / k_scale -> clip -> astype
    # [-10., 5., 10., 15.] are within float8_e4m3fn range
    expected_key = jnp.array([-10.0, 5.0, 10.0, 15.0], dtype=jnp.float8_e4m3fn)

    # Expected value: value / v_scale -> clip -> astype
    # [10., 0., -10., -15.] are within float8_e4m3fn range
    expected_value = jnp.array([10.0, 0.0, -10.0, -15.0],
                               dtype=jnp.float8_e4m3fn)

    assert jnp.array_equal(quantized_key, expected_key)
    assert jnp.array_equal(quantized_value, expected_value)

    # Test clipping
    dtype_info = jnp.finfo(kv_cache_quantized_dtype)
    minval, maxval = float(dtype_info.min), float(dtype_info.max)

    # Values that will be outside the range after scaling
    key_clip = jnp.array([minval * k_scale * 2, maxval * k_scale * 2])
    value_clip = jnp.array([maxval * v_scale * 2, minval * v_scale * 2])
    quantized_key_clip, quantized_value_clip = quantize_kv(
        key_clip, value_clip, kv_cache_quantized_dtype, k_scale, v_scale)

    # Values should be clipped to the min/max of the float8 dtype
    expected_key_clip = jnp.array([minval, maxval], dtype=jnp.float8_e4m3fn)
    expected_value_clip = jnp.array([maxval, minval], dtype=jnp.float8_e4m3fn)

    assert jnp.array_equal(quantized_key_clip, expected_key_clip)
    assert jnp.array_equal(quantized_value_clip, expected_value_clip)


def test_get_jax_dtype_from_str_dtype():
    """
    Test the get_jax_dtype_from_str_dtype function
    """
    assert get_jax_dtype_from_str_dtype("int8") == jnp.int8
    assert get_jax_dtype_from_str_dtype("bfloat16") == jnp.bfloat16
    assert get_jax_dtype_from_str_dtype("fp8") == jnp.float8_e4m3fn
    assert get_jax_dtype_from_str_dtype("fp8_e4m3") == jnp.float8_e4m3fn
    assert get_jax_dtype_from_str_dtype("fp8_e5m2") == jnp.float8_e5m2
