# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import pytest

# Import the functions to be tested
from tpu_commons.utils_jax import (GBYTES, enable_megacore, get_megacore,
                                   get_num_kv_heads_by_tp, get_padded_head_dim,
                                   hbm_usage_bytes, hbm_usage_gb)


def test_enable_and_get_megacore():
    """Tests the enable_megacore and get_megacore functions."""
    assert not get_megacore()
    enable_megacore()
    assert get_megacore()


@pytest.mark.parametrize(
    "num_kv_heads, tp_size, expected",
    [
        (8, 2, 8),  # tp_size divides num_kv_heads
        (8, 8, 8),  # tp_size equals num_kv_heads
        (4, 8, 8),  # num_kv_heads divides tp_size
        (1, 4, 4),  # num_kv_heads is 1
    ],
)
def test_get_num_kv_heads_by_tp_valid_inputs(num_kv_heads, tp_size, expected):
    """Tests get_num_kv_heads_by_tp with valid inputs."""
    assert get_num_kv_heads_by_tp(num_kv_heads, tp_size) == expected


@pytest.mark.parametrize(
    "num_kv_heads, tp_size",
    [
        (8, 3),  # Neither divides the other
        (5, 2),
    ],
)
def test_get_num_kv_heads_by_tp_invalid_inputs(num_kv_heads, tp_size):
    """Tests get_num_kv_heads_by_tp with invalid inputs (assertions)."""
    with pytest.raises(AssertionError):
        get_num_kv_heads_by_tp(num_kv_heads, tp_size)


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


@patch("tpu_commons.utils_jax.PATHWAYS_ENABLED", False)
def test_hbm_usage_bytes_pathways_disabled():
    """Tests hbm_usage_bytes when PATHWAYS_ENABLED is False."""
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


@patch("tpu_commons.utils_jax.PATHWAYS_ENABLED", True)
def test_hbm_usage_bytes_pathways_enabled():
    """Tests hbm_usage_bytes when PATHWAYS_ENABLED is True."""
    # When PATHWAYS_ENABLED is True, memory_stats() is not called.
    # The function returns a hardcoded value.
    devices = [MagicMock(), MagicMock()]  # Devices are not used in this path
    usage = hbm_usage_bytes(devices)

    # The hardcoded values from the source
    expected_usage = [(32384, 33550237184), (32384, 33550237184)]
    assert usage == expected_usage


@patch("tpu_commons.utils_jax.PATHWAYS_ENABLED", False)
def test_hbm_usage_gb_pathways_disabled():
    """Tests hbm_usage_gb when PATHWAYS_ENABLED is False."""
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


@patch("tpu_commons.utils_jax.PATHWAYS_ENABLED", True)
def test_hbm_usage_gb_pathways_enabled():
    """Tests hbm_usage_gb when PATHWAYS_ENABLED is True."""
    devices = [MagicMock()]
    usage = hbm_usage_gb(devices)

    # Hardcoded values: (32384, 33550237184)
    # 32384 / GBYTES = 0.00003016
    # 33550237184 / GBYTES = 31.25
    expected_usage = [(0.0, 31.25)]  # Rounded to 2 decimal places
    assert usage == expected_usage


@pytest.mark.parametrize(
    "head_dim, expected_padded_head_dim",
    [
        (1, 128),
        (64, 128),
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
