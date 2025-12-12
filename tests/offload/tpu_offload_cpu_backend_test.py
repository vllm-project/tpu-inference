# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tpu_inference.offload.cpu_backend import LocalCPUBackend
from tpu_inference.offload.utils import CpuChunkId


# Helper to create a mock jax array with a specific size in bytes
def create_mock_jax_array(size_in_bytes: int) -> MagicMock:
    """Creates a mock object with an 'nbytes' attribute."""
    mock_value = MagicMock()
    mock_value.nbytes = size_in_bytes
    return mock_value


class TestLocalCPUBackend:
    """Test suite for the LocalCPUBackend."""

    def test_add_and_get(self):
        """Verifies that a value can be added and then retrieved successfully."""
        backend = LocalCPUBackend(num_cpu_chunks=10)
        key = CpuChunkId(0)
        value = create_mock_jax_array(50)

        backend.add(key, value)
        retrieved_value = backend.get(key)

        assert retrieved_value == value
        assert backend.current_size_bytes == 50

        # Test with a list of JAX arrays (mocked)
        key_list = CpuChunkId(1)
        value_list = [create_mock_jax_array(20), create_mock_jax_array(30)]
        backend.add(key_list, value_list)
        retrieved_list_value = backend.get(key_list)

        assert retrieved_list_value == value_list
        assert backend.current_size_bytes == 50 + 20 + 30

        assert backend.num_saved_cpu_chunks == 2

    def test_add_invalid_chunk_id(self):
        """Verifies that adding a value with an invalid chunk_id raises a ValueError."""
        backend = LocalCPUBackend(num_cpu_chunks=10)
        value = create_mock_jax_array(50)

        with pytest.raises(ValueError):
            backend.add(CpuChunkId(-1), value)

        assert backend.num_saved_cpu_chunks == 0

    def test_reclaim_unoccupied_chunks(self):
        """Tests that unoccupied chunks are reclaimed correctly."""
        backend = LocalCPUBackend(num_cpu_chunks=10)
        key1 = CpuChunkId(0)
        key2 = CpuChunkId(1)
        key3 = CpuChunkId(2)
        value = create_mock_jax_array(10)

        backend.add(key1, value)
        backend.add(key2, value)
        backend.add(key3, value)

        assert backend.current_size_bytes == 30
        assert len(backend.cache) == 3

        # Reclaim one chunk
        backend.reclaim_unoccupied_chunks(occupied_chunk_ids=[key1, key3])

        assert backend.current_size_bytes == 20
        assert len(backend.cache) == 2
        assert key1 in backend.cache
        assert key2 not in backend.cache
        assert key3 in backend.cache

        # Reclaim all chunks
        backend.reclaim_unoccupied_chunks(occupied_chunk_ids=[])

        assert backend.current_size_bytes == 0
        assert len(backend.cache) == 0
