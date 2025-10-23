# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tpu_inference.distributed.cache_util import CacheKey
from tpu_inference.distributed.local_cpu_backend import LocalCPUBackend


# Helper to create a mock value with a specific size
def create_mock_value(size_in_bytes: int) -> MagicMock:
    """Creates a mock object with an 'nbytes' attribute."""
    mock_value = MagicMock()
    mock_value.nbytes = size_in_bytes
    return mock_value


@pytest.fixture
def clean_backend_instance():
    """
    Provides a clean instance of the LocalCPUBackend for each test.
    This is crucial because LocalCPUBackend is a singleton, and without
    resetting its internal state, tests would interfere with each other.
    """
    # Reset the singleton instance before each test.
    # By setting LocalCPUBackend._instance to None, it forces the __new__ method
    # to create a fresh, new object for every single test case, ensuring test isolation.
    LocalCPUBackend._instance = None
    LocalCPUBackend._initialized = False
    yield
    # Clean up after the test
    LocalCPUBackend._instance = None
    LocalCPUBackend._initialized = False


class TestLocalCPUBackend:
    """Test suite for the LocalCPUBackend."""

    def test_add_and_get(self, clean_backend_instance):
        """Verifies that a value can be added and then retrieved successfully."""
        # Increased size to accommodate the list test without eviction
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=150)
        key = CacheKey(model_name="test_model", chunk_hash="A")
        value = create_mock_value(50)

        backend.add(key, value)
        retrieved_value = backend.get(key)

        assert retrieved_value == value
        assert backend.current_size_bytes == 50

        # Test with a list of JAX arrays (mocked)
        key_list = CacheKey(model_name="test_model",
                            chunk_hash="list_item_hash")
        value_list = [create_mock_value(20), create_mock_value(30)]
        backend.add(key_list, value_list)
        retrieved_list_value = backend.get(key_list)

        assert retrieved_list_value == value_list
        assert backend.current_size_bytes == 50 + 20 + 30

    def test_get_updates_lru_order(self, clean_backend_instance):
        """Tests that get() moves the accessed item to the end (most recent)."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        value = create_mock_value(10)

        backend.add(key_a, value)
        backend.add(key_b, value)
        # Initial order: A, B
        assert list(backend.cache.keys()) == [key_a, key_b]

        backend.get(key_a)
        # Accessed A, so order should now be: B, A
        assert list(backend.cache.keys()) == [key_b, key_a]

    def test_contains_updates_lru_order(self, clean_backend_instance):
        """Tests that contains() moves the accessed item to the end."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        value = create_mock_value(10)

        backend.add(key_a, value)
        backend.add(key_b, value)
        # Initial order: A, B
        assert list(backend.cache.keys()) == [key_a, key_b]

        backend.contains(key_a)
        # Accessed A, so order should now be: B, A
        assert list(backend.cache.keys()) == [key_b, key_a]

    def test_eviction_on_add(self, clean_backend_instance):
        """Tests that the least recently used item is evicted when cache is full."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        key_c = CacheKey(model_name="test_model", chunk_hash="C")
        value = create_mock_value(50)

        backend.add(key_a, value)  # LRU
        backend.add(key_b, value)  # MRU
        assert backend.current_size_bytes == 100
        assert key_a in backend.cache
        assert key_b in backend.cache

        # This should evict key_a
        backend.add(key_c, value)

        assert key_a not in backend.cache
        assert key_b in backend.cache
        assert key_c in backend.cache
        assert backend.current_size_bytes == 100

    def test_cannot_add_item_larger_than_capacity(self,
                                                  clean_backend_instance):
        """Tests that an item larger than the cache's capacity is not added."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key = CacheKey(model_name="test_model", chunk_hash="large_item_hash")
        value = create_mock_value(101)

        backend.add(key, value)

        assert key not in backend.cache
        assert backend.current_size_bytes == 0

    def test_pin_on_hit(self, clean_backend_instance):
        """Tests that using contains() with pin_on_hit=True pins the key."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key = CacheKey(model_name="test_model", chunk_hash="A")
        backend.add(key, create_mock_value(10))

        assert key not in backend.pinned_keys
        backend.contains(key, pin_on_hit=True)
        assert key in backend.pinned_keys

    def test_pinned_item_is_not_evicted(self, clean_backend_instance):
        """Tests that a pinned item is protected from eviction."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        key_c = CacheKey(model_name="test_model", chunk_hash="C")
        value = create_mock_value(50)

        backend.add(key_a, value)
        backend.add(key_b, value)
        backend.contains(key_a, pin_on_hit=True)
        assert key_a in backend.pinned_keys

        # This should evict key_b, because key_a is pinned
        backend.add(key_c, value)

        assert key_a in backend.cache
        assert key_b not in backend.cache
        assert key_c in backend.cache
        assert backend.current_size_bytes == 100

    def test_unpin_makes_item_evictable(self, clean_backend_instance):
        """Tests that unpinning a key makes it eligible for eviction again."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        key_c = CacheKey(model_name="test_model", chunk_hash="C")
        value = create_mock_value(50)

        backend.add(key_a, value)
        backend.add(key_b, value)
        backend.contains(key_a,
                         pin_on_hit=True)  # Pin A, and make A most recent
        assert list(backend.cache.keys()) == [key_b, key_a]

        # Unpin A, making it the LRU evictable item
        backend.unpin_keys([key_a])
        assert key_a not in backend.pinned_keys

        # This should now evict B
        backend.add(key_c, value)

        assert key_b not in backend.cache
        assert key_a in backend.cache
        assert key_c in backend.cache

    def test_cache_full_of_pinned_items_prevents_add(self,
                                                     clean_backend_instance):
        """
        Tests that no new items can be added if the cache is full of
        pinned items.
        """
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        key_c = CacheKey(model_name="test_model", chunk_hash="C")
        value = create_mock_value(50)

        backend.add(key_a, value)
        backend.add(key_b, value)

        # Pin all items in the cache
        backend.contains(key_a, pin_on_hit=True)
        backend.contains(key_b, pin_on_hit=True)

        # Attempt to add a new item
        backend.add(key_c, value)

        assert key_c not in backend.cache
        assert key_a in backend.cache
        assert key_b in backend.cache
        assert backend.current_size_bytes == 100
