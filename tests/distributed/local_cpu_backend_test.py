# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tpu_inference.distributed.offload.cpu_backend import LocalCPUBackend
from tpu_inference.distributed.offload.utils import CacheKey


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

        assert key not in backend.pin_counts
        backend.contains(key, pin_on_hit=True)
        assert key in backend.pin_counts
        assert backend.pin_counts[key] == 1

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
        assert key_a in backend.pin_counts

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
        backend.maybe_unpin_keys([key_a])
        assert key_a not in backend.pin_counts

        # This should now evict B
        backend.add(key_c, value)

        assert key_b not in backend.cache
        assert key_a in backend.cache
        assert key_c in backend.cache

    def test_cache_full_of_pinned_items_prevents_add(clean_backend_instance):
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
        assert key_a in backend.pin_counts
        assert key_b in backend.pin_counts

    def test_pinning_same_key_multiple_times_increments_count(
            self, clean_backend_instance):
        """Verifies that pinning an already-pinned key increments its count."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key = CacheKey(model_name="test_model", chunk_hash="A")
        backend.add(key, create_mock_value(10))

        backend.contains(key, pin_on_hit=True)
        assert backend.pin_counts[key] == 1

        backend.contains(key, pin_on_hit=True)
        assert backend.pin_counts[key] == 2

    def test_unpin_decrements_count_and_removes_at_zero(
            self, clean_backend_instance):
        """Tests the core reference counting logic of the unpin_keys method."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key = CacheKey(model_name="test_model", chunk_hash="A")
        backend.add(key, create_mock_value(10))

        # Pin twice
        backend.contains(key, pin_on_hit=True)
        backend.contains(key, pin_on_hit=True)
        assert backend.pin_counts[key] == 2

        # Unpin once
        backend.maybe_unpin_keys([key])
        assert key in backend.pin_counts
        assert backend.pin_counts[key] == 1

        # Unpin again
        backend.maybe_unpin_keys([key])
        assert key not in backend.pin_counts

    def test_item_with_positive_pin_count_is_not_evicted(
            self, clean_backend_instance):
        """
        Tests that an item with a pin count > 0 is not evicted, confirming
        the race condition fix.
        """
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        key_c = CacheKey(model_name="test_model", chunk_hash="C")
        value = create_mock_value(50)

        backend.add(key_a, value)  # Will be LRU
        backend.add(key_b, value)

        # Pin key_a twice (simulating two requests)
        backend.contains(key_a, pin_on_hit=True)
        backend.contains(key_a, pin_on_hit=True)

        # Unpin key_a once (simulating one request finishing)
        backend.maybe_unpin_keys([key_a])
        assert backend.pin_counts[key_a] == 1

        # This add should trigger eviction of key_b, as key_a is still pinned.
        backend.add(key_c, value)

        assert key_a in backend.cache
        assert key_b not in backend.cache
        assert key_c in backend.cache
        assert key_a in backend.pin_counts

    def test_unpin_keys_returns_correct_counts(self, clean_backend_instance):
        """Validates the meaningful return values of unpin_keys."""
        backend = LocalCPUBackend(max_cpu_cache_size_bytes=100)
        key_a = CacheKey(model_name="test_model", chunk_hash="A")
        key_b = CacheKey(model_name="test_model", chunk_hash="B")
        value = create_mock_value(10)

        backend.add(key_a, value)
        backend.add(key_b, value)

        # Pin A twice, B once
        backend.contains(key_a, pin_on_hit=True)
        backend.contains(key_a, pin_on_hit=True)
        backend.contains(key_b, pin_on_hit=True)

        # Unpin both. A should be decremented, B should be fully unpinned.
        unpinned_count, found_count = backend.maybe_unpin_keys([key_a, key_b])

        assert found_count == 2  # Both keys were found in pin_counts
        assert unpinned_count == 1  # Only key_b's count went to 0
        assert backend.pin_counts[key_a] == 1
        assert key_b not in backend.pin_counts
