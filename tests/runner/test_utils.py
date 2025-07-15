# SPDX-License-Identifier: Apache-2.0
import time

import jax
import jax.numpy as jnp
import numpy as np  # For array comparison
import pytest
from jax._src.interpreters import pxla

from tpu_commons.runner.utils import (ForbidCompile, LatencyTracker,
                                      determine_do_sampling,
                                      get_padded_num_reqs_with_upper_limit,
                                      pad_to_multiple)


def test_determine_do_sampling():
    """Tests the determine_do_sampling function."""
    assert not determine_do_sampling(top_k=50, temperature=0.0)
    assert not determine_do_sampling(top_k=1, temperature=0.7)
    assert not determine_do_sampling(top_k=1, temperature=0.0)
    assert determine_do_sampling(top_k=10, temperature=0.5)


def test_pad_to_multiple():
    """Tests the pad_to_multiple function."""
    assert pad_to_multiple(16, 8) == 16
    assert pad_to_multiple(17, 8) == 24
    assert pad_to_multiple(24, 8) == 24
    assert pad_to_multiple(1, 8) == 8

    # Test with max_limit
    assert pad_to_multiple(17, 8, max_limit=20) == 20
    assert pad_to_multiple(17, 8, max_limit=30) == 24
    assert pad_to_multiple(25, 8, max_limit=30) == 30

    # Test with keep_one
    assert pad_to_multiple(1, 8, keep_one=True) == 1
    assert pad_to_multiple(2, 8, keep_one=True) == 8
    assert pad_to_multiple(1, 8, keep_one=False) == 8

    # Test invalid inputs
    with pytest.raises(AssertionError):
        pad_to_multiple(0, 8)
    with pytest.raises(AssertionError):
        pad_to_multiple(-5, 8)


def test_get_padded_num_reqs_with_upper_limit():
    """Tests the get_padded_num_reqs_with_upper_limit function."""
    # From utils.py, MIN_NUM_SEQS = 8
    assert get_padded_num_reqs_with_upper_limit(4, 128) == 8
    assert get_padded_num_reqs_with_upper_limit(8, 128) == 8
    assert get_padded_num_reqs_with_upper_limit(9, 128) == 16
    assert get_padded_num_reqs_with_upper_limit(16, 128) == 16
    assert get_padded_num_reqs_with_upper_limit(17, 128) == 32
    assert get_padded_num_reqs_with_upper_limit(100, 64) == 64
    assert get_padded_num_reqs_with_upper_limit(1, 128) == 8


def test_latency_tracker(caplog):
    """Tests the LatencyTracker context manager."""
    caplog.set_level("INFO")
    sleep_duration = 0.01
    with LatencyTracker("test_op") as tracker:
        time.sleep(sleep_duration)

    elapsed = tracker.end_time - tracker.start_time
    assert elapsed >= sleep_duration
    assert "Latency for 'test_op'" in caplog.text
    assert f"{elapsed:.3f} seconds" in caplog.text


# Define a fixture to clear the JAX cache before each test
@pytest.fixture(autouse=True)
def clear_jax_cache():
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture
def jitted_function():
    """Defines a jitted function for testing."""

    @jax.jit
    def my_jitted_func(x):
        return x * 2

    return my_jitted_func


@pytest.fixture
def scalar_input():
    """Sample scalar input."""
    return jnp.array(5.0, dtype=jnp.float32)


@pytest.fixture
def vector_input():
    """Sample vector input with a different shape."""
    return jnp.array([5.0, 10.0])


def test_forbid_compile_raises_error_on_first_call(jitted_function,
                                                   scalar_input):
    """Test that ForbidCompile raises an error when a compilation occurs."""
    with pytest.raises(RuntimeError, match="JAX compilation occurred"):
        with ForbidCompile():
            jitted_function(scalar_input)


def test_forbid_compile_succeeds_on_cached_call(jitted_function, scalar_input):
    """Test that ForbidCompile does not raise an error on a cached call."""
    # Warm up the cache
    jitted_function(scalar_input)
    with ForbidCompile():
        result = jitted_function(scalar_input)
    assert result == 10.0


def test_forbid_compile_restores_original_function():
    """Test that ForbidCompile restores the original JAX function after exit."""
    original_func = pxla._cached_lowering_to_hlo
    with ForbidCompile():
        pass
    assert pxla._cached_lowering_to_hlo is original_func


def test_forbid_compile_with_exception():
    """Test that ForbidCompile restores the original function even if an exception occurs."""
    original_func = pxla._cached_lowering_to_hlo
    with pytest.raises(ValueError, match="Test exception"):
        with ForbidCompile():
            raise ValueError("Test exception")
    assert pxla._cached_lowering_to_hlo is original_func


def test_recompilation_on_shape_change_only(jitted_function):
    """
    Tests that recompilation in jax.jit is primarily triggered by shape/dtype
    changes, not just value changes in strongly-typed tensors.
    """
    expected_error_message = "JAX compilation occurred but was forbidden in this context."

    # --- Inputs ---
    shape1 = (2, 2)
    shape2 = (4, 2)
    dtype = jnp.float32

    tensor1 = jnp.ones(shape1, dtype=dtype)
    tensor2_same_shape = jnp.full(shape1, 5.0, dtype=dtype)
    tensor3_diff_shape = jnp.ones(shape2, dtype=dtype)

    print("\n--- Tensor Shapes and Types ---")
    print(f"tensor1: shape {tensor1.shape}, dtype {tensor1.dtype}")
    print(
        f"tensor2_same_shape: shape {tensor2_same_shape.shape}, dtype {tensor2_same_shape.dtype}"
    )
    print(
        f"tensor3_diff_shape: shape {tensor3_diff_shape.shape}, dtype {tensor3_diff_shape.dtype}"
    )

    # --- Warm-up ---
    print("\n--- Warm-up with tensor1 ---")
    result1 = jitted_function(tensor1)
    np.testing.assert_array_equal(result1, tensor1 * 2)
    misses_after_warmup = pxla._cached_lowering_to_hlo.cache_info().misses
    assert misses_after_warmup == 1
    print(f"Cache after tensor1: {pxla._cached_lowering_to_hlo.cache_info()}")

    # --- Test 1: Same Shape, Different Values ---
    # Expected: NO recompilation, ForbidCompile should NOT raise.
    print("\n--- Testing tensor2_same_shape (same shape, diff value) ---")
    with ForbidCompile(message=expected_error_message):
        result2 = jitted_function(tensor2_same_shape)
    np.testing.assert_array_equal(result2, tensor2_same_shape * 2)
    misses_after_t2 = pxla._cached_lowering_to_hlo.cache_info().misses
    print(
        f"Cache after tensor2_same_shape: {pxla._cached_lowering_to_hlo.cache_info()}"
    )
    assert misses_after_t2 == misses_after_warmup, "Recompilation NOT expected for value change."

    # --- Test 2: Different Shape ---
    # Expected: Recompilation, ForbidCompile SHOULD raise.
    print("\n--- Testing tensor3_diff_shape (different shape) ---")
    with pytest.raises(RuntimeError, match=expected_error_message):
        with ForbidCompile(message=expected_error_message):
            jitted_function(tensor3_diff_shape)

    misses_after_t3 = pxla._cached_lowering_to_hlo.cache_info().misses
    print(
        f"Cache after tensor3_diff_shape: {pxla._cached_lowering_to_hlo.cache_info()}"
    )
    assert misses_after_t3 == misses_after_t2 + 1, "Recompilation WAS expected for shape change."

    # Optional: Verify the function works outside ForbidCompile for the new shape
    result3 = jitted_function(tensor3_diff_shape)
    np.testing.assert_array_equal(result3, tensor3_diff_shape * 2)
