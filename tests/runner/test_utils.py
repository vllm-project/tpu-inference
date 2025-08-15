# SPDX-License-Identifier: Apache-2.0
import io
import logging
import time
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from jax._src.interpreters import pxla
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons.models.jax.attention import get_kv_cache_shape_with_mesh
from tpu_commons.runner.utils import (ForbidCompile, LatencyTracker,
                                      create_kv_caches, determine_do_sampling,
                                      get_padded_num_reqs_with_upper_limit,
                                      get_padded_token_len, get_req_paddings,
                                      get_token_paddings, pad_to_multiple)


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


def test_get_paddings():
    # Bucketed padding
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    expected_paddings = [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)

    # Bucketed padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 192, 256, 320]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings

    # Exponential padding.
    max_token_size, padding_gap = 1024, 0
    expected_paddings = [16, 32, 64, 128, 256, 512, 1024]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings
    # Exponential padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 256, 512]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings


def test_get_padded_token_len():
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    paddings = get_token_paddings(min_token_size, max_token_size, padding_gap)
    assert get_padded_token_len(paddings, 1) == 16
    assert get_padded_token_len(paddings, 16) == 16
    assert get_padded_token_len(paddings, 20) == 32
    assert get_padded_token_len(paddings, 300) == 320
    assert get_padded_token_len(paddings, 512) == 512


def test_get_req_paddings():
    assert get_req_paddings(1, 32) == [8, 16, 32]
    assert get_req_paddings(8, 32) == [8, 16, 32]
    assert get_req_paddings(8, 36) == [8, 16, 32, 36]


def test_latency_tracker(caplog):
    """Tests the LatencyTracker context manager."""
    logger_name = "vllm.tpu_commons.runner.utils"
    logger = logging.getLogger(logger_name)

    original_level = logger.level
    original_propagate = logger.propagate

    # Create an in-memory stream to capture log output
    log_capture_string = io.StringIO()
    # Create a handler that writes to our in-memory stream
    capture_handler = logging.StreamHandler(log_capture_string)

    try:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(capture_handler)

        sleep_duration = 0.01
        with LatencyTracker("test_op") as tracker:
            time.sleep(sleep_duration)

        elapsed = tracker.end_time - tracker.start_time
        assert elapsed >= sleep_duration
        log_contents = log_capture_string.getvalue()

        assert "Latency for 'test_op'" in log_contents
        assert f"{elapsed:.3f} seconds" in log_contents

    finally:
        # --- IMPORTANT: Clean up and restore the logger's original state ---
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.removeHandler(capture_handler)


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
def jnp_array_input():
    return jnp.ones((2, 3))


@pytest.fixture
def jnp_array_input_same_shape():
    return jnp.zeros((2, 3))


@pytest.fixture
def jnp_array_input_new():
    return jnp.ones((3, 3))


def test_forbid_compile_raises_error_on_first_call(jitted_function,
                                                   jnp_array_input):
    """Test that ForbidCompile raises an error when a compilation occurs."""
    with pytest.raises(RuntimeError, match="JAX compilation occurred"):
        with ForbidCompile():
            jitted_function(jnp_array_input)


def test_forbid_compile_succeeds_on_cached_call(jitted_function,
                                                jnp_array_input):
    """Test that ForbidCompile does not raise an error on a cached call."""
    # Warm up the cache
    jitted_function(jnp_array_input)
    with ForbidCompile():
        jitted_function(jnp_array_input)


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


def test_forbid_compile_raises_on_new_shape(jitted_function, jnp_array_input,
                                            jnp_array_input_same_shape,
                                            jnp_array_input_new):
    """
    Tests that ForbidCompile raises a RuntimeError when a jitted function
    is called with an input shape that triggers a new compilation.
    """
    # Clear cache for a clean test state.
    pxla._cached_lowering_to_hlo.cache_clear()

    # Warm up the JIT cache with the SCALAR input.
    # This causes the first compilation and cache miss.
    jitted_function(jnp_array_input)
    misses_after_warmup = pxla._cached_lowering_to_hlo.cache_info().misses
    assert misses_after_warmup == 1

    # This call uses the same shape/dtype, so it should be a cache HIT.
    # No RuntimeError expected.
    with ForbidCompile():
        jitted_function(jnp_array_input_same_shape)
    assert pxla._cached_lowering_to_hlo.cache_info(
    ).misses == misses_after_warmup  # No new misses

    # Now, call with a VECTOR input. This has a different shape,
    # forcing a NEW compilation (cache MISS).
    # This *should* raise a RuntimeError within the ForbidCompile context.
    expected_error_message = "JAX compilation occurred but was forbidden in this context."
    with pytest.raises(RuntimeError, match=expected_error_message):
        with ForbidCompile(message=expected_error_message):
            jitted_function(jnp_array_input_new)


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
    devices = jax.devices()

    expected_sharding = NamedSharding(mesh, PartitionSpec())
    expected_dtype = jnp.bfloat16
    expected_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  num_kv_heads, head_size,
                                                  expected_dtype)

    with patch("tpu_commons.logger.init_logger",
               return_value=MagicMock()), patch(
                   "tpu_commons.runner.utils.utils.hbm_usage_gb",
                   return_value=[(0.0, 0.0), (0.0, 0.0)]):
        kv_caches = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            mesh=mesh,
            layer_names=layer_names,
            devices=devices,
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
