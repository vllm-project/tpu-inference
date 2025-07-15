# SPDX-License-Identifier: Apache-2.0
import time

import jax
import jax.numpy as jnp
import pytest
from jax._src.interpreters import pxla

from tpu_commons.runner.utils import (ForbidCompile, LatencyTracker,
                                      determine_do_sampling,
                                      get_padded_num_reqs_with_upper_limit,
                                      pad_to_multiple)


def test_determine_do_sampling():
    """Tests the determine_do_sampling function."""
    assert determine_do_sampling(top_k=50, temperature=0.0)
    assert determine_do_sampling(top_k=1, temperature=0.7)
    assert determine_do_sampling(top_k=10, temperature=0.5)
    assert not determine_do_sampling(top_k=1, temperature=0.0)


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


@pytest.mark.jax
class TestForbidCompile:
    """Tests for the ForbidCompile context manager."""

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Clear all jit caches before each test in this class."""

        @jax.jit
        def add_one(x):
            return x + 1

        @jax.jit
        def add_two(x):
            return x + 2

        @jax.jit
        def add_three(x):
            return x + 3

        add_one.clear_cache()
        add_two.clear_cache()
        add_three.clear_cache()

    def test_forbid_compile_raises_on_first_call(self):
        """Tests that compilation is forbidden on the first call."""

        @jax.jit
        def add_one(x):
            return x + 1

        with pytest.raises(RuntimeError, match="JAX compilation occurred"):
            with ForbidCompile():
                add_one(jnp.ones(1))

    def test_forbid_compile_succeeds_after_warmup(self):
        """Tests that compilation is allowed after a warm-up call."""

        @jax.jit
        def add_two(x):
            return x + 2

        # Warm-up call
        add_two(jnp.ones(1)).block_until_ready()

        # This should not raise an error
        try:
            with ForbidCompile():
                add_two(jnp.ones(1))
        except RuntimeError:
            pytest.fail("ForbidCompile raised RuntimeError unexpectedly.")

    def test_forbid_compile_raises_on_new_shape(self):
        """Tests that compilation is forbidden for new input shapes."""

        @jax.jit
        def add_three(x):
            return x + 3

        # Warm-up with one shape
        add_three(jnp.ones((1, ))).block_until_ready()

        # Call with a new shape inside the context manager
        with pytest.raises(RuntimeError, match="JAX compilation occurred"):
            with ForbidCompile():
                add_three(jnp.ones((2, )))

    def test_forbid_compile_restores_function_on_exit(self):
        """Tests that the original JAX function is restored on exit."""
        original_func = pxla._cached_lowering_to_hlo

        with ForbidCompile():
            assert pxla._cached_lowering_to_hlo is not original_func

        assert pxla._cached_lowering_to_hlo is original_func

    def test_forbid_compile_restores_function_on_exception(self):
        """Tests that the original JAX function is restored on exception."""
        original_func = pxla._cached_lowering_to_hlo

        with pytest.raises(ValueError, match="Test exception"):
            with ForbidCompile():
                assert pxla._cached_lowering_to_hlo is not original_func
                raise ValueError("Test exception")

        assert pxla._cached_lowering_to_hlo is original_func
