# Copyright 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy tests for the repetition-penalty logit op (binary_search.apply_repetition_penalty).

Runs on the JAX CPU backend (no TPU needed):
    JAX_PLATFORMS=cpu python -m pytest tests/layers/common/test_repetition_penalty.py

Asserts the JAX op matches vLLM's repetition-penalty semantics
(vllm.model_executor.layers.utils.apply_penalties /
vllm._custom_ops.apply_repetition_penalties): for tokens that appeared in the
prompt OR output, positive logits are divided by the penalty and non-positive
logits are multiplied by it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.binary_search import apply_repetition_penalty


def _reference(logits: np.ndarray, seen: np.ndarray,
               rp: np.ndarray) -> np.ndarray:
    """vLLM repetition-penalty reference (float64)."""
    logits = logits.astype(np.float64)
    rp = rp[:, None].astype(np.float64)
    penalized = np.where(logits > 0, logits / rp, logits * rp)
    return np.where(seen, penalized, logits)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("rp_val", [1.0, 1.05, 1.2, 2.0])
def test_matches_reference_scalar(seed, rp_val):
    rng = np.random.default_rng(seed)
    batch, vocab = 8, 4096
    logits = rng.standard_normal((batch, vocab)).astype(np.float32) * 5.0
    seen = rng.random((batch, vocab)) < 0.1  # ~10% of tokens marked seen
    rp = np.full((batch, ), rp_val, dtype=np.float32)

    out = np.asarray(
        apply_repetition_penalty(jnp.asarray(logits), jnp.asarray(seen),
                                 jnp.asarray(rp)))
    ref = _reference(logits, seen, rp)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)


def test_per_row_penalty():
    """Different repetition_penalty per row (the real serving case)."""
    rng = np.random.default_rng(123)
    batch, vocab = 6, 2048
    logits = rng.standard_normal((batch, vocab)).astype(np.float32) * 3.0
    seen = rng.random((batch, vocab)) < 0.2
    rp = np.array([1.0, 1.05, 1.1, 1.3, 1.5, 2.0], dtype=np.float32)

    out = np.asarray(
        apply_repetition_penalty(jnp.asarray(logits), jnp.asarray(seen),
                                 jnp.asarray(rp)))
    ref = _reference(logits, seen, rp)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)


def test_rp_one_is_identity():
    """repetition_penalty == 1.0 leaves logits unchanged (engine default)."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((4, 1000)).astype(np.float32) * 4.0
    seen = rng.random((4, 1000)) < 0.5
    rp = np.ones((4, ), dtype=np.float32)
    out = np.asarray(
        apply_repetition_penalty(jnp.asarray(logits), jnp.asarray(seen),
                                 jnp.asarray(rp)))
    np.testing.assert_allclose(out, logits, rtol=0, atol=0)


def test_unseen_tokens_unchanged():
    """Tokens not in the seen mask are never penalized."""
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((3, 512)).astype(np.float32) * 4.0
    seen = np.zeros((3, 512), dtype=bool)
    rp = np.full((3, ), 2.0, dtype=np.float32)
    out = np.asarray(
        apply_repetition_penalty(jnp.asarray(logits), jnp.asarray(seen),
                                 jnp.asarray(rp)))
    np.testing.assert_allclose(out, logits, rtol=0, atol=0)


def test_sign_direction():
    """Positive logits shrink, negative logits grow more negative (penalized)."""
    logits = jnp.asarray(np.array([[2.0, -2.0]], dtype=np.float32))
    seen = jnp.asarray(np.array([[True, True]]))
    rp = jnp.asarray(np.array([2.0], dtype=np.float32))
    out = np.asarray(apply_repetition_penalty(logits, seen, rp))
    assert out[0, 0] == pytest.approx(1.0)  # 2.0 / 2
    assert out[0, 1] == pytest.approx(-4.0)  # -2.0 * 2


def test_jit():
    logits = jnp.asarray(
        np.random.default_rng(0).standard_normal((3, 256)).astype(np.float32))
    seen = jnp.asarray(np.random.default_rng(1).random((3, 256)) < 0.3)
    rp = jnp.asarray(np.array([1.0, 1.2, 2.0], dtype=np.float32))
    fn = jax.jit(apply_repetition_penalty)
    out = np.asarray(fn(logits, seen, rp))
    ref = _reference(np.asarray(logits), np.asarray(seen), np.asarray(rp))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)
