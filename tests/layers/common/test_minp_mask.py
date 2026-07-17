# Copyright 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy tests for the min-p logit mask (``binary_search.minp_mask``).

Runs on the JAX CPU backend (no TPU needed):
    JAX_PLATFORMS=cpu python -m pytest tests/layers/common/test_minp_mask.py

``minp_mask`` implements the filter in logit space
(``logit_i >= max_logit + log(min_p)``); these tests assert it matches vLLM's
reference min-p semantics computed in probability space
(``prob_i >= min_p * max_prob``), which is the equivalence the logit-space
formulation relies on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.binary_search import minp_mask

_REPLACE = -1e12


def _reference_minp(logits: np.ndarray, min_p: np.ndarray) -> np.ndarray:
    """vLLM min-p semantics, computed in probability space (float64 reference).

    Keep token i iff ``prob_i >= min_p * max_prob``; else set to _REPLACE.
    """
    logits = logits.astype(np.float64)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    max_prob = probs.max(axis=-1, keepdims=True)
    keep = probs >= (min_p[:, None] * max_prob)
    return np.where(keep, logits, _REPLACE)


def _kept(mask_out: np.ndarray) -> np.ndarray:
    return np.asarray(mask_out) > _REPLACE / 2


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("min_p_val", [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0])
def test_minp_mask_matches_reference_scalar(seed, min_p_val):
    rng = np.random.default_rng(seed)
    batch, vocab = 8, 4096
    logits = rng.standard_normal((batch, vocab)).astype(np.float32) * 5.0
    min_p = np.full((batch, ), min_p_val, dtype=np.float32)

    out = np.asarray(
        minp_mask(jnp.asarray(logits), jnp.asarray(min_p), _REPLACE))
    ref = _reference_minp(logits, min_p)

    # The kept/removed decision must match exactly (threshold equivalence).
    np.testing.assert_array_equal(_kept(out), _kept(ref))
    # Kept logits are passed through unchanged.
    keep = _kept(ref)
    np.testing.assert_allclose(out[keep], logits[keep], rtol=0, atol=0)


def test_minp_mask_per_row_min_p():
    """Different min_p per batch row (the real serving case)."""
    rng = np.random.default_rng(123)
    batch, vocab = 6, 2048
    logits = rng.standard_normal((batch, vocab)).astype(np.float32) * 3.0
    min_p = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 1.0], dtype=np.float32)

    out = np.asarray(
        minp_mask(jnp.asarray(logits), jnp.asarray(min_p), _REPLACE))
    ref = _reference_minp(logits, min_p)
    np.testing.assert_array_equal(_kept(out), _kept(ref))


def test_minp_zero_is_disabled():
    """min_p == 0 keeps every token (log(0) = -inf threshold)."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((4, 1000)).astype(np.float32) * 4.0
    min_p = np.zeros((4, ), dtype=np.float32)
    out = np.asarray(
        minp_mask(jnp.asarray(logits), jnp.asarray(min_p), _REPLACE))
    np.testing.assert_allclose(out, logits, rtol=0, atol=0)
    assert np.isfinite(out).all()


def test_minp_one_keeps_only_argmax():
    """min_p == 1 keeps only the max-probability token(s) per row."""
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((5, 500)).astype(np.float32) * 4.0
    min_p = np.ones((5, ), dtype=np.float32)
    out = np.asarray(
        minp_mask(jnp.asarray(logits), jnp.asarray(min_p), _REPLACE))
    kept = _kept(out)
    # Exactly the argmax column is kept in each row (ties improbable here).
    assert kept.sum(axis=-1).max() == 1
    np.testing.assert_array_equal(np.argmax(out, axis=-1),
                                  np.argmax(logits, axis=-1))


def test_minp_mask_jit():
    """Works under jax.jit (min_p as a traced arg)."""
    logits = jnp.asarray(
        np.random.default_rng(0).standard_normal((3, 256)).astype(np.float32))
    min_p = jnp.asarray(np.array([0.0, 0.1, 0.7], dtype=np.float32))
    fn = jax.jit(lambda x, m: minp_mask(x, m, _REPLACE))
    out = np.asarray(fn(logits, min_p))
    ref = _reference_minp(np.asarray(logits), np.asarray(min_p))
    np.testing.assert_array_equal(_kept(out), _kept(ref))
