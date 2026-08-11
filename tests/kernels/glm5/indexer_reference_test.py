# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the GLM-5.2 lightning-indexer pure-JAX reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.glm5.indexer.reference import (
    INDEX_SKIP_TOPK_OFFSET, INDEX_TOPK_FREQ, apply_indexer_rope,
    lightning_indexer_layer, lightning_indexer_scores, select_topk_indices,
    should_recompute_indexer_layer)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# =====================================================================
# Independent (complex-rotation) reference for interleaved RoPE.
# =====================================================================
def _naive_interleaved_rope(x, positions, rope_dim, base=10000.0):
    """Independent numpy reference for GPT-J-style interleaved RoPE on the
    leading ``rope_dim`` channels, using complex-number rotation instead of
    the stack/reshape formulation under test."""
    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    out = x.copy()
    half = rope_dim // 2
    inv_freq = 1.0 / (base**(np.arange(0, rope_dim, 2) / rope_dim))
    theta = positions[:, None] * inv_freq[None, :]  # [T, half]
    if x.ndim == 3:
        theta = theta[:, None, :]
    cos, sin = np.cos(theta), np.sin(theta)
    x1 = x[..., 0:rope_dim:2]
    x2 = x[..., 1:rope_dim:2]
    z = (x1 + 1j * x2) * (cos + 1j * sin)
    out[..., 0:rope_dim:2] = z.real
    out[..., 1:rope_dim:2] = z.imag
    return out


@pytest.mark.parametrize("shape,rope_dim", [
    ((3, 128), 64),
    ((5, 4, 128), 64),
    ((1, 32, 128), 64),
    ((7, 128), 8),
])
def test_apply_indexer_rope_matches_complex_rotation(shape, rope_dim):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    num_tokens = shape[0]
    positions = jnp.arange(num_tokens, dtype=jnp.int32) + 5

    out = apply_indexer_rope(x, positions, rope_dim=rope_dim)
    ref = _naive_interleaved_rope(np.asarray(x), np.asarray(positions),
                                  rope_dim)

    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4, rtol=1e-4)
    # NoPE tail must be untouched.
    np.testing.assert_array_equal(
        np.asarray(out)[..., rope_dim:],
        np.asarray(x)[..., rope_dim:])


def test_apply_indexer_rope_position_zero_is_identity():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(4, 128)).astype(np.float32))
    positions = jnp.zeros((4, ), dtype=jnp.int32)
    out = apply_indexer_rope(x, positions, rope_dim=64)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-5)


# =====================================================================
# Naive numpy reference for the indexer scoring formula, mirroring DSv4's
# streamindex_topk_ref (tests/kernels/deepseek_v4/test_streamindex_topk.py)
# degraded to compression_ratio=1 (raw KV) plus GLM-5.2 interleaved RoPE.
# =====================================================================
def _naive_lightning_indexer_scores(
    q,
    k,
    weights,
    q_positions,
    k_positions,
    *,
    rope_dim,
    softmax_scale,
    n_head_scale,
):
    q = _naive_interleaved_rope(q, q_positions, rope_dim)
    k = _naive_interleaved_rope(k, k_positions, rope_dim)
    weights = np.asarray(weights, dtype=np.float64)
    q_positions = np.asarray(q_positions)
    k_positions = np.asarray(k_positions)

    num_tokens, num_heads, _ = q.shape
    num_kv = k.shape[0]
    scores = np.full((num_tokens, num_kv), -np.inf, dtype=np.float64)
    for t in range(num_tokens):
        for s in range(num_kv):
            if k_positions[s] > q_positions[t]:
                continue
            acc = 0.0
            for h in range(num_heads):
                inner = np.dot(q[t, h], k[s])
                acc += max(0.0, inner) * weights[t, h]
            scores[t, s] = acc * softmax_scale * n_head_scale
    return scores


@pytest.mark.parametrize(
    "num_tokens,num_kv,num_heads,head_dim,rope_dim",
    [
        (4, 4, 2, 128, 64),
        (3, 8, 4, 128, 64),
        (1, 1, 1, 128, 64),
        (5, 5, 32, 128, 64),  # GLM-5.2's real index_n_heads.
    ],
)
def test_lightning_indexer_scores_matches_naive_numpy_full_precision(
        num_tokens, num_kv, num_heads, head_dim, rope_dim):
    """Full-precision (no FP8 quant) scoring must match a from-scratch numpy
    triple loop exactly (up to float32 accumulation error)."""
    rng = np.random.default_rng(42)
    q = rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32)
    k = rng.normal(size=(num_kv, head_dim)).astype(np.float32)
    weights = rng.uniform(0.1, 1.0,
                          size=(num_tokens, num_heads)).astype(np.float32)
    # A single growing causal sequence: token t may attend to kv <= t.
    q_positions = jnp.arange(num_tokens, dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    softmax_scale = head_dim**-0.5
    n_head_scale = num_heads**-0.5

    scores = lightning_indexer_scores(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(weights),
        q_positions,
        k_positions,
        rope_dim=rope_dim,
        softmax_scale=softmax_scale,
        n_head_scale=n_head_scale,
        use_fp8_quant=False,
    )
    ref = _naive_lightning_indexer_scores(
        q,
        k,
        weights,
        np.asarray(q_positions),
        np.asarray(k_positions),
        rope_dim=rope_dim,
        softmax_scale=softmax_scale,
        n_head_scale=n_head_scale,
    )

    np.testing.assert_allclose(np.asarray(scores), ref, atol=1e-3, rtol=1e-3)


def test_lightning_indexer_scores_causal_mask():
    num_tokens, num_kv, num_heads, head_dim = 4, 6, 2, 128
    rng = np.random.default_rng(7)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, head_dim)).astype(np.float32))
    weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(num_tokens, num_heads)).astype(np.float32))
    # Decode-style: a single query token at absolute position 3, kv cache has
    # 6 raw tokens at positions 0..5 -- only positions 0..3 are valid.
    q_positions = jnp.array([3], dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    scores = lightning_indexer_scores(q[:1],
                                      k,
                                      weights[:1],
                                      q_positions,
                                      k_positions,
                                      use_fp8_quant=False)
    scores_np = np.asarray(scores)[0]
    assert np.all(np.isfinite(scores_np[:4]))
    assert np.all(scores_np[4:] == -np.inf)


@pytest.mark.parametrize("num_heads,head_dim", [(32, 128)])
def test_lightning_indexer_scores_fp8_quant_within_tolerance(
        num_heads, head_dim):
    """FP8 (ue8m0) quantized scoring should be within the same
    fp8-quantization tolerance band vLLM's own sparse-MLA tests use
    (rtol=0.065, atol=0.05) relative to the full-precision score."""
    num_tokens, num_kv = 6, 16
    rng = np.random.default_rng(123)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, head_dim)).astype(np.float32))
    weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(num_tokens, num_heads)).astype(np.float32))
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    fp32_scores = lightning_indexer_scores(q,
                                           k,
                                           weights,
                                           q_positions,
                                           k_positions,
                                           use_fp8_quant=False)
    fp8_scores = lightning_indexer_scores(q,
                                          k,
                                          weights,
                                          q_positions,
                                          k_positions,
                                          use_fp8_quant=True)

    fp32_np = np.asarray(fp32_scores)
    fp8_np = np.asarray(fp8_scores)
    finite = np.isfinite(fp32_np)
    assert finite.any()
    np.testing.assert_allclose(fp8_np[finite],
                               fp32_np[finite],
                               rtol=0.065,
                               atol=0.05)
    # Masked entries must stay masked identically regardless of quantization.
    np.testing.assert_array_equal(np.isfinite(fp8_np), finite)


# =====================================================================
# Top-k selection.
# =====================================================================
def test_select_topk_indices_basic():
    scores = jnp.array([
        [0.5, -jnp.inf, 3.0, 1.0],
        [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf],
    ])
    idxs = select_topk_indices(scores, topk=2)
    idxs_np = np.asarray(idxs)
    np.testing.assert_array_equal(idxs_np[0], [2, 3])
    np.testing.assert_array_equal(idxs_np[1], [-1, -1])


def test_select_topk_indices_pads_when_fewer_kv_than_topk():
    scores = jnp.array([[1.0, 2.0, -jnp.inf]])
    idxs = np.asarray(select_topk_indices(scores, topk=5))
    assert idxs.shape == (1, 5)
    assert list(idxs[0][:2]) == [1, 0]
    assert all(v == -1 for v in idxs[0][2:])


def test_lightning_indexer_end_to_end_topk_matches_naive_numpy():
    num_tokens, num_kv, num_heads, head_dim, rope_dim = 5, 12, 4, 128, 64
    topk = 4
    rng = np.random.default_rng(99)
    q = rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32)
    k = rng.normal(size=(num_kv, head_dim)).astype(np.float32)
    weights = rng.uniform(0.1, 1.0,
                          size=(num_tokens, num_heads)).astype(np.float32)
    q_positions = np.arange(num_tokens) + (num_kv - num_tokens)
    k_positions = np.arange(num_kv)

    softmax_scale = head_dim**-0.5
    n_head_scale = num_heads**-0.5

    scores = lightning_indexer_scores(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(weights),
        jnp.asarray(q_positions.astype(np.int32)),
        jnp.asarray(k_positions.astype(np.int32)),
        rope_dim=rope_dim,
        softmax_scale=softmax_scale,
        n_head_scale=n_head_scale,
        use_fp8_quant=False)
    got = np.asarray(select_topk_indices(scores, topk))

    ref_scores = _naive_lightning_indexer_scores(q,
                                                 k,
                                                 weights,
                                                 q_positions,
                                                 k_positions,
                                                 rope_dim=rope_dim,
                                                 softmax_scale=softmax_scale,
                                                 n_head_scale=n_head_scale)
    for t in range(num_tokens):
        order = np.argsort(-ref_scores[t])
        num_valid = int(np.isfinite(ref_scores[t]).sum())
        expected = list(order[:min(topk, num_valid)])
        got_row = [i for i in got[t] if i != -1]
        assert got_row == expected, f"token {t}: {got_row} vs {expected}"


# =====================================================================
# Shared-indexer layer-skip pattern.
# =====================================================================
def test_should_recompute_indexer_layer_matches_glm5_formula():
    # Cross-check against the literal vLLM formula (deepseek_v2.py):
    #   skip = (max(layer_id - offset + 1, 0) % freq) != 0
    for layer_id in range(20):
        expected_recompute = (max(layer_id - INDEX_SKIP_TOPK_OFFSET + 1, 0) %
                              INDEX_TOPK_FREQ == 0)
        assert should_recompute_indexer_layer(layer_id) == expected_recompute

    # GLM-5.2's real config (index_topk_freq=4, index_skip_topk_offset=3):
    # the first three layers all recompute (clamped), then every 4th layer
    # thereafter (6, 10, 14, ...).
    recompute_layers = {
        i
        for i in range(20) if should_recompute_indexer_layer(i)
    }
    assert recompute_layers == {0, 1, 2, 6, 10, 14, 18}


def test_should_recompute_indexer_layer_freq_one_always_recomputes():
    for layer_id in range(10):
        assert should_recompute_indexer_layer(layer_id,
                                              index_topk_freq=1,
                                              index_skip_topk_offset=2)


def test_lightning_indexer_layer_skip_reuses_previous_indices_verbatim():
    num_tokens, num_kv, num_heads, head_dim = 3, 8, 2, 128
    rng = np.random.default_rng(5)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, head_dim)).astype(np.float32))
    weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(num_tokens, num_heads)).astype(np.float32))
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)
    previous = jnp.array([[1, 0, -1], [2, 1, 0], [0, -1, -1]], dtype=jnp.int32)

    # layer_id=3 is a skip layer under GLM-5.2's real config.
    assert not should_recompute_indexer_layer(3)
    out = lightning_indexer_layer(3,
                                  q,
                                  k,
                                  weights,
                                  q_positions,
                                  k_positions,
                                  previous,
                                  topk=3,
                                  use_fp8_quant=False)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(previous))

    # layer_id=6 recomputes and should NOT just echo `previous`.
    assert should_recompute_indexer_layer(6)
    recomputed = lightning_indexer_layer(6,
                                         q,
                                         k,
                                         weights,
                                         q_positions,
                                         k_positions,
                                         previous,
                                         topk=3,
                                         use_fp8_quant=False)
    direct_scores = lightning_indexer_scores(q,
                                             k,
                                             weights,
                                             q_positions,
                                             k_positions,
                                             use_fp8_quant=False)
    direct = select_topk_indices(direct_scores, 3)
    np.testing.assert_array_equal(np.asarray(recomputed), np.asarray(direct))
