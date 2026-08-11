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
"""Tests for the GLM-5.2 sparse (top-k gather + dense attend) main-attention
pure-JAX reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.glm5.core_attention.reference import (
    dense_causal_attention_reference, sparse_causal_attention)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _naive_sparse_attention_per_token(q, k, v, topk_indices, q_positions,
                                      k_positions, softmax_scale):
    """Independent per-token numpy reference, mirroring the manual-SDPA-loop
    style of vLLM's test_sparse_backend_decode_correctness (one query token
    at a time, explicit valid-index gather, plain softmax)."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    topk_indices = np.asarray(topk_indices)
    q_positions = np.asarray(q_positions)
    k_positions = np.asarray(k_positions)

    num_tokens, num_heads, _ = q.shape
    v_head_dim = v.shape[-1]
    out = np.zeros((num_tokens, num_heads, v_head_dim), dtype=np.float64)

    for t in range(num_tokens):
        idxs = [
            i for i in topk_indices[t]
            if i >= 0 and k_positions[i] <= q_positions[t]
        ]
        if not idxs:
            continue
        k_sel = k[idxs]  # [n, Dqk]
        v_sel = v[idxs]  # [n, Dv]
        for h in range(num_heads):
            logits = (q[t, h] @ k_sel.T) * softmax_scale  # [n]
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            out[t, h] = probs @ v_sel
    return out


@pytest.mark.parametrize(
    "num_tokens,num_kv,num_heads,qk_dim,v_dim,topk",
    [
        (4, 10, 2, 32, 16, 6),
        (1, 1, 4, 8, 8, 4),
        (6, 20, 8, 16, 16, 8),
    ],
)
def test_sparse_causal_attention_matches_naive_numpy(num_tokens, num_kv,
                                                     num_heads, qk_dim, v_dim,
                                                     topk):
    rng = np.random.default_rng(11)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, qk_dim)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(num_kv, v_dim)).astype(np.float32))
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    # Random top-k indices per token, some padded with -1, mirroring the
    # upstream test's "half real, half -1" construction.
    topk_indices = np.full((num_tokens, topk), -1, dtype=np.int32)
    for t in range(num_tokens):
        max_valid = int(np.asarray(q_positions)[t]) + 1  # causal kv count
        num_valid = min(topk // 2 + 1, max_valid)
        chosen = rng.choice(max_valid, size=num_valid, replace=False)
        topk_indices[t, :num_valid] = chosen
    topk_indices = jnp.asarray(topk_indices)

    softmax_scale = qk_dim**-0.5
    out = sparse_causal_attention(q,
                                  k,
                                  v,
                                  topk_indices,
                                  q_positions,
                                  k_positions,
                                  softmax_scale=softmax_scale)
    ref = _naive_sparse_attention_per_token(q, k, v, topk_indices, q_positions,
                                            k_positions, softmax_scale)

    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4, rtol=1e-4)


def test_sparse_causal_attention_reduces_to_dense_when_topk_covers_all():
    num_tokens, num_kv, num_heads, qk_dim, v_dim = 5, 5, 4, 32, 32
    rng = np.random.default_rng(22)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, qk_dim)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(num_kv, v_dim)).astype(np.float32))
    q_positions = jnp.arange(num_tokens, dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    # topk_indices covers every KV position for every query token; causal
    # masking (both in sparse_causal_attention's defensive re-check and in
    # dense_causal_attention_reference) must produce identical results.
    topk_indices = jnp.broadcast_to(jnp.arange(num_kv, dtype=jnp.int32),
                                    (num_tokens, num_kv))

    sparse_out = sparse_causal_attention(q, k, v, topk_indices, q_positions,
                                         k_positions)
    dense_out = dense_causal_attention_reference(q, k, v, q_positions,
                                                 k_positions)

    np.testing.assert_allclose(np.asarray(sparse_out),
                               np.asarray(dense_out),
                               atol=1e-5,
                               rtol=1e-5)


def test_sparse_causal_attention_bf16_within_production_tolerance():
    """bf16 end-to-end tolerance target from
    tests/v1/attention/test_sparse_mla_backends.py (rtol=0.01, atol=0.01)."""
    num_tokens, num_kv, num_heads, qk_dim, v_dim, topk = 8, 24, 4, 64, 64, 12
    rng = np.random.default_rng(33)
    q_f32 = rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32)
    k_f32 = rng.normal(size=(num_kv, qk_dim)).astype(np.float32)
    v_f32 = rng.normal(size=(num_kv, v_dim)).astype(np.float32)
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    topk_indices = np.full((num_tokens, topk), -1, dtype=np.int32)
    for t in range(num_tokens):
        max_valid = int(np.asarray(q_positions)[t]) + 1
        num_valid = min(topk, max_valid)
        topk_indices[t, :num_valid] = rng.choice(max_valid,
                                                 size=num_valid,
                                                 replace=False)
    topk_indices = jnp.asarray(topk_indices)

    q_bf16 = jnp.asarray(q_f32).astype(jnp.bfloat16)
    k_bf16 = jnp.asarray(k_f32).astype(jnp.bfloat16)
    v_bf16 = jnp.asarray(v_f32).astype(jnp.bfloat16)

    bf16_out = sparse_causal_attention(q_bf16, k_bf16, v_bf16, topk_indices,
                                       q_positions, k_positions)
    fp32_ref = sparse_causal_attention(jnp.asarray(q_f32), jnp.asarray(k_f32),
                                       jnp.asarray(v_f32), topk_indices,
                                       q_positions, k_positions)

    np.testing.assert_allclose(np.asarray(bf16_out, dtype=np.float32),
                               np.asarray(fp32_ref, dtype=np.float32),
                               rtol=0.01,
                               atol=0.01)


def test_sparse_causal_attention_all_padding_row_is_finite():
    """A query token with an entirely -1 topk row (degenerate/edge case, e.g.
    a padding token) must not produce NaNs."""
    num_kv, num_heads, qk_dim, v_dim, topk = 4, 2, 8, 8, 3
    q = jnp.ones((1, num_heads, qk_dim), dtype=jnp.float32)
    k = jnp.ones((num_kv, qk_dim), dtype=jnp.float32)
    v = jnp.ones((num_kv, v_dim), dtype=jnp.float32)
    topk_indices = jnp.full((1, topk), -1, dtype=jnp.int32)
    q_positions = jnp.array([3], dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    out = sparse_causal_attention(q, k, v, topk_indices, q_positions,
                                  k_positions)
    assert np.all(np.isfinite(np.asarray(out)))
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(out))


def test_sparse_causal_attention_ignores_noncausal_selected_index():
    """If a selected index is (incorrectly) in the future, the defensive
    causal re-check inside sparse_causal_attention must still exclude it."""
    num_kv, num_heads, qk_dim, v_dim = 5, 1, 8, 8
    rng = np.random.default_rng(44)
    q = jnp.asarray(rng.normal(size=(1, num_heads, qk_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, qk_dim)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(num_kv, v_dim)).astype(np.float32))
    q_positions = jnp.array([2], dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    # Index 4 is in the future relative to q_position=2; should be masked.
    topk_with_future = jnp.array([[0, 1, 4]], dtype=jnp.int32)
    topk_causal_only = jnp.array([[0, 1, -1]], dtype=jnp.int32)

    out_with_future = sparse_causal_attention(q, k, v, topk_with_future,
                                              q_positions, k_positions)
    out_causal_only = sparse_causal_attention(q, k, v, topk_causal_only,
                                              q_positions, k_positions)

    np.testing.assert_allclose(np.asarray(out_with_future),
                               np.asarray(out_causal_only),
                               atol=1e-6)
