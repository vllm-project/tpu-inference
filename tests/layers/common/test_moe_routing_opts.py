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
"""Tests for MoE routing optimizations on CPU and TPU guards.

Verifies:
1. `_single_sort_permutation` matches the standard 3-sort reference.
2. `rowmax_topk` matches `jax.lax.top_k` on CPU across shapes and edge cases.
3. `router_topk` delegates to `rowmax_topk` on CPU platform.
4. `select(..., interpret=True)` matches `rowmax_topk` on CPU.
5. Guards for TPU-only Pallas kernels (dense_expert_moe, moe_lowtok).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert, dense_expert_moe)
from tpu_inference.kernels.experimental.moe_lowtok import (can_use_moe_lowtok,
                                                           moe_lowtok)
from tpu_inference.kernels.experimental.moe_lowtok import \
    routing_table as moe_lowtok_routing_table
from tpu_inference.kernels.router_topk import router_topk, rowmax_topk, select
from tpu_inference.layers.common.fused_moe_gmm import _single_sort_permutation

EXPERTS = 64
TOPK = 4


def _three_sort_reference(sort_keys, topk):
    token_bits = max(1, (sort_keys.shape[0] // topk - 1).bit_length())
    token_ids = jnp.arange(sort_keys.shape[0], dtype=jnp.int32) // topk
    sorted_indices = jnp.argsort(sort_keys)
    packed = jax.lax.sort(jnp.left_shift(sort_keys, token_bits) | token_ids,
                          is_stable=False)
    token_indices = jnp.bitwise_and(packed, (1 << token_bits) - 1)
    return sorted_indices, jnp.argsort(sorted_indices), token_indices


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 204])
def test_single_sort_permutation_matches_argsort(num_tokens):
    rng = np.random.default_rng(num_tokens)
    ids = np.stack(
        [rng.choice(EXPERTS, TOPK, replace=False) for _ in range(num_tokens)])
    sort_keys = jnp.asarray(ids.reshape(-1), dtype=jnp.int32)
    flat_bits = max(1, (sort_keys.shape[0] - 1).bit_length())

    actual = _single_sort_permutation(sort_keys, TOPK, flat_bits)
    expected = _three_sort_reference(sort_keys, TOPK)

    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_permutation_topk_one():
    num_tokens = 16
    keys = jnp.arange(num_tokens, dtype=jnp.int32) % EXPERTS
    flat_bits = max(1, (num_tokens - 1).bit_length())
    actual = _single_sort_permutation(keys, 1, flat_bits)
    sorted_indices = jnp.argsort(keys)
    expected = (sorted_indices, jnp.argsort(sorted_indices), sorted_indices)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_permutation_all_equal_keys():
    n = 32
    keys = jnp.zeros(n, dtype=jnp.int32)
    flat_bits = max(1, (n - 1).bit_length())
    actual = _single_sort_permutation(keys, TOPK, flat_bits)
    expected = _three_sort_reference(keys, TOPK)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("rows", [1, 8, 37, 64, 128])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_rowmax_topk_matches_lax_topk_uniform(rows, k):
    rng = np.random.default_rng(rows * 100 + k)
    scores_np = rng.random((rows, EXPERTS)).astype(np.float32)
    scores = jnp.asarray(scores_np)

    rowmax_vals, rowmax_ids = rowmax_topk(scores, k)
    lax_vals, lax_ids = jax.lax.top_k(scores, k)

    assert rowmax_vals.shape == (rows, k)
    assert rowmax_ids.shape == (rows, k)
    assert rowmax_ids.dtype == jnp.int32

    np.testing.assert_allclose(np.asarray(rowmax_vals),
                               np.asarray(lax_vals),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(rowmax_ids), np.asarray(lax_ids))


def test_rowmax_topk_softmax_distribution():
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((128, EXPERTS)).astype(np.float32) * 2.0
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)

    rowmax_vals, rowmax_ids = rowmax_topk(scores, TOPK)
    lax_vals, lax_ids = jax.lax.top_k(scores, TOPK)

    np.testing.assert_allclose(np.asarray(rowmax_vals),
                               np.asarray(lax_vals),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(rowmax_ids), np.asarray(lax_ids))


def test_router_topk_cpu_fallback():
    scores = jax.random.normal(jax.random.PRNGKey(123), (16, EXPERTS))
    rv, ri = router_topk(scores, TOPK)
    lv, li = jax.lax.top_k(scores, TOPK)

    np.testing.assert_allclose(np.asarray(rv),
                               np.asarray(lv),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(ri), np.asarray(li))


def test_select_interpret_matches_rowmax_topk():
    rng = np.random.default_rng(7)
    scores_np = rng.random((32, EXPERTS), dtype=np.float32)
    scores = jnp.asarray(scores_np)

    kw, ki = select(scores, TOPK, interpret=True)
    rw, ri = rowmax_topk(scores, TOPK)

    np.testing.assert_allclose(np.asarray(kw),
                               np.asarray(rw),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(ki), np.asarray(ri))


def test_rowmax_topk_nan_row():
    scores = jnp.full((4, EXPERTS), jnp.nan, dtype=jnp.float32)
    kw, ki = rowmax_topk(scores, TOPK)

    assert jnp.isnan(kw).all()
    # Distinct experts should be selected for each row
    for row in np.asarray(ki):
        assert len(set(row.tolist())) == TOPK


def test_can_use_dense_expert_validation():
    # Valid shape check on CPU without running Pallas kernel
    m, h, i, e = 16, 512, 512, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)

    assert can_use_dense_expert(x,
                                w1,
                                w2,
                                s1,
                                s2,
                                None,
                                None,
                                activation="silu",
                                rhs_quant_dtype=None)
    # Unsupported activation
    assert not can_use_dense_expert(
        x, w1, w2, s1, s2, None, None, activation="gelu", rhs_quant_dtype=None)
    # Unsupported hidden dimension
    assert not can_use_dense_expert(x[:, :256],
                                    w1,
                                    w2,
                                    s1,
                                    s2,
                                    None,
                                    None,
                                    activation="silu",
                                    rhs_quant_dtype=None)


def test_can_use_moe_lowtok_validation():
    # Valid shape check on CPU without running Pallas kernel
    m, h, i, e = 16, 512, 256, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)

    assert can_use_moe_lowtok(x,
                              w1,
                              w2,
                              s1,
                              s2,
                              None,
                              None,
                              activation="silu",
                              rhs_quant_dtype=None)
    # Unsupported activation
    assert not can_use_moe_lowtok(
        x, w1, w2, s1, s2, None, None, activation="gelu", rhs_quant_dtype=None)
    # Bias not allowed
    assert not can_use_moe_lowtok(x,
                                  w1,
                                  w2,
                                  s1,
                                  s2,
                                  jnp.ones(1),
                                  None,
                                  activation="silu",
                                  rhs_quant_dtype=None)


@pytest.mark.skipif(jax.devices()[0].platform != "tpu",
                    reason="requires TPU hardware")
def test_dense_expert_moe_tpu_only():
    m, h, i, e = 16, 512, 512, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)
    weights = jnp.ones((m, TOPK), jnp.float32)
    ids = jnp.zeros((m, TOPK), jnp.int32)
    out = dense_expert_moe(x,
                           w1,
                           w2,
                           s1,
                           s2,
                           weights,
                           ids,
                           num_primary_experts=e)
    assert out.shape == (m, h)


@pytest.mark.skipif(jax.devices()[0].platform != "tpu",
                    reason="requires TPU hardware")
def test_moe_lowtok_tpu_only():
    m, h, i, e = 16, 512, 256, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)
    weights = jnp.ones((m, TOPK), jnp.bfloat16)
    ids = jnp.zeros((m, TOPK), jnp.int32)
    route = moe_lowtok_routing_table(ids, weights, e)
    out = moe_lowtok(x, route, w1, s1, w2, s2)
    assert out.shape == (m, h)
