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
"""Tests for the MoE routing optimizations ported from vllm-torchtpu.

CPU tests (run anywhere):
1. `_single_sort_permutation` matches the standard 3-sort reference.
2. `rowmax_topk` matches `jax.lax.top_k` across shapes and edge cases.
3. `router_topk` (pure-JAX `rowmax_topk`) matches `jax.lax.top_k`, bit for
   bit under jit on bf16 and f32 softmax scores with ties.
4. `select(..., interpret=True)` matches `rowmax_topk` on CPU.
5. The kernels' `can_use_*` eligibility checks.

TPU tests:
6. `moe_lowtok` matches a jnp oracle of its own arithmetic to within one bf16
   ulp of the output: an f32 transcription with the MXU fp8 dot; stays within
   FP8 quantization noise of exact f32 math. Random inputs with padding rows
   and ids of -1.
7. On 8 devices, `fused_moe_func` gives the same output with each of
   TPU_MOE_SINGLE_SORT, TPU_MOE_ROUTER_TOPK=rowmax, and TPU_MOE_LOWTOK on as
   with it off, with scatter_results on and off, and with defer_all_reduce
   the EP kernel returns the shard's unreduced partial sum, as the GMM path does.
"""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _make_mesh, _mesh_id, _sharded_hidden_states, requires_8_devices)
from tpu_inference.kernels.experimental.moe_lowtok import (can_use_moe_lowtok,
                                                           moe_lowtok)
from tpu_inference.kernels.experimental.moe_lowtok import \
    routing_table as moe_lowtok_routing_table
from tpu_inference.kernels.router_topk import router_topk, rowmax_topk, select
from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.fused_moe_gmm import _single_sort_permutation
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.utils import get_mesh_shape_product

EXPERTS = 64
TOPK = 4

requires_tpu = pytest.mark.skipif(jax.devices()[0].platform != "tpu",
                                  reason="requires TPU hardware")


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


def test_router_topk_matches_lax_topk():
    scores = jax.random.normal(jax.random.PRNGKey(123), (16, EXPERTS))
    rv, ri = router_topk(scores, TOPK)
    lv, li = jax.lax.top_k(scores, TOPK)

    np.testing.assert_allclose(np.asarray(rv),
                               np.asarray(lv),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_array_equal(np.asarray(ri), np.asarray(li))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32],
                         ids=["bf16", "f32"])
def test_router_topk_bit_identical_to_lax_topk_under_jit(dtype):
    """On the same softmax scores, jitted, router_topk returns the same
    weights and ids bit for bit as jax.lax.top_k, including rows with ties
    at the top-k boundary (the lowest id wins in both). bf16 softmax over
    16 experts ties often; the ties are also forced explicitly."""
    logits = jax.random.normal(jax.random.key(0), (256, 16), jnp.float32)
    logits = logits.at[::8, :4].set(1.5)  # four-way tie at the top
    logits = logits.at[1::8, 2:4].set(logits[1::8, :1])  # tie across the cut
    scores = jax.block_until_ready(jax.nn.softmax(logits.astype(dtype), -1))

    rv, ri = jax.jit(functools.partial(router_topk, k=2))(scores)
    lv, li = jax.jit(functools.partial(jax.lax.top_k, k=2))(scores)
    assert rv.dtype == lv.dtype == dtype
    np.testing.assert_array_equal(np.asarray(ri), np.asarray(li))
    np.testing.assert_array_equal(np.asarray(rv, np.float32),
                                  np.asarray(lv, np.float32))


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


# ---------------------------------------------------------------------------
# Experimental EP decode kernel: moe_lowtok.
# ---------------------------------------------------------------------------

# Both kernels need hidden and intermediate widths that are multiples of 512.
_H = 512
_I = 512


def _fp8_experts(key, num_experts, hidden=_H, intermediate=_I):
    """Random FP8 expert weights with random per-channel f32 scales, in the
    [E, h, 2i] / [E, i, h] layout and [E, 1, 1, N] scales the kernels take."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = jax.random.normal(k1, (num_experts, hidden, 2 * intermediate),
                           jnp.float32)
    w2 = jax.random.normal(k2, (num_experts, intermediate, hidden),
                           jnp.float32)
    s1 = jax.random.uniform(k3, (num_experts, 1, 1, 2 * intermediate),
                            jnp.float32, 0.5, 1.5) / 100
    s2 = jax.random.uniform(k4, (num_experts, 1, 1, hidden), jnp.float32, 0.5,
                            1.5) / 100
    return (w1.astype(jnp.float8_e4m3fn), w2.astype(jnp.float8_e4m3fn), s1, s2)


def _routing(key, num_tokens, num_experts, topk, num_valid):
    """Random top-k ids over `num_experts` with random weights. Tokens from
    `num_valid` on are padding: weight 0. One of the first token's ids is -1,
    the value _ep_kernel_local gives an expert held by another shard."""
    k1, k2 = jax.random.split(key)
    ids = jax.vmap(
        lambda k: jax.random.choice(k, num_experts, (topk, ), replace=False))(
            jax.random.split(k1, num_tokens)).astype(jnp.int32)
    weights = jax.nn.softmax(jax.random.normal(k2, (num_tokens, topk)), -1)
    valid = (jnp.arange(num_tokens) < num_valid)[:, None]
    weights = jnp.where(valid, weights, 0.0)
    ids = ids.at[0, 0].set(-1)
    weights = weights.at[0, 0].set(0.0)
    return ids, weights.astype(jnp.float32)


def _fp8_round_trip(x):
    """Dynamic per-row FP8 quantization of 512-wide blocks and back, the
    activation quantization of gmm_v2 and of both kernels."""
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // 512, 512)
    scale = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / 448.0
    inv = jnp.where(scale == 0, 0, 1 / scale)
    q = (blocks * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    return (q * scale).reshape(rows, cols)


def _reference_moe_f32(x, w1, w2, s1, s2, weights, ids):
    """The routed MoE in exact f32 arithmetic (HIGHEST precision: TPU's
    default f32 matmul is a single bf16 pass): FP8 weights dequantized,
    activations put through the FP8 round trip, ids of -1 skipped. Both
    kernels differ from this by FP8/bf16 quantization noise, so it only
    bounds the mean error; the per-kernel oracles below are tight."""
    num_experts, _, two_i = w1.shape
    inter = two_i // 2
    hi = jax.lax.Precision.HIGHEST
    w1 = w1.astype(jnp.float32) * s1.reshape(num_experts, 1, two_i)
    w2 = w2.astype(jnp.float32) * s2.reshape(num_experts, 1, -1)
    xq = _fp8_round_trip(x.astype(jnp.float32))
    gate_up = jnp.einsum("th,ehn->etn", xq, w1, precision=hi)
    act = jax.nn.silu(gate_up[..., :inter]) * gate_up[..., inter:]
    act = jax.vmap(_fp8_round_trip)(act)
    per_expert = jnp.einsum("eti,eih->eth", act, w2, precision=hi)
    coeff = moe_lowtok_routing_table(ids, weights, num_experts)  # [t, e]
    return jnp.einsum("te,eth->th", coeff, per_expert, precision=hi)


def _fp8_dot(a, b):
    """The MXU's fp8 x fp8 -> f32 dot as both kernels issue it. It is not an
    exact f32 accumulation (it differs from exact math by ~1e-3 of the row
    scale), and XLA's lowering is bit-identical to Mosaic's, so a reference
    has to use it rather than an f32 matmul of the dequantized values."""
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _lowtok_quantize_rows(v):
    """moe_lowtok's _quantize_rows: per-row FP8 of an f32 block."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / 448.0
    return (v / scale).astype(jnp.float8_e4m3fn), scale


def _reference_lowtok(x, w1, w2, s1, s2, weights, ids):
    """moe_lowtok's arithmetic in plain jnp: f32 end to end with the MXU fp8
    dots, per-row activation quantization, scales applied after each dot,
    the routed sum accumulated in f32 and rounded to bf16 once."""
    num_experts, _, two_i = w1.shape
    inter = two_i // 2
    xq, xs = _lowtok_quantize_rows(x.astype(jnp.float32))
    route = moe_lowtok_routing_table(ids, weights, num_experts)  # [t, e]
    acc = jnp.zeros(x.shape, jnp.float32)
    for e in range(num_experts):
        h = _fp8_dot(xq, w1[e])
        gate = h[:, :inter] * xs * s1[e, 0, 0, :inter]
        up = h[:, inter:] * xs * s1[e, 0, 0, inter:]
        aq, a_s = _lowtok_quantize_rows(gate * jax.nn.sigmoid(gate) * up)
        o = _fp8_dot(aq, w2[e])
        acc = acc + (o * s2[e, 0, 0]) * (a_s * route[:, e:e + 1])
    return acc.astype(jnp.bfloat16)


# moe_lowtok's bf16 output against the oracle of the same arithmetic: the
# only rounding not pinned down is the f32 summation order of the routed
# combine, which can move the final bf16 rounding by one ulp, i.e. by at
# most 2**-7 of the row's scale.
_ONE_BF16_ULP = 2.0**-7
# Against exact f32 math, moe_lowtok (and gmm_v2) carry the FP8 activation
# quantization and the MXU fp8 dot's accumulation error through two matmul
# stages: ~1e-2 of the row scale on average, with single elements an FP8
# ulp (6%) off where a re-quantization rounds the other way.
_QUANT_NOISE_MEAN = 2e-2


def _row_normalized_error(out, expected):
    out = np.asarray(out, np.float32)
    expected = np.asarray(expected, np.float32)
    assert out.shape == expected.shape
    assert np.isfinite(out).all()
    scale = np.maximum(np.abs(expected).max(axis=1, keepdims=True), 1e-3)
    return np.abs(out / scale - expected / scale)


def test_can_use_moe_lowtok_validation():
    # Valid shape check on CPU without running Pallas kernel
    m, h, i, e = 16, 512, 256, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)
    ok = dict(activation="silu", rhs_quant_dtype=None)

    assert can_use_moe_lowtok(x, w1, w2, s1, s2, None, None, **ok)
    assert can_use_moe_lowtok(x,
                              w1,
                              w2,
                              s1,
                              s2,
                              None,
                              None,
                              num_experts=2,
                              **ok)
    # More experts per shard than the kernel's tables hold.
    assert not can_use_moe_lowtok(
        x, w1, w2, s1, s2, None, None, num_experts=129, **ok)
    # Unsupported activation
    assert not can_use_moe_lowtok(
        x, w1, w2, s1, s2, None, None, activation="gelu", rhs_quant_dtype=None)
    # Bias not allowed
    assert not can_use_moe_lowtok(x, w1, w2, s1, s2, jnp.ones(1), None, **ok)
    assert not can_use_moe_lowtok(x, w1, w2, jnp.ones(
        (e, 2, 1, 2 * i)), s2, None, None, **ok)


@requires_tpu
@pytest.mark.parametrize("num_tokens, num_valid", [(8, 8), (16, 11),
                                                   (128, 100)])
def test_ep_kernel_matches_reference(num_tokens, num_valid):
    """moe_lowtok matches a jnp oracle of its own arithmetic to within one
    bf16 ulp of the output on random weights, scales and routing: several
    experts hit per shard, padding rows with weight 0 and an id of -1; 128
    tokens is the default TPU_MOE_LOWTOK_MAX_TOKENS.

    moe_lowtok is f32 end to end (per-row FP8 activations, scales applied
    after the fp8 dot). It is also held to exact f32 math at the FP8
    quantization-noise level."""
    num_experts = 8
    k1, k2, k3 = jax.random.split(jax.random.key(num_tokens), 3)
    x = jax.random.normal(k1, (num_tokens, _H), jnp.bfloat16)
    w1, w2, s1, s2 = _fp8_experts(k2, num_experts)
    ids, weights = _routing(k3, num_tokens, num_experts, TOPK, num_valid)

    route = moe_lowtok_routing_table(ids, weights, num_experts)
    out = moe_lowtok(x, route, w1, s1, w2, s2)
    oracle = _reference_lowtok
    out = jax.block_until_ready(out)
    assert out.shape == (num_tokens, _H)
    assert out.dtype == jnp.bfloat16

    expected = oracle(x, w1, w2, s1, s2, weights, ids)
    error = _row_normalized_error(out, expected)
    assert error.max() <= _ONE_BF16_ULP, (
        f"lowtok vs its oracle: max {error.max():.4f}, "
        f"mean {error.mean():.5f} of the row scale")

    f32 = _reference_moe_f32(x, w1, w2, s1, s2, weights, ids)
    assert _row_normalized_error(out, f32).mean() <= _QUANT_NOISE_MEAN
    assert _row_normalized_error(expected, f32).mean() <= _QUANT_NOISE_MEAN
    # Padding rows (weight 0) and the -1 id contribute nothing.
    np.testing.assert_array_equal(np.asarray(out[num_valid:], np.float32), 0)
    assert np.abs(np.asarray(f32[0], np.float32)).max() > 0


# ---------------------------------------------------------------------------
# fused_moe_func with each flag on, against the same call with it off.
# ---------------------------------------------------------------------------

# EP meshes: on each, EXPERT covers every device, so each of the 16 experts
# lives on one shard (2 per shard). attn_dp is also an ATTN_DATA axis, so
# scatter_results reduces over model and scatters the tokens over attn_dp.
_EP_MESH_SIZES = [
    dict(attn_dp=2, model=4),
    dict(attn_dp=4, model=2),
]
_NUM_EXPERTS = 16
_TOPK = 2


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...), as the other MoE tests do."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


def _fused_moe_inputs(mesh: Mesh,
                      num_tokens: int,
                      fp8: bool,
                      gating_dtype=jnp.bfloat16):
    """Tokens sharded over ATTN_DATA, replicated expert weights and gating.
    fp8=True gives the FP8 weights and per-channel scales the EP kernels
    take; fp8=False the bf16 weights the other flags are tested with."""
    k1, k2 = jax.random.split(jax.random.key(num_tokens))
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    gating = jax.random.normal(k2, (num_tokens, _NUM_EXPERTS), jnp.float32)
    inputs = dict(
        hidden_states=_sharded_hidden_states(mesh, num_tokens, _H),
        gating_output=jax.device_put(gating.astype(gating_dtype), rows),
        w1_bias=None,
        w2_bias=None,
    )
    if fp8:
        w1, w2, s1, s2 = _fp8_experts(k1, _NUM_EXPERTS)
        inputs.update(w1=w1, w2=w2, w1_scale=s1, w2_scale=s2)
    else:
        k3, k4 = jax.random.split(k1)
        inputs.update(
            w1=jax.random.normal(k3,
                                 (_NUM_EXPERTS, _H, 2 * _I), jnp.bfloat16) /
            10,
            w2=jax.random.normal(k4,
                                 (_NUM_EXPERTS, _I, _H), jnp.bfloat16) / 10,
            w1_scale=None,
            w2_scale=None,
        )
    return inputs


def _run_fused_moe(mesh: Mesh,
                   env: dict,
                   num_tokens: int,
                   fp8: bool,
                   gating_dtype=jnp.bfloat16,
                   **kwargs) -> jax.Array:
    """Runs fused_moe_func with the env flags in `env` patched in. They are
    read at trace time, so JAX caches are cleared around the run."""
    jax.clear_caches()
    patches = [
        mock.patch(f"tpu_inference.envs.{name}", value)
        for name, value in env.items()
    ]
    try:
        with jax.set_mesh(mesh):
            for patch in patches:
                patch.start()
            out = fused_moe_gmm.fused_moe_func(
                **_fused_moe_inputs(mesh, num_tokens, fp8, gating_dtype),
                topk=_TOPK,
                renormalize=True,
                mesh=mesh,
                use_ep=True,
                activation="silu",
                scoring_fn="softmax",
                **kwargs,
            )
            return jax.block_until_ready(out)
    finally:
        for patch in patches:
            patch.stop()
        jax.clear_caches()


def _assert_fused_moe_close(actual, expected, atol=2e-2):
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               np.asarray(expected, np.float32),
                               rtol=2e-2,
                               atol=atol)


_ROUTING_FLAGS = {
    "single-sort": (dict(TPU_MOE_SINGLE_SORT=True), jnp.bfloat16),
    # f32 gating: with bf16 gating the softmax feeding the top-k is not
    # bit-stable itself. XLA evaluates a bf16 softmax at f32 excess precision
    # when it fuses into rowmax's reductions, but rounds its bf16 steps when
    # it has to materialize it for lax.top_k (the default path even differs
    # from its own eager result that way), so the routing weights can differ
    # by one bf16 ulp while the selected experts are the same.
    # test_router_topk_bit_identical_to_lax_topk_under_jit covers bf16 on
    # identical scores.
    "rowmax": (dict(TPU_MOE_ROUTER_TOPK="rowmax"), jnp.float32),
}


@requires_8_devices
@pytest.mark.parametrize("scatter_results", [False, True],
                         ids=["all-reduce", "reduce-scatter"])
@pytest.mark.parametrize("flag",
                         list(_ROUTING_FLAGS),
                         ids=list(_ROUTING_FLAGS))
@pytest.mark.parametrize("sizes", _EP_MESH_SIZES, ids=_mesh_id)
def test_fused_moe_routing_flag_matches_default(sizes, flag, scatter_results):
    """The packed single sort and the rowmax router top-k only change how the
    permutation and the top-k are computed, so the output is bit-identical to
    the default sorts, for 256 tokens on the GMM path."""
    mesh = _make_mesh(**sizes)
    env, gating_dtype = _ROUTING_FLAGS[flag]
    kwargs = dict(scatter_results=scatter_results,
                  num_tokens=256,
                  fp8=False,
                  gating_dtype=gating_dtype)
    expected = _run_fused_moe(mesh, {}, **kwargs)
    actual = _run_fused_moe(mesh, env, **kwargs)
    np.testing.assert_array_equal(np.asarray(actual, np.float32),
                                  np.asarray(expected, np.float32))


_EP_KERNEL_FLAGS = {
    "lowtok": dict(TPU_MOE_LOWTOK=True),
}


def _spy_ep_kernel():
    return mock.patch.object(fused_moe_gmm,
                             "_ep_dispatch_kernel",
                             wraps=fused_moe_gmm._ep_dispatch_kernel)


@requires_8_devices
@pytest.mark.parametrize("scatter_results", [False, True],
                         ids=["all-reduce", "reduce-scatter"])
@pytest.mark.parametrize("num_tokens", [16, 128])
@pytest.mark.parametrize("flag",
                         list(_EP_KERNEL_FLAGS),
                         ids=list(_EP_KERNEL_FLAGS))
@pytest.mark.parametrize("sizes", _EP_MESH_SIZES, ids=_mesh_id)
def test_fused_moe_ep_kernel_matches_gmm(sizes, flag, num_tokens,
                                         scatter_results):
    """With TPU_MOE_LOWTOK the EP kernel replaces permute + GMM + combine and
    gives the same output, all-reduced or reduce-scattered over the shards. The
    kernel and gmm_v2 quantize the activations to FP8 the same way, so they
    differ only in rounding."""
    mesh = _make_mesh(**sizes)
    kwargs = dict(scatter_results=scatter_results,
                  num_tokens=num_tokens,
                  fp8=True)
    with _spy_ep_kernel() as spy:
        expected = _run_fused_moe(mesh, {}, **kwargs)
        assert spy.call_count == 0
        actual = _run_fused_moe(mesh, _EP_KERNEL_FLAGS[flag], **kwargs)
        assert spy.call_count == 1
    assert actual.sharding.spec == expected.sharding.spec
    _assert_fused_moe_close(actual, expected)


@requires_8_devices
@pytest.mark.parametrize("flag",
                         list(_EP_KERNEL_FLAGS),
                         ids=list(_EP_KERNEL_FLAGS))
def test_fused_moe_ep_kernel_defer_all_reduce(flag):
    """With defer_all_reduce the EP kernel path returns each shard's partial
    sum unreduced, as the GMM path does, so that the caller's one all-reduce
    over shared + routed output (VllmMoERunner._fused_output_is_reduced)
    gives the reduced result rather than EP-size times it."""
    mesh = _make_mesh(attn_dp=2, model=4)
    kwargs = dict(num_tokens=16, fp8=True)
    reduced = _run_fused_moe(mesh, _EP_KERNEL_FLAGS[flag], **kwargs)
    expected = _run_fused_moe(mesh, {}, defer_all_reduce=True, **kwargs)
    with _spy_ep_kernel() as spy:
        actual = _run_fused_moe(mesh,
                                _EP_KERNEL_FLAGS[flag],
                                defer_all_reduce=True,
                                **kwargs)
        assert spy.call_count == 1

    # Per device: the same partial sum as the GMM path's.
    actual_shards = {s.device: s.data for s in actual.addressable_shards}
    expected_shards = {s.device: s.data for s in expected.addressable_shards}
    assert actual_shards.keys() == expected_shards.keys()
    for device, data in expected_shards.items():
        _assert_fused_moe_close(actual_shards[device], data)

    # Summed over the EP shards, the partials give the reduced output once.
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
    assert ep_size == len(actual_shards)
    total = sum(np.asarray(d, np.float32) for d in actual_shards.values())
    _assert_fused_moe_close(total, reduced, atol=ep_size * 2e-2)
