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
3. `router_topk` delegates to `rowmax_topk` on CPU platform.
4. `select(..., interpret=True)` matches `rowmax_topk` on CPU.
5. The kernels' `can_use_*` eligibility checks.

TPU tests:
6. `dense_expert_moe` and `moe_lowtok` match a jnp reference of the same
   arithmetic (FP8 weights and activations) on random inputs, including
   padding rows and ids of -1.
7. On 8 devices, `fused_moe_func` gives the same output with each of
   TPU_MOE_SINGLE_SORT, TPU_MOE_ROUTER_TOPK=rowmax, TPU_MOE_LOWTOK and
   TPU_MOE_DENSE_EXPERT_THRESHOLD on as with it off, with scatter_results
   on and off, and with defer_all_reduce the EP kernels return the shard's
   unreduced partial sum, as the GMM path does.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _make_mesh, _mesh_id, _sharded_hidden_states, requires_8_devices)
from tpu_inference.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert, dense_expert_moe)
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


# ---------------------------------------------------------------------------
# The two experimental EP decode kernels: dense_expert_moe and moe_lowtok.
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


def _reference_moe(x, w1, w2, s1, s2, weights, ids):
    """The routed MoE in plain jnp: FP8 weights dequantized, activations put
    through the same FP8 round trip as the kernels, the gate|up projection
    rounded to bf16 where the kernels round, ids of -1 skipped."""
    num_experts, _, two_i = w1.shape
    inter = two_i // 2
    w1 = w1.astype(jnp.float32) * s1.reshape(num_experts, 1, two_i)
    w2 = w2.astype(jnp.float32) * s2.reshape(num_experts, 1, -1)
    xq = _fp8_round_trip(x.astype(jnp.float32))
    gate_up = jnp.einsum("th,ehn->etn", xq,
                         w1).astype(jnp.bfloat16).astype(jnp.float32)
    act = jax.nn.silu(gate_up[..., :inter]) * gate_up[..., inter:]
    act = act.astype(jnp.bfloat16).astype(jnp.float32)
    act = jax.vmap(_fp8_round_trip)(act)
    per_expert = jnp.einsum("eti,eih->eth", act,
                            w2).astype(jnp.bfloat16).astype(jnp.float32)
    coeff = moe_lowtok_routing_table(ids, weights, num_experts)  # [t, e]
    return jnp.einsum("te,eth->th", coeff, per_expert)


def _assert_close_to_reference(out, expected):
    """bf16 output against the f32 reference: a few bf16 ulps of the row's
    scale, the rounding-point differences the reference does not model."""
    out = np.asarray(out, np.float32)
    expected = np.asarray(expected, np.float32)
    assert out.shape == expected.shape
    assert np.isfinite(out).all()
    scale = np.maximum(np.abs(expected).max(axis=1, keepdims=True), 1e-3)
    np.testing.assert_allclose(out / scale, expected / scale, atol=3e-2)


def test_can_use_dense_expert_validation():
    # Valid shape check on CPU without running Pallas kernel
    m, h, i, e = 16, 512, 512, 4
    x = jnp.ones((m, h), jnp.bfloat16)
    w1 = jnp.ones((e, h, 2 * i), jnp.float8_e4m3fn)
    w2 = jnp.ones((e, i, h), jnp.float8_e4m3fn)
    s1 = jnp.ones((e, 1, 1, 2 * i), jnp.float32)
    s2 = jnp.ones((e, 1, 1, h), jnp.float32)
    ok = dict(activation="silu", rhs_quant_dtype=None)

    assert can_use_dense_expert(x, w1, w2, s1, s2, None, None, **ok)
    # An EP caller passes its per-shard expert count against global weights.
    assert can_use_dense_expert(x,
                                w1,
                                w2,
                                s1,
                                s2,
                                None,
                                None,
                                num_experts=2,
                                **ok)
    assert not can_use_dense_expert(
        x, w1, w2, s1, s2, None, None, num_experts=0, **ok)
    # Unsupported activation
    assert not can_use_dense_expert(
        x, w1, w2, s1, s2, None, None, activation="gelu", rhs_quant_dtype=None)
    # Unsupported hidden dimension
    assert not can_use_dense_expert(x[:, :256], w1, w2, s1, s2, None, None, **
                                    ok)
    # Blockwise scales: the kernels take one scale per output channel.
    assert not can_use_dense_expert(x, w1, w2, jnp.ones(
        (e, 2, 1, 2 * i)), s2, None, None, **ok)


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
@pytest.mark.parametrize("kernel", ["dense_expert", "lowtok"])
@pytest.mark.parametrize("num_tokens, num_valid", [(8, 8), (16, 11),
                                                   (128, 100)])
def test_ep_kernel_matches_reference(kernel, num_tokens, num_valid):
    """Each kernel matches the jnp reference on random weights, scales and
    routing: several experts hit per shard, padding rows with weight 0 and
    an id of -1; 128 tokens is the default TPU_MOE_LOWTOK_MAX_TOKENS."""
    num_experts = 8
    k1, k2, k3 = jax.random.split(jax.random.key(num_tokens), 3)
    x = jax.random.normal(k1, (num_tokens, _H), jnp.bfloat16)
    w1, w2, s1, s2 = _fp8_experts(k2, num_experts)
    ids, weights = _routing(k3, num_tokens, num_experts, TOPK, num_valid)

    if kernel == "dense_expert":
        out = dense_expert_moe(x,
                               w1,
                               w2,
                               s1,
                               s2,
                               weights,
                               ids,
                               num_primary_experts=num_experts)
    else:
        route = moe_lowtok_routing_table(ids, weights, num_experts)
        out = moe_lowtok(x, route, w1, s1, w2, s2)
    out = jax.block_until_ready(out)
    assert out.shape == (num_tokens, _H)
    assert out.dtype == jnp.bfloat16

    expected = _reference_moe(x, w1, w2, s1, s2, weights, ids)
    _assert_close_to_reference(out, expected)
    # Padding rows (weight 0) and the -1 id contribute nothing.
    np.testing.assert_array_equal(np.asarray(out[num_valid:], np.float32), 0)
    assert np.abs(np.asarray(expected[0], np.float32)).max() > 0


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


def _fused_moe_inputs(mesh: Mesh, num_tokens: int, fp8: bool):
    """Tokens sharded over ATTN_DATA, replicated expert weights and gating.
    fp8=True gives the FP8 weights and per-channel scales the EP kernels
    take; fp8=False the bf16 weights the other flags are tested with."""
    k1, k2 = jax.random.split(jax.random.key(num_tokens))
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    gating = jax.random.normal(k2, (num_tokens, _NUM_EXPERTS), jnp.float32)
    inputs = dict(
        hidden_states=_sharded_hidden_states(mesh, num_tokens, _H),
        gating_output=jax.device_put(gating.astype(jnp.bfloat16), rows),
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


def _run_fused_moe(mesh: Mesh, env: dict, num_tokens: int, fp8: bool,
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
                **_fused_moe_inputs(mesh, num_tokens, fp8),
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
    "single-sort": dict(TPU_MOE_SINGLE_SORT=True),
    "rowmax": dict(TPU_MOE_ROUTER_TOPK="rowmax"),
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
    kwargs = dict(scatter_results=scatter_results, num_tokens=256, fp8=False)
    expected = _run_fused_moe(mesh, {}, **kwargs)
    actual = _run_fused_moe(mesh, _ROUTING_FLAGS[flag], **kwargs)
    np.testing.assert_array_equal(np.asarray(actual, np.float32),
                                  np.asarray(expected, np.float32))


_EP_KERNEL_FLAGS = {
    "lowtok": dict(TPU_MOE_LOWTOK=True),
    "dense-expert": dict(TPU_MOE_DENSE_EXPERT_THRESHOLD=128),
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
    """With TPU_MOE_LOWTOK or TPU_MOE_DENSE_EXPERT_THRESHOLD the EP kernel
    replaces permute + GMM + combine and gives the same output, all-reduced
    or reduce-scattered over the shards. The kernel and gmm_v2 quantize the
    activations to FP8 the same way, so they differ only in rounding."""
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
