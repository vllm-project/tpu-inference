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
"""Tests for small all-reduce: TPU kernel execution (skipped on CPU) and pure JAX
reference and fallback logic (runs on CPU).
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference import envs
from tpu_inference.kernels.experimental.small_collectives import \
    all_reduce as ar
from tpu_inference.kernels.experimental.small_collectives.common import (
    peer, split_rank)

from .oracle import (AXIS, N, bf16_ulp, blocks, exact_sum, run_sharded,
                     sum_tolerance, within)

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="Requires TPU hardware",
)


def _all_reduce(mesh, per_core, ranks, *, rank_operand=True, form=None):

    def body(x, r):
        return ar.small_all_reduce(x,
                                   num_cores=N,
                                   axis=AXIS,
                                   rank=r if rank_operand else None,
                                   form=form)

    return run_sharded(mesh, body, per_core, ranks)


def _check(got, per_core):
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in range(N):
        ok = within(got[core], want, tolerance)
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"


@requires_tpu
@pytest.mark.parametrize("rows", [1, 3, 4, 8, 12, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_exact_sum(mesh, rows, rank_operand):
    per_core = blocks((N, rows, 4096), seed=rows)
    _check(_all_reduce(mesh, per_core, range(N), rank_operand=rank_operand),
           per_core)


@requires_tpu
@pytest.mark.parametrize(
    "form",
    ["oneshot", "twoshot", "oneshot_xor", "hier_halves", "hier_2stage"])
@pytest.mark.parametrize("rows", [4, 16])
def test_both_forms_at_both_sizes(mesh, form, rows):
    per_core = blocks((N, rows, 4096), seed=100 + rows)
    _check(_all_reduce(mesh, per_core, range(N), form=form), per_core)


@requires_tpu
@pytest.mark.parametrize("hidden", [2048, 6144, 7168])
def test_other_widths(mesh, hidden):
    per_core = blocks((N, 16, hidden), seed=hidden)
    _check(_all_reduce(mesh, per_core, range(N)), per_core)


@requires_tpu
def test_agrees_with_psum_within_its_rounding(mesh):
    """XLA's bf16 psum may round every partial sum, (n - 1) roundings of at
    most half a step at the sum of magnitudes; the kernel rounds once."""
    per_core = blocks((N, 16, 4096), seed=7)
    psum = run_sharded(mesh, lambda x, r: jax.lax.psum(x, AXIS), per_core,
                       range(N)).astype(np.float64)
    got = _all_reduce(mesh, per_core, range(N)).astype(np.float64)
    magnitudes = np.abs(per_core.astype(np.float64)).sum(axis=0)
    tolerance = (N - 1) / 2 * bf16_ulp(magnitudes) + 2 * bf16_ulp(
        np.abs(psum[0]))
    for core in range(N):
        assert within(got[core], psum[core], tolerance).all()


@requires_tpu
def test_a_wrong_rank_order_fails_the_oracle(mesh):
    """Positive control: cores 0 and 1 told each other's index write into the
    wrong slots, and the oracle has to see it."""
    per_core = blocks((N, 16, 4096), seed=11)
    swapped = [1, 0, *range(2, N)]
    got = _all_reduce(mesh, per_core, swapped)
    want, tolerance = exact_sum(per_core), sum_tolerance(per_core)
    for core in (0, 1):
        assert not within(got[core], want, tolerance).all()


# Pure JAX CPU tests


def test_refuses_a_width_that_does_not_split():
    with pytest.raises(ValueError, match="lanes"):
        ar.small_all_reduce(jnp.zeros((16, 4000), jnp.bfloat16),
                            num_cores=N,
                            axis=AXIS)


def test_form_by_bytes():
    assert ar.form_for(4, 4096, 2) == "oneshot"
    assert ar.form_for(5, 4096, 2) == "twoshot"
    assert ar.form_for(8, 2048, 2) == "oneshot"
    assert ar.form_for_8cores(1, 4096, 2) == "oneshot_xor"
    assert ar.form_for_8cores(4, 4096, 2) == "oneshot_xor"
    assert ar.form_for_8cores(8, 4096, 2) == "hier_halves"
    assert ar.form_for_8cores(16, 4096, 2) == "twoshot"
    assert ar.form_for_8cores(24, 4096, 2) == "hier_2stage"
    assert ar.form_for_8cores(32, 4096, 2) == "hier_2stage"
    assert ar.form_for_8cores(128, 4096, 2) == "hier_2stage"


def test_reference_matches_exact_sum():
    per_core = blocks((N, 16, 4096), seed=42)
    ref = ar.reference(per_core)
    want = exact_sum(per_core)
    tol = sum_tolerance(per_core)
    assert within(ref, want, tol).all()


def test_dispatch_form_fallback_non_8cores():
    # When num_cores != 8, forms fall back to oneshot or twoshot
    assert ar.dispatch_form_for(1, 4096, 2, num_cores=4) == "oneshot"
    assert ar.dispatch_form_for(16, 4096, 2, num_cores=4) == "twoshot"
    assert ar.dispatch_form_for(24, 4096, 2, num_cores=4) == "twoshot"
    assert ar.dispatch_form_for(8, 4096, 2, num_cores=2) == "twoshot"


def test_peer_single_axis():
    assert peer(3, "model") == {"model": jnp.int32(3)}
    assert peer(3, ("model", )) == {"model": jnp.int32(3)}


def test_peer_and_split_rank_multi_axis():
    if jax.device_count() < 8:
        pytest.skip(
            "needs at least 8 host devices (set XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )

    devices = np.array(jax.devices()[:8]).reshape((2, 4))
    mesh = jax.sharding.Mesh(devices, ("attn_dp_expert", "expert"))

    @jax.shard_map(
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec("attn_dp_expert", "expert"),
        out_specs=jax.sharding.PartitionSpec("attn_dp_expert", "expert"),
        check_vma=False,
    )
    def fn(x):
        p5 = peer(5, ("attn_dp_expert", "expert"))
        p0 = peer(0, ("attn_dp_expert", "expert"))
        p7 = peer(7, ("attn_dp_expert", "expert"))

        linear_idx, _ = split_rank([], False, ("attn_dp_expert", "expert"))
        return (
            jnp.broadcast_to(linear_idx, x.shape),
            jnp.broadcast_to(p5["attn_dp_expert"], x.shape),
            jnp.broadcast_to(p5["expert"], x.shape),
            jnp.broadcast_to(p0["attn_dp_expert"], x.shape),
            jnp.broadcast_to(p0["expert"], x.shape),
            jnp.broadcast_to(p7["attn_dp_expert"], x.shape),
            jnp.broadcast_to(p7["expert"], x.shape),
        )

    dummy = jnp.zeros((2, 4), dtype=jnp.int32)
    linear_idx, p5_a, p5_e, p0_a, p0_e, p7_a, p7_e = fn(dummy)

    # Cores across 2x4 mesh should have linear ranks 0..7
    np.testing.assert_array_equal(linear_idx,
                                  np.arange(8, dtype=np.int32).reshape(2, 4))
    # peer(5) -> attn_dp_expert=1, expert=1
    assert (p5_a == 1).all()
    assert (p5_e == 1).all()
    # peer(0) -> attn_dp_expert=0, expert=0
    assert (p0_a == 0).all()
    assert (p0_e == 0).all()
    # peer(7) -> attn_dp_expert=1, expert=3
    assert (p7_a == 1).all()
    assert (p7_e == 3).all()


def test_fused_moe_gmm_fallback_on_cpu(monkeypatch):
    """When TPU_SMALL_ALL_REDUCE is enabled on CPU, moe_gmm_local safely falls back
    to jax.lax.psum. When on TPU with fitting dimensions, small_all_reduce is dispatched.
    """
    from tpu_inference.layers.common import fused_moe_gmm
    from tpu_inference.layers.common.fused_moe_gmm import moe_gmm_local

    num_tokens = 8
    topk = 2
    batch_size = num_tokens * topk
    hidden_size = 128
    inter_size = 64
    global_num_experts = 8
    local_group_size = 2
    group_offset = jnp.array([0], dtype=jnp.int32)
    group_sizes = jnp.full((global_num_experts, ), 2, dtype=jnp.int32)
    topk_argsort_revert_indices = jnp.arange(batch_size, dtype=jnp.int32)
    topk_weights = jnp.full((num_tokens, topk), 0.5, dtype=jnp.float32)

    x = jnp.ones((batch_size, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((local_group_size, hidden_size, inter_size * 2),
                  dtype=jnp.bfloat16)
    w2 = jnp.ones((local_group_size, inter_size, hidden_size),
                  dtype=jnp.bfloat16)

    def fake_gmm(lhs, rhs, *args, **kwargs):
        return jnp.ones((lhs.shape[0], hidden_size), dtype=jnp.bfloat16)

    psum_called = []

    def fake_psum(val, axis_name):
        psum_called.append(axis_name)
        return val

    # Test 1: On CPU with TPU_SMALL_ALL_REDUCE=True, it falls back to psum
    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE", True)
    with patch.object(jax.lax, "psum", side_effect=fake_psum):
        with patch.object(fused_moe_gmm, "gmm_wrapper", side_effect=fake_gmm):
            _ = moe_gmm_local(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                group_offset,
                topk_argsort_revert_indices,
                topk_weights,
                activation="silu",
                topk=topk,
                parallelism="tp",
                onehot_moe_permute_threshold=1024,
                defer_all_reduce=False,
            )
        assert len(psum_called) > 0

        # Test 2: If we mock platform to 'tpu' and dimensions fit, small_all_reduce is called
        mock_tpu_dev = MagicMock()
        mock_tpu_dev.platform = "tpu"
        ar_called = []

        def fake_small_ar(chunk, *, num_cores, axis):
            ar_called.append((chunk.shape, num_cores, axis))
            return chunk

        with patch.object(fused_moe_gmm, "gmm_wrapper", side_effect=fake_gmm), \
             patch.object(jax, "devices", return_value=[mock_tpu_dev]), \
             patch.object(fused_moe_gmm, "small_all_reduce", side_effect=fake_small_ar), \
             patch.object(jax.lax, "axis_size", return_value=1):
            # When num_cores == 1, small_all_reduce is not called (falls back to psum)
            _ = moe_gmm_local(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                group_offset,
                topk_argsort_revert_indices,
                topk_weights,
                activation="silu",
                topk=topk,
                parallelism="tp",
                onehot_moe_permute_threshold=1024,
                defer_all_reduce=False,
            )
            assert len(ar_called) == 0

        # When num_cores == 8 and hidden is multiple of 8*128 (e.g. 1024)
        w1_big = jnp.ones((local_group_size, 1024, inter_size * 2),
                          dtype=jnp.bfloat16)
        w2_big = jnp.ones((local_group_size, inter_size, 1024),
                          dtype=jnp.bfloat16)

        def fake_gmm_big(lhs, rhs, *args, **kwargs):
            return jnp.ones((lhs.shape[0], 1024), dtype=jnp.bfloat16)

        def fake_rgr(*args, **kwargs):
            return jnp.ones((8, 1024), dtype=jnp.bfloat16)

        with patch.object(fused_moe_gmm, "gmm_wrapper", side_effect=fake_gmm_big), \
             patch.object(fused_moe_gmm, "_select_ragged_gather_reduce", return_value=fake_rgr), \
             patch.object(jax, "devices", return_value=[mock_tpu_dev]), \
             patch.object(fused_moe_gmm, "small_all_reduce", side_effect=fake_small_ar), \
             patch.object(jax.lax, "axis_size", side_effect=lambda a: 8 if a == "model" else 1):
            _ = moe_gmm_local(
                x,
                w1_big,
                None,
                None,
                w2_big,
                None,
                None,
                group_sizes,
                group_offset,
                topk_argsort_revert_indices,
                topk_weights,
                activation="silu",
                topk=topk,
                parallelism="tp",
                onehot_moe_permute_threshold=0,
                defer_all_reduce=False,
            )
            assert len(ar_called) == 1
            shape, ncores, axis = ar_called[0]
            assert ncores == 8
            assert shape[1] == 1024
