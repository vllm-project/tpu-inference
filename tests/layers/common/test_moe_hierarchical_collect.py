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
"""Tests for MOE_HIERARCHICAL_COLLECT, the hierarchical MoE EP collect.

The flag tests use fake devices and run anywhere. The collect tests need 8
devices; the ones that check the real core-pairing check additionally need two
cores per chip (v7x), since elsewhere the plan always falls back. The plan
itself is tested in test_moe_hierarchical_dispatch.py.

The expert_parallel_gmm tests replace the two TPU kernels (gmm_v2 and the
SparseCore ragged_gather_reduce) with jnp stand-ins, since only the collect
after them is under test.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _MESH_SIZES, _fake_mesh, _logical_plan, _make_mesh, _mesh_id,
    _paired_devices, requires_8_devices, requires_dual_core_chips)
from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

# expert_parallel_gmm shards its routing inputs over MLP-data; keep that axis
# at 1 so the stand-in inputs stay simple.
_EP_MESH_SIZES = [s for s in _MESH_SIZES if "data" not in s]


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...) the hierarchical collect uses."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


# --- the MOE_HIERARCHICAL_COLLECT gate on fake devices ----------------------


def _traced_collect_arg(mesh,
                        plan,
                        hidden: int = 256,
                        enabled: bool = True,
                        **kwargs):
    """The hierarchical_collect argument expert_parallel_gmm hands to
    moe_gmm_local, and the logger mock. shard_map is replaced, so nothing
    runs and fake devices suffice."""
    shard_map = mock.Mock(return_value=lambda *args: None)
    with (mock.patch("tpu_inference.envs.MOE_HIERARCHICAL_COLLECT", enabled),
          mock.patch.object(fused_moe_gmm,
                            "_hierarchical_dispatch_plan",
                            return_value=plan),
          mock.patch.object(fused_moe_gmm.jax, "shard_map", shard_map),
          mock.patch.object(fused_moe_gmm, "logger") as logger):
        fused_moe_gmm.expert_parallel_gmm(SimpleNamespace(shape=(128, hidden)),
                                          SimpleNamespace(shape=(16, hidden,
                                                                 128)),
                                          None,
                                          None,
                                          None,
                                          None,
                                          None,
                                          None,
                                          None,
                                          None,
                                          activation="silu",
                                          topk=2,
                                          mesh=mesh,
                                          **kwargs)
    return shard_map.call_args.args[0].keywords["hierarchical_collect"], logger


_PLAN = (("attn_dp", ), "model", [(0, 1), (1, 0), (2, 3), (3, 2)])


@pytest.mark.parametrize(
    "enabled, kwargs, expected", [
        (True, dict(scatter_results=True), ("model", _PLAN[2])),
        (False, dict(scatter_results=True), None),
        (True, dict(scatter_results=False), None),
        (True, dict(scatter_results=True, enable_rs_kernel=True), None),
    ],
    ids=["enabled", "flag_off", "no_scatter", "rs_kernel"])
def test_collect_applies_only_to_the_scattering_collect(
        enabled, kwargs, expected):
    mesh = _fake_mesh(_paired_devices(8), attn_dp=2, model=4)
    collect, logger = _traced_collect_arg(mesh,
                                          _PLAN,
                                          enabled=enabled,
                                          **kwargs)
    assert collect == expected
    logger.warning_once.assert_not_called()


@pytest.mark.parametrize("plan, hidden, reason", [
    (None, 256, "does not apply to this mesh"),
    (_PLAN, 2880, "not a multiple of 256"),
],
                         ids=["mesh", "hidden_size"])
def test_collect_fallback_warning_names_the_reason(plan, hidden, reason):
    mesh = _fake_mesh(_paired_devices(8), attn_dp=2, model=4)
    collect, logger = _traced_collect_arg(mesh,
                                          plan,
                                          hidden=hidden,
                                          scatter_results=True)
    assert collect is None
    logger.warning_once.assert_called_once()
    message, *args = logger.warning_once.call_args.args
    assert reason in message % tuple(args)


# --- _hierarchical_collect on real devices ----------------------------------


def _as_tuple(axes) -> tuple:
    return axes if isinstance(axes, tuple) else (axes, )


def _collect_axes() -> tuple[tuple, tuple]:
    """(reduce_axes, scatter_axes) as moe_gmm_local derives them for EP."""
    expert = _as_tuple(ShardingAxisName.EXPERT)
    attn_data = _as_tuple(ShardingAxisName.ATTN_DATA)
    return (tuple(a for a in expert if a not in attn_data),
            tuple(a for a in expert if a in attn_data))


def _check_collect_matches_one_step_collect(mesh: Mesh, plan):
    _, pair_axis, perm = plan
    reduce_axes, scatter_axes = _collect_axes()
    expert = ShardingAxisName.EXPERT
    num_tokens, hidden = 64, 256
    # One [num_tokens, hidden] partial result per EP device.
    x = jax.random.normal(jax.random.key(0),
                          (mesh.devices.size * num_tokens, hidden),
                          jnp.float32)

    def run(f, out_spec):
        return jax.jit(
            jax.shard_map(f,
                          mesh=mesh,
                          in_specs=P(expert),
                          out_specs=out_spec,
                          check_vma=False))(x)

    def one_step(h):
        h = jax.lax.psum(h, reduce_axes)
        return jax.lax.psum_scatter(h,
                                    scatter_axes,
                                    scatter_dimension=0,
                                    tiled=True)

    def hierarchical(h):
        return fused_moe_gmm._hierarchical_collect(h, pair_axis, perm,
                                                   reduce_axes, scatter_axes)

    attn_data = P(ShardingAxisName.ATTN_DATA)
    np.testing.assert_allclose(run(hierarchical, attn_data),
                               run(one_step, attn_data),
                               rtol=1e-5,
                               atol=1e-5)

    # out_specs=P(ATTN_DATA) shows one replica per attention-data rank. Every
    # device in a TP group must hold the same rows, bit for bit, or the next
    # layer's TP replicas diverge.
    per_device = run(lambda h: hierarchical(h)[None],
                     P(tuple(mesh.axis_names)))
    per_device = np.asarray(per_device).reshape(mesh.devices.shape +
                                                per_device.shape[1:])
    for axis in reduce_axes:
        i = mesh.axis_names.index(axis)
        first = np.take(per_device, [0], axis=i)
        np.testing.assert_array_equal(per_device,
                                      np.broadcast_to(first, per_device.shape))


@requires_8_devices
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_collect_matches_one_step_collect(sizes):
    """The data movement itself, on any hardware: use the plan the mesh would
    get on v7x."""
    mesh = _make_mesh(**sizes)
    _check_collect_matches_one_step_collect(mesh, _logical_plan(mesh))


@requires_dual_core_chips
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_collect_matches_one_step_collect_on_dual_core_chips(sizes):
    """The mesh builder puts a chip's two cores on the model axis, so the
    real core-pairing check must pass, and the swap must run on-chip."""
    mesh = _make_mesh(**sizes)
    plan = fused_moe_gmm._hierarchical_dispatch_plan(mesh)
    assert plan == _logical_plan(mesh)
    _check_collect_matches_one_step_collect(mesh, plan)


# --- expert_parallel_gmm integration ----------------------------------------


def _fake_gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset,
              **kwargs):
    # Mixes in every local expert, so each device's partial result differs.
    return (lhs @ rhs.sum(axis=0)).astype(lhs.dtype)


def _fake_ragged_gather_reduce(x, indices, weights, mask, topk):
    rows = x[indices] * (weights * mask)[:, None]
    return rows.reshape(-1, topk, x.shape[1]).sum(axis=1)


def _ep_inputs(mesh,
               num_tokens: int = 64,
               hidden: int = 256,
               intermediate: int = 128,
               num_experts: int = 16,
               topk: int = 2):
    """Routed inputs for expert_parallel_gmm, rows sorted by expert."""
    keys = jax.random.split(jax.random.key(1), 5)
    tokens = jax.random.normal(keys[0], (num_tokens, hidden), jnp.float32)
    experts = jax.random.randint(keys[1], (num_tokens * topk, ), 0,
                                 num_experts)
    order = jnp.argsort(experts)
    data = NamedSharding(mesh, P(ShardingAxisName.MLP_DATA))
    return dict(
        x=jax.device_put(tokens[order // topk], data),
        w1=jax.random.normal(keys[2], (num_experts, hidden, intermediate),
                             jnp.float32) / 10,
        w1_scale=None,
        w1_bias=None,
        w2=jax.random.normal(keys[3], (num_experts, intermediate, hidden),
                             jnp.float32) / 10,
        w2_scale=None,
        w2_bias=None,
        group_sizes=jnp.bincount(experts, length=num_experts),
        topk_argsort_revert_indices=jnp.argsort(order),
        topk_weights=jax.random.uniform(keys[4], (num_tokens, topk)),
        activation="silu",
        topk=topk,
        mesh=mesh,
        scatter_results=True,
    )


_REAL_PLAN = object()


def _run_ep_gmm(mesh, enabled: bool, plan=_REAL_PLAN, moe_chunk_size: int = 0):
    """Run expert_parallel_gmm with the flag set to `enabled`, and `plan`
    forced unless it is _REAL_PLAN.

    Returns (output, times _hierarchical_collect was traced, logger mock).
    The flag is read at trace time, so caches are cleared to force a retrace.
    """
    jax.clear_caches()
    spy = mock.Mock(wraps=fused_moe_gmm._hierarchical_collect)
    force_plan = (
        contextlib.nullcontext() if plan is _REAL_PLAN else mock.patch.object(
            fused_moe_gmm, "_hierarchical_dispatch_plan", return_value=plan))
    with (mock.patch("tpu_inference.envs.MOE_HIERARCHICAL_COLLECT",
                     enabled), force_plan,
          mock.patch.object(fused_moe_gmm, "gmm_wrapper", _fake_gmm),
          mock.patch.object(fused_moe_gmm, "ragged_gather_reduce",
                            _fake_ragged_gather_reduce),
          mock.patch.object(fused_moe_gmm, "_hierarchical_collect",
                            spy), mock.patch.object(fused_moe_gmm, "logger") as
          logger):
        out = fused_moe_gmm.expert_parallel_gmm(**_ep_inputs(mesh),
                                                moe_chunk_size=moe_chunk_size)
    return out, spy.call_count, logger


@requires_8_devices
@pytest.mark.parametrize("moe_chunk_size", [0, 8],
                         ids=["unchunked", "chunked"])
@pytest.mark.parametrize("sizes", _EP_MESH_SIZES, ids=_mesh_id)
def test_expert_parallel_gmm_output_is_unchanged(sizes, moe_chunk_size):
    """Same output with the flag on, including when the collect runs once per
    chunk on tokens reordered by _permute_tokens_for_chunked_rs."""
    mesh = _make_mesh(**sizes)
    # 64 tokens; each chunk is moe_chunk_size tokens per attention-data rank,
    # so 4 chunks at attention-data size 2 and 2 chunks at size 4.
    attn_data_size = mesh.shape["attn_dp"] * mesh.shape["attn_dp_expert"]
    num_chunks = (64 //
                  (moe_chunk_size * attn_data_size) if moe_chunk_size else 1)
    plan = _logical_plan(mesh)
    expected, calls, _ = _run_ep_gmm(mesh,
                                     enabled=False,
                                     plan=plan,
                                     moe_chunk_size=moe_chunk_size)
    assert calls == 0

    # Force the plan so the hierarchical path is exercised on any hardware.
    actual, calls, logger = _run_ep_gmm(mesh,
                                        enabled=True,
                                        plan=plan,
                                        moe_chunk_size=moe_chunk_size)
    assert calls == num_chunks
    logger.warning_once.assert_not_called()
    assert actual.sharding.is_equivalent_to(
        NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA)), actual.ndim)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@requires_dual_core_chips
@pytest.mark.parametrize("sizes", _EP_MESH_SIZES, ids=_mesh_id)
def test_expert_parallel_gmm_applies_on_dual_core_chips(sizes):
    mesh = _make_mesh(**sizes)
    expected, _, _ = _run_ep_gmm(mesh, enabled=False)
    actual, calls, logger = _run_ep_gmm(mesh, enabled=True)
    assert calls == 1
    logger.warning_once.assert_not_called()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@requires_dual_core_chips
def test_expert_parallel_gmm_falls_back_when_model_axis_spans_chips():
    # Core-major device order: model neighbours sit on different chips.
    devices = sorted(jax.devices(),
                     key=lambda d: (getattr(d, "core_on_chip", 0), d.id))
    shape = tuple(dict(attn_dp=2, model=4).get(a, 1) for a in MESH_AXIS_NAMES)
    mesh = Mesh(np.array(devices[:8]).reshape(shape), MESH_AXIS_NAMES)
    _, calls, logger = _run_ep_gmm(mesh, enabled=True)
    assert calls == 0
    logger.warning_once.assert_called_once()
