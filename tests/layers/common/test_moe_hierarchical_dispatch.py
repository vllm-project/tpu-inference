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
"""Tests for MOE_HIERARCHICAL_DISPATCH, the hierarchical MoE dispatch gather.

The plan tests use fake devices and run anywhere. The gather tests need 8
devices; the ones that check the real core-pairing check additionally need two
cores per chip (v7x), since elsewhere the plan always falls back. CI runs this
file on v7x-8 with MOE_HIERARCHICAL_DISPATCH_REQUIRE_DUAL_CORE=1, which turns
missing hardware into a failure instead of a skip.
"""

import math
import os
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D,
                                                  ShardingAxisName,
                                                  ShardingAxisName2D,
                                                  ShardingAxisNameBase)

_NUM_DEVICES = len(jax.devices())
_HAS_DUAL_CORE_CHIPS = _NUM_DEVICES >= 8 and any(
    getattr(d, "core_on_chip", 0) for d in jax.devices())
_REQUIRE_DUAL_CORE = "MOE_HIERARCHICAL_DISPATCH_REQUIRE_DUAL_CORE"

requires_8_devices = pytest.mark.skipif(_NUM_DEVICES < 8,
                                        reason="needs 8 devices")
requires_dual_core_chips = pytest.mark.skipif(
    not _HAS_DUAL_CORE_CHIPS,
    reason="needs 8 devices with two cores per chip (e.g. v7x-8)")

# Real-device mesh layouts the hierarchical gather is meant for.
_MESH_SIZES = [
    dict(attn_dp=2, model=4),
    dict(attn_dp=4, model=2),
    dict(data=2, attn_dp=2, model=2),
    dict(attn_dp=2, attn_dp_expert=2, model=2),
]


def _mesh_id(sizes: dict) -> str:
    return "x".join(f"{a}{n}" for a, n in sizes.items())


@pytest.mark.skipif(os.environ.get(_REQUIRE_DUAL_CORE) != "1",
                    reason=f"{_REQUIRE_DUAL_CORE} is not set")
def test_required_hardware_is_present():
    # Without this, a CI step that lands on the wrong hardware would skip every
    # real-device test below and still pass.
    assert _HAS_DUAL_CORE_CHIPS, (
        f"{_REQUIRE_DUAL_CORE}=1 but found {_NUM_DEVICES} devices "
        f"({jax.devices()[0].device_kind}); the real-device tests need 8 "
        "devices with two cores per chip and would all be skipped.")


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...) the hierarchical gather uses."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


# --- _hierarchical_dispatch_plan on fake devices ----------------------------


def _fake_device(chip: int, core: int, slice_index=None) -> SimpleNamespace:
    device = SimpleNamespace(coords=(chip, 0, 0), core_on_chip=core)
    if slice_index is not None:  # only multi-slice TPU devices have one
        device.slice_index = slice_index
    return device


def _paired_devices(n: int) -> list[SimpleNamespace]:
    """n devices in mesh order where devices 2k, 2k+1 are the cores of chip k."""
    return [_fake_device(i // 2, i % 2) for i in range(n)]


def _fake_mesh(devices, axis_names=MESH_AXIS_NAMES, **sizes):
    shape = {a: sizes.get(a, 1) for a in axis_names}
    devs = np.empty(len(devices), dtype=object)
    devs[:] = devices
    return SimpleNamespace(shape=shape,
                           axis_names=axis_names,
                           devices=devs.reshape(tuple(shape.values())))


@pytest.mark.parametrize("attn_dp, model, perm", [
    (2, 4, [(0, 1), (1, 0), (2, 3), (3, 2)]),
    (4, 2, [(0, 1), (1, 0)]),
    (8, 4, [(0, 1), (1, 0), (2, 3), (3, 2)]),
])
def test_plan_pairs_the_cores_of_each_chip(attn_dp, model, perm):
    mesh = _fake_mesh(_paired_devices(attn_dp * model),
                      attn_dp=attn_dp,
                      model=model)
    plan = fused_moe_gmm._hierarchical_dispatch_plan(mesh)
    assert plan == (("attn_dp", ), "model", perm)


def test_plan_leaves_mlp_data_axis_sharded():
    mesh = _fake_mesh(_paired_devices(8), data=2, attn_dp=2, model=2)
    plan = fused_moe_gmm._hierarchical_dispatch_plan(mesh)
    assert plan == (("attn_dp", ), "model", [(0, 1), (1, 0)])


def test_plan_gathers_over_every_sharded_attention_axis():
    mesh = _fake_mesh(_paired_devices(8), attn_dp=2, attn_dp_expert=2, model=2)
    step1, pair_axis, _ = fused_moe_gmm._hierarchical_dispatch_plan(mesh)
    assert step1 == ("attn_dp", "attn_dp_expert")
    assert pair_axis == "model"


def test_plan_accepts_pairs_within_each_slice():
    # One attention-data rank per slice; coords repeat across slices.
    devices = [
        _fake_device(i // 2 % 2, i % 2, slice_index=i // 4) for i in range(8)
    ]
    mesh = _fake_mesh(devices, attn_dp=2, model=4)
    assert fused_moe_gmm._hierarchical_dispatch_plan(mesh) is not None


@pytest.mark.parametrize(
    "devices",
    [
        # One core per chip (v6e): no two devices share a chip.
        [_fake_device(i, 0) for i in range(8)],
        # Core-major order: model neighbours are the same core of two chips.
        [_fake_device(i % 4, i // 4) for i in range(8)],
        # Paired in the first attention-data rank only.
        _paired_devices(4) + [
            _fake_device(2, 0),
            _fake_device(3, 0),
            _fake_device(2, 1),
            _fake_device(3, 1)
        ],
        # Same chip but not distinct cores.
        [_fake_device(i // 2, 0) for i in range(8)],
        # No topology information (e.g. CPU devices).
        [SimpleNamespace() for _ in range(8)],
        # Same coords and distinct cores, but on two different slices.
        [_fake_device(i // 2, i % 2, slice_index=i % 2) for i in range(8)],
    ],
    ids=[
        "single_core_chips", "pairs_span_chips", "paired_in_one_rank_only",
        "same_core", "no_coords", "pairs_span_slices"
    ])
def test_plan_falls_back_without_on_chip_pairs(devices):
    mesh = _fake_mesh(devices, attn_dp=2, model=4)
    assert fused_moe_gmm._hierarchical_dispatch_plan(mesh) is None


@pytest.mark.parametrize(
    "sizes", [
        dict(model=8),
        dict(attn_dp=2, expert=2, model=2),
        dict(attn_dp=2, model=3),
    ],
    ids=["no_attention_data_sharding", "two_replicated_axes", "odd_pair_axis"])
def test_plan_falls_back_on_unsupported_mesh_shape(sizes):
    mesh = _fake_mesh(_paired_devices(math.prod(sizes.values())), **sizes)
    assert fused_moe_gmm._hierarchical_dispatch_plan(mesh) is None


def test_plan_falls_back_on_2d_mesh():
    # With 2D axis names the attention-data axis is the MLP-data axis, which
    # stays sharded, so there is nothing to gather over.
    mesh = _fake_mesh(_paired_devices(8),
                      axis_names=MESH_AXIS_NAMES_2D,
                      data=2,
                      model=4)
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisName2D):
        assert fused_moe_gmm._hierarchical_dispatch_plan(mesh) is None


@pytest.mark.parametrize("plan, hidden, reason", [
    (None, 256, "does not apply to this mesh"),
    ((("attn_dp", ), "model", [(0, 1),
                               (1, 0)]), 2880, "not a multiple of 256"),
],
                         ids=["mesh", "hidden_size"])
def test_gather_fallback_warning_names_the_reason(plan, hidden, reason):
    mesh = _fake_mesh(_paired_devices(4), attn_dp=2, model=2)
    x = SimpleNamespace(shape=(64, hidden))
    with mock.patch.object(fused_moe_gmm, "_hierarchical_dispatch_plan",
                           return_value=plan), \
            mock.patch.object(fused_moe_gmm, "logger") as logger:
        assert fused_moe_gmm._apply_hierarchical_dispatch_gather(x, mesh) is x
    logger.warning_once.assert_called_once()
    message, *args = logger.warning_once.call_args.args
    assert reason in message % tuple(args)


# --- _apply_hierarchical_dispatch_gather on real devices --------------------


def _make_mesh(**sizes) -> Mesh:
    shape = tuple(sizes.get(a, 1) for a in MESH_AXIS_NAMES)
    devices = sorted(jax.devices(), key=lambda d: d.id)[:math.prod(shape)]
    return jax.make_mesh(shape,
                         MESH_AXIS_NAMES, (AxisType.Auto, ) * len(shape),
                         devices=devices)


def _sharded_hidden_states(mesh: Mesh,
                           num_tokens: int = 64,
                           hidden: int = 256,
                           seed: int = 0) -> jax.Array:
    x = jax.random.normal(jax.random.key(seed), (num_tokens, hidden),
                          jnp.bfloat16)
    return jax.device_put(
        x, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))


def _assert_bitwise_equal(actual: jax.Array, expected: jax.Array):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(expected).view(np.uint16))


def _logical_plan(mesh: Mesh):
    """The plan for `mesh` if its model axis did pair the cores of a chip."""
    return fused_moe_gmm._hierarchical_dispatch_plan(
        _fake_mesh(_paired_devices(mesh.devices.size), **mesh.shape))


def _run_gather(x: jax.Array, mesh: Mesh):
    gather = jax.jit(
        lambda h: fused_moe_gmm._apply_hierarchical_dispatch_gather(h, mesh))
    return gather(x), gather.lower(x).as_text()


@requires_8_devices
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_gather_matches_one_step_gather(sizes):
    """The data movement itself, on any hardware: force the plan the mesh
    would get on v7x and check the result is the input, bit for bit, laid out
    as the one-step gather would leave it."""
    mesh = _make_mesh(**sizes)
    x = _sharded_hidden_states(mesh)
    with mock.patch.object(fused_moe_gmm,
                           "_hierarchical_dispatch_plan",
                           return_value=_logical_plan(mesh)):
        out, hlo = _run_gather(x, mesh)

    assert "collective_permute" in hlo
    assert out.sharding.is_equivalent_to(
        NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)), out.ndim)
    _assert_bitwise_equal(out, x)


@requires_dual_core_chips
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_gather_applies_on_dual_core_chips(sizes):
    """The mesh builder puts a chip's two cores on the model axis, so the
    real core-pairing check must pass and the gather must take two steps."""
    mesh = _make_mesh(**sizes)
    plan = fused_moe_gmm._hierarchical_dispatch_plan(mesh)
    assert plan == _logical_plan(mesh)

    x = _sharded_hidden_states(mesh)
    out, hlo = _run_gather(x, mesh)
    assert "collective_permute" in hlo
    _assert_bitwise_equal(out, x)


@requires_dual_core_chips
def test_gather_falls_back_when_model_axis_spans_chips():
    # Core-major device order: model neighbours sit on different chips.
    devices = sorted(jax.devices(),
                     key=lambda d: (getattr(d, "core_on_chip", 0), d.id))
    shape = tuple(dict(attn_dp=2, model=4).get(a, 1) for a in MESH_AXIS_NAMES)
    mesh = Mesh(np.array(devices[:8]).reshape(shape), MESH_AXIS_NAMES)
    x = _sharded_hidden_states(mesh)
    assert fused_moe_gmm._apply_hierarchical_dispatch_gather(x, mesh) is x


@requires_dual_core_chips
@pytest.mark.parametrize("hidden", [255, 2880],
                         ids=["odd", "half_not_lane_aligned"])
def test_gather_falls_back_on_misaligned_hidden_size(hidden):
    mesh = _make_mesh(attn_dp=2, model=4)
    assert fused_moe_gmm._hierarchical_dispatch_plan(mesh) is not None
    x = _sharded_hidden_states(mesh, hidden=hidden)
    assert fused_moe_gmm._apply_hierarchical_dispatch_gather(x, mesh) is x


# --- fused_moe_func integration ---------------------------------------------


def _moe_inputs(mesh: Mesh,
                num_tokens: int = 64,
                hidden: int = 256,
                intermediate: int = 1024,
                num_experts: int = 16):
    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    return dict(
        hidden_states=_sharded_hidden_states(mesh, num_tokens, hidden),
        w1=jax.random.normal(k1, (num_experts, hidden, 2 * intermediate),
                             jnp.bfloat16) / 10,
        w2=jax.random.normal(k2, (num_experts, intermediate, hidden),
                             jnp.bfloat16) / 10,
        gating_output=jax.device_put(
            jax.random.normal(k3, (num_tokens, num_experts), jnp.bfloat16),
            rows),
    )


def _run_fused_moe(mesh: Mesh, enabled: bool, **kwargs):
    """Run fused_moe_func with the flag set to `enabled`.

    Returns (output, times the hierarchical gather was traced). The flag
    is read at trace time, so caches are cleared to force a retrace, and again
    afterwards so no later test gets this run's executable from the cache.
    """
    jax.clear_caches()
    spy = mock.Mock(wraps=fused_moe_gmm._apply_hierarchical_dispatch_gather)
    try:
        with (mock.patch("tpu_inference.envs.MOE_HIERARCHICAL_DISPATCH",
                         enabled),
              mock.patch.object(fused_moe_gmm,
                                "_apply_hierarchical_dispatch_gather",
                                spy), jax.set_mesh(mesh)):
            out = fused_moe_gmm.fused_moe_func(
                **_moe_inputs(mesh),
                w1_scale=None,
                w2_scale=None,
                w1_bias=None,
                w2_bias=None,
                topk=2,
                renormalize=True,
                mesh=mesh,
                activation="silu",
                scoring_fn="softmax",
                **kwargs,
            )
            out = jax.block_until_ready(out)
    finally:
        jax.clear_caches()
    return out, spy.call_count


def _warned_ignored(logger: mock.Mock) -> bool:
    return any("is set but ignored" in call.args[0]
               for call in logger.warning_once.call_args_list)


@requires_8_devices
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_fused_moe_output_is_unchanged(sizes):
    mesh = _make_mesh(**sizes)
    expected, calls = _run_fused_moe(mesh, enabled=False, use_ep=True)
    assert calls == 0

    # Force the plan so the hierarchical path is exercised on any hardware.
    with mock.patch.object(fused_moe_gmm,
                           "_hierarchical_dispatch_plan",
                           return_value=_logical_plan(mesh)), \
            mock.patch.object(fused_moe_gmm, "logger") as logger:
        actual, calls = _run_fused_moe(mesh, enabled=True, use_ep=True)
    assert calls == 1
    assert not _warned_ignored(logger)
    _assert_bitwise_equal(actual, expected)


@requires_8_devices
@pytest.mark.parametrize("kwargs", [
    dict(use_ep=False),
    dict(use_ep=True, all_gather_fp8=True),
],
                         ids=["tensor_parallel", "fp8_all_gather"])
def test_fused_moe_skips_hierarchical_gather(kwargs):
    # Only the expert-parallel path uses it, and the fp8 all-gather wins; the
    # flag being ignored is logged.
    mesh = _make_mesh(attn_dp=2, model=4)
    with mock.patch.object(fused_moe_gmm, "logger") as logger:
        _, calls = _run_fused_moe(mesh, enabled=True, **kwargs)
    assert calls == 0
    assert _warned_ignored(logger)
