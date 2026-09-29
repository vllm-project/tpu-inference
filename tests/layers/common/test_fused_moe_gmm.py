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
"""Tests for fused_moe_func's static activation input scales.

The tests need a TPU on which gmm_v2 quantizes activations to fp8 for fp8
weights (fp8_ops_per_second > 0, e.g. v6e, v7x).
"""

import math
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_NUM_DEVICES = len(jax.devices())

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu"
    or pltpu.get_tpu_info().fp8_ops_per_second == 0,
    reason="needs a TPU with fp8 ops (e.g. v6e, v7x)")


@pytest.fixture(autouse=True)
def base_sharding_axes():
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


def _make_mesh(**sizes) -> Mesh:
    shape = tuple(sizes.get(a, 1) for a in MESH_AXIS_NAMES)
    devices = sorted(jax.devices(), key=lambda d: d.id)[:math.prod(shape)]
    return jax.make_mesh(shape,
                         MESH_AXIS_NAMES, (AxisType.Auto, ) * len(shape),
                         devices=devices)


def _quantize(w: jax.Array, block_k: int | None):
    """[E, K, N] -> fp8 weight and its [E, K // block_k, 1, N] scale."""
    num_experts, k, n = w.shape
    block_k = block_k or k
    w32 = w.astype(jnp.float32).reshape(num_experts, k // block_k, block_k, n)
    scale = jnp.max(jnp.abs(w32), axis=2, keepdims=True) / _FP8_MAX
    w_q = (w32 / scale).astype(jnp.float8_e4m3fn).reshape(num_experts, k, n)
    return w_q, scale


def _run(mesh: Mesh,
         *,
         quantized: bool = True,
         clip: float | None = None,
         block_k: int | None = None,
         **kwargs) -> np.ndarray:
    num_tokens, hidden, intermediate, num_experts = 64, 256, 1024, 16
    kx, k1, k2, kg = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(kx, (num_tokens, hidden), jnp.bfloat16)
    if clip is not None:
        x = jnp.clip(x, -clip, clip)
    w1 = jax.random.normal(k1, (num_experts, hidden, 2 * intermediate),
                           jnp.bfloat16) / 16
    w2 = jax.random.normal(k2, (num_experts, intermediate, hidden),
                           jnp.bfloat16) / 16
    w1_scale = w2_scale = None
    if quantized:
        w1, w1_scale = _quantize(w1, block_k)
        w2, w2_scale = _quantize(w2, block_k)
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    with jax.set_mesh(mesh):
        out = fused_moe_gmm.fused_moe_func(
            hidden_states=jax.device_put(x, rows),
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=None,
            w2_bias=None,
            gating_output=jax.device_put(
                jax.random.normal(kg, (num_tokens, num_experts), jnp.bfloat16),
                rows),
            topk=2,
            renormalize=True,
            mesh=mesh,
            use_ep=True,
            activation="silu",
            scoring_fn="softmax",
            **kwargs)
        return np.asarray(jax.block_until_ready(out).astype(jnp.float32))


def _scale(value: float) -> jax.Array:
    return jnp.full((1, 1), value, jnp.float32)


def _rel_err(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


@pytest.mark.parametrize("sizes", [
    pytest.param({}, id="1dev"),
    pytest.param(dict(attn_dp=2, model=4),
                 id="8dev_ep",
                 marks=pytest.mark.skipif(_NUM_DEVICES < 8,
                                          reason="needs 8 devices")),
])
def test_static_input_scale_is_used_and_accurate(sizes):
    mesh = _make_mesh(**sizes)
    reference = _run(mesh, quantized=False)
    dynamic = _run(mesh)
    # 224 / 448: the fixed,-224,224 convention of MaxText's fp8 recipes.
    static = _run(mesh, w1_input_scale=_scale(0.5), w2_input_scale=_scale(0.5))

    assert not np.array_equal(static, dynamic), "static scale was not used"
    assert _rel_err(dynamic, reference) < 0.1
    assert _rel_err(static, reference) < 0.1


def test_static_input_scale_saturates_at_fixed_range():
    # A static scale does not adapt to the input: GMM1 inputs beyond
    # +-(scale * fp8_max) saturate, exactly as if they had been clipped.
    mesh = _make_mesh()
    clip = 0.25
    kwargs = dict(w1_input_scale=_scale(clip / _FP8_MAX),
                  w2_input_scale=_scale(0.5))
    np.testing.assert_array_equal(_run(mesh, **kwargs),
                                  _run(mesh, clip=clip, **kwargs))


def test_input_scale_is_ignored_for_narrow_weight_scale_blocks():
    # Weight-scale blocks narrower than the MXU make gmm_v2 dequantize the
    # weights before the matmul, so the activations are never quantized and
    # gmm_v2 would reject an lhs_scale.
    mesh = _make_mesh()
    expected = _run(mesh, block_k=128)
    with mock.patch.object(fused_moe_gmm, "logger") as logger:
        actual = _run(mesh,
                      block_k=128,
                      w1_input_scale=_scale(0.5),
                      w2_input_scale=_scale(0.5))
    np.testing.assert_array_equal(actual, expected)
    assert any("is ignored" in call.args[0]
               for call in logger.warning_once.call_args_list)
