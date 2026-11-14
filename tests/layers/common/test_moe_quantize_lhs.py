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
"""Tests for fused_moe_func's quantize_lhs option.

The gmm_wrapper tests run anywhere. The numerical fused_moe_func test needs 8
TPU devices (gmm_v2 is a TPU kernel); CI runs it in the v7x multi-chip MoE step.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _make_mesh, _sharded_hidden_states, requires_8_devices)
from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...), as the other MoE tests do."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


@pytest.mark.parametrize("quantize_lhs", [True, False])
def test_gmm_wrapper_forwards_quantize_lhs(quantize_lhs):
    with mock.patch.object(fused_moe_gmm, "gmm_v2") as gmm_v2:
        fused_moe_gmm.gmm_wrapper(lhs="lhs",
                                  rhs="rhs",
                                  rhs_scale=None,
                                  rhs_bias=None,
                                  group_sizes="group_sizes",
                                  group_offset=[0],
                                  quantize_lhs=quantize_lhs)
    assert gmm_v2.call_args.kwargs["maybe_quantize_lhs"] is quantize_lhs


def test_gmm_wrapper_quantizes_lhs_by_default():
    with mock.patch.object(fused_moe_gmm, "gmm_v2") as gmm_v2:
        fused_moe_gmm.gmm_wrapper(lhs="lhs",
                                  rhs="rhs",
                                  rhs_scale=None,
                                  rhs_bias=None,
                                  group_sizes="group_sizes",
                                  group_offset=[0])
    assert gmm_v2.call_args.kwargs["maybe_quantize_lhs"] is True


def _quantize_per_channel(w: jax.Array, qtype) -> tuple[jax.Array, jax.Array]:
    """[E, K, N] -> (q [E, K, N], scale [E, 1, 1, N]) per output channel."""
    w32 = w.astype(jnp.float32)
    qmax = float(jnp.finfo(qtype).max)
    scale = jnp.max(jnp.abs(w32), axis=1, keepdims=True) / qmax
    q = jnp.clip(w32 / scale, -qmax, qmax).astype(qtype)
    return q, scale[:, :, None, :]


def _run(mesh, quantize_lhs: bool, w1, w2, w1_scale, w2_scale, hidden, gating):
    jax.clear_caches()
    try:
        with jax.set_mesh(mesh):
            out = fused_moe_gmm.fused_moe_func(hidden_states=hidden,
                                               w1=w1,
                                               w2=w2,
                                               w1_scale=w1_scale,
                                               w2_scale=w2_scale,
                                               w1_bias=None,
                                               w2_bias=None,
                                               gating_output=gating,
                                               topk=2,
                                               renormalize=True,
                                               mesh=mesh,
                                               use_ep=True,
                                               activation="silu",
                                               scoring_fn="softmax",
                                               quantize_lhs=quantize_lhs)
            return np.asarray(jax.block_until_ready(out), np.float32)
    finally:
        jax.clear_caches()


@requires_8_devices
@pytest.mark.parametrize("qtype", [jnp.float8_e4m3fn, jnp.float4_e2m1fn],
                         ids=["fp8", "fp4"])
def test_quantize_lhs_false_matches_dequantized_weights(qtype):
    """quantize_lhs=False (bf16 LHS) matches running the same quantized
    weights dequantized to bf16, much more closely than the default dynamic
    FP8 LHS quantization does."""
    mesh = _make_mesh(attn_dp=2, model=4)
    num_tokens, hidden_size, intermediate, num_experts = 512, 256, 512, 16
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    w1 = jax.random.normal(k1, (num_experts, hidden_size, 2 * intermediate),
                           jnp.bfloat16) / 10
    w2 = jax.random.normal(k2, (num_experts, intermediate, hidden_size),
                           jnp.bfloat16) / 10
    w1_q, w1_scale = _quantize_per_channel(w1, qtype)
    w2_q, w2_scale = _quantize_per_channel(w2, qtype)
    w1_dq = (w1_q.astype(jnp.float32) * w1_scale[:, :, 0, :]).astype(
        jnp.bfloat16)
    w2_dq = (w2_q.astype(jnp.float32) * w2_scale[:, :, 0, :]).astype(
        jnp.bfloat16)
    hidden = _sharded_hidden_states(mesh, num_tokens, hidden_size)
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    gating = jax.device_put(
        jax.random.normal(k3, (num_tokens, num_experts), jnp.bfloat16), rows)

    reference = _run(mesh, True, w1_dq, w2_dq, None, None, hidden, gating)
    bf16_lhs = _run(mesh, False, w1_q, w2_q, w1_scale, w2_scale, hidden,
                    gating)
    fp8_lhs = _run(mesh, True, w1_q, w2_q, w1_scale, w2_scale, hidden, gating)

    def rel_err(x):
        return np.linalg.norm(x - reference) / np.linalg.norm(reference)

    assert rel_err(bf16_lhs) < 1e-2
    assert rel_err(bf16_lhs) < rel_err(fp8_lhs) / 4


def test_quantize_lhs_false_rejects_fused_rs_kernel():
    mesh = _make_mesh()
    kwargs = dict(hidden_states=jnp.zeros((8, 256), jnp.bfloat16),
                  w1=jnp.zeros((2, 256, 1024), jnp.bfloat16),
                  w2=jnp.zeros((2, 512, 256), jnp.bfloat16),
                  w1_scale=None,
                  w2_scale=None,
                  w1_bias=None,
                  w2_bias=None,
                  gating_output=jnp.zeros((8, 2), jnp.bfloat16),
                  topk=1,
                  renormalize=True,
                  mesh=mesh,
                  use_ep=True,
                  activation="silu",
                  scoring_fn="softmax",
                  use_gmm_fused_rs_kernel=True,
                  quantize_lhs=False)
    with pytest.raises(ValueError, match="quantize_lhs=False"):
        fused_moe_gmm.fused_moe_func.__wrapped__(**kwargs)
