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
"""Structure-level tests for sparse_moe_func's shard_map spec construction.

The non-qwix quantized path (MXFP4 experts arriving pre-quantized from the
checkpoint) must pass each expert kernel to shard_map as a (value, scale)
pair with a matching pair of PartitionSpecs; the unquantized path must pass
plain arrays with single specs. These tests stub shard_map itself, so they
run on CPU and pin the argument/in_specs pytree structure rather than the
numerics (covered by the distributed backend tests).
"""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.layers.common.process_weights.moe_weights import \
    UnfusedMoEWeights
from tpu_inference.layers.jax.moe import sparse_moe

T, D, E, F, TOPK = 4, 8, 4, 16, 2


def _layer():
    return SimpleNamespace(
        qwix_quantized_weight_dtype=None,
        edf_sharding=(None, None, None),
        efd_sharding=(None, None, None),
        activation_ffw_td=(None, None),
    )


def _weights(with_scales: tuple[bool, bool, bool]) -> UnfusedMoEWeights:
    w_edf = jnp.zeros((E, D, F), dtype=jnp.bfloat16)
    w_efd = jnp.zeros((E, F, D), dtype=jnp.bfloat16)
    scale_edf = jnp.ones((E, 1, F), dtype=jnp.float32)
    scale_efd = jnp.ones((E, 1, D), dtype=jnp.float32)
    return UnfusedMoEWeights(
        w1_weight=w_edf,
        w1_weight_scale=scale_edf if with_scales[0] else None,
        w1_bias=None,
        w2_weight=w_edf,
        w2_weight_scale=scale_edf if with_scales[1] else None,
        w2_bias=None,
        w3_weight=w_efd,
        w3_weight_scale=scale_efd if with_scales[2] else None,
        w3_bias=None,
    )


def _call_with_stubbed_shard_map(weights):
    """Run sparse_moe_func with shard_map stubbed out; capture its inputs."""
    captured = {}

    def fake_shard_map(f, mesh=None, in_specs=None, out_specs=None, **kwargs):
        captured["in_specs"] = in_specs

        def mapped(*args):
            captured["args"] = args
            return jnp.zeros((T, D), dtype=jnp.bfloat16)

        return mapped

    x_TD = jnp.zeros((T, D), dtype=jnp.bfloat16)
    gating = (jnp.zeros(
        (T, TOPK), dtype=jnp.float32), jnp.zeros((T, TOPK), dtype=jnp.int32))
    mesh = Mesh(jax.devices()[:1], ("x", ))
    with mock.patch.object(jax.experimental.shard_map, "shard_map",
                           fake_shard_map):
        sparse_moe.sparse_moe_func(weights, x_TD, gating, _layer(), mesh)
    return captured


def test_checkpoint_quantized_weights_become_value_scale_pairs():
    captured = _call_with_stubbed_shard_map(_weights((True, True, True)))

    # in_specs: (layer, x_TD, router_weights, indices, w1, w2, w3) -- each
    # quantized kernel spec is a pair of PartitionSpecs, value + scale.
    for kernel_spec in captured["in_specs"][4:7]:
        assert isinstance(kernel_spec, tuple) and len(kernel_spec) == 2
        assert all(isinstance(s, PartitionSpec) for s in kernel_spec)

    # The kernel arguments mirror that structure as (value, scale) tuples.
    for kernel_arg in captured["args"][4:7]:
        assert isinstance(kernel_arg, tuple) and len(kernel_arg) == 2


def test_unquantized_weights_stay_plain_arrays():
    captured = _call_with_stubbed_shard_map(_weights((False, False, False)))

    for kernel_spec in captured["in_specs"][4:7]:
        assert isinstance(kernel_spec, PartitionSpec)
    for kernel_arg in captured["args"][4:7]:
        assert isinstance(kernel_arg, jax.Array)


@pytest.mark.parametrize("with_scales", [
    (True, False, False),
    (True, True, False),
    (False, False, True),
])
def test_partial_weight_scales_fail_fast(with_scales):
    with pytest.raises(AssertionError, match=r"\[moe\].*all three"):
        _call_with_stubbed_shard_map(_weights(with_scales))
