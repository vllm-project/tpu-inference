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
"""Tests for MOE_TC_PERMUTE, the EP permute and combine on the TensorCore.

The capacity tests run anywhere. The fused_moe_func tests need 8 TPU devices
(gmm_v2 is a TPU kernel); their reference is the one-hot permute path
(ONEHOT_MOE_PERMUTE_THRESHOLD), which doesn't use the SparseCore either.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _MESH_SIZES, _make_mesh, _mesh_id, _sharded_hidden_states,
    requires_8_devices)
from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.utils import get_mesh_shape_product


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...) the other MoE tests use."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


@pytest.mark.parametrize(
    "num_rows, num_shards, factor, expected",
    [
        (3840, 32, 2.0, 512),  # 384 decode tokens x top-10: the 512 floor
        (40960, 32, 2.0, 2560),  # a 4096-token prefill step: 2 x 1280
        (40960, 32, 3.0, 3840),
        (1000, 8, 2.0, 512),
        (256, 8, 2.0, 256),  # never more than all the rows
    ])
def test_capacity(num_rows, num_shards, factor, expected):
    with mock.patch("tpu_inference.envs.MOE_TC_PERMUTE_CAPACITY_FACTOR",
                    factor):
        assert fused_moe_gmm.tc_permute_capacity(num_rows,
                                                 num_shards) == expected


@pytest.mark.parametrize(
    "num_tokens, num_rows, hot_block",
    [
        (4096, 1024, None),  # rows spread over the blocks: one window each
        (2304, 768, None),  # tokens not a multiple of the 512-token block
        (4096, 1024, 3),  # most rows in block 3: several windows
    ])
def test_banded_combine(num_tokens, num_rows, hot_block):
    """_tc_banded_combine against a scatter-add of the weighted rows."""
    rng = np.random.default_rng(0)
    row_tokens = rng.integers(0, num_tokens, num_rows)
    if hot_block is not None:
        hot = rng.integers(hot_block * 512, (hot_block + 1) * 512, num_rows)
        row_tokens = np.where(rng.random(num_rows) < 0.75, hot, row_tokens)
    row_valid = rng.random(num_rows) < 0.9
    rows = jnp.asarray(rng.standard_normal((num_rows, 256)), jnp.bfloat16)
    weights = jnp.asarray(rng.random(num_rows), jnp.float32)
    combine = jax.jit(fused_moe_gmm._tc_banded_combine, static_argnums=4)
    out = combine(rows, jnp.asarray(row_tokens, jnp.int32), weights,
                  jnp.asarray(row_valid), num_tokens)
    weighted = (np.asarray(rows, np.float32) *
                np.asarray(weights.astype(jnp.bfloat16), np.float32)[:, None])
    expected = np.zeros((num_tokens, 256), np.float32)
    np.add.at(expected, row_tokens[row_valid], weighted[row_valid])
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)


def _moe_inputs(mesh: Mesh,
                num_tokens: int,
                skew: bool,
                hidden: int = 256,
                intermediate: int = 512,
                num_experts: int = 16):
    k1, k2, k3 = jax.random.split(jax.random.key(3), 3)
    gating = jax.random.normal(k3, (num_tokens, num_experts), jnp.float32)
    if skew:
        # Every token picks the first shard's experts, so that shard has all
        # the rows: more than the compact buffer holds.
        per_shard = num_experts // get_mesh_shape_product(
            mesh, ShardingAxisName.EXPERT)
        gating = gating.at[:, :per_shard].add(10.0)
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    return dict(
        hidden_states=_sharded_hidden_states(mesh, num_tokens, hidden),
        w1=jax.random.normal(k1, (num_experts, hidden, 2 * intermediate),
                             jnp.bfloat16) / 10,
        w2=jax.random.normal(k2, (num_experts, intermediate, hidden),
                             jnp.bfloat16) / 10,
        gating_output=jax.device_put(gating.astype(jnp.bfloat16), rows),
    )


def _run_fused_moe(mesh: Mesh, tc_permute: bool, num_tokens: int,
                   skew: bool) -> jax.Array:
    """fused_moe_func with MOE_TC_PERMUTE, or else the one-hot permute. The
    flag is read at trace time, so caches are cleared around the run."""
    jax.clear_caches()
    try:
        with (mock.patch("tpu_inference.envs.MOE_TC_PERMUTE",
                         tc_permute), jax.set_mesh(mesh)):
            out = fused_moe_gmm.fused_moe_func(
                **_moe_inputs(mesh, num_tokens, skew),
                w1_scale=None,
                w2_scale=None,
                w1_bias=None,
                w2_bias=None,
                topk=2,
                renormalize=True,
                mesh=mesh,
                use_ep=True,
                activation="silu",
                scoring_fn="softmax",
                onehot_moe_permute_threshold=0 if tc_permute else 10**8,
            )
            return jax.block_until_ready(out)
    finally:
        jax.clear_caches()


@requires_8_devices
@pytest.mark.parametrize("skew", [False, True], ids=["compact", "fallback"])
@pytest.mark.parametrize(
    "num_tokens",
    [512, 1536, 4096, 4352],
    ids=[
        "onehot-gather", "plain-gather", "banded-combine",
        "banded-combine-partial-block"
    ],
)
@pytest.mark.parametrize("sizes", _MESH_SIZES, ids=_mesh_id)
def test_fused_moe_matches_onehot_permute(sizes, num_tokens, skew):
    """Same output as the one-hot permute. 512 tokens x top-2 over 8 shards
    gives a 512-row buffer, 1536 tokens a 768-row one; from 4096 tokens the
    combine is banded. With skew one shard has every row and takes the
    all-rows fallback (combined in 2-4 chunks)."""
    mesh = _make_mesh(**sizes)
    expected = _run_fused_moe(mesh, False, num_tokens, skew)
    actual = _run_fused_moe(mesh, True, num_tokens, skew)
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               np.asarray(expected, np.float32),
                               rtol=2e-2,
                               atol=2e-2)
