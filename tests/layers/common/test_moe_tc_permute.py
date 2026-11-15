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
    """Use the multi-axis names (attn_dp, ...), as the other MoE tests do."""
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
        (40960, 32, 1.1, 1408),  # 1.1 x 1280 is 1408 plus float error
    ])
def test_capacity(num_rows, num_shards, factor, expected):
    with mock.patch("tpu_inference.envs.MOE_TC_PERMUTE_CAPACITY_FACTOR",
                    factor):
        assert fused_moe_gmm.tc_permute_capacity(num_rows,
                                                 num_shards) == expected


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_capacity_rejects_non_positive_factor(factor):
    with (mock.patch("tpu_inference.envs.MOE_TC_PERMUTE_CAPACITY_FACTOR",
                     factor), pytest.raises(ValueError)):
        fused_moe_gmm.tc_permute_capacity(40960, 32)


def _banded_case(rng, num_tokens, num_rows, hot_block):
    """Random inputs for _tc_banded_combine, and its expected output from a
    scatter-add. With hot_block, most rows go to that 512-token block."""
    row_tokens = rng.integers(0, num_tokens, num_rows)
    if hot_block is not None:
        hot = rng.integers(hot_block * 512, (hot_block + 1) * 512, num_rows)
        row_tokens = np.where(rng.random(num_rows) < 0.75, hot, row_tokens)
    row_valid = rng.random(num_rows) < 0.9
    rows = jnp.asarray(rng.standard_normal((num_rows, 256)), jnp.bfloat16)
    weights = jnp.asarray(rng.random(num_rows), jnp.float32)
    weighted = (np.asarray(rows, np.float32) *
                np.asarray(weights.astype(jnp.bfloat16), np.float32)[:, None])
    expected = np.zeros((num_tokens, 256), np.float32)
    np.add.at(expected, row_tokens[row_valid], weighted[row_valid])
    args = (rows, jnp.asarray(row_tokens,
                              jnp.int32), weights, jnp.asarray(row_valid))
    return args, expected


@pytest.mark.parametrize(
    "num_tokens, num_rows, hot_block",
    [
        (4096, 1024, None),  # rows spread over the blocks: one window each
        (2304, 768, None),  # tokens not a multiple of the 512-token block
        (4096, 1024, 3),  # most rows in block 3: several windows
    ])
def test_banded_combine(num_tokens, num_rows, hot_block):
    """_tc_banded_combine matches a scatter-add, alone and when adding into
    an accumulator that holds another chunk's result."""
    rng = np.random.default_rng(0)
    args, expected = _banded_case(rng, num_tokens, num_rows, hot_block)
    args2, expected2 = _banded_case(rng, num_tokens, num_rows, hot_block)
    combine = jax.jit(fused_moe_gmm._tc_banded_combine, static_argnums=4)
    out = combine(*args, num_tokens)
    assert out.shape == fused_moe_gmm._tc_banded_blocks(num_tokens) + (256, )
    assert out.dtype == jnp.bfloat16

    # bf16 sums: one rounding for the result, one more per added chunk.

    def flat(x):
        return np.asarray(x, np.float32).reshape(-1, 256)[:num_tokens]

    np.testing.assert_allclose(flat(out), expected, rtol=1e-2, atol=1e-2)
    both = combine(*args2, num_tokens, out)
    np.testing.assert_allclose(flat(both),
                               expected + expected2,
                               rtol=1e-2,
                               atol=2e-2)


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
    """Runs fused_moe_func with MOE_TC_PERMUTE on, or off (the one-hot
    permute). The flag is read at trace time, so JAX caches are cleared
    around the run."""
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
@pytest.mark.parametrize("skew", [False, True], ids=["compact", "overflow"])
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
    """MOE_TC_PERMUTE gives the same output as the one-hot permute.

    512 tokens use the one-hot gather, 1536 the plain gather, and 4096 and
    4352 the banded combine (4352 ends in a partial block). With skew, one
    shard gets every row, more than its buffer holds, so it loops over 2-4
    buffer-fulls. On the data=2 mesh each data shard gets half the tokens: at
    512 its 512 rows fit the smallest buffer, so nothing is compacted, and
    the combine is banded only from 8192 tokens (see
    test_fused_moe_banded_data_mesh).

    Without skew the output is bit-identical. With skew, the shard rounds each
    chunk's sum to bf16 before adding the next, so the output can move a
    little more; hence the looser atol.
    """
    mesh = _make_mesh(**sizes)
    expected = _run_fused_moe(mesh, False, num_tokens, skew)
    actual = _run_fused_moe(mesh, True, num_tokens, skew)
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               np.asarray(expected, np.float32),
                               rtol=2e-2,
                               atol=6e-2 if skew else 2e-2)


@requires_8_devices
@pytest.mark.parametrize("skew", [False, True], ids=["compact", "overflow"])
def test_fused_moe_banded_data_mesh(skew):
    """The banded combine on the data=2 mesh: 8192 tokens give each data
    shard 4096, where the banded combine starts."""
    test_fused_moe_matches_onehot_permute(dict(data=2, attn_dp=2, model=2),
                                          8192, skew)
