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
"""KDA under attention data parallelism.

Under attention DP the runner publishes ragged-batch metadata as a per-rank
concatenation: `query_start_loc` contributes `reqs_per_rank + 1` entries per
rank (offsets restarting at 0) while `seq_lens` / `has_initial_state` /
`mamba_state_indices` contribute `reqs_per_rank`. Evaluated on that packed
array, `query_start_loc[1:] - query_start_loc[:-1]` is neither the right
length nor the right values, which is why `kda_attention` runs its
state-carrying part inside a `shard_map` over `ShardingAxisName.ATTN_DATA` —
the same treatment `run_jax_gdn_attention` gives the GDN kernel.

Slot ids in the packed metadata are rank-local: the runner rebases the
globally allocated slot ids per DP rank (`global % local_slots` in
`TPURunner._prepare_inputs`) before publishing
`AttentionMetadata.mamba_state_indices`, so each rank's slice of the packed
array indexes that rank's shard of the state pool directly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.kda_attention import (KDAParams, KDAState,
                                                       kda_attention)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)

DP_SIZE = 2
REQS_PER_RANK = 2
BLOCKS_PER_RANK = 3  # slot 0 is the null block
TOKENS_PER_RANK = 16
LENGTHS = [8, 4]  # two prefills per rank; the rest of the block is padding

H, D, W = 2, 32, 4
HP = H * D
HIDDEN = 64
EPS = 1e-5
LOWER_BOUND = -5.0


@pytest.fixture
def dp_mesh():
    """A mesh whose only nontrivial axis is `attn_dp`, so ATTN_DATA splits
    2 ways and every other sharding axis is a no-op."""
    if len(jax.devices()) < DP_SIZE:
        pytest.skip(f"needs >= {DP_SIZE} devices")
    shape = tuple(DP_SIZE if name == "attn_dp" else 1
                  for name in MESH_AXIS_NAMES)
    devices = np.array(jax.devices()[:DP_SIZE]).reshape(shape)
    # `kda_attention` partitions on the multi-axis names, which only the base
    # (non-2D) sharding class defines. Not installed as the default mesh: the
    # single-rank reference calls below must stay unsharded.
    saved = sharding_mod.ShardingAxisName._cls
    sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase
    try:
        yield Mesh(devices, axis_names=MESH_AXIS_NAMES)
    finally:
        sharding_mod.ShardingAxisName._cls = saved


def _params():
    rngs = iter(jax.random.split(jax.random.key(11), 16))

    def n(shape, scale=0.05):
        return jax.random.normal(next(rngs), shape, jnp.float32) * scale

    return KDAParams(
        q_proj=n((HP, HIDDEN)),
        k_proj=n((HP, HIDDEN)),
        v_proj=n((HP, HIDDEN)),
        q_conv_weight=n((HP, 1, W)),
        k_conv_weight=n((HP, 1, W)),
        v_conv_weight=n((HP, 1, W)),
        A_log=n((H, ), 1.0),
        dt_bias=n((HP, ), 1.0),
        f_a_proj=n((D, HIDDEN)),
        f_b_proj=n((HP, D)),
        b_proj=n((H, HIDDEN)),
        o_norm_weight=jnp.ones((D, ), jnp.float32),
        o_proj=n((HIDDEN, HP)),
        g_proj=n((HP, HIDDEN)),
    )


def _rank_inputs(rank, has_init_flags):
    """One rank's view: tokens, its shard of the state pool, its metadata.

    Slot ids are rank-local (slots 1..REQS_PER_RANK), exactly as the runner
    publishes them in `AttentionMetadata.mamba_state_indices` (it rebases
    global slot ids per DP rank before publishing).
    """
    rngs = iter(jax.random.split(jax.random.key(200 + rank), 8))
    x = jax.random.normal(next(rngs), (TOKENS_PER_RANK, HIDDEN), jnp.float32)

    def state(shape):
        # Nonzero slot contents, so resuming and zero-initializing differ.
        s = jax.random.normal(next(rngs), shape, jnp.float32) * 0.1
        return s.at[0].set(0.0)  # null block

    return dict(
        x=x,
        state=KDAState(
            conv_q=state((BLOCKS_PER_RANK, W - 1, HP)),
            conv_k=state((BLOCKS_PER_RANK, W - 1, HP)),
            conv_v=state((BLOCKS_PER_RANK, W - 1, HP)),
            recurrent=state((BLOCKS_PER_RANK, H, D, D)),
        ),
        query_start_loc=jnp.asarray([0] + list(np.cumsum(LENGTHS)), jnp.int32),
        state_indices=jnp.arange(1, REQS_PER_RANK + 1, dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, REQS_PER_RANK], jnp.int32),
        has_initial_state=jnp.asarray(has_init_flags, jnp.int32),
    )


def _call(inputs, params, mesh=None):
    return kda_attention(inputs["x"],
                         params,
                         inputs["state"],
                         inputs["state_indices"],
                         inputs["query_start_loc"],
                         inputs["distribution"],
                         inputs["has_initial_state"].astype(jnp.bool_),
                         num_heads=H,
                         head_dim=D,
                         kernel_size=W,
                         gate_lower_bound=LOWER_BOUND,
                         rms_norm_eps=EPS,
                         mesh=mesh)


# Rank 0 resumes its first request and starts its second fresh; rank 1 is the
# mirror image. Any rank-agnostic handling of the flag gets one of them wrong.
FLAGS = [[1, 0], [0, 1]]


def _pack(per_rank, key):
    return jnp.concatenate([r[key] for r in per_rank], axis=0)


def _packed_state(per_rank):
    return KDAState(*[
        jnp.concatenate([getattr(r["state"], f) for r in per_rank], axis=0)
        for f in KDAState._fields
    ])


def test_kda_attention_dp_matches_per_rank_single_device_runs(dp_mesh):
    """The sharded call over packed metadata == each rank running alone."""
    params = _params()
    per_rank = [_rank_inputs(r, FLAGS[r]) for r in range(DP_SIZE)]

    # Reference: each rank alone, on the same rank-local slot ids.
    ref = []
    for rank in range(DP_SIZE):
        local = dict(_rank_inputs(rank, FLAGS[rank]))
        new_state, out = jax.jit(lambda i, p: _call(i, p))(local, params)
        ref.append((new_state, np.asarray(out)))

    packed = dict(
        x=_pack(per_rank, "x"),
        state=_packed_state(per_rank),
        query_start_loc=_pack(per_rank, "query_start_loc"),
        state_indices=_pack(per_rank, "state_indices"),
        distribution=_pack(per_rank, "distribution"),
        has_initial_state=_pack(per_rank, "has_initial_state"),
    )
    # The asymmetry that makes a global `query_start_loc` diff invalid: one
    # extra offset entry per rank.
    assert packed["query_start_loc"].shape == (DP_SIZE * (REQS_PER_RANK + 1), )
    assert packed["has_initial_state"].shape == (DP_SIZE * REQS_PER_RANK, )

    new_state, out = _call(packed, params, mesh=dp_mesh)
    out = np.asarray(out)

    for rank, (ref_state, ref_out) in enumerate(ref):
        tok = slice(rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK)
        blk = slice(rank * BLOCKS_PER_RANK, (rank + 1) * BLOCKS_PER_RANK)
        np.testing.assert_allclose(out[tok],
                                   ref_out,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   err_msg=f"output, rank {rank}")
        for field in KDAState._fields:
            np.testing.assert_allclose(np.asarray(getattr(new_state,
                                                          field))[blk],
                                       np.asarray(getattr(ref_state, field)),
                                       rtol=1e-5,
                                       atol=1e-5,
                                       err_msg=f"{field}, rank {rank}")


def test_kda_attention_dp_isolates_ranks(dp_mesh):
    """Flipping one rank's resume flag changes that rank's output only."""
    params = _params()

    def run(flags):
        per_rank = [_rank_inputs(r, flags[r]) for r in range(DP_SIZE)]
        packed = dict(
            x=_pack(per_rank, "x"),
            state=_packed_state(per_rank),
            query_start_loc=_pack(per_rank, "query_start_loc"),
            state_indices=_pack(per_rank, "state_indices"),
            distribution=_pack(per_rank, "distribution"),
            has_initial_state=_pack(per_rank, "has_initial_state"),
        )
        _, out = _call(packed, params, mesh=dp_mesh)
        return np.asarray(out)

    base = run(FLAGS)
    flipped = run([FLAGS[0], [1, 1]])  # rank 1 now resumes both requests

    r0 = slice(0, TOKENS_PER_RANK)
    r1 = slice(TOKENS_PER_RANK, 2 * TOKENS_PER_RANK)
    np.testing.assert_array_equal(base[r0], flipped[r0])
    assert not np.allclose(base[r1], flipped[r1]), (
        "rank 1's resume flag had no effect on rank 1's output")


def test_kda_attention_packed_metadata_needs_the_shard_map():
    """Without the shard_map the packed layout is a shape error, not a
    silently wrong number — this pins the failure the 48B hit."""
    params = _params()
    per_rank = [_rank_inputs(r, FLAGS[r]) for r in range(DP_SIZE)]
    packed = dict(
        x=_pack(per_rank, "x"),
        state=_packed_state(per_rank),
        query_start_loc=_pack(per_rank, "query_start_loc"),
        state_indices=_pack(per_rank, "state_indices"),
        distribution=_pack(per_rank, "distribution"),
        has_initial_state=_pack(per_rank, "has_initial_state"),
    )
    with pytest.raises(TypeError, match="concatenate|broadcast"):
        _call(packed, params, mesh=None)
