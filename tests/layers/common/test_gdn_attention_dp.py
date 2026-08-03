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
"""GDN under attention data parallelism.

`run_jax_gdn_attention` shards its ragged-batch metadata over
`ShardingAxisName.ATTN_DATA`, so each rank's kernel invocation must see
exactly what it would see running alone. This pins that equivalence for the
runner-published `has_initial_state` in particular: it arrives as one global
`(max_num_seqs,)` array and is sliced per rank by the shard_map, unlike
`query_start_loc`, whose per-rank blocks carry an extra entry each.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_inference.kernels.gdn.v3 import wrapper
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)

DP_SIZE = 2
REQS_PER_RANK = 2
BLOCKS_PER_RANK = 3  # slot 0 is the null block
TOKENS_PER_RANK = 16
# Two prefills per rank; the tail of each rank's token budget is padding.
LENGTHS = [8, 4]

N_KQ = 2
N_V = 2
D_K = 128
D_V = 128
KERNEL_SIZE = 4
KQ_DIM = N_KQ * D_K
CONV_DIM = 2 * KQ_DIM + N_V * D_V


@pytest.fixture
def dp_mesh():
    """A mesh whose only nontrivial axis is `attn_dp`, so ATTN_DATA splits
    2 ways and every other sharding axis is a no-op."""
    if len(jax.devices()) < DP_SIZE:
        pytest.skip(f"needs >= {DP_SIZE} devices")
    shape = tuple(DP_SIZE if name == "attn_dp" else 1
                  for name in MESH_AXIS_NAMES)
    devices = np.array(jax.devices()[:DP_SIZE]).reshape(shape)
    # `run_jax_gdn_attention` partitions on the multi-axis names, which only
    # the base (non-2D) sharding class defines.
    saved = sharding_mod.ShardingAxisName._cls
    sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase
    try:
        # Deliberately NOT installed as the default mesh: the single-device
        # reference calls below must stay unsharded, and a default multi-device
        # mesh makes XLA try to auto-partition their Mosaic kernel (which it
        # cannot). `run_jax_gdn_attention` takes the mesh explicitly for its
        # shard_map, so nothing here needs it ambient.
        yield Mesh(devices, axis_names=MESH_AXIS_NAMES)
    finally:
        sharding_mod.ShardingAxisName._cls = saved


def _rank_inputs(rank, has_init_flags):
    """Everything one rank's kernel call consumes, in rank-local indexing."""
    rngs = iter(jax.random.split(jax.random.key(100 + rank), 8))
    qkv = jax.random.normal(next(rngs), (TOKENS_PER_RANK, CONV_DIM))
    b = jax.random.normal(next(rngs), (TOKENS_PER_RANK, N_V))
    a = jax.random.normal(next(rngs), (TOKENS_PER_RANK, N_V))
    # Nonzero slot contents, so resuming and zero-initializing differ.
    conv_state = jax.random.normal(
        next(rngs), (BLOCKS_PER_RANK, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state = jax.random.normal(next(rngs),
                                        (BLOCKS_PER_RANK, N_V, D_K, D_V))
    conv_state = conv_state.at[0].set(0.0)  # null block
    recurrent_state = recurrent_state.at[0].set(0.0)

    qsl = jnp.asarray([0] + list(np.cumsum(LENGTHS)), jnp.int32)
    return dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        query_start_loc=qsl,
        # Slots 1..REQS_PER_RANK; rank-local, as `_prepare_inputs` publishes.
        state_indices=jnp.arange(1, REQS_PER_RANK + 1, dtype=jnp.int32),
        distribution=jnp.asarray([0, REQS_PER_RANK, REQS_PER_RANK], jnp.int32),
        has_initial_state=jnp.asarray(has_init_flags, jnp.int32),
    )


def _weights():
    rngs = iter(jax.random.split(jax.random.key(7), 4))
    return dict(
        conv_weight=jax.random.normal(next(rngs), (CONV_DIM, 1, KERNEL_SIZE)),
        conv_bias=jax.random.normal(next(rngs), (CONV_DIM, )),
        a_log=jax.random.normal(next(rngs), (N_V, )),
        dt_bias=jax.random.normal(next(rngs), (N_V, )),
    )


# Rank 0 resumes its first request and starts its second fresh; rank 1 is the
# mirror image. Any rank-agnostic handling of the flag gets one of them wrong.
FLAGS = [[1, 0], [0, 1]]


def test_attention_dp_gdn_matches_per_rank_single_device_runs(dp_mesh):
    weights = _weights()
    per_rank = [_rank_inputs(r, FLAGS[r]) for r in range(DP_SIZE)]

    # Reference: each rank on its own, no sharding involved.
    ref = []
    for inputs in per_rank:
        (conv_out, rec_out), out = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )(inputs["qkv"],
          inputs["b"],
          inputs["a"],
          inputs["conv_state"],
          inputs["recurrent_state"],
          weights["conv_weight"],
          weights["conv_bias"],
          weights["a_log"],
          weights["dt_bias"],
          inputs["query_start_loc"],
          inputs["state_indices"],
          inputs["distribution"],
          inputs["has_initial_state"],
          n_kq=N_KQ,
          n_v=N_V,
          d_k=D_K,
          d_v=D_V,
          kernel_size=KERNEL_SIZE)
        ref.append(
            (np.asarray(conv_out), np.asarray(rec_out), np.asarray(out)))

    # Under test: one sharded call over the globally packed metadata. Note
    # `query_start_loc` concatenates REQS_PER_RANK + 1 entries per rank while
    # `has_initial_state` concatenates REQS_PER_RANK -- the asymmetry that
    # makes a global `seq_lens - query_lens` derivation invalid.
    def cat(key):
        return jnp.concatenate([r[key] for r in per_rank], axis=0)

    packed_qsl = cat("query_start_loc")
    packed_flags = cat("has_initial_state")
    assert packed_qsl.shape == (DP_SIZE * (REQS_PER_RANK + 1), )
    assert packed_flags.shape == (DP_SIZE * REQS_PER_RANK, )

    (conv_out, rec_out), out = run_jax_gdn_attention(
        cat("qkv"),
        cat("b"),
        cat("a"),
        cat("conv_state"),
        cat("recurrent_state"),
        weights["conv_weight"],
        weights["conv_bias"],
        weights["a_log"],
        weights["dt_bias"],
        cat("state_indices"),
        packed_qsl,
        cat("distribution"),
        packed_flags,
        N_KQ,
        N_V,
        D_K,
        D_V,
        KERNEL_SIZE,
        mesh=dp_mesh,
    )

    conv_out, rec_out, out = map(np.asarray, (conv_out, rec_out, out))
    for rank, (ref_conv, ref_rec, ref_out) in enumerate(ref):
        tok = slice(rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK)
        blk = slice(rank * BLOCKS_PER_RANK, (rank + 1) * BLOCKS_PER_RANK)
        np.testing.assert_allclose(out[tok],
                                   ref_out,
                                   rtol=2e-2,
                                   atol=2e-2,
                                   err_msg=f"output, rank {rank}")
        np.testing.assert_allclose(conv_out[blk],
                                   ref_conv,
                                   rtol=2e-2,
                                   atol=2e-2,
                                   err_msg=f"conv state, rank {rank}")
        np.testing.assert_allclose(rec_out[blk],
                                   ref_rec,
                                   rtol=2e-2,
                                   atol=2e-2,
                                   err_msg=f"recurrent state, rank {rank}")


def test_flipping_one_ranks_flag_changes_only_that_rank(dp_mesh):
    """Guards against the flag being read rank-agnostically (e.g. broadcast
    from rank 0): rank 1's flags change, rank 0's output must not."""
    weights = _weights()

    def cat(rows, key):
        return jnp.concatenate([r[key] for r in rows], axis=0)

    outs = []
    for rank1_flags in ([0, 1], [1, 1]):
        rows = [_rank_inputs(0, FLAGS[0]), _rank_inputs(1, rank1_flags)]
        _, out = run_jax_gdn_attention(
            cat(rows, "qkv"),
            cat(rows, "b"),
            cat(rows, "a"),
            cat(rows, "conv_state"),
            cat(rows, "recurrent_state"),
            weights["conv_weight"],
            weights["conv_bias"],
            weights["a_log"],
            weights["dt_bias"],
            cat(rows, "state_indices"),
            cat(rows, "query_start_loc"),
            cat(rows, "distribution"),
            cat(rows, "has_initial_state"),
            N_KQ,
            N_V,
            D_K,
            D_V,
            KERNEL_SIZE,
            mesh=dp_mesh,
        )
        outs.append(np.asarray(out))

    rank0 = slice(0, TOKENS_PER_RANK)
    rank1_first_seq = slice(TOKENS_PER_RANK, TOKENS_PER_RANK + LENGTHS[0])
    np.testing.assert_allclose(outs[0][rank0], outs[1][rank0], rtol=0, atol=0)
    assert not np.allclose(
        outs[0][rank1_first_seq],
        outs[1][rank1_first_seq],
        rtol=1e-3,
        atol=1e-3), ("rank 1's first request resumed vs zeroed its "
                     "state but produced identical output")
