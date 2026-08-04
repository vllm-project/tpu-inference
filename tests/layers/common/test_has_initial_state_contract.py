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
"""Pins the `AttentionMetadata.has_initial_state` contract.

Linear-attention ops (GDN, KDA) need to know which persistent-batch slots
resume conv/recurrent state from an earlier step. The tempting derivation is
``(seq_lens - query_lens) > 0`` with
``query_lens = query_start_loc[1:] - query_start_loc[:-1]``. These tests pin
why that derivation is only conditionally valid, and that the runner-published
field is the unconditional answer.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.gdn.v3 import config, metadata


def _runner_layout(num_computed, num_scheduled, max_reqs_per_rank):
    """Emulates how `TPUModelRunner._prepare_inputs` packs metadata under DP.

    `num_computed` / `num_scheduled` are per-rank lists of per-request values.
    Returns `(query_start_loc, seq_lens, has_initial_state)` in the packed
    global layout: `query_start_loc` gets `max_reqs_per_rank + 1` entries per
    rank (each rank's block restarts at 0), while the other two get
    `max_reqs_per_rank`.
    """
    dp_size = len(num_computed)
    qsl = np.zeros(dp_size * (max_reqs_per_rank + 1), dtype=np.int32)
    seq_lens = np.zeros(dp_size * max_reqs_per_rank, dtype=np.int32)
    has_init = np.zeros(dp_size * max_reqs_per_rank, dtype=np.int32)
    for rank, (computed,
               scheduled) in enumerate(zip(num_computed, num_scheduled)):
        n = len(computed)
        qsl_off = rank * (max_reqs_per_rank + 1)
        req_off = rank * max_reqs_per_rank
        np.cumsum(scheduled, out=qsl[qsl_off + 1:qsl_off + n + 1])
        qsl[qsl_off + n + 1:qsl_off + max_reqs_per_rank + 1] = qsl[qsl_off + n]
        seq_lens[req_off:req_off +
                 n] = np.array(computed) + np.array(scheduled)
        has_init[req_off:req_off + n] = np.array(computed) > 0
    return qsl, seq_lens, has_init


def _rank_shard(packed, rank, per_rank):
    """One rank's view of a packed array, as `shard_map(in_specs=P(ATTN_DATA))`
    slices it: contiguous equal-sized blocks along axis 0."""
    return packed[rank * per_rank:(rank + 1) * per_rank]


# Two ranks hold requests mid-stream (nonzero computed tokens => resuming),
# two hold fresh prefills, and ranks are unevenly filled so padding is
# exercised. 8 ranks x 4 slots = 32 slots, mirroring max_num_seqs=32 at dp=8.
DP_SIZE = 8
MAX_REQS_PER_RANK = 4
NUM_COMPUTED = [[0, 0], [16, 0, 4], [0], [7, 7, 7, 7], [0, 32], [], [64], [0]]
NUM_SCHEDULED = [[8, 8], [1, 12, 1], [5], [1, 1, 1, 1], [3, 1], [], [1], [9]]


def test_global_derivation_has_the_wrong_shape_under_attention_dp():
    """The (32,) vs (39,) break: the derivation cannot even be evaluated."""
    qsl, seq_lens, _ = _runner_layout(NUM_COMPUTED, NUM_SCHEDULED,
                                      MAX_REQS_PER_RANK)

    # `query_start_loc` carries one extra entry *per rank*, not one overall.
    assert qsl.shape == (DP_SIZE * MAX_REQS_PER_RANK + DP_SIZE, ) == (40, )
    assert seq_lens.shape == (DP_SIZE * MAX_REQS_PER_RANK, ) == (32, )

    query_lens = qsl[1:] - qsl[:-1]
    assert query_lens.shape == (39, )
    with pytest.raises(TypeError,
                       match=r"incompatible shapes.*\(32,\).*\(39,\)"):
        _ = jnp.asarray(seq_lens) - jnp.asarray(query_lens)


def test_global_derivation_is_valid_per_dp_rank_shard():
    """Why the GDN kernel path never hit the break.

    `run_jax_gdn_attention` hands `query_start_loc` to a `shard_map` whose
    in_spec is `P(ATTN_DATA)`, so the kernel-side code sees one rank's block:
    `max_reqs_per_rank + 1` offsets restarting at 0, against
    `max_reqs_per_rank` sequence lengths. The shapes line up and the values
    are right. The break is reachable only from ops that run on the *packed*
    arrays, i.e. outside such a shard_map -- which is where the JAX-native
    model path (KDA) runs.
    """
    qsl, seq_lens, has_init = _runner_layout(NUM_COMPUTED, NUM_SCHEDULED,
                                             MAX_REQS_PER_RANK)

    for rank in range(DP_SIZE):
        qsl_r = _rank_shard(qsl, rank, MAX_REQS_PER_RANK + 1)
        seq_lens_r = _rank_shard(seq_lens, rank, MAX_REQS_PER_RANK)
        derived = (seq_lens_r - (qsl_r[1:] - qsl_r[:-1])) > 0
        np.testing.assert_array_equal(
            derived,
            _rank_shard(has_init, rank, MAX_REQS_PER_RANK).astype(bool),
            f"rank {rank}")


def test_published_flag_matches_the_per_rank_derivation_everywhere():
    """The migration is behavior-preserving for the paths that were correct."""
    qsl, seq_lens, has_init = _runner_layout(NUM_COMPUTED, NUM_SCHEDULED,
                                             MAX_REQS_PER_RANK)
    derived = np.concatenate([
        (_rank_shard(seq_lens, r, MAX_REQS_PER_RANK) -
         (_rank_shard(qsl, r, MAX_REQS_PER_RANK + 1)[1:] -
          _rank_shard(qsl, r, MAX_REQS_PER_RANK + 1)[:-1])) > 0
        for r in range(DP_SIZE)
    ])
    np.testing.assert_array_equal(derived, has_init.astype(bool))


def test_dp1_derivation_agrees_with_the_published_flag():
    """At dp_size == 1 the packed layout degenerates to the classic one, which
    is why the derivation shipped correct for so long."""
    qsl, seq_lens, has_init = _runner_layout([[0, 3, 20]], [[6, 2, 1]],
                                             max_reqs_per_rank=3)
    derived = (seq_lens - (qsl[1:] - qsl[:-1])) > 0
    np.testing.assert_array_equal(derived, has_init.astype(bool))
    np.testing.assert_array_equal(has_init, [0, 1, 1])


def _gdn_cfg(mode, batch_size, tile_size):
    return config.GDNConfig(
        mode=mode,
        batch_size=batch_size,
        kernel_size=4,
        tile_size=tile_size,
        dim_size=64,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=32,
        v_head_dim=32,
        dtypes=config.Dtypes(
            act_in=jnp.float32,
            act_out=jnp.float32,
            compute=jnp.float32,
            recurrent_state=jnp.float32,
            conv_state=jnp.float32,
        ),
    )


@pytest.mark.parametrize("flags", [[1, 0, 1], [0, 0, 0], [1, 1, 1]])
def test_gdn_batched_metadata_takes_the_flag_verbatim(flags):
    """`compute_batched_seq_metadata` must publish the caller's flag, not a
    re-derivation. All three sequences here have `query_len == 1`, so any
    derivation from `query_start_loc` alone would give the same answer for
    every slot regardless of `flags`."""
    num_seqs = len(flags)
    meta = metadata.compute_batched_seq_metadata(
        cfg=_gdn_cfg(config.GDNMode.BATCHED,
                     batch_size=num_seqs,
                     tile_size=num_seqs),
        has_initial_state=jnp.asarray(flags, jnp.int32),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32),
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        read_offsets=jnp.zeros(num_seqs, jnp.int32),
        end_seq=jnp.int32(num_seqs),
    )
    np.testing.assert_array_equal(np.asarray(meta.s_idx_has_initial_state),
                                  np.asarray(flags, bool))


@pytest.mark.parametrize("flags", [[1, 0], [0, 1]])
def test_gdn_per_seq_metadata_takes_the_flag_verbatim(flags):
    """Same for the prefill/mixed path, which additionally rolls the flag by
    `start_seq` so it stays aligned with the rolled sequence order."""
    num_seqs = len(flags)
    meta = metadata.compute_per_seq_metadata(
        cfg=_gdn_cfg(config.GDNMode.PER_SEQ, batch_size=8, tile_size=4),
        has_initial_state=jnp.asarray(flags, jnp.int32),
        query_start_loc=jnp.asarray([0, 4, 8], jnp.int32),
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        start_seq=jnp.int32(0),
        end_seq=jnp.int32(num_seqs),
    )
    np.testing.assert_array_equal(np.asarray(meta.s_idx_has_initial_state),
                                  np.asarray(flags, bool))
