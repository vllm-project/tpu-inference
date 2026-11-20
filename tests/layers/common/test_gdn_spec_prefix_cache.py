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
"""Mamba prefix caching (align mode) + speculative decoding for GDN layers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import (
    MAMBA_READ_OFFSET_FROM_BLOCK_TABLE, AttentionMetadata, MambaMetadata)
from tpu_inference.layers.common.gdn_attention import (
    compute_align_spec_state_indices, copy_gdn_state_slots,
    resolve_gdn_state_slots)
from tpu_inference.layers.common.sharding import ShardingAxisName

BLOCK_SIZE = 16


def _axis_names(axis) -> tuple[str, ...]:
    return (axis, ) if isinstance(axis, str) else tuple(axis)


@pytest.fixture
def mesh():
    """Single-device mesh carrying every axis the GDN shard_maps name."""
    names = []
    for axis in (ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD):
        for name in _axis_names(axis):
            if name not in names:
                names.append(name)
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(names))
    return Mesh(devices, tuple(names))


def _indices(block_tables,
             query_lens,
             seq_lens,
             is_window,
             scratch_bases,
             read_offsets,
             num_local_rows=100):
    out = compute_align_spec_state_indices(
        jnp.asarray(block_tables, dtype=jnp.int32),
        jnp.asarray(query_lens, dtype=jnp.int32),
        jnp.asarray(seq_lens, dtype=jnp.int32),
        jnp.asarray(is_window, dtype=bool),
        jnp.asarray(scratch_bases, dtype=jnp.int32),
        jnp.asarray(read_offsets, dtype=jnp.int32),
        BLOCK_SIZE,
        num_local_rows=num_local_rows,
    )
    return [np.asarray(x) for x in out]


def test_window_resumes_from_scratch_and_copies_boundary():
    """A window that resumes from its scratch checkpoint and covers a block
    boundary copies the boundary checkpoint into the block of that boundary.
    """
    # 14 computed tokens + a 4-token window (seq_len 18) crosses 16 at window
    # position 16 - 14 - 1 = 1.
    state, read, src, dst, mask = _indices(
        block_tables=[[10, 11]],
        query_lens=[4],
        seq_lens=[18],
        is_window=[True],
        scratch_bases=[40],
        read_offsets=[2],
    )
    assert state.tolist() == [40]  # checkpoints go to the scratch group
    assert read.tolist() == [42]  # scratch base + read offset
    assert mask.tolist() == [True]
    assert src.tolist() == [41]
    assert dst.tolist() == [10]  # block holding tokens [0, 16)


def test_window_resumes_from_block_after_prefill():
    """MAMBA_READ_OFFSET_FROM_BLOCK_TABLE (after a prefill chunk or a prefix
    hit) resumes from the block covering the last computed token; no boundary
    inside the window means no copy."""
    state, read, _, _, mask = _indices(
        block_tables=[[20, 21, 22]],
        query_lens=[2],
        seq_lens=[34],
        is_window=[True],
        scratch_bases=[44],
        read_offsets=[MAMBA_READ_OFFSET_FROM_BLOCK_TABLE],
    )
    assert state.tolist() == [44]
    assert read.tolist() == [21]  # token 31 lives in block column 1
    assert mask.tolist() == [False]


def test_window_ending_on_boundary_copies_last_position():
    """A window whose last position is the boundary copies that position."""
    _, _, src, dst, mask = _indices(
        block_tables=[[20, 21, 22]],
        query_lens=[4],
        seq_lens=[32],
        is_window=[True],
        scratch_bases=[44],
        read_offsets=[0],
    )
    assert mask.tolist() == [True]
    assert src.tolist() == [44 + 3]
    assert dst.tolist() == [21]


def test_prefill_uses_block_table_slots():
    """Prefill chunks keep align-mode slots: read the block of the last
    computed token, write the block of the last scheduled token, never copy.
    """
    state, read, _, _, mask = _indices(
        block_tables=[[30, 31, 32]],
        query_lens=[16],
        seq_lens=[32],
        is_window=[False],
        scratch_bases=[48],
        read_offsets=[MAMBA_READ_OFFSET_FROM_BLOCK_TABLE],
    )
    assert state.tolist() == [31]
    assert read.tolist() == [30]
    assert mask.tolist() == [False]


def test_prefill_after_short_window_chunk_resumes_from_scratch():
    """A prefill chunk that follows a short chunk run as a window resumes
    from the window's scratch checkpoint."""
    state, read, _, _, mask = _indices(
        block_tables=[[30, 31, 32]],
        query_lens=[16],
        seq_lens=[35],
        is_window=[False],
        scratch_bases=[48],
        read_offsets=[2],
    )
    assert state.tolist() == [32]
    assert read.tolist() == [50]
    assert mask.tolist() == [False]


def test_window_crossing_boundary_at_first_position():
    """A window whose first position (offset 0) completes a block copies
    `scratch_base + 0` into that block."""
    # 15 computed tokens + 4-token window (seq_len 19) crosses 16 at offset 0.
    state, read, src, dst, mask = _indices(
        block_tables=[[10, 11]],
        query_lens=[4],
        seq_lens=[19],
        is_window=[True],
        scratch_bases=[40],
        read_offsets=[MAMBA_READ_OFFSET_FROM_BLOCK_TABLE],
    )
    assert state.tolist() == [40]
    assert read.tolist() == [10]
    assert mask.tolist() == [True]
    assert src.tolist() == [40]
    assert dst.tolist() == [10]


def test_single_token_window_boundary_and_non_boundary():
    """1-token decode windows (e.g. 0 draft tokens on an ngram miss) copy
    only when the single token lands on a block boundary."""
    _, _, src, dst, mask = _indices(
        block_tables=[[10, 11], [20, 21]],
        query_lens=[1, 1],
        seq_lens=[16, 17],
        is_window=[True, True],
        scratch_bases=[40, 44],
        read_offsets=[0, 0],
    )
    assert mask.tolist() == [True, False]
    assert int(src[0]) == 40
    assert int(dst[0]) == 10


def test_padded_requests_never_copy():
    """Padding rows (no tokens, null scratch base) must not copy."""
    _, _, _, _, mask = _indices(
        block_tables=[[0, 0]],
        query_lens=[0],
        seq_lens=[0],
        is_window=[False],
        scratch_bases=[0],
        read_offsets=[0],
    )
    assert mask.tolist() == [False]


def test_copy_mask_guards_isolate_invalid_window_copies():
    """Each copy_mask guard independently suppresses copying even when the
    request is marked as a verify window."""
    # Row 0: valid boundary crossing (control -> True)
    # Row 1: scratch_bases == 0 (null scratch -> False)
    # Row 2: copy_dst == 0 (null block in block_tables -> False)
    # Row 3: query_lens == 0 (empty window -> False)
    _, _, _, _, mask = _indices(
        block_tables=[[10, 11], [12, 13], [0, 15], [16, 17]],
        query_lens=[4, 4, 4, 0],
        seq_lens=[18, 18, 18, 16],
        is_window=[True, True, True, True],
        scratch_bases=[40, 0, 44, 48],
        read_offsets=[0, 0, 0, 0],
    )
    assert mask.tolist() == [True, False, False, False]


def test_compute_align_spec_state_indices_clamps_out_of_range_ids():
    """Block-table columns and rank-local row ids clamp below num_local_rows."""
    # Row 0: window with out-of-range scratch_base and block id
    # Row 1: prefill whose seq_len exceeds block_tables columns and block id
    #        exceeds num_local_rows - 1
    state, read, src, dst, mask = _indices(
        block_tables=[[90, 95], [10, 80]],
        query_lens=[4, 16],
        seq_lens=[18, 64],
        is_window=[True, False],
        scratch_bases=[50, 40],
        read_offsets=[5, MAMBA_READ_OFFSET_FROM_BLOCK_TABLE],
        num_local_rows=52,
    )
    assert state.tolist() == [50, 51]  # row 1 col clamped to 1 (80 -> 51)
    assert read.tolist() == [
        51, 51
    ]  # row 0: 50+5 -> 51; row 1: col 2 -> 1 (80 -> 51)
    assert mask.tolist() == [True, False]
    assert int(src[0]) == 51  # 50 + 1 -> 51
    assert int(dst[0]) == 51  # 90 -> 51


def test_copy_gdn_state_slots_copies_only_masked_pairs(mesh):
    num_rows, dim, n_v, d_k, d_v = 12, 8, 2, 4, 4
    conv = jnp.arange(num_rows * 3 * dim,
                      dtype=jnp.float32).reshape(num_rows, 3, dim)
    rec = jnp.arange(num_rows * n_v * d_k * d_v,
                     dtype=jnp.float32).reshape(num_rows, n_v, d_k, d_v)
    src = jnp.array([8, 9, 10, 11], dtype=jnp.int32)
    dst = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    mask = jnp.array([True, False, True, False])

    new_conv, new_rec = jax.jit(copy_gdn_state_slots,
                                static_argnames=("mesh", ))(conv,
                                                            rec,
                                                            src,
                                                            dst,
                                                            mask,
                                                            mesh=mesh)

    expected_conv = np.asarray(conv).copy()
    expected_rec = np.asarray(rec).copy()
    expected_conv[1] = expected_conv[8]
    expected_conv[3] = expected_conv[10]
    expected_rec[1] = expected_rec[8]
    expected_rec[3] = expected_rec[10]
    np.testing.assert_array_equal(np.asarray(new_conv), expected_conv)
    np.testing.assert_array_equal(np.asarray(new_rec), expected_rec)


def test_copy_gdn_state_slots_noop_without_mask(mesh):
    conv = jnp.arange(4 * 3 * 8, dtype=jnp.float32).reshape(4, 3, 8)
    rec = jnp.arange(4 * 2 * 4 * 4, dtype=jnp.float32).reshape(4, 2, 4, 4)
    zeros = jnp.zeros((2, ), dtype=jnp.int32)
    new_conv, new_rec = copy_gdn_state_slots(conv,
                                             rec,
                                             zeros,
                                             zeros + 1,
                                             jnp.zeros((2, ), dtype=bool),
                                             mesh=mesh)
    np.testing.assert_array_equal(np.asarray(new_conv), np.asarray(conv))
    np.testing.assert_array_equal(np.asarray(new_rec), np.asarray(rec))


def test_resolve_gdn_state_slots_dp_align_spec():
    """resolve_gdn_state_slots slices per-DP-rank block tables, scratch bases,
    and read offsets, and emits per-rank boundary copy descriptors."""
    # dp_size=2, max_num_reqs_per_dp=2, padded_num_reqs_per_dp=1
    md = AttentionMetadata(
        input_positions=jnp.zeros(8, dtype=jnp.int32),
        block_tables=jnp.array(
            [[10, 11, 12], [99, 99, 99], [20, 21, 22], [99, 99, 99]],
            dtype=jnp.int32),
        seq_lens=jnp.array([18, 0, 34, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 4, 0, 2, 2], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.int32),
        mamba=MambaMetadata(
            state_indices=jnp.array([40, 0, 44, 0], dtype=jnp.int32),
            read_offsets=jnp.array(
                [2, 0, MAMBA_READ_OFFSET_FROM_BLOCK_TABLE, 0],
                dtype=jnp.int32),
        ),
        padded_num_reqs=2,
    )
    seq_lens_sliced = jnp.array([18, 34], dtype=jnp.int32)
    qsl_sliced = jnp.array([0, 4, 0, 2], dtype=jnp.int32)

    slots = resolve_gdn_state_slots(
        md,
        seq_lens=seq_lens_sliced,
        query_start_loc=qsl_sliced,
        padded_num_reqs_per_dp=1,
        dp_size=2,
        num_local_rows=64,
        mamba_block_size=BLOCK_SIZE,
        num_spec_tokens=3,
    )
    assert slots.state_indices.tolist() == [40, 44]
    assert slots.read_state_indices.tolist() == [42, 21]
    assert slots.read_offsets.tolist() == [0, 0]
    assert slots.num_spec_tokens == 3
    assert slots.boundary_copies is not None
    assert slots.boundary_copies.mask.tolist() == [True, False]
    assert int(slots.boundary_copies.src[0]) == 41
    assert int(slots.boundary_copies.dst[0]) == 10


def test_resolve_gdn_state_slots_non_align_forwards_and_clamps_read_offsets():
    """In non-align mode (mamba_block_size=0), positive read_offsets are
    forwarded directly while MAMBA_READ_OFFSET_FROM_BLOCK_TABLE (-1) clamps to
    0 so the request reads its base slot."""
    md = AttentionMetadata(
        input_positions=jnp.zeros(8, dtype=jnp.int32),
        seq_lens=jnp.array([20, 16], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 8], dtype=jnp.int32),
        request_distribution=jnp.array([2, 2, 2], dtype=jnp.int32),
        mamba=MambaMetadata(
            state_indices=jnp.array([5, 9], dtype=jnp.int32),
            read_offsets=jnp.array([2, MAMBA_READ_OFFSET_FROM_BLOCK_TABLE],
                                   dtype=jnp.int32),
        ),
        padded_num_reqs=2,
    )
    slots = resolve_gdn_state_slots(
        md,
        seq_lens=jnp.array([20, 16], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 8], dtype=jnp.int32),
        padded_num_reqs_per_dp=2,
        dp_size=1,
        num_local_rows=16,
        mamba_block_size=0,
        num_spec_tokens=3,
    )
    assert slots.state_indices.tolist() == [5, 9]
    assert slots.read_state_indices.tolist() == [5, 9]
    assert slots.read_offsets.tolist() == [2, 0]
    assert slots.num_spec_tokens == 3
    assert slots.boundary_copies is None


def test_resolve_gdn_state_slots_rejects_window_spanning_multiple_boundaries():
    """Align + spec decode requires num_spec_tokens < mamba_block_size so a
    single verify window cannot cross more than one block boundary."""
    md = AttentionMetadata(
        input_positions=jnp.zeros(17, dtype=jnp.int32),
        block_tables=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        seq_lens=jnp.array([32], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 17], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
        mamba=MambaMetadata(
            state_indices=jnp.array([10], dtype=jnp.int32),
            read_offsets=jnp.array([0], dtype=jnp.int32),
        ),
        padded_num_reqs=1,
    )
    with pytest.raises(ValueError, match="num_spec_tokens.*mamba_block_size"):
        resolve_gdn_state_slots(
            md,
            seq_lens=md.seq_lens,
            query_start_loc=md.query_start_loc,
            padded_num_reqs_per_dp=1,
            dp_size=1,
            num_local_rows=32,
            mamba_block_size=BLOCK_SIZE,
            num_spec_tokens=BLOCK_SIZE,
        )


@pytest.mark.parametrize("mamba_block_size", [0, BLOCK_SIZE])
def test_resolve_gdn_state_slots_requires_mamba_metadata_for_spec_decode(
        mamba_block_size):
    """Missing MambaMetadata fields with num_spec_tokens > 0 fail fast instead
    of silently falling back to num_spec_tokens=0."""
    md = AttentionMetadata(
        input_positions=jnp.zeros(4, dtype=jnp.int32),
        block_tables=jnp.array([[1, 2]], dtype=jnp.int32),
        seq_lens=jnp.array([8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
        mamba=MambaMetadata(state_indices=jnp.array([4], dtype=jnp.int32),
                            read_offsets=None),
        padded_num_reqs=1,
    )
    with pytest.raises(ValueError, match="read_offsets"):
        resolve_gdn_state_slots(
            md,
            seq_lens=md.seq_lens,
            query_start_loc=md.query_start_loc,
            padded_num_reqs_per_dp=1,
            dp_size=1,
            num_local_rows=16,
            mamba_block_size=mamba_block_size,
            num_spec_tokens=3,
        )


def test_resolve_gdn_state_slots_dp_mixed_window_and_prefill():
    """Mixed decode-window and prefill batches across DP ranks derive
    per-rank `is_window` from `request_distribution`."""
    # dp_size=2, max_num_reqs_per_dp=3, padded_num_reqs_per_dp=2
    # Rank 0: 1 window + 1 prefill (request_distribution=[1, 2, 2])
    # Rank 1: 0 windows + 2 prefills (request_distribution=[0, 2, 2])
    md = AttentionMetadata(
        input_positions=jnp.zeros(44, dtype=jnp.int32),
        block_tables=jnp.array(
            [
                [10, 11, 12],
                [13, 14, 15],
                [99, 99, 99],
                [20, 21, 22],
                [23, 24, 25],
                [99, 99, 99],
            ],
            dtype=jnp.int32,
        ),
        seq_lens=jnp.array([18, 32, 0, 35, 8, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 20, 20, 0, 16, 24, 24],
                                  dtype=jnp.int32),
        request_distribution=jnp.array([1, 2, 2, 0, 2, 2], dtype=jnp.int32),
        mamba=MambaMetadata(
            state_indices=jnp.array([40, 44, 0, 48, 52, 0], dtype=jnp.int32),
            read_offsets=jnp.array(
                [
                    2,
                    MAMBA_READ_OFFSET_FROM_BLOCK_TABLE,
                    0,
                    1,
                    MAMBA_READ_OFFSET_FROM_BLOCK_TABLE,
                    0,
                ],
                dtype=jnp.int32,
            ),
        ),
        padded_num_reqs=4,
    )
    slots = resolve_gdn_state_slots(
        md,
        seq_lens=jnp.array([18, 32, 35, 8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 20, 0, 16, 24], dtype=jnp.int32),
        padded_num_reqs_per_dp=2,
        dp_size=2,
        num_local_rows=64,
        mamba_block_size=BLOCK_SIZE,
        num_spec_tokens=3,
    )
    # Rank 0 req 0 (window): write scratch 40, read 40+2=42, copy 41 -> 10
    # Rank 0 req 1 (prefill 16->32): write block 14, read block 13, no copy
    # Rank 1 req 0 (prefill 19->35 after window): write block 22, read 48+1=49
    # Rank 1 req 1 (first prefill 0->8): write block 23, read block 23
    assert slots.state_indices.tolist() == [40, 14, 22, 23]
    assert slots.read_state_indices.tolist() == [42, 13, 49, 23]
    assert slots.read_offsets.tolist() == [0, 0, 0, 0]
    assert slots.boundary_copies is not None
    assert slots.boundary_copies.mask.tolist() == [True, False, False, False]
    assert int(slots.boundary_copies.src[0]) == 41
    assert int(slots.boundary_copies.dst[0]) == 10


def test_resolve_gdn_state_slots_dp_plain_align_1d_block_tables(mesh):
    """Plain align mode (num_spec_tokens=0) supports 1D flattened block_tables
    across DP ranks with no spec kwargs or boundary copies."""
    block_tables_1d = jnp.array(
        [10, 11, 99, 99, 20, 21, 99, 99],
        dtype=jnp.int32,
    )
    md = AttentionMetadata(
        input_positions=jnp.zeros(24, dtype=jnp.int32),
        block_tables=block_tables_1d,
        seq_lens=jnp.array([32, 0, 8, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 16, 16, 0, 8, 8], dtype=jnp.int32),
        request_distribution=jnp.array([0, 1, 1, 0, 1, 1], dtype=jnp.int32),
        mamba=None,
        padded_num_reqs=2,
    )
    slots = resolve_gdn_state_slots(
        md,
        seq_lens=jnp.array([32, 8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 16, 0, 8], dtype=jnp.int32),
        padded_num_reqs_per_dp=1,
        dp_size=2,
        num_local_rows=64,
        mamba_block_size=BLOCK_SIZE,
        num_spec_tokens=0,
    )
    assert slots.state_indices.tolist() == [11, 20]
    assert slots.read_state_indices.tolist() == [10, 20]
    assert slots.read_offsets is None
    assert slots.num_spec_tokens == 0
    assert slots.boundary_copies is None
    kwargs = slots.kernel_kwargs()
    assert "read_offsets" not in kwargs
    assert "num_spec_tokens" not in kwargs

    conv = jnp.arange(4 * 3 * 8, dtype=jnp.float32).reshape(4, 3, 8)
    rec = jnp.arange(4 * 2 * 4 * 4, dtype=jnp.float32).reshape(4, 2, 4, 4)
    out_conv, out_rec = slots.apply_boundary_copies(conv, rec, mesh)
    assert out_conv is conv
    assert out_rec is rec


def test_resolve_gdn_state_slots_rejects_mismatched_seq_lens_shape():
    md = AttentionMetadata(
        input_positions=jnp.zeros(4, dtype=jnp.int32),
        block_tables=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        seq_lens=jnp.array([8, 8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 8], dtype=jnp.int32),
        request_distribution=jnp.array([2, 2, 2], dtype=jnp.int32),
        mamba=MambaMetadata(state_indices=jnp.array([4, 8], dtype=jnp.int32)),
        padded_num_reqs=2,
    )
    with pytest.raises(ValueError, match="Expected DP-truncated seq_lens"):
        resolve_gdn_state_slots(
            md,
            seq_lens=jnp.array([8, 8], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
            padded_num_reqs_per_dp=1,
            dp_size=1,
            num_local_rows=16,
            mamba_block_size=0,
            num_spec_tokens=0,
        )


@pytest.mark.parametrize("mamba_block_size,num_spec_tokens,mamba", [
    (0, 0, None),
    (0, 0, MambaMetadata(state_indices=None, read_offsets=None)),
    (BLOCK_SIZE, 3, None),
    (BLOCK_SIZE, 3,
     MambaMetadata(state_indices=None,
                   read_offsets=jnp.array([0], dtype=jnp.int32))),
])
def test_resolve_gdn_state_slots_requires_state_indices(
        mamba_block_size, num_spec_tokens, mamba):
    md = AttentionMetadata(
        input_positions=jnp.zeros(4, dtype=jnp.int32),
        block_tables=jnp.array([[1, 2]], dtype=jnp.int32),
        seq_lens=jnp.array([8], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
        mamba=mamba,
        padded_num_reqs=1,
    )
    with pytest.raises(ValueError, match="state_indices"):
        resolve_gdn_state_slots(
            md,
            seq_lens=md.seq_lens,
            query_start_loc=md.query_start_loc,
            padded_num_reqs_per_dp=1,
            dp_size=1,
            num_local_rows=16,
            mamba_block_size=mamba_block_size,
            num_spec_tokens=num_spec_tokens,
        )
