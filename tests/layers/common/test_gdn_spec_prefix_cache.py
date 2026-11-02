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

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
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


def _indices(block_tables, query_lens, seq_lens, is_window, scratch_bases,
             read_offsets):
    out = compute_align_spec_state_indices(
        jnp.asarray(block_tables, dtype=jnp.int32),
        jnp.asarray(query_lens, dtype=jnp.int32),
        jnp.asarray(seq_lens, dtype=jnp.int32),
        jnp.asarray(is_window, dtype=bool),
        jnp.asarray(scratch_bases, dtype=jnp.int32),
        jnp.asarray(read_offsets, dtype=jnp.int32),
        BLOCK_SIZE,
        num_local_rows=100,
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
    """Offset -1 (after a prefill chunk or a prefix hit) resumes from the
    block covering the last computed token; no boundary inside the window
    means no copy."""
    state, read, _, _, mask = _indices(
        block_tables=[[20, 21, 22]],
        query_lens=[2],
        seq_lens=[34],
        is_window=[True],
        scratch_bases=[44],
        read_offsets=[-1],
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
        read_offsets=[-1],
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
    conv = jnp.ones((4, 3, 8))
    rec = jnp.ones((4, 2, 4, 4))
    zeros = jnp.zeros((2, ), dtype=jnp.int32)
    new_conv, new_rec = copy_gdn_state_slots(conv,
                                             rec,
                                             zeros,
                                             zeros + 1,
                                             jnp.zeros((2, ), dtype=bool),
                                             mesh=mesh)
    np.testing.assert_array_equal(np.asarray(new_conv), np.asarray(conv))
    np.testing.assert_array_equal(np.asarray(new_rec), np.asarray(rec))


def test_resolve_gdn_state_slots_dp_align_spec(mesh):
    """resolve_gdn_state_slots slices per-DP-rank block tables, scratch bases,
    and read offsets, and applies boundary copies."""
    # dp_size=2, max_num_reqs_per_dp=2, padded_num_reqs_per_dp=1
    md = AttentionMetadata(
        input_positions=jnp.zeros(8, dtype=jnp.int32),
        block_tables=jnp.array([[10, 11], [99, 99], [20, 21], [99, 99]],
                               dtype=jnp.int32),
        seq_lens=jnp.array([18, 0, 34, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 4, 0, 2, 2], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.int32),
        mamba_state_indices=jnp.array([40, 0, 44, 0], dtype=jnp.int32),
        gdn_read_offsets=jnp.array([2, 0, -1, 0], dtype=jnp.int32),
        num_spec_tokens=3,
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
    )
    assert slots.state_indices.tolist() == [40, 44]
    assert slots.read_state_indices.tolist() == [42, 21]
    assert slots.read_offsets.tolist() == [0, 0]
    assert slots.num_spec_tokens == 3
    assert slots.boundary_copies is not None
    src, dst, mask = [np.asarray(x).tolist() for x in slots.boundary_copies]
    assert mask == [True, False]
    assert src[0] == 41
    assert dst[0] == 10


def test_resolve_gdn_state_slots_non_align_clamps_negative_read_offset():
    """In non-align mode (mamba_block_size=0), a -1 read offset from a prefill
    chunk is clamped to 0 so the request reads its base slot."""
    md = AttentionMetadata(
        input_positions=jnp.zeros(4, dtype=jnp.int32),
        seq_lens=jnp.array([20], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
        mamba_state_indices=jnp.array([5], dtype=jnp.int32),
        gdn_read_offsets=jnp.array([-1], dtype=jnp.int32),
        num_spec_tokens=3,
        padded_num_reqs=1,
    )
    slots = resolve_gdn_state_slots(
        md,
        seq_lens=jnp.array([20], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
        padded_num_reqs_per_dp=1,
        dp_size=1,
        num_local_rows=16,
        mamba_block_size=0,
    )
    assert slots.state_indices.tolist() == [5]
    assert slots.read_state_indices.tolist() == [5]
    assert slots.read_offsets.tolist() == [0]
    assert slots.num_spec_tokens == 3
    assert slots.boundary_copies is None


