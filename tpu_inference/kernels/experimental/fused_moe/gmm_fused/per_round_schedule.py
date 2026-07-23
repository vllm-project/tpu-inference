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
"""Per-round push schedule for AG+GMM1 EP MoE kernel.

Computes, for each chip's local view, a static send schedule that drives a
push-based all-to-all interleaved with the GMM1 compute pipeline.

Each round `r` corresponds to one gm tile on the receiver. In round `r`, every
sender pushes a (possibly variable) number of rows to each receiver. The
invariant is that the total number of rows arriving at any single receiver in
round `r` equals exactly that receiver's gm tile size (≤ tile_m).

The schedule is computed via a one-time `jax.lax.all_gather` of `lhs_indices`,
`group_sizes`, and `group_offset` (all tiny int32 arrays).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

# =============================================================================
# JAX reimplementation of fill_metadata (per chip).
# =============================================================================
# Mirrors the SMEM-mutating loop in gmm_v2_gather_scatter.py but produces JAX
# arrays so we can reason about every chip's gm tiling from outside the kernel.


def _fill_metadata_one_chip(
        group_sizes: jax.Array,  # int32[size_lhs_group]
        group_offset: jax.Array,  # int32 scalar — may differ across chips
        *,
        size_group: int,  # static
        tile_m: int,  # static
        size_lhs_sublane: int,  # static
        max_num_gm: int,  # static upper bound
):
    """Compute (gm_to_m_offset, gm_to_group_id, num_gm) for one chip."""
    max_num_group = group_offset + size_group
    init_off = jnp.zeros((max_num_gm + 1, ), dtype=jnp.int32)
    init_gid = jnp.zeros((max_num_gm, ), dtype=jnp.int32)

    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset, off_arr, gid_arr = carry
        group_id = lhs_group_id - group_offset
        group_size = group_sizes[lhs_group_id]
        end_m_offset = start_m_offset + group_size
        local_offset_outer = start_m_offset % size_lhs_sublane
        aligned_group_size = group_size + local_offset_outer
        curr_num_gm = (aligned_group_size + tile_m - 1) // tile_m
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        def inner_tm_loop(tm_id, inner_carry):
            curr_m_offset, off_in, gid_in = inner_carry
            local_off = curr_m_offset % size_lhs_sublane
            tm_size = jnp.minimum(tile_m - local_off,
                                  end_m_offset - curr_m_offset)
            gid_in = gid_in.at[tm_id].set(group_id)
            next_m_offset = curr_m_offset + tm_size
            off_in = off_in.at[tm_id].set(curr_m_offset)
            off_in = off_in.at[tm_id + 1].set(next_m_offset)
            return (next_m_offset, off_in, gid_in)

        _, off_arr, gid_arr = lax.fori_loop(
            num_gm,
            next_num_gm,
            inner_tm_loop,
            (start_m_offset, off_arr, gid_arr),
        )
        return (next_num_gm, end_m_offset, off_arr, gid_arr)

    num_gm, _, off_arr, gid_arr = lax.fori_loop(
        0,
        max_num_group,
        outer_group_loop,
        (jnp.int32(0), jnp.int32(0), init_off, init_gid),
    )
    return off_arr, gid_arr, num_gm


def _fill_metadata_all_chips(
    all_group_sizes: jax.Array,  # int32[ep_size, size_lhs_group]
    all_group_offsets: jax.Array,  # int32[ep_size]
    *,
    size_group: int,
    tile_m: int,
    size_lhs_sublane: int,
    max_num_gm: int,
):
    """Vectorize _fill_metadata_one_chip over the EP axis."""
    return jax.vmap(lambda gs, go: _fill_metadata_one_chip(
        gs,
        go,
        size_group=size_group,
        tile_m=tile_m,
        size_lhs_sublane=size_lhs_sublane,
        max_num_gm=max_num_gm,
    ))(all_group_sizes, all_group_offsets)


# =============================================================================
# Per-round send schedule.
# =============================================================================


def compute_max_num_gm(size_m: int, size_group: int, tile_m: int) -> int:
    """Static upper bound on num_gm — same formula used everywhere."""
    return size_group + (size_m + tile_m - 1) // tile_m - 1


def compute_per_round_send_schedule(
        lhs_indices: jax.Array,  # int32[size_m] — local chip's indices
        group_sizes: jax.
    Array,  # int32[size_lhs_group] — local chip's group sizes
        group_offset: jax.
    Array,  # int32[1] — local chip's group offset (may differ per chip)
        *,
        ep_axis_name: str,
        ep_size: int,
        chunk_size: int,
        tile_m: int,
        size_group: int,  # static — same on every chip
        size_lhs_sublane: int,  # static
):
    """Build the per-round send schedule for a push-based AG+GMM1 pipeline.

    Returns:
      send_count[c, r]:        int32 — rows this chip pushes to chip c in round r.
      send_local_off[c, r, k]: int32 — local row offset (in [0, chunk_size)).
      send_dest_pos[c, r, k]:  int32 — destination row position within chip c's
                               round-r staging slot (= m_start_local + p).
      my_num_gm:               int32[1] — actual num_gm for this chip.
      max_num_gm_static:       Python int — static upper bound.
    """
    size_m = int(lhs_indices.shape[0])
    max_num_gm = compute_max_num_gm(size_m, size_group, tile_m)

    my_id = lax.axis_index(ep_axis_name)

    # All-gather routing tables across the EP axis (small int32 arrays).
    all_lhs_indices = lax.all_gather(lhs_indices,
                                     ep_axis_name)  # (ep_size, size_m)
    all_group_sizes = lax.all_gather(group_sizes,
                                     ep_axis_name)  # (ep_size, size_lhs_group)
    all_group_offsets_2d = lax.all_gather(group_offset,
                                          ep_axis_name)  # (ep_size, 1)
    all_group_offsets = all_group_offsets_2d[:, 0]  # (ep_size,)

    # Per-chip gm tiling.
    gm_to_m_offset, _gm_to_group_id, num_gm_per_chip = _fill_metadata_all_chips(
        all_group_sizes,
        all_group_offsets,
        size_group=size_group,
        tile_m=tile_m,
        size_lhs_sublane=size_lhs_sublane,
        max_num_gm=max_num_gm,
    )

    p_arange = jnp.arange(tile_m, dtype=jnp.int32)
    starts = gm_to_m_offset[:, :max_num_gm]  # (ep_size, max_num_gm)
    ends = gm_to_m_offset[:, 1:max_num_gm + 1]  # (ep_size, max_num_gm)
    m_offsets_3d = starts[..., None] + p_arange[None, None, :]
    valid_mask = m_offsets_3d < ends[...,
                                     None]  # (ep_size, max_num_gm, tile_m)

    safe_m = jnp.minimum(m_offsets_3d, size_m - 1)
    global_ids = jax.vmap(lambda idx_row, sm: idx_row[sm])(all_lhs_indices,
                                                           safe_m)

    source_chip = global_ids // chunk_size
    local_offset = global_ids % chunk_size
    is_mine = jnp.logical_and(source_chip == my_id, valid_mask)

    # Stable partition: place "mine" entries first.
    sort_key = jnp.where(
        is_mine,
        p_arange[None, None, :],
        tile_m + p_arange[None, None, :],
    )
    sort_perm = jnp.argsort(sort_key, axis=-1)  # (ep_size, max_num_gm, tile_m)

    # Receiver staging convention mirrors `dma_gather_gm_start`
    # (gmm_v2_gather_scatter.py:840): tile rows occupy
    # [m_start_local, m_start_local + expected_rows) within the slot.
    m_start_local = (starts % size_lhs_sublane).astype(jnp.int32)
    p_full = jnp.broadcast_to(p_arange[None, None, :],
                              (ep_size, max_num_gm, tile_m))
    dest_pos_unsorted = m_start_local[..., None] + p_full

    send_local_off = jnp.take_along_axis(local_offset, sort_perm, axis=-1)
    send_dest_pos = jnp.take_along_axis(dest_pos_unsorted, sort_perm, axis=-1)
    send_count = jnp.sum(is_mine.astype(jnp.int32), axis=-1)

    my_num_gm = num_gm_per_chip[my_id].reshape((1, )).astype(jnp.int32)

    return (
        send_count.astype(jnp.int32),
        send_local_off.astype(jnp.int32),
        send_dest_pos.astype(jnp.int32),
        my_num_gm,
        max_num_gm,
    )


# =============================================================================
# Per-(sender, target, token) dedup schedule with persistent HBM cache.
# =============================================================================


def compute_dedup_send_schedule(
    lhs_indices: jax.Array,  # int32[size_m] — local chip's indices
    group_sizes: jax.Array,  # int32[size_lhs_group] — local chip's group sizes
    group_offset: jax.Array,  # int32[1] — local chip's group offset
    *,
    ep_axis_name: str,
    ep_size: int,
    chunk_size: int,
    tile_m: int,
    size_group: int,
    size_lhs_sublane: int,
):
    """Build a dedup-aware send schedule.

    For each (sender s, target c, global_token_id g), `g` is sent over ICI
    AT MOST ONCE — at the earliest round r where c needs g. The receiver
    keeps a persistent HBM cache of shape `(num_tokens_total, K1/NL, NL)`
    indexed directly by `global_token_id`. Subsequent rounds that need the
    same token re-read from the cache (zero ICI traffic).

    Returns:
      send_count_new[c, r]:        int32 — NEW (first-use) tokens this chip
                                   pushes to chip c in round r.
      send_local_off_new[c, r, k]: int32 — local row to read (in
                                   [0, chunk_size)).
      send_global_id_new[c, r, k]: int32 — destination cache slot in chip
                                   c's HBM cache (= global token id).
      recv_count_new[r]:           int32 — total NEW token arrivals at
                                   THIS chip in round r (used for the
                                   receiver-side recv-sem wait).
      my_num_gm:                   int32[1] — actual num_gm for this chip.
      max_num_gm_static:           Python int — static upper bound.
    """
    size_m = int(lhs_indices.shape[0])
    num_tokens_total = chunk_size * ep_size
    max_num_gm = compute_max_num_gm(size_m, size_group, tile_m)

    my_id = lax.axis_index(ep_axis_name)

    all_lhs_indices = lax.all_gather(lhs_indices, ep_axis_name)
    all_group_sizes = lax.all_gather(group_sizes, ep_axis_name)
    all_group_offsets_2d = lax.all_gather(group_offset, ep_axis_name)
    all_group_offsets = all_group_offsets_2d[:, 0]

    gm_to_m_offset, _gm_to_group_id, num_gm_per_chip = _fill_metadata_all_chips(
        all_group_sizes,
        all_group_offsets,
        size_group=size_group,
        tile_m=tile_m,
        size_lhs_sublane=size_lhs_sublane,
        max_num_gm=max_num_gm,
    )

    p_arange = jnp.arange(tile_m, dtype=jnp.int32)
    r_arange = jnp.arange(max_num_gm, dtype=jnp.int32)
    starts = gm_to_m_offset[:, :max_num_gm]
    ends = gm_to_m_offset[:, 1:max_num_gm + 1]
    m_offsets_3d = starts[..., None] + p_arange[None, None, :]
    valid_mask = m_offsets_3d < ends[...,
                                     None]  # (ep_size, max_num_gm, tile_m)

    safe_m = jnp.minimum(m_offsets_3d, size_m - 1)
    global_ids = jax.vmap(lambda idx_row, sm: idx_row[sm])(all_lhs_indices,
                                                           safe_m)
    # (ep_size, max_num_gm, tile_m)

    source_chip = global_ids // chunk_size

    # ---- First-use determination ----
    # Encode (r, p) as a single ordered rank = r * tile_m + p. For each
    # (c, global_id), the first-use is the smallest rank over all valid
    # (r, p) with global_ids[c, r, p] == global_id. Computed by
    # per-chip scatter-min into a (num_tokens_total,) array.
    sentinel = jnp.iinfo(jnp.int32).max
    rank_3d = (r_arange[None, :, None] * tile_m +
               p_arange[None, None, :]).astype(jnp.int32)
    rank_3d = jnp.broadcast_to(rank_3d, (ep_size, max_num_gm, tile_m))
    masked_rank = jnp.where(valid_mask, rank_3d, sentinel)

    flat_global = global_ids.reshape(ep_size, max_num_gm * tile_m)
    flat_rank = masked_rank.reshape(ep_size, max_num_gm * tile_m)

    init_min = jnp.full((ep_size, num_tokens_total), sentinel, dtype=jnp.int32)

    def _scatter_min_one_chip(init_row, idx, vals):
        # Use .min() to keep the smallest rank per token id.
        return init_row.at[idx].min(vals)

    min_rank_per_token = jax.vmap(_scatter_min_one_chip)(
        init_min, flat_global, flat_rank)  # (ep_size, num_tokens_total)

    # Look up min_rank for each (c, r, p)'s global_id and compare to its rank.
    gathered_min = jax.vmap(lambda mr, g: mr[g])(
        min_rank_per_token,
        global_ids.reshape(ep_size, -1)).reshape(ep_size, max_num_gm, tile_m)
    is_first_use = jnp.logical_and(masked_rank == gathered_min, valid_mask)
    # (ep_size, max_num_gm, tile_m)

    # ---- Sender side: my chip pushes only first-use entries it owns ----
    is_my_send = jnp.logical_and(is_first_use, source_chip == my_id)

    sort_key = jnp.where(
        is_my_send,
        p_arange[None, None, :],
        tile_m + p_arange[None, None, :],
    )
    sort_perm = jnp.argsort(sort_key, axis=-1)

    local_offset = global_ids % chunk_size
    send_local_off_new = jnp.take_along_axis(local_offset, sort_perm, axis=-1)
    send_global_id_new = jnp.take_along_axis(global_ids, sort_perm, axis=-1)
    send_count_new = jnp.sum(is_my_send.astype(jnp.int32), axis=-1)

    # ---- Receiver side: total new arrivals per round on this chip ----
    my_first_use = is_first_use[my_id]  # (max_num_gm, tile_m)
    recv_count_new = jnp.sum(my_first_use.astype(jnp.int32), axis=-1)
    # (max_num_gm,)

    my_num_gm = num_gm_per_chip[my_id].reshape((1, )).astype(jnp.int32)

    return (
        send_count_new.astype(jnp.int32),
        send_local_off_new.astype(jnp.int32),
        send_global_id_new.astype(jnp.int32),
        recv_count_new.astype(jnp.int32),
        my_num_gm,
        max_num_gm,
    )
