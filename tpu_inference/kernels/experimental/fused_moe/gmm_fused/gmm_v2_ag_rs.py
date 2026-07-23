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
"""Dedup-aware push-based AG fused with GMM1 + activation (EP MoE).

Each (sender s, target c, global_token_id g) tuple crosses the ICI link at
most once. Each receiver maintains a persistent HBM cache of shape
(num_tokens_total, K1/NL, NL) indexed directly by `global_token_id`. The
first time chip B needs token T originating from chip A, A pushes T into
B's cache slot T. Subsequent rounds that need T on B read straight from
the cache (zero ICI).

The schedule is precomputed in JAX (compute_dedup_send_schedule). The
kernel itself just executes:

  Bootstrap: push round 0 NEW tokens to all targets' cache slots.
  Per round r in [0, max_num_gm):
    - Prefetch: push round (r+1) NEW tokens (slot (r+1) % 2 send_sem).
    - Drain send_sem[(r+1) % 2, *] from its prior use, two rounds back.
    - (r < my_num_gm) Wait recv_sem[r % 2] for recv_count_new[r] bytes.
    - (r < my_num_gm) Per-row indexed gather from cache_ref to VMEM via
      dma_gather_gm_start (using lhs_indices; identical to the original
      inner_kernel_gather pattern).
    - (r < my_num_gm) Wait gather, then run inner_kernel matmul.

The receiver's gather works whether the token landed in this round or in
an earlier round — the cache is the canonical source.
"""

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
from jax import lax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# isort: off
# yapf: disable
from ..gmm_v2_gather_scatter import (
    FusedWeightsRef, MetadataRef, WeightsRef, align_to, apply_act_fn,
    calculate_tiling, dma_gather_gm_start, dma_gather_gm_wait,
    dma_scatter_gm_start, dma_scatter_gm_wait, fill_metadata,
    generate_block_specs, get_cost_estimate, get_scope_name, inner_kernel,
    make_gmm_configs, zero_out_end, zero_out_end_3d, zero_out_start,
    zero_out_start_3d)
# yapf: enable
# isort: on
from .per_round_schedule import compute_dedup_send_schedule

# =============================================================================
# Push helpers (NEW tokens only, target = cache_ref[global_id])
# =============================================================================


def _push_round_to_target(
    *,
    hidden_states_ref,  # 3D HBM: (chunk_size, padded_K1/NL, NL)
    target_cache_ref,  # 3D HBM: (num_tokens_total, padded_K1/NL, NL)
    send_count_for_target,  # int32 scalar (SMEM)
    send_local_off_for_target,  # int32[tile_m] (SMEM)
    send_global_id_for_target,  # int32[tile_m] (SMEM) — destination cache slot
    target_id,  # int32 scalar
    is_local_target,  # bool scalar
    send_sem,
    recv_sem,
    ep_axis_name: str,
):
    """Push send_count_for_target NEW tokens to the given target chip.

    Source row = local hidden_states[send_local_off_for_target[i]].
    Destination = target's cache_ref[send_global_id_for_target[i]].

    Local self-target uses make_async_copy with recv_sem so the receiver-
    side wait drains uniformly across local + remote contributions.
    Remote uses make_async_remote_copy with both send_sem and recv_sem.
    """

    def _send_row(i, _):
        local_off = send_local_off_for_target[i]
        cache_slot = send_global_id_for_target[i]

        @pl.when(is_local_target)
        def _local():
            pltpu.make_async_copy(
                src_ref=hidden_states_ref.at[pl.ds(local_off, 1), :, :],
                dst_ref=target_cache_ref.at[pl.ds(cache_slot, 1), :, :],
                sem=recv_sem,
            ).start()

        @pl.when(jnp.logical_not(is_local_target))
        def _remote():
            pltpu.make_async_remote_copy(
                src_ref=hidden_states_ref.at[pl.ds(local_off, 1), :, :],
                dst_ref=target_cache_ref.at[pl.ds(cache_slot, 1), :, :],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id={
                    ep_axis_name: target_id
                },
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

        return _

    lax.fori_loop(0, send_count_for_target, _send_row, 0)


def _push_round(
    *,
    round_id,
    slot,  # send_sem slot index (round % 2)
    my_id,
    hidden_states_ref,
    cache_ref,
    send_count_ref,
    send_local_off_ref,
    send_global_id_ref,
    send_sem_ref,
    recv_sem_ref,
    ep_size: int,
    ep_axis_name: str,
):
    """Issue all outgoing DMAs for one round, to all targets."""
    for c in range(ep_size):
        target_id = jnp.int32(c)
        is_local_target = my_id == target_id

        send_count_for_target = send_count_ref[c, round_id]
        send_local_off_for_target = send_local_off_ref.at[c, round_id, :]
        send_global_id_for_target = send_global_id_ref.at[c, round_id, :]

        _push_round_to_target(
            hidden_states_ref=hidden_states_ref,
            target_cache_ref=cache_ref,
            send_count_for_target=send_count_for_target,
            send_local_off_for_target=send_local_off_for_target,
            send_global_id_for_target=send_global_id_for_target,
            target_id=target_id,
            is_local_target=is_local_target,
            send_sem=send_sem_ref.at[slot, c],
            recv_sem=recv_sem_ref.at[slot],
            ep_axis_name=ep_axis_name,
        )


def _drain_send_sem_for_round(
    *,
    slot,
    round_id,
    my_id,
    cache_ref,  # any HBM buffer used as a byte-count proxy
    send_sem_ref,
    send_count_ref,
    ep_size: int,
):
    """Drain the per-(slot, target) send_sem for `round_id` so the (slot,
    target) pair can be reused two rounds later.

    The drain byte-count equals `send_count[c, round_id] * row_bytes`.
    Skips:
      * (slot, my_id) — local self-target writes use recv_sem.
      * (slot, c) with send_count[c, round_id] == 0 — no DMA was issued.
    """
    for c in range(ep_size):
        target_id = jnp.int32(c)
        is_local_target = my_id == target_id
        n = send_count_ref[c, round_id]

        @pl.when(jnp.logical_and(jnp.logical_not(is_local_target), n > 0))
        def _drain_one():
            # Use cache_ref[0, :n] as a byte-count proxy. The drain only
            # cares about TOTAL bytes accumulated, not which physical
            # bytes — same trick as `dma_gather_gm_wait`.
            pltpu.make_async_copy(
                src_ref=cache_ref.at[pl.ds(0, n), :, :],
                dst_ref=cache_ref.at[pl.ds(0, n), :, :],
                sem=send_sem_ref.at[slot, c],
            ).wait()


def _drain_recv_sem_for_round(
    *,
    slot,
    round_id,
    cache_ref,
    recv_sem_ref,
    recv_count_new_ref,
):
    """Drain recv_sem[slot] by recv_count_new[round_id] * row_bytes.

    The new arrivals land at scattered cache slots indexed by global_id, so
    we cannot bound them with a contiguous slice of staging. We drain by
    total byte count using any contiguous region of cache_ref of the right
    size as a byte-count proxy.
    """
    n = recv_count_new_ref[round_id]

    @pl.when(n > 0)
    def _drain():
        pltpu.make_async_copy(
            src_ref=cache_ref.at[pl.ds(0, n), :, :],
            dst_ref=cache_ref.at[pl.ds(0, n), :, :],
            sem=recv_sem_ref.at[slot],
        ).wait()


# =============================================================================
# Kernel: dedup push + cache + per-row gather + GMM1 + Activation
# =============================================================================


def kernel_main_ag_gmm1(
    # Scalar prefetch (8)
    lhs_group_sizes_ref,
    group_offset_ref,
    lhs_indices_ref,
    send_count_ref,  # SMEM (ep_size, max_num_gm)
    send_local_off_ref,  # SMEM (ep_size, max_num_gm, tile_m)
    send_global_id_ref,  # SMEM (ep_size, max_num_gm, tile_m) — cache slot
    recv_count_new_ref,  # SMEM (max_num_gm,) — new arrivals per round
    my_num_gm_ref,  # SMEM (1,)
    # In (2)
    hidden_states_ref,  # 3D HBM: local shard (chunk_size, padded_K1/NL, NL)
    rhs_ref,  # HBM: W1 weights via WeightsRef
    # Out (2)
    out_ref,  # HBM: (size_m, out_N)
    cache_ref,  # HBM: (num_tokens_total, padded_K1/NL, NL)
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref,
    zero_ref,
    semaphore_ref,
    gather_sem_ref,  # DMA(2,) — per-row gather from cache to VMEM
    gathered_lhs_2x_ref,  # VMEM: (2, tile_m, padded_K1/NL, NL)
    recv_sem_ref,  # DMA(2,) — incoming push completion (per slot)
    send_sem_ref,  # DMA(2, ep_size) — outgoing push completion
    *,
    cfgs,
    ep_size: int,
    ep_axis_name: str,
    max_num_gm: int,
):
    """Dedup push-based AG fused with GMM1 + activation."""
    my_id = lax.axis_index(ep_axis_name)
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = cfgs.tiles.tile_k
    num_k = pl.cdiv(cfgs.dims.size_k, tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight
        rhs_weight = rhs_weight.bitcast(jnp.uint32)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    num_gm_real = fill_metadata(lhs_group_sizes_ref,
                                group_offset_ref,
                                metadata_ref,
                                cfgs=cfgs)

    # Pad metadata for indices >= num_gm_real so emit_pipeline's RHS index
    # map returns a valid group id on padding rounds.
    last_valid_offset = metadata_ref.gm_id_to_m_offset[num_gm_real]
    for _i in range(max_num_gm):
        i = jnp.int32(_i)

        @pl.when(i >= num_gm_real)
        def _pad():
            metadata_ref.gm_id_to_group_id[i] = jnp.int32(0)
            metadata_ref.gm_id_to_m_offset[i] = last_valid_offset
            metadata_ref.gm_id_to_m_offset[i + 1] = last_valid_offset

    in_specs, out_specs = generate_block_specs(metadata_ref, cfgs)
    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:],
                                  rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        _, rhs_spec_orig = in_specs
        in_specs = (
            in_specs[0],
            FusedWeightsRef(gate=rhs_spec_orig, up=rhs_spec_orig),
        )
    _, rhs_in_spec = in_specs

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm_real,
            dims=cfgs.dims,
        )

    my_num_gm = my_num_gm_ref[0]

    # ---- Bootstrap ----
    # 1) Push round 0 NEW tokens to all targets' caches.
    # 2) Wait recv_sem[0] for round 0's incoming arrivals (no-op if 0).
    # 3) Pre-start gather for tile 0 into VMEM[0]. This kicks off the
    #    round-0 per-row indexed gather DMAs concurrently with the inner
    #    loop's prefetch of round 1.
    with jax.named_scope("dedup_push_bootstrap_r0"):
        _push_round(
            round_id=jnp.int32(0),
            slot=jnp.int32(0),
            my_id=my_id,
            hidden_states_ref=hidden_states_ref,
            cache_ref=cache_ref,
            send_count_ref=send_count_ref,
            send_local_off_ref=send_local_off_ref,
            send_global_id_ref=send_global_id_ref,
            send_sem_ref=send_sem_ref,
            recv_sem_ref=recv_sem_ref,
            ep_size=ep_size,
            ep_axis_name=ep_axis_name,
        )

        _drain_recv_sem_for_round(
            slot=jnp.int32(0),
            round_id=jnp.int32(0),
            cache_ref=cache_ref,
            recv_sem_ref=recv_sem_ref,
            recv_count_new_ref=recv_count_new_ref,
        )

        # Pre-start gather for tile 0 (no-op if my_num_gm == 0; metadata
        # padding makes m_end == m_start so the row-loop is empty).
        @pl.when(my_num_gm > 0)
        def _bootstrap_gather():
            dma_gather_gm_start(
                cache_ref,
                gathered_lhs_2x_ref.at[0],
                lhs_indices_ref,
                gather_sem_ref.at[0],
                jnp.int32(0),
                metadata_ref,
            )

    # ---- Per-round inner ----
    # At gm_id=r is_first_kn:
    #   (a) Prefetch push round r+1 (with send_sem drain from 2 rounds back).
    #   (b) Wait recv_sem[(r+1)%2] for round r+1 arrivals.
    #   (c) Start gather for tile r+1 into VMEM[(r+1)%2] — async; runs
    #       concurrently with this round's matmul.
    #   (d) Wait gather for current tile r (sem_id) — bootstrap
    #       pre-started r=0; previous iterations pre-started r>=1.
    # The matmul body runs gated on r < my_num_gm.
    def inner_per_round(
        tiled_rhs_ref,
        tiled_out_ref,
        partial_out_ref_in,
        acc_ref_in,
        metadata_ref_in,
    ):
        gm_id = pl.program_id(0)
        n_id = pl.program_id(1)
        k_id = pl.program_id(2)
        num_gm_grid = pl.num_programs(0)

        sem_id = gm_id % 2
        is_first_kn = jnp.logical_and(k_id == 0, n_id == 0)

        # (a) Prefetch sends for round (gm_id + 1).
        @pl.when(jnp.logical_and(is_first_kn, gm_id + 1 < num_gm_grid))
        def _prefetch_next():
            next_round = gm_id + 1
            next_slot = 1 - sem_id

            @pl.when(gm_id + 1 >= 2)
            def _drain():
                _drain_send_sem_for_round(
                    slot=next_slot,
                    round_id=gm_id - 1,
                    my_id=my_id,
                    cache_ref=cache_ref,
                    send_sem_ref=send_sem_ref,
                    send_count_ref=send_count_ref,
                    ep_size=ep_size,
                )

            _push_round(
                round_id=next_round,
                slot=next_slot,
                my_id=my_id,
                hidden_states_ref=hidden_states_ref,
                cache_ref=cache_ref,
                send_count_ref=send_count_ref,
                send_local_off_ref=send_local_off_ref,
                send_global_id_ref=send_global_id_ref,
                send_sem_ref=send_sem_ref,
                recv_sem_ref=recv_sem_ref,
                ep_size=ep_size,
                ep_axis_name=ep_axis_name,
            )

        # (b)+(c) Pre-start gather for round r+1: wait its recv_sem, then
        #         start the per-row gather into the OTHER VMEM slot. This
        #         overlaps with current matmul.
        @pl.when(
            jnp.logical_and(
                is_first_kn,
                jnp.logical_and(gm_id + 1 < num_gm_grid, gm_id + 1
                                < my_num_gm),
            ))
        def _prefetch_gather_next():
            next_slot = 1 - sem_id
            _drain_recv_sem_for_round(
                slot=next_slot,
                round_id=gm_id + 1,
                cache_ref=cache_ref,
                recv_sem_ref=recv_sem_ref,
                recv_count_new_ref=recv_count_new_ref,
            )
            dma_gather_gm_start(
                cache_ref,
                gathered_lhs_2x_ref.at[next_slot],
                lhs_indices_ref,
                gather_sem_ref.at[next_slot],
                gm_id + 1,
                metadata_ref_in,
            )

        # (d) Wait gather for current tile r before consuming VMEM[sem_id].
        @pl.when(jnp.logical_and(is_first_kn, gm_id < my_num_gm))
        def _wait_current_gather():
            dma_gather_gm_wait(
                gathered_lhs_2x_ref.at[sem_id],
                gather_sem_ref.at[sem_id],
                gm_id,
                metadata_ref_in,
            )

        # Matmul body — only run for r < my_num_gm.
        @pl.when(gm_id < my_num_gm)
        def _matmul():
            gathered_lhs_k_slice = gathered_lhs_2x_ref.at[
                sem_id, :,
                pl.ds(k_id * (tile_k // num_lanes), tile_k // num_lanes), :]
            gathered_lhs_data = gathered_lhs_k_slice[...].reshape(
                -1, cfgs.dims.size_lhs_sublane, tile_k)
            inner_kernel(
                gathered_lhs_data,
                tiled_rhs_ref,
                tiled_out_ref,
                partial_out_ref_in,
                acc_ref_in,
                metadata_ref_in,
                cfgs=cfgs,
            )

    pipeline_fn = pltpu.emit_pipeline(
        inner_per_round,
        grid=(max_num_gm, num_n, num_k),
        in_specs=[rhs_in_spec],
        out_specs=out_specs,
    )
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(rhs_ref, out_in, scratches=scratches)

    # ---- Epilogue: drain in-flight send sems for the last two rounds ----
    with jax.named_scope("dedup_epilogue_drain"):
        if max_num_gm >= 2:
            last_round_a = jnp.int32(max_num_gm - 1)
            last_round_b = jnp.int32(max_num_gm - 2)
            slot_a = (max_num_gm - 1) % 2
            slot_b = (max_num_gm - 2) % 2
            _drain_send_sem_for_round(
                slot=jnp.int32(slot_a),
                round_id=last_round_a,
                my_id=my_id,
                cache_ref=cache_ref,
                send_sem_ref=send_sem_ref,
                send_count_ref=send_count_ref,
                ep_size=ep_size,
            )
            _drain_send_sem_for_round(
                slot=jnp.int32(slot_b),
                round_id=last_round_b,
                my_id=my_id,
                cache_ref=cache_ref,
                send_sem_ref=send_sem_ref,
                send_count_ref=send_count_ref,
                ep_size=ep_size,
            )
        elif max_num_gm == 1:
            _drain_send_sem_for_round(
                slot=jnp.int32(0),
                round_id=jnp.int32(0),
                my_id=my_id,
                cache_ref=cache_ref,
                send_sem_ref=send_sem_ref,
                send_count_ref=send_count_ref,
                ep_size=ep_size,
            )

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


@jax.jit(static_argnames=[
    "fuse_act",
    "ep_size",
    "ep_axis_name",
], )
def gmm_v2_ag_gmm1(
    hidden_states_shard: jax.Array,  # (chunk_size, size_k1) — local shard
    w1: jax.Array,  # (size_group, size_k1, size_n1)
    group_sizes: jax.Array,  # int32[size_lhs_group]
    lhs_indices: jax.Array,  # int32[size_m]
    *,
    w1_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    fuse_act: str | None = "silu",
    ep_size: int,
    ep_axis_name: str,
) -> jax.Array:
    """Dedup push-based AG fused with GMM1 + activation."""
    chunk_size = hidden_states_shard.shape[0]
    size_k1 = hidden_states_shard.shape[1]
    num_tokens_total = chunk_size * ep_size

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)

    num_lanes = pltpu.get_tpu_info().num_lanes
    sls = pltpu.get_tpu_info().get_sublane_tiling(hidden_states_shard.dtype)

    tile_k_unit = num_lanes * sls
    padded_k1 = align_to(size_k1, tile_k_unit)
    if padded_k1 != size_k1:
        k_pad = padded_k1 - size_k1
        hidden_states_shard = jnp.pad(hidden_states_shard,
                                      ((0, 0), (0, k_pad)))

    size_m = lhs_indices.shape[0]
    lhs_for_config = jax.ShapeDtypeStruct((size_m, w1.shape[1]),
                                          hidden_states_shard.dtype)

    _fallback_fuse_act = None
    if fuse_act is not None:
        _nl = pltpu.get_tpu_info().num_lanes
        if w1.shape[2] % (2 * _nl) != 0:
            _fallback_fuse_act = fuse_act
            fuse_act = None

    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    cfgs = make_gmm_configs(
        lhs_for_config,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        tile_info=calculate_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        lhs_indices=lhs_indices,
        original_k=size_k1 if padded_k1 != size_k1 else None,
        fuse_act=fuse_act,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    if padded_k1 != size_k1:
        if tiles.tile_k % tile_k_unit != 0:
            aligned_tile_k = (tiles.tile_k // tile_k_unit) * tile_k_unit
            if aligned_tile_k == 0:
                aligned_tile_k = tile_k_unit
            tiles = dataclasses.replace(tiles, tile_k=aligned_tile_k)
            cfgs = dataclasses.replace(cfgs, tiles=tiles)

    _out_sls = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
    out_n = (align_to(dims.size_n, num_lanes * _out_sls) if cfgs.fuse_act
             is None else align_to(cfgs.out_size_n, num_lanes * _out_sls))

    rhs_scale_spec = rhs_bias_spec = None
    if w1_scale is not None:
        w1_scale = w1_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if w1_bias is not None:
        w1_bias = w1_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    rhs_weights = WeightsRef(weight=w1, scale=w1_scale, bias=w1_bias)
    rhs_in_spec = WeightsRef(
        weight=pl.BlockSpec(memory_space=pltpu.HBM),
        scale=rhs_scale_spec,
        bias=rhs_bias_spec,
    )

    hidden_3d = hidden_states_shard.reshape(chunk_size, padded_k1 // num_lanes,
                                            num_lanes)

    # Dedup send schedule (each (sender, target, token) sent at most once).
    (
        send_count_new,
        send_local_off_new,
        send_global_id_new,
        recv_count_new,
        my_num_gm,
        max_num_gm_static,
    ) = compute_dedup_send_schedule(
        lhs_indices,
        group_sizes,
        group_offset,
        ep_axis_name=ep_axis_name,
        ep_size=ep_size,
        chunk_size=chunk_size,
        tile_m=tiles.tile_m,
        size_group=dims.size_group,
        size_lhs_sublane=dims.size_lhs_sublane,
    )

    max_num_gm = max_num_gm_static
    num_n = pl.cdiv(cfgs.out_size_n, tiles.tile_n)
    partial_out_n = num_n * tiles.tile_n
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n

    target_zero_ref_bytes = 2 * 1024 * 1024
    out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
    tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
    tile_zero_m = min(tile_zero_m, dims.size_m)

    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, partial_out_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((1, )),  # zero semaphore
        pltpu.SemaphoreType.DMA((2, )),  # gather_sem
        pltpu.VMEM(
            (2, tiles.tile_m, padded_k1 // num_lanes, num_lanes),
            cfgs.out_dtype,
        ),  # gathered_lhs_2x
        pltpu.SemaphoreType.DMA((2, )),  # recv_sem (per slot)
        pltpu.SemaphoreType.DMA((2, ep_size)),  # send_sem (slot x target)
    ]

    out_init = [
        jax.ShapeDtypeStruct((dims.size_m, out_n), cfgs.out_dtype),
        # Persistent HBM cache: indexed by global_token_id.
        jax.ShapeDtypeStruct(
            (num_tokens_total, padded_k1 // num_lanes, num_lanes),
            hidden_states_shard.dtype,
        ),
    ]

    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    result = pl.pallas_call(
        functools.partial(
            kernel_main_ag_gmm1,
            cfgs=cfgs,
            ep_size=ep_size,
            ep_axis_name=ep_axis_name,
            max_num_gm=max_num_gm,
        ),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=8,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                rhs_in_spec,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache_ref
            ],
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=get_scope_name(dims, tiles) + "-ag_gmm1_dedup",
        cost_estimate=get_cost_estimate(cfgs),
    )(
        group_sizes,
        group_offset,
        lhs_indices,
        send_count_new,
        send_local_off_new,
        send_global_id_new,
        recv_count_new,
        my_num_gm,
        hidden_3d,
        rhs_weights,
    )[0]

    result = result[:, :cfgs.out_size_n]

    if _fallback_fuse_act is not None:
        from ..gmm_v2_gather_scatter import apply_act_fn

        result = apply_act_fn(result, _fallback_fuse_act)

    return result


# =============================================================================
# GMM with ICI direct-write scatter (single GMM2 kernel)
# Manual fori_loop approach — drain at top of loop for overlap.
# =============================================================================


def kernel_main_scatter_ici_dedup(
    # Scalar prefetch (6)
    lhs_group_sizes_ref,
    group_offset_ref,
    output_indices_ref,
    topk_weights_ref,  # (M,) float32 — per-row weights for weighted accumulation
    is_last_ref,  # (M,) int32 — 1 iff last occurrence of token on this chip
    total_recv_count_ref,  # (1,) int32 — total remote rows this chip receives
    # In (2)
    lhs_ref,  # HBM: (size_m, K//NL, NL) — 3D for DMA
    rhs_ref,  # HBM: WeightsRef (size_group, size_k, size_n)
    # Out (2)
    out_buf_ref,  # 3D HBM: (chunk_size * ep_size, N//NL, NL) — ICI target
    accumulator_ref,  # 3D HBM: (num_tokens, N//NL, NL) — running weighted sum
    # Scratch
    metadata_ref,
    fused_metadata_ref,  # single-slot MetadataRef for inner_kernel
    tiled_out_2x_ref,  # (2, tile_m, aligned_n) — double-buffered matmul output
    scatter_staging_ref,  # (tile_m, n_cols, num_lanes) — single staging buffer
    partial_out_ref,  # (sls, tile_n) — dummy for scatter_mode inner_kernel
    acc_ref,  # (tile_m, tile_n) — matmul accumulator
    count_ref,  # SMEM(3,): [0]=gm_id, [1]=send_count, [2]=local_count
    semaphore_ref,  # DMA(1,) for out_buf zero_out
    acc_zero_sem_ref,  # DMA(1,) for accumulator zero_out
    send_sem_ref,  # DMA(1,) for ICI remote sends
    local_write_sem_ref,  # DMA(1,) for local out_buf writes
    recv_sem_ref,  # DMA(1,) for incoming remote writes
    staging_sem_ref,  # DMA(1,) for accumulator scatter-back
    output_sem_ref,  # DMA(1,) for accumulator gather
    w_buf_ref,  # VMEM: (num_w_bufs, packed_k, tile_n) — weight buffer
    w_scale_buf_ref,  # VMEM: (num_w_bufs, nqb, 1, tile_n) — scale buffer
    w_bias_buf_ref,  # VMEM: (1, tile_n) — bias buffer
    w_sem_ref,  # DMA(num_w_bufs,) — weight DMA semaphores
    gathered_lhs_2x_ref,  # VMEM: (2, tile_m, K//NL, NL) — double-buffered LHS
    gather_sem_ref,  # DMA(2,) — LHS gather semaphores
    acc_gather_2x_ref,  # VMEM: (2, tile_m, N//NL, NL) — double-buffered acc gather
    *,
    cfgs,
    ep_size,
    chunk_size,
    ep_axis_name,
    num_w_bufs: int,
):
    """GMM kernel with ICI direct-write scatter + dedup (is_last).

    Uses weighted accumulation: acc[token] += gmm_result * topk_weight.
    Only ICI-sends when is_last_for_token == 1 (fully reduced result).
    This reduces ICI sends from ~112/tile to ~14/tile (1/top_k).
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    my_id = lax.axis_index(ep_axis_name)
    out_dtype = cfgs.out_dtype
    tile_m = cfgs.tiles.tile_m
    tile_k = cfgs.tiles.tile_k
    tile_n = cfgs.tiles.tile_n

    num_k = pl.cdiv(cfgs.dims.size_k, tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, tile_n)

    # Pack along K for quantized rhs.
    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    packing = cfgs.rhs_cfgs.packing
    pk = tile_k // packing
    has_scale = cfgs.rhs_cfgs.has_scale
    has_bias = cfgs.rhs_cfgs.has_bias
    nqb = cfgs.num_quant_blocks_per_tile_k if has_scale else 0
    w_dma_n = tile_n * 2 if cfgs.fuse_act else tile_n

    rhs_packed = rhs_ref.weight

    # --- Weight DMA helpers (same pattern as nodedup) ---
    @jax.named_scope("start_w_dma")
    def start_w_dma(buf_id, expert_id, n_id, k_id=0):
        pltpu.make_async_copy(
            src_ref=rhs_packed.at[expert_id,
                                  pl.ds(k_id * pk, pk),
                                  pl.ds(n_id * w_dma_n, w_dma_n)],
            dst_ref=w_buf_ref.at[buf_id],
            sem=w_sem_ref.at[buf_id],
        ).start()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=rhs_ref.scale.at[expert_id,
                                         pl.ds(k_id * nqb, nqb), :,
                                         pl.ds(n_id * w_dma_n, w_dma_n)],
                dst_ref=w_scale_buf_ref.at[buf_id],
                sem=w_sem_ref.at[buf_id],
            ).start()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=rhs_ref.bias.at[expert_id, :,
                                        pl.ds(n_id * w_dma_n, w_dma_n)],
                dst_ref=w_bias_buf_ref,
                sem=w_sem_ref.at[buf_id],
            ).start()

    @jax.named_scope("wait_w_dma")
    def wait_w_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w_buf_ref.at[buf_id],
            dst_ref=w_buf_ref.at[buf_id],
            sem=w_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w_scale_buf_ref.at[buf_id],
                dst_ref=w_scale_buf_ref.at[buf_id],
                sem=w_sem_ref.at[buf_id],
            ).wait()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w_bias_buf_ref,
                dst_ref=w_bias_buf_ref,
                sem=w_sem_ref.at[buf_id],
            ).wait()

    @jax.named_scope("compute_tile")
    def compute_tile(buf_id, n_id, k_id, gm_id):
        sem_id = gm_id % 2
        # Read LHS k-slice from VMEM gathered_lhs_2x_ref (pre-loaded via DMA).
        k_cols = tile_k // num_lanes
        k_offset = k_id * k_cols
        sls_check = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
        if k_cols >= sls_check and k_cols % sls_check == 0:
            # Direct ref slice — efficient, no redundant read.
            lhs_tile = gathered_lhs_2x_ref[sem_id, :,
                                           pl.ds(k_offset, k_cols), :].reshape(
                                               -1, cfgs.dims.size_lhs_sublane,
                                               tile_k)
        else:
            # Fallback for tiny K: read full buffer, slice at JAX level.
            lhs_full = gathered_lhs_2x_ref.at[sem_id][...]
            lhs_k_slice = lhs_full[:, k_offset:k_offset + k_cols, :]
            lhs_tile = lhs_k_slice.reshape(-1, cfgs.dims.size_lhs_sublane,
                                           tile_k)

        w_tile = w_buf_ref.at[buf_id]
        w_sc = w_scale_buf_ref.at[buf_id] if has_scale else None
        w_bias_tile = w_bias_buf_ref if has_bias else None
        w_weights = WeightsRef(weight=w_tile, scale=w_sc, bias=w_bias_tile)
        if cfgs.fuse_act is not None:
            w_up = WeightsRef(
                weight=w_tile.at[:, tile_n:],
                scale=w_sc.at[:, :, tile_n:] if w_sc is not None else None,
                bias=w_bias_tile.at[:, pl.ds(tile_n, tile_n)]
                if has_bias else None,
            )
            w_gate = WeightsRef(
                weight=w_tile.at[:, :tile_n],
                scale=w_sc.at[:, :, :tile_n] if w_sc is not None else None,
                bias=w_bias_tile.at[:, pl.ds(0, tile_n)] if has_bias else None,
            )
            w_weights = FusedWeightsRef(gate=w_gate, up=w_up)
        inner_kernel(
            lhs_tile,
            w_weights,
            tiled_out_2x_ref.at[sem_id, :,
                                pl.ds(n_id * tile_n, tile_n)],
            partial_out_ref,
            acc_ref,
            fused_metadata_ref,
            cfgs=cfgs,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k,
            _gm_id=0,
            _n_id=n_id,
        )

    # --- Sync barrier ---
    @jax.named_scope("sync_barrier")
    def sync_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(ep_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id={ep_axis_name: jnp.int32(i)},
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, ep_size)

    sync_barrier()

    # Fill metadata.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    # Zero-init async start (both out_buf and accumulator).
    if cfgs.zero_init:
        with jax.named_scope("zero_init_start"):
            zero_size_out = zero_out_start_3d(
                out_buf_ref,
                scatter_staging_ref,
                semaphore_ref,
            )
            zero_size_acc = zero_out_start_3d(
                accumulator_ref,
                scatter_staging_ref,
                acc_zero_sem_ref,
            )

    # Initialize send/local counts.
    count_ref[0] = jnp.int32(0)  # gm_id
    count_ref[1] = jnp.int32(0)  # send_count
    count_ref[2] = jnp.int32(0)  # local_count

    total_w_steps = num_n * num_k
    can_cache_w = num_w_bufs >= total_w_steps

    # --- Main gm loop (nodedup pattern) ---
    @jax.named_scope("gm_loop_body")
    def gm_loop_body(gm_id, _):
        sem_id = gm_id % 2

        # 1. Wait for previous accumulator scatter before reusing acc_gather_2x_ref.
        @pl.when(gm_id > 0)
        def _():
            with jax.named_scope("acc_scatter_wait_prev"):
                dma_scatter_gm_wait(
                    acc_gather_2x_ref.at[1 - sem_id],
                    staging_sem_ref.at[0],
                    gm_id - 1,
                    metadata_ref,
                )

        # 2. Setup metadata for this gm tile.
        fused_metadata_ref.gm_id_to_group_id[
            0] = metadata_ref.gm_id_to_group_id[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[
            0] = metadata_ref.gm_id_to_m_offset[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[
            1] = metadata_ref.gm_id_to_m_offset[gm_id + 1]
        expert_id = fused_metadata_ref.gm_id_to_group_id[0]

        # 2b. LHS gather: contiguous DMA from HBM to VMEM.
        # Bootstrap: start gather for tile 0 on first iteration.
        @pl.when(gm_id == 0)
        def _():
            with jax.named_scope("lhs_gather_bootstrap"):
                m_start_0 = metadata_ref.gm_id_to_m_offset[0]
                sls_0 = cfgs.dims.size_lhs_sublane
                m_aligned_0 = m_start_0 - m_start_0 % sls_0
                pltpu.make_async_copy(
                    src_ref=lhs_ref.at[pl.ds(m_aligned_0, tile_m), :, :],
                    dst_ref=gathered_lhs_2x_ref.at[0],
                    sem=gather_sem_ref.at[0],
                ).start()

        # Prefetch next tile's LHS.
        @pl.when(gm_id + 1 < num_gm)
        def _():
            with jax.named_scope("lhs_gather_prefetch"):
                m_start_next = metadata_ref.gm_id_to_m_offset[gm_id + 1]
                sls_n = cfgs.dims.size_lhs_sublane
                m_aligned_next = m_start_next - m_start_next % sls_n
                pltpu.make_async_copy(
                    src_ref=lhs_ref.at[pl.ds(m_aligned_next, tile_m), :, :],
                    dst_ref=gathered_lhs_2x_ref.at[1 - sem_id],
                    sem=gather_sem_ref.at[1 - sem_id],
                ).start()

        # Wait for current tile's LHS gather.
        with jax.named_scope("lhs_gather_wait"):
            pltpu.make_async_copy(
                src_ref=gathered_lhs_2x_ref.at[sem_id],
                dst_ref=gathered_lhs_2x_ref.at[sem_id],
                sem=gather_sem_ref.at[sem_id],
            ).wait()

        # 3. Weight DMA with same-expert caching.
        prev_gm_clamped = jnp.maximum(gm_id - 1, 0)
        prev_expert = metadata_ref.gm_id_to_group_id[prev_gm_clamped]
        is_new_expert = jnp.logical_or(gm_id == 0, prev_expert != expert_id)
        is_same_w = jnp.logical_and(jnp.logical_not(is_new_expert),
                                    jnp.bool_(can_cache_w))

        @pl.when(gm_id == 0)
        def _():
            for _i in range(min(num_w_bufs, total_w_steps)):
                start_w_dma(_i, expert_id, _i // num_k, _i % num_k)

        # 4. Matmul loop over (N, K) tiles with weight DMA pipelining.
        for step in range(total_w_steps):
            _n = step // num_k
            _k = step % num_k
            buf_id = step % num_w_bufs

            @pl.when(jnp.logical_not(is_same_w))
            def _():
                wait_w_dma(buf_id)

            if step + num_w_bufs < total_w_steps:
                ns = step + num_w_bufs
                start_w_dma(ns % num_w_bufs, expert_id, ns // num_k,
                            ns % num_k)

            compute_tile(buf_id, _n, _k, gm_id)

            # Cross-gm weight prefetch after first step.
            if step == 0:

                @pl.when(gm_id + 1 < num_gm)
                def _():
                    next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
                    next_same = jnp.logical_and(next_e == expert_id,
                                                jnp.bool_(can_cache_w))

                    @pl.when(jnp.logical_not(next_same))
                    def _():
                        start_w_dma(0, next_e, 0, 0)

        # Cross-gm prefetch remaining buffers.
        @pl.when(gm_id + 1 < num_gm)
        def _():
            next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
            next_same = jnp.logical_and(next_e == expert_id,
                                        jnp.bool_(can_cache_w))

            @pl.when(jnp.logical_not(next_same))
            def _():
                for _i in range(1, min(num_w_bufs, total_w_steps)):
                    start_w_dma(_i, next_e, _i // num_k, _i % num_k)

        # 5. Deferred zero-init wait on first tile (both out_buf and accumulator).
        if cfgs.zero_init:

            @pl.when(gm_id == 0)
            def _():
                with jax.named_scope("zero_out_end_deferred"):
                    zero_out_end_3d(out_buf_ref, semaphore_ref, zero_size_out)
                    zero_out_end_3d(accumulator_ref, acc_zero_sem_ref,
                                    zero_size_acc)

        # 6. Accumulator gather: load current acc values for this tile's rows.
        with jax.named_scope("acc_gather_start"):
            dma_gather_gm_start(
                accumulator_ref,
                acc_gather_2x_ref.at[sem_id],
                output_indices_ref,
                output_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

        # 7. Reshape GMM output to staging, then do weighted RMW + conditional ICI send.
        m_st = metadata_ref.gm_id_to_m_offset[gm_id]
        m_en = metadata_ref.gm_id_to_m_offset[gm_id + 1]
        _sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
        _ml = m_st % _sls
        num_valid = m_en - m_st

        with jax.named_scope("reshape_to_staging"):
            scatter_staging_ref[...] = tiled_out_2x_ref[sem_id][...].reshape(
                scatter_staging_ref.shape)

        # Wait for accumulator gather to complete.
        with jax.named_scope("acc_gather_wait"):
            dma_gather_gm_wait(
                acc_gather_2x_ref.at[sem_id],
                output_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

        # Per-row weighted RMW + conditional ICI send (dedup pattern).
        acc_slot = acc_gather_2x_ref.at[sem_id]

        @jax.named_scope("weighted_add_and_send")
        def _do_weighted_add_and_send():

            def _row_fn(i, carry):
                send_sz, local_sz = carry
                row_idx = _ml + i
                w = topk_weights_ref[m_st + i].astype(jnp.float32)
                token_g = output_indices_ref[m_st + i]
                is_last = is_last_ref[m_st + i]
                dest_chip = token_g // chunk_size
                local_row = token_g % chunk_size
                write_pos = local_row * ep_size + my_id
                is_local = dest_chip == my_id

                # Weighted RMW: acc[token] += staging[row] * weight
                old_r = acc_slot.at[pl.ds(row_idx, 1), :, :]
                new_r = scatter_staging_ref.at[pl.ds(row_idx, 1), :, :]
                result = (old_r[...].astype(jnp.float32) +
                          new_r[...].astype(jnp.float32) * w)
                acc_slot.at[pl.ds(row_idx, 1), :, :][...] = result.astype(
                    out_dtype)

                # Conditional ICI send (only when is_last == 1).
                @pl.when(jnp.logical_and(is_last == 1, ~is_local))
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=acc_slot.at[pl.ds(row_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        send_sem=send_sem_ref.at[0],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={
                            ep_axis_name: dest_chip
                        },
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                @pl.when(jnp.logical_and(is_last == 1, is_local))
                def _():
                    pltpu.make_async_copy(
                        src_ref=acc_slot.at[pl.ds(row_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        sem=local_write_sem_ref.at[0],
                    ).start()

                is_last_remote = jnp.logical_and(is_last == 1, ~is_local)
                is_last_local = jnp.logical_and(is_last == 1, is_local)
                return (
                    send_sz +
                    lax.select(is_last_remote, jnp.int32(1), jnp.int32(0)),
                    local_sz +
                    lax.select(is_last_local, jnp.int32(1), jnp.int32(0)),
                )

            return lax.fori_loop(0, num_valid, _row_fn,
                                 (jnp.int32(0), jnp.int32(0)))

        sz_send, sz_local = _do_weighted_add_and_send()
        count_ref[1] = count_ref[1] + sz_send
        count_ref[2] = count_ref[2] + sz_local

        # Scatter running sum back to accumulator HBM.
        with jax.named_scope("acc_scatter_start"):
            dma_scatter_gm_start(
                acc_gather_2x_ref.at[sem_id],
                accumulator_ref,
                output_indices_ref,
                staging_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

        return _

    lax.fori_loop(0, num_gm, gm_loop_body, None)

    # Handle num_gm == 0 edge case for deferred zero-init.
    if cfgs.zero_init:

        @pl.when(num_gm == 0)
        def _():
            zero_out_end_3d(out_buf_ref, semaphore_ref, zero_size_out)
            zero_out_end_3d(accumulator_ref, acc_zero_sem_ref, zero_size_acc)

    # Epilogue: wait for last accumulator scatter.
    @pl.when(num_gm > 0)
    def _():
        with jax.named_scope("final_acc_scatter_wait"):
            last_sem_id = (num_gm - 1) % 2
            dma_scatter_gm_wait(
                acc_gather_2x_ref.at[last_sem_id],
                staging_sem_ref.at[0],
                num_gm - 1,
                metadata_ref,
            )

    # Final barrier.
    sync_barrier()

    # Drain send_sem (sparse — only is_last rows sent).
    with jax.named_scope("drain_send_sem"):
        total_send = count_ref[1]
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[pl.ds(0, total_send), :, :],
            dst_ref=out_buf_ref.at[pl.ds(0, total_send), :, :],
            sem=send_sem_ref.at[0],
        ).wait()

    # Drain local_write_sem.
    with jax.named_scope("drain_local_write_sem"):
        total_local = count_ref[2]
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[pl.ds(0, total_local), :, :],
            dst_ref=out_buf_ref.at[pl.ds(0, total_local), :, :],
            sem=local_write_sem_ref.at[0],
        ).wait()

    # Drain recv_sem.
    with jax.named_scope("drain_recv_sem"):
        total_recv = total_recv_count_ref[0]
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[pl.ds(0, total_recv), :, :],
            dst_ref=out_buf_ref.at[pl.ds(0, total_recv), :, :],
            sem=recv_sem_ref.at[0],
        ).wait()


@jax.jit(static_argnames=[
    "fuse_act",
    "ep_size",
    "ep_axis_name",
    "chunk_size",
    "num_tokens",
    "vmem_limit_bytes",
    "maybe_quantize_lhs",
], )
def gmm_v2_scatter_ici_dedup(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    output_indices: jax.Array,  # int32[size_m] — global scatter positions
    total_recv_count: jax.Array,  # int32[1] — remote rows this chip receives
    topk_weights: jax.
    Array,  # float32[size_m] — per-row weights for accumulation
    is_last_for_token: jax.
    Array,  # int32[size_m] — 1 iff last occurrence on this chip
    *,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    fuse_act: str | None = None,
    ep_size: int,
    ep_axis_name: str,
    chunk_size: int,
    num_tokens: int,  # total tokens across all chips (for accumulator sizing)
    vmem_limit_bytes: int | None = None,
    maybe_quantize_lhs: bool = True,
) -> jax.Array:
    """GMM with ICI direct-write scatter + dedup (is_last).

    Uses weighted accumulation: acc[token] += gmm_result * topk_weight.
    Only ICI-sends when is_last_for_token == 1 (fully reduced result).
    Output: (chunk_size * ep_size, size_n) — each token has ep_size slots.
    Post-kernel: sum(reshape(chunk_size, ep_size, N), axis=1).

    Must be called inside shard_map with ep_axis_name mesh axis.

    Args:
        lhs: LHS matrix [size_m, size_k].
        rhs: RHS matrix [size_group, size_k, size_n].
        group_sizes: Group sizes [size_lhs_group].
        output_indices: int32[size_m] — global scatter positions.
            dest_chip = oi // chunk_size, local_row = oi % chunk_size.
        total_recv_count: int32[1] — number of remote rows this chip receives.
        rhs_scale: Optional per-block scale.
        rhs_bias: Optional bias.
        group_offset: Optional group offset.
        fuse_act: Optional activation fusion.
        ep_size: Number of expert-parallel chips.
        ep_axis_name: Mesh axis name for expert parallelism.
        chunk_size: Tokens per chip (output_size // ep_size).
        vmem_limit_bytes: Optional VMEM limit.
        maybe_quantize_lhs: Whether to quantize LHS.

    Returns:
        Output of shape [chunk_size, size_n] (this chip's portion).
    """
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    num_lanes = pltpu.get_tpu_info().num_lanes

    # Check fuse_act alignment.
    _fallback_fuse_act = None
    if fuse_act is not None:
        if rhs.shape[2] % (2 * num_lanes) != 0:
            _fallback_fuse_act = fuse_act
            fuse_act = None

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=calculate_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=True,
        output_indices=output_indices,
        fuse_act=fuse_act,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    # Align N for DMA scatter.
    _out_sls = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
    aligned_n = align_to(dims.size_n, num_lanes * _out_sls)

    # Ensure tile_n divides aligned_n.
    if aligned_n % tiles.tile_n != 0:
        _mxu_size = pltpu.get_tpu_info().mxu_column_size
        _num_n = pl.cdiv(aligned_n, tiles.tile_n)
        _adj_tile_n = (aligned_n // _num_n // _mxu_size) * _mxu_size
        if _adj_tile_n > 0:
            tiles = dataclasses.replace(tiles, tile_n=_adj_tile_n)
            cfgs = dataclasses.replace(cfgs, tiles=tiles)

    n_cols = aligned_n // num_lanes

    # Weight buffer config (same as nodedup pattern).
    is_quantized = cfgs.rhs_cfgs.quant_dtype is not None
    packing = cfgs.rhs_cfgs.packing
    pk = tiles.tile_k // packing
    has_scale = cfgs.rhs_cfgs.has_scale
    nqb = cfgs.num_quant_blocks_per_tile_k if has_scale else 1
    w_dma_n = tiles.tile_n * 2 if cfgs.fuse_act else tiles.tile_n
    num_w_bufs = 2  # double-buffered weights

    # Scratch shapes matching kernel_main_scatter_ici params.
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n

    scratch_shapes = [
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        # fused_metadata_ref (single-slot for inner_kernel)
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((1, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((2, ), jnp.int32),
        ),
        # tiled_out_2x_ref — double-buffered compute buffer
        pltpu.VMEM((2, tiles.tile_m, aligned_n), cfgs.out_dtype),
        # scatter_staging_ref — single buffer (no triple-buffering needed with dedup)
        pltpu.VMEM((tiles.tile_m, n_cols, num_lanes), cfgs.out_dtype),
        # partial_out_ref (dummy for scatter_mode inner_kernel)
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # count_ref — SMEM: [0]=gm_id, [1]=send_count, [2]=local_count
        pltpu.SMEM((3, ), jnp.int32),
        # semaphore_ref for out_buf zero_out
        pltpu.SemaphoreType.DMA((1, )),
        # acc_zero_sem_ref for accumulator zero_out
        pltpu.SemaphoreType.DMA((1, )),
        # send_sem_ref — single (sparse sends)
        pltpu.SemaphoreType.DMA((1, )),
        # local_write_sem_ref — single
        pltpu.SemaphoreType.DMA((1, )),
        # recv_sem_ref — for incoming remote writes
        pltpu.SemaphoreType.DMA((1, )),
        # staging_sem_ref — for accumulator scatter-back
        pltpu.SemaphoreType.DMA((1, )),
        # output_sem_ref — for accumulator gather
        pltpu.SemaphoreType.DMA((1, )),
        # w_buf_ref — weight buffer
        pltpu.VMEM(
            (num_w_bufs, pk, w_dma_n),
            jnp.uint32 if is_quantized else rhs.dtype,
        ),
        # w_scale_buf_ref
        pltpu.VMEM(
            (num_w_bufs, nqb, 1, w_dma_n),
            jnp.float32,
        ),
        # w_bias_buf_ref
        pltpu.VMEM((1, w_dma_n), jnp.float32),
        # w_sem_ref — weight DMA semaphores
        pltpu.SemaphoreType.DMA((num_w_bufs, )),
        # gathered_lhs_2x_ref — double-buffered LHS only (K//NL cols)
        pltpu.VMEM(
            (2, tiles.tile_m, dims.size_k // num_lanes, num_lanes),
            cfgs.out_dtype,
        ),
        # gather_sem_ref — LHS gather semaphores
        pltpu.SemaphoreType.DMA((2, )),
        # acc_gather_2x_ref — double-buffered accumulator gather (N//NL cols)
        pltpu.VMEM(
            (2, tiles.tile_m, n_cols, num_lanes),
            cfgs.out_dtype,
        ),
    ]

    # Output shapes: out_buf (ICI target) + accumulator (running sum).
    out_buf_init = jax.ShapeDtypeStruct(
        (chunk_size * ep_size, n_cols, num_lanes), cfgs.out_dtype)
    accumulator_init = jax.ShapeDtypeStruct((num_tokens, n_cols, num_lanes),
                                            cfgs.out_dtype)

    # Prepare RHS specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
    rhs_in_spec = WeightsRef(
        weight=pl.BlockSpec(memory_space=pltpu.HBM),
        scale=rhs_scale_spec,
        bias=rhs_bias_spec,
    )

    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
        collective_id=0,
    )

    total_recv_arr = total_recv_count.astype(jnp.int32).reshape((1, ))
    topk_w_flat = topk_weights.astype(jnp.float32).flatten()

    # Reshape LHS to 3D for contiguous DMA inside kernel.
    lhs_3d = lhs.reshape(dims.size_m, dims.size_k // num_lanes, num_lanes)

    out_buf, _accumulator = pl.pallas_call(
        functools.partial(
            kernel_main_scatter_ici_dedup,
            cfgs=cfgs,
            ep_size=ep_size,
            chunk_size=chunk_size,
            ep_axis_name=ep_axis_name,
            num_w_bufs=num_w_bufs,
        ),
        out_shape=[out_buf_init, accumulator_init],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # lhs_3d
                rhs_in_spec,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # out_buf
                pl.BlockSpec(memory_space=pltpu.HBM),  # accumulator
            ],
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=get_scope_name(dims, tiles) + "-scatter_ici_dedup",
        cost_estimate=get_cost_estimate(cfgs),
    )(
        group_sizes,
        group_offset,
        output_indices,
        topk_w_flat,
        is_last_for_token,
        total_recv_arr,
        lhs_3d,
        rhs_weights,
    )

    result = out_buf.reshape(chunk_size * ep_size,
                             aligned_n)[:, :cfgs.out_size_n]

    if _fallback_fuse_act is not None:
        result = apply_act_fn(result, _fallback_fuse_act)

    return result


# =============================================================================
# GMM with ICI direct-write scatter — NODEDUP version (ALL rows sent)
# Simple triple-buffered staging, no accumulator, no is_last gating.
# =============================================================================


def kernel_main_scatter_ici_nodedup(
    # Scalar prefetch (4)
    lhs_group_sizes_ref,
    group_offset_ref,
    output_indices_ref,
    total_recv_count_ref,  # (1,) int32 — total remote rows this chip receives
    # In (2)
    lhs_ref,  # HBM: (size_m, K//NL, NL) — 3D for DMA
    rhs_ref,  # HBM: WeightsRef (size_group, size_k, size_n)
    # Out (1)
    out_buf_ref,  # 3D HBM: (chunk_size, N//NL, NL) — ICI target
    # Scratch
    metadata_ref,
    fused_metadata_ref,  # single-slot MetadataRef for inner_kernel
    tiled_out_2x_ref,  # (2, tile_m, aligned_n) — double-buffered matmul output
    scatter_staging_3x_ref,  # (3, tile_m, n_cols, num_lanes) — triple-buffered staging
    partial_out_ref,  # (sls, tile_n) — dummy for scatter_mode inner_kernel
    acc_ref,  # (tile_m, tile_n) — matmul accumulator
    count_ref,  # SMEM(7,): [0]=gm_id, [1..3]=send counts, [4..6]=local counts
    semaphore_ref,  # DMA(1,) for out_buf zero_out
    send_sems_ref,  # DMA(3,) per-staging-slot send sems
    local_write_sems_ref,  # DMA(3,) per-staging-slot local write sems
    recv_sem_ref,  # DMA(1,) for incoming remote writes
    w_buf_ref,  # VMEM: (num_w_bufs, packed_k, tile_n) — weight buffer
    w_scale_buf_ref,  # VMEM: (num_w_bufs, nqb, 1, tile_n) — scale buffer
    w_bias_buf_ref,  # VMEM: (1, tile_n) — bias buffer
    w_sem_ref,  # DMA(num_w_bufs,) — weight DMA semaphores
    gathered_lhs_2x_ref,  # VMEM: (2, tile_m, K//NL, NL) — double-buffered LHS
    gather_sem_ref,  # DMA(2,) — LHS gather semaphores
    *,
    cfgs,
    ep_size,
    chunk_size,
    ep_axis_name,
    num_w_bufs: int,
):
    """GMM kernel with ICI direct-write scatter (nodedup — ALL rows sent).

    Every GMM output row is ICI-sent to the destination chip. No accumulator,
    no is_last gating, no topk_weights. Simple triple-buffered staging with
    drain at top of loop.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    my_id = lax.axis_index(ep_axis_name)
    out_dtype = cfgs.out_dtype
    tile_m = cfgs.tiles.tile_m
    tile_k = cfgs.tiles.tile_k
    tile_n = cfgs.tiles.tile_n

    num_k = pl.cdiv(cfgs.dims.size_k, tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, tile_n)

    # Pack along K for quantized rhs.
    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    packing = cfgs.rhs_cfgs.packing
    pk = tile_k // packing
    has_scale = cfgs.rhs_cfgs.has_scale
    has_bias = cfgs.rhs_cfgs.has_bias
    nqb = cfgs.num_quant_blocks_per_tile_k if has_scale else 0
    w_dma_n = tile_n * 2 if cfgs.fuse_act else tile_n

    rhs_packed = rhs_ref.weight

    # --- Weight DMA helpers ---
    @jax.named_scope("start_w_dma")
    def start_w_dma(buf_id, expert_id, n_id, k_id=0):
        pltpu.make_async_copy(
            src_ref=rhs_packed.at[expert_id,
                                  pl.ds(k_id * pk, pk),
                                  pl.ds(n_id * w_dma_n, w_dma_n)],
            dst_ref=w_buf_ref.at[buf_id],
            sem=w_sem_ref.at[buf_id],
        ).start()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=rhs_ref.scale.at[expert_id,
                                         pl.ds(k_id * nqb, nqb), :,
                                         pl.ds(n_id * w_dma_n, w_dma_n)],
                dst_ref=w_scale_buf_ref.at[buf_id],
                sem=w_sem_ref.at[buf_id],
            ).start()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=rhs_ref.bias.at[expert_id, :,
                                        pl.ds(n_id * w_dma_n, w_dma_n)],
                dst_ref=w_bias_buf_ref,
                sem=w_sem_ref.at[buf_id],
            ).start()

    @jax.named_scope("wait_w_dma")
    def wait_w_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w_buf_ref.at[buf_id],
            dst_ref=w_buf_ref.at[buf_id],
            sem=w_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w_scale_buf_ref.at[buf_id],
                dst_ref=w_scale_buf_ref.at[buf_id],
                sem=w_sem_ref.at[buf_id],
            ).wait()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w_bias_buf_ref,
                dst_ref=w_bias_buf_ref,
                sem=w_sem_ref.at[buf_id],
            ).wait()

    @jax.named_scope("compute_tile")
    def compute_tile(buf_id, n_id, k_id, gm_id):
        sem_id = gm_id % 2
        # Read LHS k-slice from VMEM gathered_lhs_2x_ref (pre-loaded via DMA).
        k_cols = tile_k // num_lanes
        k_offset = k_id * k_cols
        sls_check = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
        if k_cols >= sls_check and k_cols % sls_check == 0:
            # Direct ref slice — efficient, no redundant read.
            lhs_tile = gathered_lhs_2x_ref[sem_id, :,
                                           pl.ds(k_offset, k_cols), :].reshape(
                                               -1, cfgs.dims.size_lhs_sublane,
                                               tile_k)
        else:
            # Fallback for tiny K: read full buffer, slice at JAX level.
            lhs_full = gathered_lhs_2x_ref.at[sem_id][...]
            lhs_k_slice = lhs_full[:, k_offset:k_offset + k_cols, :]
            lhs_tile = lhs_k_slice.reshape(-1, cfgs.dims.size_lhs_sublane,
                                           tile_k)

        w_tile = w_buf_ref.at[buf_id]
        w_sc = w_scale_buf_ref.at[buf_id] if has_scale else None
        w_bias_tile = w_bias_buf_ref if has_bias else None
        w_weights = WeightsRef(weight=w_tile, scale=w_sc, bias=w_bias_tile)
        if cfgs.fuse_act is not None:
            w_up = WeightsRef(
                weight=w_tile.at[:, tile_n:],
                scale=w_sc.at[:, :, tile_n:] if w_sc is not None else None,
                bias=w_bias_tile.at[:, pl.ds(tile_n, tile_n)]
                if has_bias else None,
            )
            w_gate = WeightsRef(
                weight=w_tile.at[:, :tile_n],
                scale=w_sc.at[:, :, :tile_n] if w_sc is not None else None,
                bias=w_bias_tile.at[:, pl.ds(0, tile_n)] if has_bias else None,
            )
            w_weights = FusedWeightsRef(gate=w_gate, up=w_up)
        inner_kernel(
            lhs_tile,
            w_weights,
            tiled_out_2x_ref.at[sem_id, :,
                                pl.ds(n_id * tile_n, tile_n)],
            partial_out_ref,
            acc_ref,
            fused_metadata_ref,
            cfgs=cfgs,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k,
            _gm_id=0,
            _n_id=n_id,
        )

    # --- Sync barrier ---
    @jax.named_scope("sync_barrier")
    def sync_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(ep_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id={ep_axis_name: jnp.int32(i)},
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, ep_size)

    sync_barrier()

    # Fill metadata.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    # Zero-init async start.
    if cfgs.zero_init:
        with jax.named_scope("zero_init_start"):
            zero_size_out = zero_out_start_3d(
                out_buf_ref,
                scatter_staging_3x_ref.at[0],
                semaphore_ref,
            )

    # Initialize per-slot DMA counts to 0.
    for _s in range(3):
        count_ref[1 + _s] = jnp.int32(0)  # send counts
        count_ref[4 + _s] = jnp.int32(0)  # local write counts

    total_w_steps = num_n * num_k
    can_cache_w = num_w_bufs >= total_w_steps

    # --- Main gm loop ---
    @jax.named_scope("gm_loop_body")
    def gm_loop_body(gm_id, _):
        sem_id = gm_id % 2
        stg_id = gm_id % 3

        # Drain the staging slot we're about to reuse. With triple buffering,
        # this slot was last used 3 iterations ago — giving 2 full iterations
        # of compute overlap for its DMAs to complete.
        @jax.named_scope("drain_prev_dmas")
        @pl.when(gm_id >= 3)
        def _():
            prev_send = count_ref[1 + stg_id]
            pltpu.make_async_copy(
                src_ref=scatter_staging_3x_ref.at[stg_id,
                                                  pl.ds(0, prev_send), :, :],
                dst_ref=scatter_staging_3x_ref.at[stg_id,
                                                  pl.ds(0, prev_send), :, :],
                sem=send_sems_ref.at[stg_id],
            ).wait()
            prev_local = count_ref[4 + stg_id]
            pltpu.make_async_copy(
                src_ref=out_buf_ref.at[pl.ds(0, prev_local), :, :],
                dst_ref=out_buf_ref.at[pl.ds(0, prev_local), :, :],
                sem=local_write_sems_ref.at[stg_id],
            ).wait()
            # Reset counts for this slot's new use.
            count_ref[1 + stg_id] = jnp.int32(0)
            count_ref[4 + stg_id] = jnp.int32(0)

        # 1. Setup metadata for this gm tile.
        fused_metadata_ref.gm_id_to_group_id[
            0] = metadata_ref.gm_id_to_group_id[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[
            0] = metadata_ref.gm_id_to_m_offset[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[
            1] = metadata_ref.gm_id_to_m_offset[gm_id + 1]
        expert_id = fused_metadata_ref.gm_id_to_group_id[0]

        # 2. LHS gather: contiguous DMA from HBM to VMEM.
        # Bootstrap: start gather for tile 0 on first iteration.
        @pl.when(gm_id == 0)
        def _():
            with jax.named_scope("lhs_gather_bootstrap"):
                m_start_0 = metadata_ref.gm_id_to_m_offset[0]
                sls_0 = cfgs.dims.size_lhs_sublane
                m_aligned_0 = m_start_0 - m_start_0 % sls_0
                pltpu.make_async_copy(
                    src_ref=lhs_ref.at[pl.ds(m_aligned_0, tile_m), :, :],
                    dst_ref=gathered_lhs_2x_ref.at[0],
                    sem=gather_sem_ref.at[0],
                ).start()

        # Prefetch next tile's LHS.
        @pl.when(gm_id + 1 < num_gm)
        def _():
            with jax.named_scope("lhs_gather_prefetch"):
                m_start_next = metadata_ref.gm_id_to_m_offset[gm_id + 1]
                sls_n = cfgs.dims.size_lhs_sublane
                m_aligned_next = m_start_next - m_start_next % sls_n
                pltpu.make_async_copy(
                    src_ref=lhs_ref.at[pl.ds(m_aligned_next, tile_m), :, :],
                    dst_ref=gathered_lhs_2x_ref.at[1 - sem_id],
                    sem=gather_sem_ref.at[1 - sem_id],
                ).start()

        # Wait for current tile's LHS gather.
        with jax.named_scope("lhs_gather_wait"):
            pltpu.make_async_copy(
                src_ref=gathered_lhs_2x_ref.at[sem_id],
                dst_ref=gathered_lhs_2x_ref.at[sem_id],
                sem=gather_sem_ref.at[sem_id],
            ).wait()

        # 3. Weight DMA with same-expert caching.
        prev_gm_clamped = jnp.maximum(gm_id - 1, 0)
        prev_expert = metadata_ref.gm_id_to_group_id[prev_gm_clamped]
        is_new_expert = jnp.logical_or(gm_id == 0, prev_expert != expert_id)
        is_same_w = jnp.logical_and(jnp.logical_not(is_new_expert),
                                    jnp.bool_(can_cache_w))

        @pl.when(gm_id == 0)
        def _():
            for _i in range(min(num_w_bufs, total_w_steps)):
                start_w_dma(_i, expert_id, _i // num_k, _i % num_k)

        # 4. Matmul loop over (N, K) tiles with weight DMA pipelining.
        for step in range(total_w_steps):
            _n = step // num_k
            _k = step % num_k
            buf_id = step % num_w_bufs

            @pl.when(jnp.logical_not(is_same_w))
            def _():
                wait_w_dma(buf_id)

            if step + num_w_bufs < total_w_steps:
                ns = step + num_w_bufs
                start_w_dma(ns % num_w_bufs, expert_id, ns // num_k,
                            ns % num_k)

            compute_tile(buf_id, _n, _k, gm_id)

            # Cross-gm weight prefetch after first step.
            if step == 0:

                @pl.when(gm_id + 1 < num_gm)
                def _():
                    next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
                    next_same = jnp.logical_and(next_e == expert_id,
                                                jnp.bool_(can_cache_w))

                    @pl.when(jnp.logical_not(next_same))
                    def _():
                        start_w_dma(0, next_e, 0, 0)

        # Cross-gm prefetch remaining buffers.
        @pl.when(gm_id + 1 < num_gm)
        def _():
            next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
            next_same = jnp.logical_and(next_e == expert_id,
                                        jnp.bool_(can_cache_w))

            @pl.when(jnp.logical_not(next_same))
            def _():
                for _i in range(1, min(num_w_bufs, total_w_steps)):
                    start_w_dma(_i, next_e, _i // num_k, _i % num_k)

        # 5. Deferred zero-init wait on first tile.
        if cfgs.zero_init:

            @pl.when(gm_id == 0)
            def _():
                with jax.named_scope("zero_out_end_deferred"):
                    zero_out_end_3d(out_buf_ref, semaphore_ref, zero_size_out)

        # 6. Reshape GMM output to staging.
        m_st = metadata_ref.gm_id_to_m_offset[gm_id]
        m_en = metadata_ref.gm_id_to_m_offset[gm_id + 1]
        _sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
        _ml = m_st % _sls
        num_valid = m_en - m_st

        with jax.named_scope("reshape_to_staging"):
            scatter_staging_3x_ref[stg_id] = tiled_out_2x_ref[sem_id][
                ...].reshape(
                    scatter_staging_3x_ref.shape[1],
                    scatter_staging_3x_ref.shape[2],
                    scatter_staging_3x_ref.shape[3],
                )

        # 7. Per-row ICI send — ALL rows (no is_last gating).
        @jax.named_scope("direct_write_rows")
        def _do_direct_write():

            def _write_row(i, carry):
                send_sz, local_sz = carry
                row_idx = _ml + i
                token_g = output_indices_ref[m_st + i]
                dest_chip = token_g // chunk_size
                local_row = token_g % chunk_size
                write_pos = local_row
                is_local = dest_chip == my_id

                @pl.when(~is_local)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            stg_id, pl.ds(row_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        send_sem=send_sems_ref.at[stg_id],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={
                            ep_axis_name: dest_chip
                        },
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                @pl.when(is_local)
                def _():
                    pltpu.make_async_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            stg_id, pl.ds(row_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        sem=local_write_sems_ref.at[stg_id],
                    ).start()

                return (
                    send_sz +
                    lax.select(~is_local, jnp.int32(1), jnp.int32(0)),
                    local_sz +
                    lax.select(is_local, jnp.int32(1), jnp.int32(0)),
                )

            return lax.fori_loop(0, num_valid, _write_row,
                                 (jnp.int32(0), jnp.int32(0)))

        send_sz, local_sz = _do_direct_write()
        count_ref[1 + stg_id] = count_ref[1 + stg_id] + send_sz
        count_ref[4 + stg_id] = count_ref[4 + stg_id] + local_sz

        return _

    lax.fori_loop(0, num_gm, gm_loop_body, None)

    # Handle num_gm == 0 edge case for deferred zero-init.
    if cfgs.zero_init:

        @pl.when(num_gm == 0)
        def _():
            zero_out_end_3d(out_buf_ref, semaphore_ref, zero_size_out)

    # Epilogue: drain remaining per-slot DMA counts.
    @jax.named_scope("epilogue_drain")
    @pl.when(num_gm > 0)
    def _():
        for _slot in range(3):

            @pl.when(num_gm > _slot)
            def _():
                remaining_send = count_ref[1 + _slot]
                pltpu.make_async_copy(
                    src_ref=scatter_staging_3x_ref.at[
                        _slot, pl.ds(0, remaining_send), :, :],
                    dst_ref=scatter_staging_3x_ref.at[
                        _slot, pl.ds(0, remaining_send), :, :],
                    sem=send_sems_ref.at[_slot],
                ).wait()
                remaining_local = count_ref[4 + _slot]
                pltpu.make_async_copy(
                    src_ref=out_buf_ref.at[pl.ds(0, remaining_local), :, :],
                    dst_ref=out_buf_ref.at[pl.ds(0, remaining_local), :, :],
                    sem=local_write_sems_ref.at[_slot],
                ).wait()

    # Final barrier.
    sync_barrier()

    # Drain recv_sem.
    with jax.named_scope("drain_recv_sem"):
        total_recv = total_recv_count_ref[0]
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[pl.ds(0, total_recv), :, :],
            dst_ref=out_buf_ref.at[pl.ds(0, total_recv), :, :],
            sem=recv_sem_ref.at[0],
        ).wait()


@jax.jit(static_argnames=[
    "fuse_act",
    "ep_size",
    "ep_axis_name",
    "chunk_size",
    "vmem_limit_bytes",
    "maybe_quantize_lhs",
], )
def gmm_v2_scatter_ici_nodedup(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    output_indices: jax.Array,  # int32[size_m] — global scatter positions
    total_recv_count: jax.Array,  # int32[1] — remote rows this chip receives
    *,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    fuse_act: str | None = None,
    ep_size: int,
    ep_axis_name: str,
    chunk_size: int,
    vmem_limit_bytes: int | None = None,
    maybe_quantize_lhs: bool = True,
) -> jax.Array:
    """GMM with ICI direct-write scatter (nodedup — ALL rows sent).

    Every GMM output row is ICI-sent to the destination chip. No accumulator,
    no is_last gating, no topk_weights.
    Output: (chunk_size, size_n) — this chip's portion, no post-kernel reduction.

    Must be called inside shard_map with ep_axis_name mesh axis.

    Args:
        lhs: LHS matrix [size_m, size_k].
        rhs: RHS matrix [size_group, size_k, size_n].
        group_sizes: Group sizes [size_lhs_group].
        output_indices: int32[size_m] — global scatter positions.
            dest_chip = oi // chunk_size, local_row = oi % chunk_size.
        total_recv_count: int32[1] — number of remote rows this chip receives.
        rhs_scale: Optional per-block scale.
        rhs_bias: Optional bias.
        group_offset: Optional group offset.
        fuse_act: Optional activation fusion.
        ep_size: Number of expert-parallel chips.
        ep_axis_name: Mesh axis name for expert parallelism.
        chunk_size: Tokens per chip (output_size // ep_size).
        vmem_limit_bytes: Optional VMEM limit.
        maybe_quantize_lhs: Whether to quantize LHS.

    Returns:
        Output of shape [chunk_size, size_n] (this chip's portion).
    """
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    num_lanes = pltpu.get_tpu_info().num_lanes

    # Check fuse_act alignment.
    _fallback_fuse_act = None
    if fuse_act is not None:
        if rhs.shape[2] % (2 * num_lanes) != 0:
            _fallback_fuse_act = fuse_act
            fuse_act = None

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=calculate_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=True,
        output_indices=output_indices,
        fuse_act=fuse_act,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    # Align N for DMA scatter.
    _out_sls = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
    aligned_n = align_to(dims.size_n, num_lanes * _out_sls)

    # Ensure tile_n divides aligned_n.
    if aligned_n % tiles.tile_n != 0:
        _mxu_size = pltpu.get_tpu_info().mxu_column_size
        _num_n = pl.cdiv(aligned_n, tiles.tile_n)
        _adj_tile_n = (aligned_n // _num_n // _mxu_size) * _mxu_size
        if _adj_tile_n > 0:
            tiles = dataclasses.replace(tiles, tile_n=_adj_tile_n)
            cfgs = dataclasses.replace(cfgs, tiles=tiles)

    n_cols = aligned_n // num_lanes

    # Weight buffer config.
    is_quantized = cfgs.rhs_cfgs.quant_dtype is not None
    packing = cfgs.rhs_cfgs.packing
    pk = tiles.tile_k // packing
    has_scale = cfgs.rhs_cfgs.has_scale
    nqb = cfgs.num_quant_blocks_per_tile_k if has_scale else 1
    w_dma_n = tiles.tile_n * 2 if cfgs.fuse_act else tiles.tile_n
    num_w_bufs = 2  # double-buffered weights

    # Scratch shapes.
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n

    scratch_shapes = [
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        # fused_metadata_ref (single-slot for inner_kernel)
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((1, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((2, ), jnp.int32),
        ),
        # tiled_out_2x_ref — double-buffered compute buffer
        pltpu.VMEM((2, tiles.tile_m, aligned_n), cfgs.out_dtype),
        # scatter_staging_3x_ref — triple-buffered staging
        pltpu.VMEM((3, tiles.tile_m, n_cols, num_lanes), cfgs.out_dtype),
        # partial_out_ref (dummy for scatter_mode inner_kernel)
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # count_ref — SMEM: [0]=gm_id, [1..3]=send counts, [4..6]=local counts
        pltpu.SMEM((7, ), jnp.int32),
        # semaphore_ref for out_buf zero_out
        pltpu.SemaphoreType.DMA((1, )),
        # send_sems_ref — triple-buffered
        pltpu.SemaphoreType.DMA((3, )),
        # local_write_sems_ref — triple-buffered
        pltpu.SemaphoreType.DMA((3, )),
        # recv_sem_ref — for incoming remote writes
        pltpu.SemaphoreType.DMA((1, )),
        # w_buf_ref — weight buffer
        pltpu.VMEM(
            (num_w_bufs, pk, w_dma_n),
            jnp.uint32 if is_quantized else rhs.dtype,
        ),
        # w_scale_buf_ref
        pltpu.VMEM(
            (num_w_bufs, nqb, 1, w_dma_n),
            jnp.float32,
        ),
        # w_bias_buf_ref
        pltpu.VMEM((1, w_dma_n), jnp.float32),
        # w_sem_ref — weight DMA semaphores
        pltpu.SemaphoreType.DMA((num_w_bufs, )),
        # gathered_lhs_2x_ref — double-buffered LHS only (K//NL cols)
        pltpu.VMEM(
            (2, tiles.tile_m, dims.size_k // num_lanes, num_lanes),
            cfgs.out_dtype,
        ),
        # gather_sem_ref — LHS gather semaphores
        pltpu.SemaphoreType.DMA((2, )),
    ]

    # Output shape: out_buf only (no accumulator).
    out_buf_init = jax.ShapeDtypeStruct((chunk_size, n_cols, num_lanes),
                                        cfgs.out_dtype)

    # Prepare RHS specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)
    rhs_in_spec = WeightsRef(
        weight=pl.BlockSpec(memory_space=pltpu.HBM),
        scale=rhs_scale_spec,
        bias=rhs_bias_spec,
    )

    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
        collective_id=0,
    )

    total_recv_arr = total_recv_count.astype(jnp.int32).reshape((1, ))

    # Reshape LHS to 3D for contiguous DMA inside kernel.
    lhs_3d = lhs.reshape(dims.size_m, dims.size_k // num_lanes, num_lanes)

    out_buf = pl.pallas_call(
        functools.partial(
            kernel_main_scatter_ici_nodedup,
            cfgs=cfgs,
            ep_size=ep_size,
            chunk_size=chunk_size,
            ep_axis_name=ep_axis_name,
            num_w_bufs=num_w_bufs,
        ),
        out_shape=out_buf_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # lhs_3d
                rhs_in_spec,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),  # out_buf
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=get_scope_name(dims, tiles) + "-scatter_ici_nodedup",
        cost_estimate=get_cost_estimate(cfgs),
    )(group_sizes, group_offset, output_indices, total_recv_arr, lhs_3d,
      rhs_weights)

    result = out_buf.reshape(chunk_size, aligned_n)[:, :cfgs.out_size_n]

    if _fallback_fuse_act is not None:
        result = apply_act_fn(result, _fallback_fuse_act)

    return result
