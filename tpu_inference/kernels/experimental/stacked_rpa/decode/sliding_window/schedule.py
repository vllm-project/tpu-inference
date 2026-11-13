# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Compact metadata schedule for one-block sliding-window decode."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import \
    configs as common_configs
from tpu_inference.kernels.experimental.stacked_rpa import utils
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    block_sizes as decode_block_sizes
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    config as decode_config

_CELL_Q_SRC = 0
_CELL_Q_SIZE = 1
_CELL_PROCESSED_Q = 2
_CELL_PROCESSED_KV = 3
_CELL_EFFECTIVE_KV = 4
_CELL_SIZE = 5

_DMA_NEW_SRC_HBM = 0
_DMA_NEW_DST_VMEM = 1
_DMA_NEW_DST_HBM = 2
_DMA_NEW_SRC_VMEM = 3
_DMA_NEW_FLAGS = 4
_FETCH_VALID = 1
_UPDATE_VALID = 2


@dataclasses.dataclass(frozen=True)
class SchedulePlan:
    """Static capacity and windowing plan for the compact decode schedule."""

    max_steps_ub: int
    worst_steps: int
    fits_one_window: bool
    sched_window: int
    num_sched_windows: int
    total_steps_ub: int
    seq_page_table_size: int
    compute_bkv_size: int
    num_shared_groups: int = 1

    @classmethod
    def create(
        cls,
        cfgs: decode_config.DecodeConfig,
        *,
        num_shared_groups: int = 1,
    ) -> "SchedulePlan":
        validate_config(cfgs)
        if num_shared_groups < 1:
            raise ValueError(
                f"num_shared_groups must be >= 1, got {num_shared_groups}.")
        seq_page_table_size = -(-cfgs.serve.pages_per_seq // 1024) * 1024
        max_steps_ub = calculate_max_steps_ub(
            cfgs,
            num_buffers=1,
            seq_page_table_size=seq_page_table_size,
            num_shared_groups=num_shared_groups,
        )
        worst_steps = -(-cfgs.serve.num_seqs // cfgs.batch_size)
        fits_one_window = worst_steps <= max_steps_ub
        sched_window = (max(1, worst_steps)
                        if fits_one_window else calculate_max_steps_ub(
                            cfgs,
                            num_buffers=2,
                            seq_page_table_size=seq_page_table_size,
                            num_shared_groups=num_shared_groups,
                        ))
        num_sched_windows = max(1, -(-worst_steps // sched_window))
        num_lanes = pltpu.get_tpu_info().num_lanes
        compute_bkv_size = min(
            cfgs.bkv_sz,
            utils.align_to(
                cfgs.model.sliding_window + cfgs.decode_q_len + num_lanes - 1,
                num_lanes,
            ),
        )
        return cls(
            max_steps_ub=max_steps_ub,
            worst_steps=worst_steps,
            fits_one_window=fits_one_window,
            sched_window=sched_window,
            num_sched_windows=num_sched_windows,
            total_steps_ub=num_sched_windows * sched_window,
            seq_page_table_size=seq_page_table_size,
            compute_bkv_size=compute_bkv_size,
            num_shared_groups=num_shared_groups,
        )


def validate_config(cfgs: decode_config.DecodeConfig) -> None:
    """Validate the compact one-query-block, one-KV-block contract."""
    cfgs.validate_decode()
    if cfgs.model.sliding_window is None:
        raise ValueError(
            "Sliding-window decode requires model.sliding_window.")
    required_bkv = decode_block_sizes.required_window_bkv_size(
        cfgs.model,
        cfgs.serve,
        cfgs.decode_q_len,
    )
    if cfgs.bkv_sz < required_bkv:
        raise ValueError(
            "Sliding-window decode requires one anchored KV block.")
    if cfgs.n_buffer != 2:
        raise ValueError(
            "Sliding-window decode requires double buffering, got "
            f"n_buffer={cfgs.n_buffer}.")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Map a padded physical 1-D buffer to a logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        n = int(np.prod(shape))
        n_pad = max(1024, -(-n // 1024) * 1024)
        return cls(data=jax.ShapeDtypeStruct((n_pad, ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)
        return sum(stride * idx for stride, idx in zip(strides, indices))

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


def _schedule_wrappers(cfgs: decode_config.DecodeConfig,
                       steps: int) -> dict[str, SmemWrapper]:
    """Build the padded data leaves that define the compact physical schema."""
    return {
        "cell":
        SmemWrapper.create_shape_dtype((steps, cfgs.batch_size, _CELL_SIZE)),
        "dma_kv_cache":
        SmemWrapper.create_shape_dtype(
            (steps, cfgs.batch_size, cfgs.bkv_p_cache, 2)),
        "dma_kv_new":
        SmemWrapper.create_shape_dtype(
            (steps, cfgs.batch_size, cfgs.bkv_p_new, cfgs.dma_kv_new_size)),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SlidingWindowSchedule:
    """Metadata needed by a one-query-block, one-KV-block decode step."""

    # [steps, batch, (q_src, q_size, processed_q, processed_kv, effective_kv)]
    cell: SmemWrapper
    # [steps, batch, cache pages, (physical_page, transfer_size)]
    dma_kv_cache: SmemWrapper
    # [steps, batch, new pages, (src_hbm, dst_vmem, dst_hbm, src_vmem, flags)]
    dma_kv_new: SmemWrapper
    actual_steps: Any

    cfgs: decode_config.DecodeConfig = dataclasses.field(metadata=dict(
        static=True))
    plan: SchedulePlan = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(
        cls,
        cfgs: decode_config.DecodeConfig,
        steps: int | None = None,
        *,
        plan: SchedulePlan | None = None,
    ) -> "SlidingWindowSchedule":
        validate_config(cfgs)
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        if steps is None:
            steps = plan.max_steps_ub
        wrappers = _schedule_wrappers(cfgs, steps)
        return cls(
            **wrappers,
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            cfgs=cfgs,
            plan=plan,
        )

    @classmethod
    def create_hbm_shape_dtype(
        cls,
        cfgs: decode_config.DecodeConfig,
        *,
        plan: SchedulePlan | None = None,
    ) -> "SlidingWindowSchedule":
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        window = cls.create_shape_dtype(cfgs,
                                        steps=plan.sched_window,
                                        plan=plan)
        num_windows = plan.num_sched_windows

        def grow(value):
            if not isinstance(value, SmemWrapper):
                return value
            return SmemWrapper(
                data=jax.ShapeDtypeStruct(
                    (value.data.shape[0] * num_windows, ), value.data.dtype),
                shape=value.shape,
            )

        replacements = {
            field.name: grow(getattr(window, field.name))
            for field in dataclasses.fields(window)
            if isinstance(getattr(window, field.name), SmemWrapper)
        }
        return dataclasses.replace(window, **replacements)

    def get_cell_metadata(self, step, batch_idx):
        return tuple(self.cell[step, batch_idx, i] for i in range(_CELL_SIZE))

    def get_dma_q(self, step, batch_idx):
        return (
            self.cell[step, batch_idx, _CELL_Q_SRC],
            self.cell[step, batch_idx, _CELL_Q_SIZE],
        )

    def get_dma_kv_cache(self, step, batch_idx, page_idx):
        physical_page = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_vmem = page_idx * self.cfgs.serve.page_size
        size = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        return physical_page, dst_vmem, size

    def get_dma_fetch_kv_new(self, step, batch_idx, page_idx):
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_SRC_HBM]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx,
                                   _DMA_NEW_DST_VMEM]
        valid = (self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_FLAGS]
                 & _FETCH_VALID)
        return src_hbm, dst_vmem, valid

    def get_dma_update_kv_new(self, step, batch_idx, page_idx):
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_DST_HBM]
        src_vmem = self.dma_kv_new[step, batch_idx, page_idx,
                                   _DMA_NEW_SRC_VMEM]
        flags = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_FLAGS]
        valid = (flags & _UPDATE_VALID) >> 1
        width, mask = self.cfgs.wb_lane_bits
        wb_lane = ((flags >> 2) & mask) * self.cfgs.num_lanes
        wb_size = ((flags >> (2 + width)) & mask) * self.cfgs.num_lanes
        return dst_hbm, src_vmem, valid, wb_lane, wb_size

    def scratch_shapes(self):
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def in_specs(self):

        def wrapper(value):
            memory_space = pltpu.SMEM if value.size == 1 else pltpu.HBM
            return pl.BlockSpec(memory_space=memory_space)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        return jax.tree.map(lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
                            self)


# Reserve room for compiler padding, semaphores, and small spill allocations.
SMEM_HEADROOM_BYTES = 256 * 1024


def _shape_tree_nbytes(tree) -> int:
    """Return the physical byte size of every shaped leaf in ``tree``."""
    return sum(
        int(np.prod(leaf.shape)) * jnp.dtype(leaf.dtype).itemsize
        for leaf in jax.tree_util.tree_leaves(tree))


def _metadata_fixed_smem_shapes(
    cfgs: decode_config.DecodeConfig,
    seq_page_table_size: int,
) -> dict[str, jax.ShapeDtypeStruct]:
    """Describe non-schedule SMEM owned by the compact metadata builder."""
    int32 = jnp.int32
    return {
        "cu_q_lens":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs + 1, ), int32),
        "kv_lens":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs, ), int32),
        "distribution":
        jax.ShapeDtypeStruct((len(common_configs.RpaCase), ), int32),
        "sequence_order":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs, ), int32),
        "seq_page_table":
        jax.ShapeDtypeStruct((seq_page_table_size, ), int32),
    }


def schedule_smem_usage_bytes(
    cfgs: decode_config.DecodeConfig,
    steps: int,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
    num_shared_groups: int = 1,
) -> int:
    """Return the compact schema footprint for equal-sized SMEM buffers."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be >= 1, got {num_buffers}.")
    if num_shared_groups < 1:
        raise ValueError(
            f"num_shared_groups must be >= 1, got {num_shared_groups}.")
    if seq_page_table_size is None:
        seq_page_table_size = -(-cfgs.serve.pages_per_seq // 1024) * 1024

    schedule_shapes = _schedule_wrappers(cfgs, steps)
    schedule_shapes["actual_steps"] = jax.ShapeDtypeStruct((1, ), jnp.int32)
    fixed_bytes = _shape_tree_nbytes(
        _metadata_fixed_smem_shapes(cfgs, seq_page_table_size))
    schedule_bytes = num_buffers * _shape_tree_nbytes(schedule_shapes)
    if num_shared_groups == 1:
        return fixed_bytes + schedule_bytes

    # The shared builder stages one page table and the two physical-page columns
    # for every local KV-cache group. All other schedule columns are emitted once.
    extra_page_tables = (num_shared_groups - 1) * seq_page_table_size * 4
    physical_columns = (num_shared_groups * steps * cfgs.batch_size *
                        (cfgs.bkv_p_cache + cfgs.bkv_p_new) * 4)
    return fixed_bytes + schedule_bytes + extra_page_tables + physical_columns


def calculate_max_steps_ub(
    cfgs: decode_config.DecodeConfig,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
    num_shared_groups: int = 1,
) -> int:
    """Find the largest lane-aligned compact schedule that fits in SMEM."""
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be >= 1, got {num_buffers}.")
    if seq_page_table_size is None:
        seq_page_table_size = -(-cfgs.serve.pages_per_seq // 1024) * 1024

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    smem_budget = tpu_info.smem_capacity_bytes - SMEM_HEADROOM_BYTES

    def fits(num_lane_groups: int) -> bool:
        return (schedule_smem_usage_bytes(
            cfgs,
            num_lane_groups * num_lanes,
            num_buffers=num_buffers,
            seq_page_table_size=seq_page_table_size,
            num_shared_groups=num_shared_groups,
        ) <= smem_budget)

    if not fits(1):
        raise ValueError(
            "Sliding-window decode schedule cannot fit one lane-aligned step "
            f"group in SMEM: {smem_budget=} {num_lanes=} {num_buffers=}.")

    lower = 1
    upper = 2
    while fits(upper):
        lower = upper
        upper *= 2

    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if fits(midpoint):
            lower = midpoint
        else:
            upper = midpoint

    return lower * num_lanes


def _emit_kv_new_metadata(
    *,
    cfgs: decode_config.DecodeConfig,
    bkv_sz_cache,
    new_sz,
    fill_dma_kv_new,
) -> None:
    if cfgs.decode_q_len > 1:
        base_page = bkv_sz_cache // cfgs.serve.page_size
        for i in range(cfgs.bkv_p_new):
            slot_start = (base_page + i) * cfgs.serve.page_size
            slot_end = slot_start + cfgs.serve.page_size
            dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
            end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
            dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
            fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)
    else:
        slot_start = (bkv_sz_cache //
                      cfgs.serve.page_size) * cfgs.serve.page_size
        fill_dma_kv_new(0, bkv_sz_cache, new_sz, slot_start)


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sequence_order_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    schedule: SlidingWindowSchedule,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
    window_lo: int = 0,
    shared: Any | None = None,
) -> None:
    """Emit one compact schedule cell per active decode sequence.

    ``shared`` carries per-group page-table and physical-page scratch refs. The
    structural schedule is emitted once; only cache physical pages and update
    destination pages vary across local KV-cache groups.
    """
    validate_config(cfgs)
    n = cfgs.batch_size
    window_size = schedule.plan.sched_window
    pps_pad = schedule.plan.seq_page_table_size
    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    if shared is not None:
        pkv_cache_refs, pkv_new_refs, page_table_refs, page_indices_hbm_refs = shared
        num_groups = len(page_table_refs)

    def emit_sequence(order_idx, _):
        s_idx = sequence_order_ref[order_idx]
        global_cell = order_idx - start_seq_idx
        step = global_cell // n
        target_lane = global_cell % n
        local = jnp.clip(step - window_lo, 0, window_size - 1)
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        in_window = jnp.logical_and(
            jnp.logical_and(step >= window_lo, step < window_lo + window_size),
            q_len > 0,
        )

        @pl.when(in_window)
        def _emit():
            seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
            if shared is None:
                page_copy = pltpu.make_async_copy(
                    page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
                    seq_page_table_ref.at[pl.ds(0, pps_pad)],
                    page_dma_sem.at[0],
                )
                page_copy.start()
                page_copy.wait()
            else:
                page_copies = []
                for group in range(num_groups):
                    page_copy = pltpu.make_async_copy(
                        page_indices_hbm_refs[group].at[pl.ds(
                            seq_off, pps_pad)],
                        page_table_refs[group].at[pl.ds(0, pps_pad)],
                        page_dma_sem.at[group],
                    )
                    page_copy.start()
                    page_copies.append(page_copy)
                for page_copy in page_copies:
                    page_copy.wait()

            k_len = kv_lens_ref[s_idx]
            processed_q = k_len - q_len
            processed_kv = utils.window_anchor_tok(
                k_len,
                q_len,
                cfgs.model.sliding_window,
                cfgs.serve.page_size_log2,
            )
            kv_p_start = processed_kv >> cfgs.serve.page_size_log2
            kv_left = jnp.maximum(k_len - processed_kv, 0)
            if cfgs.update_kv_cache:
                kv_left_from_cache = jnp.maximum(kv_left - q_len, 0)
            else:
                kv_left_from_cache = kv_left
            kv_left_from_new = jnp.maximum(kv_left - kv_left_from_cache, 0)
            bkv_sz_cache = jnp.minimum(kv_left_from_cache, cfgs.bkv_sz)
            new_sz = jnp.minimum(
                cfgs.bkv_sz - bkv_sz_cache,
                kv_left_from_new,
            )

            schedule.cell[local, target_lane, _CELL_Q_SRC] = q_start
            schedule.cell[local, target_lane,
                          _CELL_Q_SIZE] = jnp.minimum(q_len, cfgs.bq_sz)
            schedule.cell[local, target_lane, _CELL_PROCESSED_Q] = processed_q
            schedule.cell[local, target_lane,
                          _CELL_PROCESSED_KV] = processed_kv
            schedule.cell[local, target_lane, _CELL_EFFECTIVE_KV] = k_len

            for i in range(cfgs.bkv_p_cache):
                dst_vmem = i << cfgs.serve.page_size_log2
                dma_sz = jnp.clip(
                    kv_left_from_cache - dst_vmem,
                    0,
                    cfgs.serve.page_size,
                )
                dma_sz = utils.align_to(dma_sz, cfgs.num_lanes)
                page_idx = jnp.minimum(kv_p_start + i,
                                       cfgs.serve.pages_per_seq - 1)
                if shared is None:
                    src_hbm = seq_page_table_ref[page_idx]
                else:
                    physical_idx = (local * n +
                                    target_lane) * cfgs.bkv_p_cache + i
                    for group in range(num_groups):
                        pkv_cache_refs[group][physical_idx] = page_table_refs[
                            group][page_idx]
                    src_hbm = page_table_refs[0][page_idx]
                schedule.dma_kv_cache[local, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[local, target_lane, i, 1] = dma_sz

            def fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start):
                cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
                hbm_token_idx_base = q_end - kv_left_from_new
                new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
                num_pages_to_fetch = jnp.where(
                    new_sz > 0,
                    (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
                    0,
                )
                fetch_valid = jnp.where(i < num_pages_to_fetch, _FETCH_VALID,
                                        0)
                src_hbm = (hbm_token_idx_base -
                           new_tok_offset) + i * cfgs.serve.page_size
                fetch_vmem = (cache_pages + i) * cfgs.serve.page_size
                p_idx = jnp.minimum(
                    (processed_kv + slot_start) >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq - 1,
                )
                if shared is None:
                    dst_hbm = seq_page_table_ref[p_idx]
                else:
                    physical_idx = (local * n +
                                    target_lane) * cfgs.bkv_p_new + i
                    for group in range(num_groups):
                        pkv_new_refs[group][physical_idx] = page_table_refs[
                            group][p_idx]
                    dst_hbm = page_table_refs[0][p_idx]
                lane_lo = dst_vmem - slot_start
                wb_lane = (lane_lo // cfgs.num_lanes) * cfgs.num_lanes
                wb_end = pl.cdiv(lane_lo + dma_sz,
                                 cfgs.num_lanes) * cfgs.num_lanes
                wb_size = jnp.where(dma_sz > 0, wb_end - wb_lane, 0)
                width, _ = cfgs.wb_lane_bits
                update_valid = jnp.where(dma_sz > 0, _UPDATE_VALID, 0)
                flags = (fetch_valid | update_valid
                         | ((wb_lane // cfgs.num_lanes) << 2)
                         | ((wb_size // cfgs.num_lanes) << (2 + width)))
                schedule.dma_kv_new[local, target_lane, i,
                                    _DMA_NEW_SRC_HBM] = src_hbm
                schedule.dma_kv_new[local, target_lane, i,
                                    _DMA_NEW_DST_VMEM] = (fetch_vmem)
                schedule.dma_kv_new[local, target_lane, i,
                                    _DMA_NEW_DST_HBM] = dst_hbm
                schedule.dma_kv_new[local, target_lane, i,
                                    _DMA_NEW_SRC_VMEM] = (slot_start)
                schedule.dma_kv_new[local, target_lane, i,
                                    _DMA_NEW_FLAGS] = flags

            _emit_kv_new_metadata(
                cfgs=cfgs,
                bkv_sz_cache=bkv_sz_cache,
                new_sz=new_sz,
                fill_dma_kv_new=fill_dma_kv_new,
            )

        return None

    jax.lax.fori_loop(start_seq_idx, end_seq_idx, emit_sequence, None)


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sequence_order_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    schedule_hbm_ref: SlidingWindowSchedule,
    schedule_ref: SlidingWindowSchedule,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
) -> None:
    """Build compact sliding-window metadata one SMEM window at a time."""
    plan = schedule_ref.plan
    window_size = plan.sched_window
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)
    mask_upper = min(plan.worst_steps, window_size)

    def mask_window(step, _):
        for b_idx in range(cfgs.batch_size):
            for i in range(_CELL_SIZE):
                schedule_ref.cell[step, b_idx, i] = 0
            for i in range(cfgs.bkv_p_cache):
                for j in range(2):
                    schedule_ref.dma_kv_cache[step, b_idx, i, j] = 0
            for i in range(cfgs.bkv_p_new):
                for j in range(cfgs.dma_kv_new_size):
                    schedule_ref.dma_kv_new[step, b_idx, i, j] = 0
        return None

    def build_window(window_idx):
        window_lo = window_idx * window_size
        jax.lax.fori_loop(0, mask_upper, mask_window, None)
        compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            sequence_order_ref,
            page_indices_hbm_ref,
            schedule_ref,
            seq_page_table_ref,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
        )

        copies = []
        for hbm_ref, smem_ref in zip(flat_hbm, flat_smem):
            if hbm_ref.shape[0] > 1:
                smem_len = smem_ref.shape[0]
                raw_stride = smem_len // window_size
                copy_len = min(
                    ((mask_upper * raw_stride + 1023) // 1024) * 1024,
                    smem_len,
                )
                copy = pltpu.make_async_copy(
                    smem_ref.at[pl.ds(0, copy_len)],
                    hbm_ref.at[pl.ds(
                        pl.multiple_of(window_idx * smem_len, smem_len),
                        copy_len,
                    )],
                    dma_sem.at[0],
                )
                copy.start()
                copies.append(copy)
        jax.tree.map(lambda copy: copy.wait(), copies)

    build_window(0)
    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    actual_steps = pl.cdiv(end_seq_idx - start_seq_idx, cfgs.batch_size)
    schedule_ref.actual_steps[0] = actual_steps

    for hbm_ref, smem_ref in zip(flat_hbm, flat_smem):
        if hbm_ref.shape[0] == 1:
            copy = pltpu.make_async_copy(
                smem_ref.at[pl.ds(0, 1)],
                hbm_ref.at[pl.ds(0, 1)],
                dma_sem.at[0],
            )
            copy.start()
            copy.wait()

    num_windows_actual = jnp.minimum(
        pl.cdiv(actual_steps, window_size),
        plan.num_sched_windows,
    )

    def build_remaining(window_idx, _):
        build_window(window_idx)
        return None

    jax.lax.fori_loop(1, num_windows_actual, build_remaining, None)


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    page_indices: jax.Array,
    cfgs: decode_config.DecodeConfig,
    *,
    interpret: bool = False,
    sequence_order_override: jax.Array | None = None,
) -> SlidingWindowSchedule:
    """Generate the compact HBM schedule for standalone window decode."""
    validate_config(cfgs)
    plan = SchedulePlan.create(cfgs)
    hbm_shaped = SlidingWindowSchedule.create_hbm_shape_dtype(cfgs, plan=plan)
    smem_shaped = SlidingWindowSchedule.create_shape_dtype(
        cfgs, steps=plan.sched_window, plan=plan)
    page_indices_padded = jnp.pad(
        page_indices.reshape(cfgs.serve.num_seqs, cfgs.serve.pages_per_seq),
        ((0, 0), (0, plan.seq_page_table_size - cfgs.serve.pages_per_seq)),
    ).reshape(-1)
    sequence_order = (jnp.arange(cfgs.serve.num_seqs, dtype=jnp.int32)
                      if sequence_order_override is None else
                      sequence_order_override.astype(jnp.int32))

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=hbm_shaped,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=hbm_shaped.out_specs(),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.SMEM((plan.seq_page_table_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule_sliding_window",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        sequence_order,
        page_indices_padded,
    )


def rpa_metadata_schedule_kernel_shared(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sequence_order_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    schedule_hbm_ref,
    schedule_ref: SlidingWindowSchedule,
    dma_sem: jax.Ref,
    page_table_refs,
    page_dma_sem: jax.Ref,
    pkv_cache_refs,
    pkv_new_refs,
    *,
    cfgs: decode_config.DecodeConfig,
    num_groups: int,
) -> None:
    """Build all local-group decode schedules from one structural schedule."""
    plan = schedule_ref.plan
    if plan.num_shared_groups != num_groups:
        raise ValueError(
            f"Schedule plan has {plan.num_shared_groups} shared groups, got "
            f"{num_groups}.")
    n = cfgs.batch_size
    window_size = plan.sched_window
    page_indices_hbm_refs = tuple(page_indices_hbm_ref.at[group]
                                  for group in range(num_groups))
    shared = (
        pkv_cache_refs,
        pkv_new_refs,
        page_table_refs,
        page_indices_hbm_refs,
    )
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)
    flat_hbm_per_group = [
        jax.tree_util.tree_leaves(schedule_hbm_ref[group])
        for group in range(num_groups)
    ]
    mask_upper = min(plan.worst_steps, window_size)

    def mask_window(step, _):
        for b_idx in range(cfgs.batch_size):
            for i in range(_CELL_SIZE):
                schedule_ref.cell[step, b_idx, i] = 0
            for i in range(cfgs.bkv_p_cache):
                for j in range(2):
                    schedule_ref.dma_kv_cache[step, b_idx, i, j] = 0
            for i in range(cfgs.bkv_p_new):
                for j in range(cfgs.dma_kv_new_size):
                    schedule_ref.dma_kv_new[step, b_idx, i, j] = 0
        return None

    def splice_physical_pages(group, step):
        for b_idx in range(cfgs.batch_size):

            @pl.when(schedule_ref.cell[step, b_idx, _CELL_Q_SIZE] > 0)
            def _active(_b_idx=b_idx):
                for i in range(cfgs.bkv_p_cache):
                    schedule_ref.dma_kv_cache[
                        step, _b_idx, i,
                        0] = pkv_cache_refs[group][(step * n + _b_idx) *
                                                   cfgs.bkv_p_cache + i]
                for i in range(cfgs.bkv_p_new):
                    schedule_ref.dma_kv_new[
                        step, _b_idx, i,
                        _DMA_NEW_DST_HBM] = (pkv_new_refs[group][
                            (step * n + _b_idx) * cfgs.bkv_p_new + i])

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    actual_steps = pl.cdiv(end_seq_idx - start_seq_idx, cfgs.batch_size)
    schedule_ref.actual_steps[0] = actual_steps

    def build_window(window_idx):
        window_lo = window_idx * window_size
        jax.lax.fori_loop(0, mask_upper, mask_window, None)
        compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            sequence_order_ref,
            page_indices_hbm_ref,
            schedule_ref,
            None,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
            shared=shared,
        )
        active_steps = jnp.clip(actual_steps - window_lo, 0, window_size)

        for group in range(num_groups):

            def splice_step(step, _, _group=group):
                splice_physical_pages(_group, step)
                return None

            jax.lax.fori_loop(0, active_steps, splice_step, None)
            copies = []
            for hbm_ref, smem_ref in zip(flat_hbm_per_group[group], flat_smem):
                if hbm_ref.shape[0] > 1:
                    smem_len = smem_ref.shape[0]
                    raw_stride = smem_len // window_size
                    copy_len = min(
                        ((mask_upper * raw_stride + 1023) // 1024) * 1024,
                        smem_len,
                    )
                    copy = pltpu.make_async_copy(
                        smem_ref.at[pl.ds(0, copy_len)],
                        hbm_ref.at[pl.ds(
                            pl.multiple_of(window_idx * smem_len, smem_len),
                            copy_len,
                        )],
                        dma_sem.at[0],
                    )
                    copy.start()
                    copies.append(copy)
            for copy in copies:
                copy.wait()

    build_window(0)
    for group in range(num_groups):
        copy = pltpu.make_async_copy(
            schedule_ref.actual_steps.at[pl.ds(0, 1)],
            schedule_hbm_ref[group].actual_steps.at[pl.ds(0, 1)],
            dma_sem.at[0],
        )
        copy.start()
        copy.wait()

    num_windows_actual = jnp.minimum(
        pl.cdiv(actual_steps, window_size),
        plan.num_sched_windows,
    )

    def build_remaining(window_idx, _):
        build_window(window_idx)
        return None

    jax.lax.fori_loop(1, num_windows_actual, build_remaining, None)


def generate_rpa_metadata_shared(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    page_indices_list: list[jax.Array],
    cfgs: decode_config.DecodeConfig,
    *,
    interpret: bool = False,
    sequence_order_override: jax.Array | None = None,
) -> tuple[SlidingWindowSchedule, ...]:
    """Build one structural decode schedule for several local cache groups."""
    validate_config(cfgs)
    num_groups = len(page_indices_list)
    if num_groups < 1:
        raise ValueError(
            "page_indices_list must contain at least one cache group.")
    expected_shape = page_indices_list[0].shape
    if any(page_indices.shape != expected_shape
           for page_indices in page_indices_list):
        raise ValueError(
            "All local cache groups must use the same page-table shape.")

    plan = SchedulePlan.create(cfgs, num_shared_groups=num_groups)
    hbm_shaped = SlidingWindowSchedule.create_hbm_shape_dtype(cfgs, plan=plan)
    smem_shaped = SlidingWindowSchedule.create_shape_dtype(
        cfgs,
        steps=plan.sched_window,
        plan=plan,
    )

    def pad_page_indices(page_indices):
        return jnp.pad(
            page_indices.reshape(cfgs.serve.num_seqs,
                                 cfgs.serve.pages_per_seq),
            ((0, 0), (0, plan.seq_page_table_size - cfgs.serve.pages_per_seq)),
        ).reshape(-1)

    page_indices_stacked = jnp.stack(
        [pad_page_indices(page_indices) for page_indices in page_indices_list],
        axis=0,
    )
    sequence_order = (jnp.arange(cfgs.serve.num_seqs, dtype=jnp.int32)
                      if sequence_order_override is None else
                      sequence_order_override.astype(jnp.int32))
    window = plan.sched_window

    schedules = pl.pallas_call(
        functools.partial(
            rpa_metadata_schedule_kernel_shared,
            cfgs=cfgs,
            num_groups=num_groups,
        ),
        out_shape=(tuple(hbm_shaped for _ in range(num_groups)), ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=(tuple(hbm_shaped.out_specs()
                             for _ in range(num_groups)), ),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1, )),
                tuple(
                    pltpu.SMEM((plan.seq_page_table_size, ), jnp.int32)
                    for _ in range(num_groups)),
                pltpu.SemaphoreType.DMA((num_groups, )),
                tuple(
                    pltpu.SMEM(
                        (window * cfgs.batch_size * cfgs.bkv_p_cache, ),
                        jnp.int32,
                    ) for _ in range(num_groups)),
                tuple(
                    pltpu.SMEM(
                        (window * cfgs.batch_size * cfgs.bkv_p_new, ),
                        jnp.int32,
                    ) for _ in range(num_groups)),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule_sliding_window_shared",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        sequence_order,
        page_indices_stacked,
    )
    return schedules[0]
