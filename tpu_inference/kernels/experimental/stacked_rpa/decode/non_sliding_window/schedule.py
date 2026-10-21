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
"""General split-K scheduler retained for global and legacy decode callers."""

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
    config as decode_config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Map a padded physical 1-D buffer to a logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        # Keep each full-window DMA aligned to the 1-D int32 tile.
        n = int(np.prod(shape))
        n_pad = max(1024, -(-n // 1024) * 1024)
        return cls(data=jax.ShapeDtypeStruct((n_pad, ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


@dataclasses.dataclass(frozen=True)
class SchedulePlan:
    """Static sizing and windowing plan for the global decode schedule."""

    max_steps_ub: int
    worst_steps: int
    fits_one_window: bool
    sched_window: int
    num_sched_windows: int
    total_steps_ub: int
    seq_page_table_size: int

    @classmethod
    def create(cls, cfgs: decode_config.DecodeConfig) -> "SchedulePlan":
        """Derive global decode bounds from the physical schedule schema."""
        cfgs.validate_decode()
        if cfgs.model.sliding_window is not None:
            raise ValueError("Global decode does not accept a sliding window.")
        seq_page_table_size = -(-cfgs.serve.pages_per_seq // 1024) * 1024
        max_steps_ub = calculate_max_steps_ub(
            cfgs,
            num_buffers=1,
            seq_page_table_size=seq_page_table_size,
        )

        max_model_len = cfgs.serve.pages_per_seq * cfgs.serve.page_size
        num_k = -(-max_model_len // cfgs.bkv_sz)
        num_q = -(-cfgs.serve.total_q_tokens // cfgs.bq_sz)
        total_work = max(cfgs.serve.num_seqs * num_k, num_q * num_k)
        worst_steps = -(-total_work // cfgs.batch_size)
        worst_steps += num_k + cfgs.n_buffer + 1

        fits_one_window = worst_steps <= max_steps_ub
        sched_window = (max_steps_ub
                        if fits_one_window else calculate_max_steps_ub(
                            cfgs,
                            num_buffers=2,
                            seq_page_table_size=seq_page_table_size,
                        ))
        num_sched_windows = max(1, -(-worst_steps // sched_window))
        return cls(
            max_steps_ub=max_steps_ub,
            worst_steps=worst_steps,
            fits_one_window=fits_one_window,
            sched_window=sched_window,
            num_sched_windows=num_sched_windows,
            total_steps_ub=num_sched_windows * sched_window,
            seq_page_table_size=seq_page_table_size,
        )


def _schedule_wrappers(cfgs: decode_config.DecodeConfig,
                       steps: int) -> dict[str, SmemWrapper]:
    """Build the padded data leaves that define the physical schema."""
    idx_wrapper = SmemWrapper.create_shape_dtype((steps, cfgs.batch_size))
    return {
        "s_idx":
        idx_wrapper,
        "k_idx":
        idx_wrapper,
        "is_last_k":
        idx_wrapper,
        "do_writeback":
        idx_wrapper,
        "skip_mask":
        idx_wrapper,
        "combine_span":
        idx_wrapper,
        "is_final":
        idx_wrapper,
        "dma_q":
        SmemWrapper.create_shape_dtype((steps, cfgs.batch_size, 2)),
        "dma_kv_cache":
        SmemWrapper.create_shape_dtype(
            (steps, cfgs.batch_size, cfgs.bkv_p_cache, 3)),
        "dma_kv_new":
        SmemWrapper.create_shape_dtype((
            steps,
            cfgs.batch_size,
            cfgs.bkv_p_new,
            cfgs.dma_kv_new_size,
        )),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DecodeSchedule:
    """Decode metadata schema with one implicit query block per sequence."""

    s_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    skip_mask: SmemWrapper  # [steps, batch]
    combine_span: SmemWrapper  # [steps, batch]
    is_final: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
    dma_kv_new: SmemWrapper  # [steps, batch, bkv_p_new, 5]
    actual_steps: Any  # [1]

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
    ) -> "DecodeSchedule":
        cfgs.validate_decode()
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        if steps is None:
            steps = plan.max_steps_ub
        return cls(
            **_schedule_wrappers(cfgs, steps),
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
    ) -> "DecodeSchedule":
        """Create HBM leaves as padded, independently aligned SMEM windows."""
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

    def get_dma_kv_cache(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        physical_page = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_vmem = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        size = self.dma_kv_cache[step, batch_idx, page_idx, 2]
        return physical_page, dst_vmem, size

    def get_dma_fetch_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, 0]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx, 1]
        valid = self.dma_kv_new[step, batch_idx, page_idx, 4] & 1
        return src_hbm, dst_vmem, valid

    def get_dma_update_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, 2]
        src_vmem = self.dma_kv_new[step, batch_idx, page_idx, 3]
        packed = self.dma_kv_new[step, batch_idx, page_idx, 4]
        valid = (packed >> 1) & 1
        width, mask = self.cfgs.wb_lane_bits
        wb_lane = ((packed >> 2) & mask) * self.cfgs.num_lanes
        wb_size = ((packed >> (2 + width)) & mask) * self.cfgs.num_lanes
        return dst_hbm, src_vmem, valid, wb_lane, wb_size

    def get_dma_q(
            self, step: jax.typing.ArrayLike,
            batch_idx: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        src_hbm = self.dma_q[step, batch_idx, 0]
        size = self.dma_q[step, batch_idx, 1]
        return src_hbm, size

    def get_q_idx(self, _step: jax.typing.ArrayLike,
                  _batch_idx: jax.typing.ArrayLike) -> int:
        """Return decode's single implicit query-block index."""
        return 0

    def scratch_shapes(self):
        """Return the schedule PyTree as SMEM scratch-memory descriptors."""
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def in_specs(self):
        """Keep large schedule leaves in HBM and scalar-prefetch actual_steps."""

        def wrapper(value):
            memory_space = pltpu.SMEM if value.size == 1 else pltpu.HBM
            return pl.BlockSpec(memory_space=memory_space)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        """Materialize every schedule-builder output in HBM."""
        return jax.tree.map(lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
                            self)


# Keep explicit headroom for allocations that are not described by shape/dtype
# trees, including compiler padding, semaphores, and small spill slots.
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
    """Describe non-schedule SMEM owned by the metadata builder."""
    int32 = jnp.int32
    return {
        "cu_q_lens":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs + 1, ), int32),
        "kv_lens":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs, ), int32),
        "distribution":
        jax.ShapeDtypeStruct((len(common_configs.RpaCase), ), int32),
        "sorted_seq_idx":
        jax.ShapeDtypeStruct((cfgs.serve.num_seqs, ), int32),
        "lane_lengths":
        jax.ShapeDtypeStruct((cfgs.batch_size, ), int32),
        "seq_page_table":
        jax.ShapeDtypeStruct((seq_page_table_size, ), int32),
    }


def schedule_smem_usage_bytes(
    cfgs: decode_config.DecodeConfig,
    steps: int,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
) -> int:
    """Return the schema-derived footprint for equally sized SMEM buffers."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be >= 1, got {num_buffers}.")
    if seq_page_table_size is None:
        seq_page_table_size = -(-cfgs.serve.pages_per_seq // 1024) * 1024

    schedule_shapes = _schedule_wrappers(cfgs, steps)
    schedule_shapes["actual_steps"] = jax.ShapeDtypeStruct((1, ), jnp.int32)
    return _shape_tree_nbytes(
        _metadata_fixed_smem_shapes(cfgs, seq_page_table_size)) + (
            num_buffers * _shape_tree_nbytes(schedule_shapes))


def calculate_max_steps_ub(
    cfgs: decode_config.DecodeConfig,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
) -> int:
    """Find the largest lane-aligned per-buffer capacity that fits in SMEM."""
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
        ) <= smem_budget)

    if not fits(1):
        raise ValueError(
            "Decode schedule cannot fit one lane-aligned step group in SMEM: "
            f"{smem_budget=} {num_lanes=} {num_buffers=}.")

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


def get_kv_block_start(
    k_idx,
    *,
    cfgs: decode_config.DecodeConfig,
    k_len,
    q_len,
):
    """Return decode's token and logical-page starts for a KV block."""
    del k_len, q_len
    return k_idx * cfgs.bkv_sz, k_idx * cfgs.bkv_p


def emit_kv_new_metadata(
    *,
    cfgs: decode_config.DecodeConfig,
    bkv_sz_cache,
    new_sz,
    fill_dma_kv_new,
) -> None:
    """Emit the bounded new-KV DMA records used by decode."""
    if cfgs.decode_q_len > 1:
        # The new tokens sit at the cache boundary and span a small bounded leaf.
        base_page = bkv_sz_cache // cfgs.serve.page_size
        for i in range(cfgs.bkv_p_new):
            slot_start = (base_page + i) * cfgs.serve.page_size
            slot_end = slot_start + cfgs.serve.page_size
            dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
            end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
            dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
            fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)
    elif cfgs.bkv_p_new < cfgs.bkv_p:
        assert cfgs.bkv_p_new == 1
        slot_start = (bkv_sz_cache //
                      cfgs.serve.page_size) * cfgs.serve.page_size
        fill_dma_kv_new(0, bkv_sz_cache, new_sz, slot_start)
    else:
        for i in range(max(cfgs.bkv_p, cfgs.bkv_p_new)):
            slot_start = i * cfgs.serve.page_size
            slot_end = slot_start + cfgs.serve.page_size
            dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
            end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
            dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
            fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: DecodeSchedule,
    lane_lengths_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
    window_lo: int = 0,
    sorted_seq_idx_ref: jax.Ref | None = None,
) -> None:
    """Place decode requests and encode each cell's DMA metadata."""
    cfgs.validate_decode()
    if sorted_seq_idx_ref is None:
        raise ValueError("Decode scheduling requires a sorted sequence order.")

    n = cfgs.batch_size
    window_size = schedule.plan.sched_window
    pps_pad = schedule.plan.seq_page_table_size

    def copy_page_table(s_idx):
        seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
        page_copy = pltpu.make_async_copy(
            page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
            seq_page_table_ref.at[pl.ds(0, pps_pad)],
            page_dma_sem.at[0],
        )
        page_copy.start()
        page_copy.wait()

    @jax.named_scope("emit_cell_metadata")
    def emit_cell_metadata(
        k_idx,
        step,
        *,
        target_lane,
        s_idx,
        q_end,
        q_src,
        q_sz_task,
        k_len,
        q_len,
        end_k_idx,
        kv_len_start,
        kv_p_start,
    ):
        in_window = jnp.logical_and(step >= window_lo, step
                                    < window_lo + window_size)
        local = jnp.clip(step - window_lo, 0, window_size - 1)

        kv_left = k_len - kv_len_start
        if cfgs.update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        else:
            kv_left_frm_cache = kv_left

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)
        do_writeback = jnp.where(cfgs.update_kv_cache & (new_sz > 0), 1, 0)

        min_q_pos = k_len - q_len
        skip_mask = jnp.where(
            (not cfgs.disable_skip_mask)
            & (kv_len_start + cfgs.bkv_sz - 1 <= min_q_pos),
            1,
            0,
        )

        def fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start):
            cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
            hbm_token_idx_base = q_end - kv_left_frm_new
            new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
            num_pages_to_fetch = jnp.where(
                new_sz > 0,
                (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
                0,
            )

            fetch_dma_valid = jnp.where(i < num_pages_to_fetch, 1, 0)
            new_page_start = (hbm_token_idx_base -
                              new_tok_offset) + i * cfgs.serve.page_size
            fetch_vmem = (cache_pages + i) * cfgs.serve.page_size

            p_idx = jnp.minimum(
                (kv_len_start + slot_start) >> cfgs.serve.page_size_log2,
                cfgs.serve.pages_per_seq - 1,
            )
            dst_hbm = seq_page_table_ref[p_idx]
            lane_lo = dst_vmem - slot_start
            wb_lane = (lane_lo // cfgs.num_lanes) * cfgs.num_lanes
            wb_end = pl.cdiv(lane_lo + dma_sz, cfgs.num_lanes) * cfgs.num_lanes
            wb_size = jnp.where(dma_sz > 0, wb_end - wb_lane, 0)
            width, _ = cfgs.wb_lane_bits
            packed_dma_valid = (fetch_dma_valid
                                | (jnp.where(dma_sz > 0, 1, 0) << 1)
                                | ((wb_lane // cfgs.num_lanes) << 2)
                                | ((wb_size // cfgs.num_lanes) << (2 + width)))

            schedule.dma_kv_new[local, target_lane, i, 0] = new_page_start
            schedule.dma_kv_new[local, target_lane, i, 1] = fetch_vmem
            schedule.dma_kv_new[local, target_lane, i, 2] = dst_hbm
            schedule.dma_kv_new[local, target_lane, i, 3] = slot_start
            schedule.dma_kv_new[local, target_lane, i, 4] = packed_dma_valid

        @pl.when(in_window)
        def _emit_writes():
            schedule.s_idx[local, target_lane] = s_idx
            schedule.k_idx[local, target_lane] = k_idx
            schedule.is_last_k[local, target_lane] = jnp.where(
                k_idx == end_k_idx - 1, 1, 0)

            schedule.dma_q[local, target_lane, 0] = q_src
            schedule.dma_q[local, target_lane, 1] = q_sz_task

            for i in range(cfgs.bkv_p_cache):
                dst_vmem = i << cfgs.serve.page_size_log2
                dma_sz = jnp.clip(kv_left_frm_cache - dst_vmem, 0,
                                  cfgs.serve.page_size)
                dma_sz = utils.align_to(dma_sz, cfgs.num_lanes)
                src_hbm = seq_page_table_ref[jnp.minimum(
                    kv_p_start + i, cfgs.serve.pages_per_seq - 1)]
                schedule.dma_kv_cache[local, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[local, target_lane, i, 1] = dst_vmem
                schedule.dma_kv_cache[local, target_lane, i, 2] = dma_sz

            schedule.do_writeback[local, target_lane] = do_writeback
            schedule.skip_mask[local, target_lane] = skip_mask

            emit_kv_new_metadata(
                cfgs=cfgs,
                bkv_sz_cache=bkv_sz_cache,
                new_sz=new_sz,
                fill_dma_kv_new=fill_dma_kv_new,
            )

    def write_group_metadata(step, cell, span, is_final, active):
        in_window = jnp.logical_and(step >= window_lo, step
                                    < window_lo + window_size)
        local = jnp.clip(step - window_lo, 0, window_size - 1)

        @pl.when(jnp.logical_and(in_window, active))
        def _write():
            schedule.combine_span[local, cell] = span
            schedule.is_final[local, cell] = is_final

    @jax.named_scope("dense_seq_loop")
    def dense_seq_loop(order_idx, cursor):
        s_idx = sorted_seq_idx_ref[order_idx]
        copy_page_table(s_idx)

        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        k_len = kv_lens_ref[s_idx]
        q_len = q_end - q_start
        num_k = pl.cdiv(k_len, cfgs.bkv_sz)
        q_sz_task = jnp.clip(q_len, 0, cfgs.bq_sz)

        start_k_idx = 0
        end_k_idx = jnp.minimum(
            num_k,
            (k_len - q_len + q_sz_task - 1) // cfgs.bkv_sz + 1,
        )

        # Empty decode slots are legal padding; they must not advance the cursor.
        end_k_idx = jnp.where(q_len > 0, end_k_idx, start_k_idx)
        num_k_for_q = jnp.maximum(end_k_idx - start_k_idx, 0)
        cursor_end = cursor + num_k_for_q

        def k_loop(k_idx, _):
            global_cell = cursor + (k_idx - start_k_idx)
            kv_len_start, kv_p_start = get_kv_block_start(k_idx,
                                                          cfgs=cfgs,
                                                          k_len=k_len,
                                                          q_len=q_len)
            emit_cell_metadata(
                k_idx,
                global_cell // n,
                target_lane=global_cell % n,
                s_idx=s_idx,
                q_end=q_end,
                q_src=q_start,
                q_sz_task=q_sz_task,
                k_len=k_len,
                q_len=q_len,
                end_k_idx=end_k_idx,
                kv_len_start=kv_len_start,
                kv_p_start=kv_p_start,
            )
            return None

        jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop, None)

        first_step = cursor // n
        last_step = jnp.maximum(cursor_end - 1, cursor) // n

        def step_metadata(step, _):
            lo = jnp.maximum(cursor, step * n)
            hi = jnp.minimum(cursor_end, (step + 1) * n)
            root = lo - step * n
            span = hi - lo
            is_final = jnp.where(step == last_step, 1, 0)
            write_group_metadata(step, root, span, is_final, num_k_for_q > 0)
            return None

        @pl.when(num_k_for_q > 0)
        def _emit_group_metadata():
            jax.lax.fori_loop(first_step, last_step + 1, step_metadata, None)

        return cursor_end

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    final_cursor = jax.lax.fori_loop(start_seq_idx, end_seq_idx,
                                     dense_seq_loop, jnp.int32(0))
    lane_lengths_ref[0] = pl.cdiv(final_cursor, n)


def rpa_metadata_schedule_kernel(
    # Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    # HBM input streamed one sequence at a time during the build.
    page_indices_hbm_ref: jax.Ref,
    # Output and scratch.
    schedule_hbm_ref: DecodeSchedule,
    schedule_ref: DecodeSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
    sorted_seq_idx_ref: jax.Ref | None = None,
) -> None:
    """Build decode metadata one SMEM window at a time and stream it to HBM."""
    plan = schedule_ref.plan
    window_size = plan.sched_window
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)
    mask_upper = min(plan.worst_steps, window_size)

    @jax.named_scope("mask_window")
    def mask_window(step, _):
        for b_idx in range(cfgs.batch_size):
            schedule_ref.s_idx[step, b_idx] = -1
            schedule_ref.k_idx[step, b_idx] = 0
            schedule_ref.is_last_k[step, b_idx] = 0
            schedule_ref.do_writeback[step, b_idx] = 0
            schedule_ref.skip_mask[step, b_idx] = 0
            schedule_ref.combine_span[step, b_idx] = 0
            schedule_ref.is_final[step, b_idx] = 0
            schedule_ref.dma_q[step, b_idx, 0] = 0
            schedule_ref.dma_q[step, b_idx, 1] = 0
            for i in range(cfgs.bkv_p_cache):
                schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
                schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
                schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0
            for i in range(cfgs.bkv_p_new):
                for j in range(cfgs.dma_kv_new_size):
                    schedule_ref.dma_kv_new[step, b_idx, i, j] = 0

    def build_window(window_idx):
        window_lo = window_idx * window_size

        for b_idx in range(cfgs.batch_size):
            lane_lengths_ref[b_idx] = 0
        jax.lax.fori_loop(0, mask_upper, mask_window, None)

        compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            schedule_ref,
            lane_lengths_ref,
            page_indices_hbm_ref,
            seq_page_table_ref,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
            sorted_seq_idx_ref=sorted_seq_idx_ref,
        )

        dma_list = []
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
                dma_list.append(copy)
        jax.tree.map(lambda copy: copy.wait(), dma_list)

    build_window(0)
    max_steps = 0
    for b_idx in range(cfgs.batch_size):
        max_steps = jnp.maximum(max_steps, lane_lengths_ref[b_idx])
    schedule_ref.actual_steps[0] = max_steps

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
        pl.cdiv(max_steps, window_size),
        plan.num_sched_windows,
    )

    def build_remaining(window_idx, _):
        build_window(window_idx)
        return None

    jax.lax.fori_loop(1, num_windows_actual, build_remaining, None)


def rpa_metadata_schedule_kernel_stacked(
    # Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sorted_seq_idx_ref: jax.Ref,
    # HBM input streamed one sequence at a time during the build.
    page_indices_hbm_ref: jax.Ref,
    # Output and scratch.
    schedule_hbm_ref: DecodeSchedule,
    schedule_ref: DecodeSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
) -> None:
    """Stacked builder entry with the decode sequence order scalar-prefetched."""
    rpa_metadata_schedule_kernel(
        cu_q_lens_ref,
        kv_lens_ref,
        distribution_ref,
        page_indices_hbm_ref,
        schedule_hbm_ref,
        schedule_ref,
        lane_lengths_ref,
        dma_sem,
        seq_page_table_ref,
        page_dma_sem,
        cfgs=cfgs,
        sorted_seq_idx_ref=sorted_seq_idx_ref,
    )


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    page_indices: jax.Array,
    cfgs: decode_config.DecodeConfig,
    *,
    interpret=False,
    sorted_seq_idx_override: jax.Array | None = None,
) -> DecodeSchedule:
    """Generate the HBM-resident dense decode schedule."""
    cfgs.validate_decode()

    plan = SchedulePlan.create(cfgs)
    hbm_shaped = DecodeSchedule.create_hbm_shape_dtype(cfgs, plan=plan)
    smem_shaped = DecodeSchedule.create_shape_dtype(cfgs,
                                                    steps=plan.sched_window,
                                                    plan=plan)
    fixed_smem = _metadata_fixed_smem_shapes(cfgs, plan.seq_page_table_size)

    page_indices_padded = jnp.pad(
        page_indices.reshape(cfgs.serve.num_seqs, cfgs.serve.pages_per_seq),
        ((0, 0), (0, plan.seq_page_table_size - cfgs.serve.pages_per_seq)),
    ).reshape(-1)

    seq_idx = jnp.arange(cfgs.serve.num_seqs, dtype=jnp.int32)
    start_order, end_order = cfgs.mode.get_range(distribution)
    in_mode = jnp.logical_and(seq_idx >= start_order, seq_idx < end_order)
    sort_key = jnp.where(in_mode, kv_lens, -1).astype(jnp.int32)
    local_perm = jnp.argsort(-sort_key).astype(jnp.int32)
    gather_pos = jnp.clip(seq_idx - start_order, 0, cfgs.serve.num_seqs - 1)
    sorted_seq_idx = local_perm[gather_pos]
    if sorted_seq_idx_override is not None:
        sorted_seq_idx = sorted_seq_idx_override.astype(jnp.int32)

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel_stacked, cfgs=cfgs),
        out_shape=hbm_shaped,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=hbm_shaped.out_specs(),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SMEM(
                    fixed_smem["lane_lengths"].shape,
                    fixed_smem["lane_lengths"].dtype,
                ),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.SMEM(
                    fixed_smem["seq_page_table"].shape,
                    fixed_smem["seq_page_table"].dtype,
                ),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule_stacked",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        sorted_seq_idx,
        page_indices_padded,
    )
