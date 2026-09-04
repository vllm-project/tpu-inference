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

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, utils


class FieldOffset:
    """A Python descriptor that generates the `.at[pos + offset]` lazy lookup.

    This is necessary because JAX does not support dynamically slicing a
    range (e.g. `data.at[pos:pos+4]`) using traced indices inside a loop,
    but it natively supports retrieving/updating single dynamically-indexed
    elements (e.g. `data.at[pos+1]`).
    """

    def __init__(self, offset: int | None = None, is_abstract: bool = False):
        if offset is None:
            assert is_abstract

        self.offset = offset
        self.__isabstractmethod__ = is_abstract

    def __get__(self, obj, objtype=None):
        return obj.data.at[obj.pos + self.offset]


@dataclasses.dataclass(frozen=True)
class DmaNew(ABC):
    data: Any
    pos: Any

    # HBM address to fetch new KV tokens from
    fetch_hbm = FieldOffset(is_abstract=True)
    # VMEM offset within the block where new tokens are placed
    fetch_vmem = FieldOffset(is_abstract=True)
    # HBM address to write the updated KV cache block back to
    wb_hbm = FieldOffset(is_abstract=True)
    # VMEM offset of the cache block to write back
    wb_vmem = FieldOffset(is_abstract=True)
    # Bitpacked: Flags for whether to fetch or write back new tokens.
    _flags = FieldOffset(is_abstract=True)

    @staticmethod
    @abstractmethod
    def num_fields() -> int:
        ...

    @abstractmethod
    def set_flags(self, fetch_val, wb_val):
        ...

    @property
    @abstractmethod
    def fetch_val(self):
        ...

    @property
    @abstractmethod
    def wb_val(self):
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeqAlongLaneDmaNew(DmaNew):
    fetch_hbm = FieldOffset(0)
    fetch_vmem = FieldOffset(1)
    wb_hbm = FieldOffset(2)
    wb_vmem = FieldOffset(3)
    _flags = FieldOffset(4)

    @staticmethod
    def num_fields() -> int:
        return 5

    @property
    def fetch_val(self):
        return self._flags[...] & 1

    @property
    def wb_val(self):
        return (self._flags[...] >> 1) & 1

    def set_flags(self, fetch_val, wb_val):
        self._flags[...] = fetch_val | (wb_val << 1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNew(DmaNew):
    wb_hbm = FieldOffset(0)
    fetch_hbm = FieldOffset(1)
    fetch_vmem = FieldOffset(2)
    wb_vmem = FieldOffset(2)
    _flags = FieldOffset(3)

    @staticmethod
    def num_fields() -> int:
        return 4

    @property
    def fetch_val(self):
        return self._flags[...]

    @property
    def wb_val(self):
        return self._flags[...]

    def set_flags(self, fetch_val, wb_val):
        self._flags[...] = fetch_val


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1-D data into logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        return cls(data=jax.ShapeDtypeStruct((np.prod(shape), ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices, )
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemArrayOfStructs(SmemWrapper):
    """Maps physical 1-D data into logical Array of Structs."""

    struct_cls: type[DmaNew] = dataclasses.field(metadata=dict(static=True))
    struct_size: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape, struct_cls, struct_size):  # pytype: disable=bad-override
        assert struct_size == struct_cls.num_fields()
        return cls(
            data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size, ),
                                      jnp.int32),
            shape=shape,
            struct_cls=struct_cls,
            struct_size=struct_size,
        )

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos * self.struct_size

    def __getitem__(self, indices):
        pos_start = self._get_pos(indices)
        return self.struct_cls(self.data, pos_start)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
    """Container for metadata arrays with integrated shape/spec logic."""

    s_idx: SmemWrapper  # [steps, batch]
    q_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
    dma_kv_new: SmemArrayOfStructs  # [steps, batch, bkv_p_new]
    total_wait_kv_in: SmemWrapper  # [steps]
    total_wait_kv_out: SmemWrapper  # [steps]
    total_wait_q_in: SmemWrapper  # [steps]
    total_wait_o_out: SmemWrapper  # [steps]
    total_wait_lse_out: SmemWrapper  # [steps]
    actual_steps: Any  # [1]

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(
        cls,
        cfgs: configs.RpaConfigs,
        dma_kv_new_struct_cls: type[DmaNew] = SeqAlongLaneDmaNew,
        multiplier: int = 1,
    ):
        effective_max_steps = cfgs.max_steps_ub * multiplier

        idx_wrapper = SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size))

        steps_wrapper = SmemWrapper.create_shape_dtype((effective_max_steps, ))

        return cls(
            s_idx=idx_wrapper,
            q_idx=idx_wrapper,
            k_idx=idx_wrapper,
            is_last_k=idx_wrapper,
            do_writeback=idx_wrapper,
            dma_q=SmemWrapper.create_shape_dtype(
                (effective_max_steps, cfgs.batch_size, 2)),
            dma_kv_cache=SmemWrapper.create_shape_dtype(
                (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_cache, 3)),
            dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
                (
                    effective_max_steps,
                    cfgs.batch_size,
                    cfgs.bkv_p_new,
                ),
                struct_cls=dma_kv_new_struct_cls,
                struct_size=cfgs.dma_kv_new_size,
            ),
            total_wait_kv_in=steps_wrapper,
            total_wait_kv_out=steps_wrapper,
            total_wait_q_in=steps_wrapper,
            total_wait_o_out=steps_wrapper,
            total_wait_lse_out=steps_wrapper,
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            cfgs=cfgs,
        )

    def get_dma_kv_cache(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # 0: src_hbm, 1: dst_vmem, 2: size
        src_off = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_off = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        sz = self.dma_kv_cache[step, batch_idx, page_idx, 2]
        return src_off, dst_off, sz

    def get_dma_q(
            self, step: jax.typing.ArrayLike,
            batch_idx: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        # 0: src_hbm, 1: size
        src_hbm = self.dma_q[step, batch_idx, 0]
        sz = self.dma_q[step, batch_idx, 1]
        return src_hbm, sz

    def scratch_shapes(self):
        """Returns a Pytree of SMEM scratch memory."""

        return jax.tree.map(
            lambda x: pltpu.SMEM(x.shape, x.dtype),
            self,
        )

    def in_specs(self):
        """Returns a Pytree of input BlockSpecs."""

        def wrapper(x):
            if x.size == 1:
                return pl.BlockSpec(memory_space=pltpu.SMEM)
            else:
                # Since we use maximum upper bound when allocating scheduler data,
                # it is not feasible to use scalar prefetch and fetch entire scheduler
                # data into the kernel. Instead, we stored it to HBM first and perform
                # dynamic sized DMA inside the kernel using actual number of steps.
                return pl.BlockSpec(memory_space=pltpu.HBM)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        """Returns a Pytree of output BlockSpecs."""

        return jax.tree.map(
            lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
            self,
        )


def _mask_out_steps(
    step: jax.typing.ArrayLike,
    schedule_smem: RpaSchedule,
    b_idx: jax.typing.ArrayLike,
):
    """Mask out unvisited metadata schedule entries at (step, b_idx)."""
    schedule_smem.s_idx[step, b_idx] = -1
    schedule_smem.q_idx[step, b_idx] = 0
    schedule_smem.k_idx[step, b_idx] = 0
    schedule_smem.is_last_k[step, b_idx] = 0
    schedule_smem.do_writeback[step, b_idx] = 0

    schedule_smem.dma_q[step, b_idx, 0] = 0
    schedule_smem.dma_q[step, b_idx, 1] = 0

    for i in range(schedule_smem.cfgs.bkv_p_cache):
        schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
        schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0
        schedule_smem.dma_kv_cache[step, b_idx, i, 2] = 0

    for i in range(schedule_smem.cfgs.bkv_p_new):
        dma_entry = schedule_smem.dma_kv_new[step, b_idx, i]
        dma_entry.fetch_hbm[...] = 0
        dma_entry.fetch_vmem[...] = 0
        dma_entry.wb_hbm[...] = 0
        dma_entry.wb_vmem[...] = 0
        dma_entry.set_flags(0, 0)

    schedule_smem.total_wait_kv_in[step] = 0
    schedule_smem.total_wait_kv_out[step] = 0
    schedule_smem.total_wait_q_in[step] = 0
    schedule_smem.total_wait_o_out[step] = 0
    schedule_smem.total_wait_lse_out[step] = 0


def _write_schedule_to_hbm(
    schedule_smem: RpaSchedule,
    schedule_hbm: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    num_steps: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Writes `num_steps` of metadata from `schedule_smem` to `schedule_hbm` at `hbm_offset`."""
    hbm_offset_aligned = pl.multiple_of(hbm_offset, 128)  # pytype: disable=bad-argument-type
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm)
    flat_smem = jax.tree_util.tree_leaves(schedule_smem)
    dma_list = []
    for h, s in zip(flat_hbm, flat_smem):
        element_size = s.shape[0] // cfgs.max_steps_ub
        if h.shape[0] > 1:
            write_size = num_steps * element_size
            write_size = utils.align_to(write_size, 128)
            output_offset = hbm_offset_aligned * element_size
        else:
            write_size = h.shape[0]
            output_offset = 0

        copy = pltpu.make_async_copy(
            s.at[pl.ds(0, write_size)],
            h.at[pl.ds(output_offset, write_size)],
            dma_sem.at[0],
        )
        dma_list.append(copy)

    jax.tree.map(lambda x: x.start(), dma_list)
    jax.tree.map(lambda x: x.wait(), dma_list)


def _compute_waits(
    schedule: RpaSchedule,
    start_step: jax.typing.ArrayLike,
    end_step: jax.typing.ArrayLike,
    *,
    cfgs: configs.RpaConfigs,
):
    """Computes total wait row/lane counts for a range of steps."""

    kv_bytes_per_token = cfgs.kv_bytes_per_token
    q_bytes_per_token = cfgs.q_bytes_per_token
    o_bytes_per_token = cfgs.o_bytes_per_token

    # Reshape buffers to (-1, 128) in u32 to perform contiguous 512-byte lane
    # waits (1 row of 128 u32 words = 512 bytes).
    num_lanes = pltpu.get_tpu_info().num_lanes
    dma_chunk_size = num_lanes * 4

    @jax.named_scope("compute_waits")
    def body(step, _):
        # KV IN
        kv_in_tokens = 0
        for b in range(cfgs.batch_size):
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                for i in range(cfgs.bkv_p_cache):
                    _, _, dma_valid = schedule.get_dma_kv_cache(step, b, i)
                    kv_in_tokens += dma_valid * cfgs.serve.page_size
                for i in range(cfgs.bkv_p_new):
                    dma_entry = schedule.dma_kv_new[step, b, i]
                    kv_in_tokens += jnp.where(dma_entry.fetch_val > 0,
                                              cfgs.serve.page_size, 0)
            else:
                for i in range(cfgs.bkv_p_cache):
                    _, _, sz = schedule.get_dma_kv_cache(step, b, i)
                    kv_in_tokens += sz
                total_new_sz = 0
                for i in range(cfgs.bkv_p_new):
                    dma_entry = schedule.dma_kv_new[step, b, i]
                    total_new_sz += dma_entry.fetch_val
                kv_in_tokens += total_new_sz

        schedule.total_wait_kv_in[step] = (
            kv_in_tokens * kv_bytes_per_token) // dma_chunk_size

        # KV OUT
        kv_out_tokens = 0
        for b in range(cfgs.batch_size):
            do_writeback = schedule.do_writeback[step, b] == 1
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                for i in range(cfgs.bkv_p_new):
                    dma_entry = schedule.dma_kv_new[step, b, i]
                    kv_out_tokens += jnp.where(
                        do_writeback & (dma_entry.wb_val > 0),
                        cfgs.serve.page_size, 0)
            else:
                for i in range(cfgs.bkv_p_new):
                    dma_entry = schedule.dma_kv_new[step, b, i]
                    kv_out_tokens += jnp.where(do_writeback, dma_entry.wb_val,
                                               0)

        schedule.total_wait_kv_out[step] = (
            kv_out_tokens * kv_bytes_per_token) // dma_chunk_size

        # Q IN
        q_in_tokens = 0
        for b in range(cfgs.batch_size):
            _, q_sz = schedule.get_dma_q(step, b)
            q_in_tokens += q_sz
        schedule.total_wait_q_in[step] = (q_in_tokens *
                                          q_bytes_per_token) // dma_chunk_size

        # O OUT
        o_out_tokens = 0
        for b in range(cfgs.batch_size):
            is_last_k = schedule.is_last_k[step, b] == 1
            _, q_sz = schedule.get_dma_q(step, b)
            o_out_tokens += jnp.where(is_last_k, q_sz, 0)
        schedule.total_wait_o_out[step] = (o_out_tokens *
                                           o_bytes_per_token) // dma_chunk_size

        # LSE OUT
        lse_bytes_per_token = (cfgs.model.num_kv_heads * cfgs.lse_row_stride *
                               128 * cfgs.serve.dtype_out.itemsize)
        schedule.total_wait_lse_out[step] = (
            o_out_tokens * lse_bytes_per_token) // dma_chunk_size

    jax.lax.fori_loop(start_step, end_step, body, None)


def flush_to_hbm(
    count: jax.typing.ArrayLike,
    schedule: RpaSchedule,
    schedule_hbm_ref: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):

    last_step = utils.align_to(count, cfgs.batch_size)

    @pl.loop(count, last_step)
    @jax.named_scope("mask_out_steps")
    def body(idx):
        step, b_idx = divmod(idx, cfgs.batch_size)
        _mask_out_steps(step, b_idx=b_idx, schedule_smem=schedule)

    _compute_waits(schedule, 0, last_step // cfgs.batch_size, cfgs=cfgs)
    _write_schedule_to_hbm(
        schedule,
        schedule_hbm_ref,
        hbm_offset,
        cfgs.max_steps_ub,
        dma_sem,
        cfgs=cfgs,
    )

    return hbm_offset + cfgs.max_steps_ub, 0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
    hbm_offset: jax.Array | int
    count: jax.Array | int


class BaseMetadataComputer:
    """Base metadata scheduler computing seq -> q -> k loops."""

    @classmethod
    def get_rpa_schedule(cls,
                         cfgs: configs.RpaConfigs,
                         multiplier: int = 1) -> RpaSchedule:
        """Returns the RpaSchedule shape/dtype struct for this metadata computer."""
        dma_kv_new_struct_cls = (HeadAlongSublaneDmaNew if cfgs.serve.kv_layout
                                 == configs.KVLayout.HEAD_ALONG_SUBLANE else
                                 SeqAlongLaneDmaNew)
        return RpaSchedule.create_shape_dtype(
            cfgs,
            dma_kv_new_struct_cls=dma_kv_new_struct_cls,
            multiplier=multiplier,
        )

    def __init__(
            self,
            schedule: RpaSchedule,
            schedule_hbm_ref: RpaSchedule,
            dma_sem: jax.Ref,
            *,
            cfgs: configs.RpaConfigs,
            extra_refs: Sequence[jax.Ref] = (),
            **kwargs,
    ):
        self.schedule = schedule
        self.schedule_hbm_ref = schedule_hbm_ref
        self.dma_sem = dma_sem
        self.cfgs = cfgs
        self.extra_refs = extra_refs

    @jax.named_scope("k_loop")
    def k_loop(
        self,
        k_idx,
        carry: LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
    ) -> LoopCarry:
        cfgs = self.cfgs
        schedule = self.schedule
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)

        schedule.s_idx[step, target_lane] = s_idx
        schedule.q_idx[step, target_lane] = q_idx
        schedule.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule.is_last_k[step, target_lane] = is_last_k

        schedule.dma_q[step, target_lane, 0] = q_src
        schedule.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        k_len = kv_cache_len + kv_new_len
        kv_left = k_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
        p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = kv_left_frm_cache - dst_vmem
            dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

            src_hbm = jnp.minimum(p_offset + i,
                                  cfgs.serve.num_page_indices - 1)

            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                dma_valid = jnp.where(dma_sz > 0, 1, 0)
                schedule.dma_kv_cache[step, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                schedule.dma_kv_cache[step, target_lane, i, 2] = dma_valid
            else:
                schedule.dma_kv_cache[step, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                schedule.dma_kv_cache[step, target_lane, i, 2] = dma_sz

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - kv_cache_len)) // cfgs.bq_sz

        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        schedule.do_writeback[step, target_lane] = do_writeback
        src_hbm = q_end - kv_left_frm_new

        def fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start):
            dma_entry = schedule.dma_kv_new[step, target_lane, i]
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
                hbm_token_idx_base = q_end - kv_left_frm_new
                new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
                # If new_sz = 150, new_tok_offset = 120, page_size = 128.  The new
                # tokens occupy indices 120 through 269 relative to the HBM page
                # boundaries.  This spans 3 pages: [120-127], [128-255], and [256-269].
                # (120 + 150 - 1) // 128 + 1 = 269 // 128 + 1 = 3 pages.
                num_pages_to_fetch = jnp.where(
                    new_sz > 0,
                    (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
                    0,
                )
                fetch_val = jnp.where(i < num_pages_to_fetch, 1, 0)
                new_page_start = (hbm_token_idx_base -
                                  new_tok_offset) + i * cfgs.serve.page_size
                # Fetched pages of new tokens are placed sequentially in VMEM
                # immediately following the existing cached pages. E.g., if
                # cache_pages=2, new pages go to offsets 2*page_size, 3*page_size, etc.
                fetch_vmem = (cache_pages + i) * cfgs.serve.page_size
                p_idx = jnp.minimum(
                    (kv_len_start + slot_start) >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq - 1,
                )
                dst_hbm = s_idx * cfgs.serve.pages_per_seq + p_idx
                wb_val = jnp.where(dma_sz > 0, 1, 0)

                dma_entry.fetch_hbm[...] = new_page_start
                dma_entry.fetch_vmem[...] = fetch_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.wb_vmem[...] = slot_start
                dma_entry.set_flags(fetch_val, wb_val)
            else:
                p_idx = jnp.minimum(
                    (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq - 1,
                )
                p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
                dst_hbm = ((s_idx * cfgs.serve.pages_per_seq + p_idx) <<
                           cfgs.serve.page_size_log2) | p_off

                dma_entry.fetch_hbm[...] = src_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, dma_sz)

        if cfgs.block.bq_sz == 1:
            # Decode path
            assert cfgs.bkv_p_new == 1
            slot_start = (bkv_sz_cache //
                          cfgs.serve.page_size) * cfgs.serve.page_size
            fill_dma_kv_new(0, bkv_sz_cache, new_sz, slot_start)
        else:
            iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
            for i in range(iters):
                slot_start = i * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size

                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

                fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)

        return self.advance_carry(carry)

    def advance_carry(self, carry: LoopCarry) -> LoopCarry:
        new_count = carry.count + 1
        return jax.lax.cond(
            new_count < self.cfgs.max_steps_ub * self.cfgs.batch_size,
            lambda c: c,
            self.flush_carry,
            LoopCarry(carry.hbm_offset, new_count),
        )

    def flush_carry(self, carry: LoopCarry) -> LoopCarry:
        new_hbm_offset, _ = flush_to_hbm(
            carry.count,
            self.schedule,
            self.schedule_hbm_ref,
            carry.hbm_offset,
            self.dma_sem,
            cfgs=self.cfgs,
        )
        return LoopCarry(new_hbm_offset, 0)

    @jax.named_scope("q_loop")
    def q_loop(
        self,
        q_idx,
        carry: LoopCarry,
        *,
        s_idx,
        q_start,
        q_end,
        q_offset,
        kv_cache_len,
        kv_new_len,
        num_k,
    ) -> LoopCarry:
        cfgs = self.cfgs
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = q_offset + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

        end_k_idx_causal = (q_offset + q_idx * cfgs.bq_sz + q_sz_task -
                            1) // cfgs.bkv_sz + 1
        end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

        if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
            start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)

        k_loop_fn = functools.partial(
            self.k_loop,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            end_k_idx=end_k_idx,
        )

        return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

    @jax.named_scope("seq_loop")
    def seq_loop(
        self,
        s_idx,
        carry: LoopCarry,
        *,
        cu_q_lens_ref,
        q_offsets_ref,
        kv_cache_lens_ref,
        kv_new_lens_ref,
    ) -> LoopCarry:
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        q_offset = q_offsets_ref[s_idx]
        kv_cache_len = kv_cache_lens_ref[s_idx]
        kv_new_len = kv_new_lens_ref[s_idx]
        k_len = kv_cache_len + kv_new_len

        num_q = pl.cdiv(q_len, self.cfgs.bq_sz)
        num_k = pl.cdiv(k_len, self.cfgs.bkv_sz)

        q_loop_fn = functools.partial(
            self.q_loop,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            num_k=num_k,
        )

        return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

    def compute(
        self,
        cu_q_lens_ref: jax.Ref,
        q_offsets_ref: jax.Ref,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        distribution_ref: jax.Ref,
    ) -> LoopCarry:
        """Generates the metadata schedule across all sequences."""
        seq_loop_fn = functools.partial(
            self.seq_loop,
            cu_q_lens_ref=cu_q_lens_ref,
            q_offsets_ref=q_offsets_ref,
            kv_cache_lens_ref=kv_cache_lens_ref,
            kv_new_lens_ref=kv_new_lens_ref,
        )
        start_seq_idx, end_seq_idx = self.cfgs.mode.get_range(distribution_ref)  # pytype: disable=bad-argument-type
        init_carry = LoopCarry(0, 0)
        return jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop_fn,
                                 init_carry)


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    extra_scalars_ref: Sequence[jax.Ref],
    # Outputs.
    schedule_hbm_ref: RpaSchedule,
    # Scratch.
    schedule_ref: RpaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[BaseMetadataComputer] = BaseMetadataComputer,
):
    """Generates the HBM-to-VMEM DMA schedule.

    This kernel:
    1. Iterates through each (potentially ragged) sequence
    2. Breaks Queries (Q) and Key-Values (KV) into blocks (bq_sz, bkv_sz).
    3. Assigns tasks to 'lanes' (TPU batch items) based on current lane occupancy
      to ensure balanced execution across the batch dimension.
    4. Encodes DMA offsets:
      - dma_q: HBM start index and size for Query blocks.
      - dma_kv_cache: Paged indices for existing KV tokens.
      - dma_kv_new: offsets for new tokens being added to the cache.
      - do_writeback: boolean flag indicating if a block should be flushed to
        HBM (ie does this block contain new tokens to add to KV cache).

    Args:
      cu_q_lens_ref: [max_num_seqs + 1]. Cumulative sum of each sequence's query
        length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
        b=cu_q_lens[i+1] represents q/k/v of sequence i.
      q_offsets_ref: [max_num_seqs]. Starting Q index for each sequence.
      kv_cache_lens_ref: [max_num_seqs]. Existing kv cache length of each
        sequence.
      kv_new_lens_ref: [max_num_seqs]. New kv length of each sequence.
      distribution_ref: [3]. Cumulative sum of number of decode, prefill, and
        mixed
      extra_scalars_ref: Additional scalar refs for custom metadata computers.
      schedule_hbm_ref: HBM memory that will store output of the kernel.
      schedule_ref: Scratch memory where schedule results gets written.
      dma_sem: Semaphore used for writing scheduler output to HBM.
      cfgs: Configuration of the kernel.
      computer_cls: Metadata computer class to use for schedule generation.
    """
    # Step 1: Compute and fill scheduler metadata.
    computer = computer_cls(
        schedule=schedule_ref,
        schedule_hbm_ref=schedule_hbm_ref,
        dma_sem=dma_sem,
        cfgs=cfgs,
        extra_refs=extra_scalars_ref,
    )
    loop_carry = computer.compute(
        cu_q_lens_ref=cu_q_lens_ref,
        q_offsets_ref=q_offsets_ref,
        kv_cache_lens_ref=kv_cache_lens_ref,
        kv_new_lens_ref=kv_new_lens_ref,
        distribution_ref=distribution_ref,
    )
    count = loop_carry.count
    hbm_offset = loop_carry.hbm_offset
    steps = pl.cdiv(count, cfgs.batch_size) + hbm_offset
    schedule_ref.actual_steps[0] = steps

    # Step 3: Mask out unvisited steps.
    flush_to_hbm(
        count,
        schedule_ref,
        schedule_hbm_ref,
        hbm_offset,
        dma_sem,
        cfgs=cfgs,
    )


def generate_rpa_metadata(
        cu_q_lens: jax.Array,
        q_offsets: jax.Array,
        kv_cache_lens: jax.Array,
        kv_new_lens: jax.Array,
        distribution: jax.Array,
        cfgs: configs.RpaConfigs,
        *,
        computer_cls: type[BaseMetadataComputer] = BaseMetadataComputer,
        extra_scalars: Sequence[jax.Array] = (),
        interpret=False,
) -> RpaSchedule:
    """Generates RPA metadata schedule using the specified computer_cls."""
    schedule_shaped_dtype = computer_cls.get_rpa_schedule(cfgs)
    schedule_hbm = computer_cls.get_rpa_schedule(
        cfgs, multiplier=cfgs.max_schedule_size_multiplier)

    return pl.pallas_call(
        functools.partial(
            rpa_metadata_schedule_kernel,
            cfgs=cfgs,
            computer_cls=computer_cls,
        ),
        out_shape=schedule_hbm,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=[],
            out_specs=schedule_hbm.out_specs(),
            scratch_shapes=[
                schedule_shaped_dtype.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        extra_scalars,
    )
