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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs, schedule

# `pltpu.BufferType` compatibility shim. The JAX that ships with the build image
# (jax==0.9.2) does not expose `pltpu.BufferType`; the real `buffer_type` value is
# passed through from `emit_pipeline` to `BufferedRef.create`, so the annotations
# and asserts below are advisory only. Provide a permissive stand-in whose enum
# members compare equal to anything, turning those asserts into no-ops (matching
# the upstream base, which kept them commented out). Remove once the image ships
# a JAX that exposes `pltpu.BufferType`.
if not hasattr(pltpu, "BufferType"):

    class _AlwaysEq:

        def __eq__(self, _other):
            return True

        def __hash__(self):
            return 0

    class _BufferTypeCompat:

        def __getattr__(self, _name):
            return _AlwaysEq()

    pltpu.BufferType = _BufferTypeCompat()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pltpu.BufferedRef):
    """Helper class to safely bypass buffer_count checks during creation."""

    def __post_init__(self):
        # pallas doesn't allow you to set n_buffer > 2 for output refs, so
        # we override to bypass this check.
        pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KVBufferedRefSeqAlongLane(_BypassRef):
    """Handles fetching and updating KV cache using SEQ_ALONG_LANE memory layout."""

    cfgs: configs.RpaConfigs = dataclasses.field(default=None,
                                                 metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RpaConfigs,
    ):
        assert buffer_type == pltpu.BufferType.INPUT_OUTPUT

        standard_ref = _BypassRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("kv_copy_in")
    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # src_ref: (kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref)
        kv_cache_hbm, new_kv_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = jnp.maximum(grid_indices[0], 0)

        vmem_dst_lane = self.window_ref.at[slot]
        for b in range(self.cfgs.batch_size):
            # TODO(perf, kv-page-coalescing): this issues one make_async_copy per
            # KV page (bkv_p_cache descriptors per batch item), so at long context
            # kv_copy_in spends scalar-core time issuing many tiny descriptors and
            # caps DMA/compute overlap. From the fp8 128k trace, kv_copy_in is
            # ~13% of the RPAd op with the HBM engine idle ~half the time (the
            # kernel is co-bottlenecked on this + flash compute, ~50% HBM util).
            # Coalesce runs of pages that are contiguous in HBM AND land
            # contiguously in VMEM into a single larger async copy (one descriptor
            # per run) to cut issue occupancy. Requires a build-side run-length
            # pass over the schedule's physical page list (page-major in HBM vs
            # token-contiguous in VMEM), then emitting one copy per contiguous run.
            for i in range(self.cfgs.bkv_p_cache):
                p_idx, dst_off, dma_valid = schedule_ref.get_dma_kv_cache(
                    block_idx, b, i)
                hbm_p_idx = p_idx  # schedule stores the physical page directly
                sz = dma_valid * self.cfgs.serve.page_size
                dst_off = pl.multiple_of(dst_off, 128)
                sz = pl.multiple_of(sz, 128)
                pltpu.make_async_copy(
                    kv_cache_hbm.at[hbm_p_idx, :, :,
                                    pl.ds(0, sz)],
                    vmem_dst_lane.at[b, :, :, pl.ds(dst_off, sz)],
                    sem,
                ).start()

            for i in range(self.cfgs.bkv_p_new):
                src_new_off, dst_vmem_off, dma_valid = (
                    schedule_ref.get_dma_fetch_kv_new(block_idx, b, i))
                sz = dma_valid * self.cfgs.serve.page_size
                src_new_off = pl.multiple_of(src_new_off, 128)
                dst_vmem_off = pl.multiple_of(dst_vmem_off, 128)
                sz = pl.multiple_of(sz, 128)
                pltpu.make_async_copy(
                    new_kv_hbm.at[:, :, pl.ds(src_new_off, sz)],
                    vmem_dst_lane.at[b, :, :, pl.ds(dst_vmem_off, sz)],
                    sem,
                ).start()

    @jax.named_scope("kv_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        vmem_src_lane = self.window_ref.at[slot]
        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            for i in range(self.cfgs.bkv_p_new):
                dst_hbm_p, src_vmem_off, dma_valid = schedule_ref.get_dma_update_kv_new(
                    block_idx, b, i)
                hbm_p_idx = dst_hbm_p  # physical page (folded in at build)
                sz = jnp.where(do_writeback,
                               dma_valid * self.cfgs.serve.page_size, 0)
                src_vmem_off = pl.multiple_of(src_vmem_off, 128)
                sz = pl.multiple_of(sz, 128)
                pltpu.make_async_copy(
                    vmem_src_lane.at[b, :, :, pl.ds(src_vmem_off, sz)],
                    kv_out_ref.at[hbm_p_idx, :, :,
                                  pl.ds(0, sz)],
                    sem,
                ).start()

    @jax.named_scope("kv_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, _, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        for b in range(self.cfgs.batch_size):
            total_pages_b = 0
            for i in range(self.cfgs.bkv_p_cache):
                _, _, dma_valid = schedule_ref.get_dma_kv_cache(
                    block_idx, b, i)
                total_pages_b += dma_valid
            for i in range(self.cfgs.bkv_p_new):
                _, _, dma_valid = schedule_ref.get_dma_fetch_kv_new(
                    block_idx, b, i)
                total_pages_b += jnp.where(dma_valid > 0, 1, 0)

            sz = total_pages_b * self.cfgs.serve.page_size
            sz = pl.multiple_of(sz, 128)
            pltpu.make_async_copy(
                vmem_dst.at[b, :, :, pl.ds(0, sz)],
                vmem_dst.at[b, :, :, pl.ds(0, sz)],
                sem,
            ).wait()

    @jax.named_scope("kv_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            total_pages_b = 0
            for i in range(self.cfgs.bkv_p_new):
                _, _, dma_valid = schedule_ref.get_dma_update_kv_new(
                    block_idx, b, i)
                total_pages_b += jnp.where(do_writeback, dma_valid, 0)

            sz = total_pages_b * self.cfgs.serve.page_size
            sz = pl.multiple_of(sz, 128)
            dst_hbm_p, _, _ = schedule_ref.get_dma_update_kv_new(
                block_idx, b, 0)
            hbm_p_idx = dst_hbm_p  # physical page (folded in at build)
            pltpu.make_async_copy(
                kv_out_ref.at[hbm_p_idx, :, :, pl.ds(0, sz)],
                kv_out_ref.at[hbm_p_idx, :, :, pl.ds(0, sz)],
                sem,
            ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingORef(pltpu.BufferedRef):
    """Handles normalizing and storing the final attention output."""

    cfgs: configs.RpaConfigs = dataclasses.field(default=None,
                                                 metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RpaConfigs,
    ):
        assert buffer_type == pltpu.BufferType.OUTPUT

        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("o_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # dst_ref: (o_hbm, schedule_ref)
        o_hbm, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        dma_list = []
        for b in range(self.cfgs.batch_size):
            if self.cfgs.is_stacked and self.cfgs.dense_pack:
                out_flag = jnp.logical_and(
                    schedule_ref.combine_span[block_idx, b] > 0,
                    schedule_ref.is_final[block_idx, b] == 1,
                )
            elif self.cfgs.is_stacked:
                out_flag = schedule_ref.combine_span[block_idx, b] > 0
            else:
                out_flag = schedule_ref.is_last_k[block_idx, b] == 1
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            q_sz = jnp.where(out_flag, q_sz, 0)
            dma_list.append((q_src, q_sz, b))

        for i in range(len(dma_list)):
            q_src, q_sz, b = dma_list[i]
            pltpu.make_async_copy(
                vmem_src.at[b, :, pl.ds(0, q_sz)],
                o_hbm.at[:, pl.ds(q_src, q_sz)],
                sem,
            ).start()

    @jax.named_scope("o_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # dst_ref: (o_hbm, schedule_ref)
        o_hbm, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            if self.cfgs.is_stacked and self.cfgs.dense_pack:
                out_flag = jnp.logical_and(
                    schedule_ref.combine_span[block_idx, b] > 0,
                    schedule_ref.is_final[block_idx, b] == 1,
                )
            elif self.cfgs.is_stacked:
                out_flag = schedule_ref.combine_span[block_idx, b] > 0
            else:
                out_flag = schedule_ref.is_last_k[block_idx, b] == 1
            _, q_sz = schedule_ref.get_dma_q(block_idx, b)
            q_sz = jnp.where(out_flag, q_sz, 0)
            total_sz += q_sz

        flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
        pltpu.make_async_copy(
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingVisibilityRef(pltpu.BufferedRef):
    """Handles fetching visibility blocks using precomputed Q metadata."""

    cfgs: configs.RpaConfigs = dataclasses.field(default=None,
                                                 metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RpaConfigs,
    ):
        assert buffer_type == pltpu.BufferType.INPUT

        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("visibility_copy_in")
    def copy_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        if not self.cfgs.has_visibility:
            return

        visibility_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        for b in range(self.cfgs.batch_size):
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            pltpu.make_async_copy(
                visibility_hbm.at[pl.ds(q_src, q_sz)],
                vmem_dst.at[b, pl.ds(0, q_sz)],
                sem,
            ).start()

    @jax.named_scope("visibility_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        if not self.cfgs.has_visibility:
            return

        _, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            _, q_sz = schedule_ref.get_dma_q(block_idx, b)
            total_sz += q_sz

        flat_vmem = vmem_dst.reshape((-1, 128))
        pltpu.make_async_copy(
            flat_vmem.at[pl.ds(0, total_sz)],
            flat_vmem.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingQRef(pltpu.BufferedRef):
    """Handles fetching Q blocks using precomputed metadata."""

    cfgs: configs.RpaConfigs = dataclasses.field(default=None,
                                                 metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RpaConfigs,
    ):
        assert buffer_type == pltpu.BufferType.INPUT

        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("q_copy_in")
    def copy_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # src_ref: (q_hbm, schedule_ref)
        q_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        dma_list = []
        for b in range(self.cfgs.batch_size):
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            dma_list.append((q_src, q_sz, b))

        for i in range(len(dma_list)):
            q_src, q_sz, b = dma_list[i]
            pltpu.make_async_copy(
                q_hbm.at[:, pl.ds(q_src, q_sz)],
                vmem_dst.at[b, :, pl.ds(0, q_sz)],
                sem,
            ).start()

    @jax.named_scope("q_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RpaSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            _, q_sz = schedule_ref.get_dma_q(block_idx, b)
            total_sz += q_sz

        # Flatten to 2D: (Total_Rows, Head_Dim)
        # vmem_dst is (Batch, Heads, Q, Head_Dim). We copy Heads * q_sz rows.
        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[3:]))
        pltpu.make_async_copy(
            flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
            flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
            sem,
        ).wait()
