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
"""Buffered pipeline refs for one-block sliding-window prefill."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa.prefill import config
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import \
    schedule


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SlidingWindowKVRef(pltpu.BufferedRef):
    """Fetch and update one page-anchored sequence-along-lane KV tile."""

    cfgs: config.PrefillConfig = dataclasses.field(default=None,
                                                   metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: config.PrefillConfig,
    ):
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
                field.name: getattr(standard_ref, field.name)
                for field in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("kv_copy_in")
    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_cache_hbm, new_kv_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = jnp.maximum(grid_indices[0], 0)
        vmem_dst = self.window_ref.at[slot]
        new_fetch_lanes = min(self.cfgs.serve.page_size, new_kv_hbm.shape[-1])

        for batch_idx in range(self.cfgs.batch_size):
            for page_idx in range(self.cfgs.bkv_p_cache):
                physical_page, dst_vmem, size = schedule_ref.get_dma_kv_cache(
                    block_idx, batch_idx, page_idx)
                size = pl.multiple_of(size, self.cfgs.num_lanes)
                dst_vmem = pl.multiple_of(dst_vmem, self.cfgs.num_lanes)
                pltpu.make_async_copy(
                    kv_cache_hbm.at[physical_page, :, :,
                                    pl.ds(0, size)],
                    vmem_dst.at[batch_idx, :, :,
                                pl.ds(dst_vmem, size)],
                    sem,
                ).start()

            for page_idx in range(self.cfgs.bkv_p_new):
                src_hbm, dst_vmem, valid = schedule_ref.get_dma_fetch_kv_new(
                    block_idx, batch_idx, page_idx)
                remaining = new_kv_hbm.shape[-1] - src_hbm
                fetch_size = jnp.minimum(new_fetch_lanes,
                                         jnp.maximum(remaining, 0))
                size = pl.multiple_of(valid * fetch_size, self.cfgs.num_lanes)
                src_hbm = pl.multiple_of(src_hbm, self.cfgs.num_lanes)
                dst_vmem = pl.multiple_of(dst_vmem, self.cfgs.num_lanes)
                pltpu.make_async_copy(
                    new_kv_hbm.at[:, :, pl.ds(src_hbm, size)],
                    vmem_dst.at[batch_idx, :, :,
                                pl.ds(dst_vmem, size)],
                    sem,
                ).start()

    @jax.named_scope("kv_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, new_kv_hbm, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        for batch_idx in range(self.cfgs.batch_size):
            total_size = 0
            for page_idx in range(self.cfgs.bkv_p_cache):
                _, _, cache_size = schedule_ref.get_dma_kv_cache(
                    block_idx, batch_idx, page_idx)
                total_size += cache_size
            new_fetch_lanes = min(self.cfgs.serve.page_size,
                                  new_kv_hbm.shape[-1])
            for page_idx in range(self.cfgs.bkv_p_new):
                src_hbm, _, valid = schedule_ref.get_dma_fetch_kv_new(
                    block_idx, batch_idx, page_idx)
                remaining = new_kv_hbm.shape[-1] - src_hbm
                fetch_size = jnp.minimum(new_fetch_lanes,
                                         jnp.maximum(remaining, 0))
                total_size += jnp.where(valid > 0, fetch_size, 0)

            size = pl.multiple_of(total_size, self.cfgs.num_lanes)
            pltpu.make_async_copy(
                vmem_dst.at[batch_idx, :, :, pl.ds(0, size)],
                vmem_dst.at[batch_idx, :, :, pl.ds(0, size)],
                sem,
            ).wait()

    @jax.named_scope("kv_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]
        vmem_src = self.window_ref.at[slot]

        for batch_idx in range(self.cfgs.batch_size):
            for page_idx in range(self.cfgs.bkv_p_new):
                (dst_hbm, src_vmem, valid, wb_lane,
                 wb_size) = schedule_ref.get_dma_update_kv_new(
                     block_idx, batch_idx, page_idx)
                size = pl.multiple_of(valid * wb_size, self.cfgs.num_lanes)
                src_vmem = pl.multiple_of(src_vmem + wb_lane,
                                          self.cfgs.num_lanes)
                wb_lane = pl.multiple_of(wb_lane, self.cfgs.num_lanes)
                pltpu.make_async_copy(
                    vmem_src.at[batch_idx, :, :,
                                pl.ds(src_vmem, size)],
                    kv_out_ref.at[dst_hbm, :, :,
                                  pl.ds(wb_lane, size)],
                    sem,
                ).start()

    @jax.named_scope("kv_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        for batch_idx in range(self.cfgs.batch_size):
            size = 0
            for page_idx in range(self.cfgs.bkv_p_new):
                _, _, valid, _, wb_size = schedule_ref.get_dma_update_kv_new(
                    block_idx, batch_idx, page_idx)
                size += valid * wb_size
            size = pl.multiple_of(size, self.cfgs.num_lanes)
            dst_hbm, _, _, _, _ = schedule_ref.get_dma_update_kv_new(
                block_idx, batch_idx, 0)
            pltpu.make_async_copy(
                kv_out_ref.at[dst_hbm, :, :, pl.ds(0, size)],
                kv_out_ref.at[dst_hbm, :, :, pl.ds(0, size)],
                sem,
            ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SlidingWindowQRef(pltpu.BufferedRef):
    """Fetch each compact schedule cell's Q block."""

    cfgs: config.PrefillConfig = dataclasses.field(default=None,
                                                   metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: config.PrefillConfig,
    ):
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
                field.name: getattr(standard_ref, field.name)
                for field in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("q_copy_in")
    def copy_in(
        self,
        src_ref: tuple[jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        q_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        for batch_idx in range(self.cfgs.batch_size):
            q_src, q_size = schedule_ref.get_dma_q(block_idx, batch_idx)
            pltpu.make_async_copy(
                q_hbm.at[:, pl.ds(q_src, q_size)],
                vmem_dst.at[batch_idx, :, pl.ds(0, q_size)],
                sem,
            ).start()

    @jax.named_scope("q_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]
        total_size = 0
        for batch_idx in range(self.cfgs.batch_size):
            _, q_size = schedule_ref.get_dma_q(block_idx, batch_idx)
            total_size += q_size

        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[3:]))
        size = total_size * vmem_dst.shape[1]
        pltpu.make_async_copy(
            flat_vmem.at[pl.ds(0, size)],
            flat_vmem.at[pl.ds(0, size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SlidingWindowORef(pltpu.BufferedRef):
    """Write every nonempty compact-schedule Q-block output."""

    cfgs: config.PrefillConfig = dataclasses.field(default=None,
                                                   metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: config.PrefillConfig,
    ):
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
                field.name: getattr(standard_ref, field.name)
                for field in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @jax.named_scope("o_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        o_hbm, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        for batch_idx in range(self.cfgs.batch_size):
            q_src, q_size = schedule_ref.get_dma_q(block_idx, batch_idx)
            pltpu.make_async_copy(
                vmem_src.at[batch_idx, :, pl.ds(0, q_size)],
                o_hbm.at[:, pl.ds(q_src, q_size)],
                sem,
            ).start()

    @jax.named_scope("o_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.SlidingWindowSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        o_hbm, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]
        total_size = 0
        for batch_idx in range(self.cfgs.batch_size):
            _, q_size = schedule_ref.get_dma_q(block_idx, batch_idx)
            total_size += q_size

        flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
        size = total_size * o_hbm.shape[0]
        pltpu.make_async_copy(
            flat_ref.at[pl.ds(0, size)],
            flat_ref.at[pl.ds(0, size)],
            sem,
        ).wait()
