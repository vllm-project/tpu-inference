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
"""Buffered references for non-sliding-window decode."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    config as decode_config
from tpu_inference.kernels.experimental.stacked_rpa.decode.non_sliding_window import \
    schedule


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KVBufferedRefSeqAlongLane(pltpu.BufferedRef):
    """Fetch and update KV cache in SEQ_ALONG_LANE layout."""

    cfgs: decode_config.DecodeConfig = dataclasses.field(
        default=None, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: decode_config.DecodeConfig,
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
        src_ref: tuple[jax.Ref, jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_cache_hbm, new_kv_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = jnp.maximum(grid_indices[0], 0)
        vmem_dst_lane = self.window_ref.at[slot]
        new_fetch_lanes = min(self.cfgs.serve.page_size, new_kv_hbm.shape[-1])

        for b_idx in range(self.cfgs.batch_size):
            for page_idx in range(self.cfgs.bkv_p_cache):
                physical_page, dst_off, size = schedule_ref.get_dma_kv_cache(
                    block_idx, b_idx, page_idx)
                size = pl.multiple_of(size, self.cfgs.num_lanes)
                dst_off = pl.multiple_of(dst_off, 128)
                pltpu.make_async_copy(
                    kv_cache_hbm.at[physical_page, :, :,
                                    pl.ds(0, size)],
                    vmem_dst_lane.at[b_idx, :, :,
                                     pl.ds(dst_off, size)],
                    sem,
                ).start()

            n_new = 0 if self.cfgs.new_kv_resident else self.cfgs.bkv_p_new
            for page_idx in range(n_new):
                src_off, dst_off, dma_valid = schedule_ref.get_dma_fetch_kv_new(
                    block_idx, b_idx, page_idx)
                remaining = new_kv_hbm.shape[-1] - src_off
                fetch_size = jnp.minimum(new_fetch_lanes,
                                         jnp.maximum(remaining, 0))
                size = pl.multiple_of(dma_valid * fetch_size,
                                      self.cfgs.num_lanes)
                src_off = pl.multiple_of(src_off, 128)
                dst_off = pl.multiple_of(dst_off, 128)
                pltpu.make_async_copy(
                    new_kv_hbm.at[:, :, pl.ds(src_off, size)],
                    vmem_dst_lane.at[b_idx, :, :,
                                     pl.ds(dst_off, size)],
                    sem,
                ).start()

    @jax.named_scope("kv_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]
        vmem_src_lane = self.window_ref.at[slot]

        for b_idx in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b_idx] == 1
            for page_idx in range(self.cfgs.bkv_p_new):
                (physical_page, src_off, dma_valid, wb_lane,
                 wb_size) = schedule_ref.get_dma_update_kv_new(
                     block_idx, b_idx, page_idx)
                size = jnp.where(
                    do_writeback,
                    dma_valid * wb_size,
                    0,
                )
                size = pl.multiple_of(size, self.cfgs.num_lanes)
                src_off = pl.multiple_of(src_off + wb_lane,
                                         self.cfgs.num_lanes)
                wb_lane = pl.multiple_of(wb_lane, self.cfgs.num_lanes)
                pltpu.make_async_copy(
                    vmem_src_lane.at[b_idx, :, :,
                                     pl.ds(src_off, size)],
                    kv_out_ref.at[physical_page, :, :,
                                  pl.ds(wb_lane, size)],
                    sem,
                ).start()

    @jax.named_scope("kv_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, new_kv_hbm, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = grid_indices[0]
        vmem_dst = self.window_ref.at[slot]

        for b_idx in range(self.cfgs.batch_size):
            total_size = 0
            for page_idx in range(self.cfgs.bkv_p_cache):
                _, _, cache_size = schedule_ref.get_dma_kv_cache(
                    block_idx, b_idx, page_idx)
                total_size += cache_size
            n_new = 0 if self.cfgs.new_kv_resident else self.cfgs.bkv_p_new
            new_fetch_lanes = min(self.cfgs.serve.page_size,
                                  new_kv_hbm.shape[-1])
            for page_idx in range(n_new):
                src_off, _, dma_valid = schedule_ref.get_dma_fetch_kv_new(
                    block_idx, b_idx, page_idx)
                remaining = new_kv_hbm.shape[-1] - src_off
                fetch_size = jnp.minimum(new_fetch_lanes,
                                         jnp.maximum(remaining, 0))
                total_size += jnp.where(dma_valid > 0, fetch_size, 0)

            size = pl.multiple_of(total_size, self.cfgs.num_lanes)
            pltpu.make_async_copy(
                vmem_dst.at[b_idx, :, :, pl.ds(0, size)],
                vmem_dst.at[b_idx, :, :, pl.ds(0, size)],
                sem,
            ).wait()

    @jax.named_scope("kv_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        for b_idx in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b_idx] == 1
            size = 0
            for page_idx in range(self.cfgs.bkv_p_new):
                _, _, dma_valid, _, wb_size = (
                    schedule_ref.get_dma_update_kv_new(block_idx, b_idx,
                                                       page_idx))
                size += jnp.where(do_writeback, dma_valid * wb_size, 0)

            size = pl.multiple_of(size, self.cfgs.num_lanes)
            physical_page, _, _, _, _ = schedule_ref.get_dma_update_kv_new(
                block_idx, b_idx, 0)
            pltpu.make_async_copy(
                kv_out_ref.at[physical_page, :, :,
                              pl.ds(0, size)],
                kv_out_ref.at[physical_page, :, :,
                              pl.ds(0, size)],
                sem,
            ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DenseBatchingORef(pltpu.BufferedRef):
    """Copy completed dense-pack cells to the output buffer."""

    cfgs: decode_config.DecodeConfig = dataclasses.field(
        default=None, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: decode_config.DecodeConfig,
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

    def _output_flag(self, schedule_ref, block_idx, b_idx):
        return jnp.logical_and(
            schedule_ref.combine_span[block_idx, b_idx] > 0,
            schedule_ref.is_final[block_idx, b_idx] == 1,
        )

    @jax.named_scope("o_copy_out")
    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        o_hbm, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]
        vmem_src = self.window_ref.at[slot]

        dma_list = []
        for b_idx in range(self.cfgs.batch_size):
            output = self._output_flag(schedule_ref, block_idx, b_idx)
            q_src, q_size = schedule_ref.get_dma_q(block_idx, b_idx)
            q_size = jnp.where(output, q_size, 0)
            dma_list.append((q_src, q_size, b_idx))

        for q_src, q_size, b_idx in dma_list:
            pltpu.make_async_copy(
                vmem_src.at[b_idx, :, pl.ds(0, q_size)],
                o_hbm.at[:, pl.ds(q_src, q_size)],
                sem,
            ).start()

    @jax.named_scope("o_wait_out")
    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        o_hbm, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_size = 0
        for b_idx in range(self.cfgs.batch_size):
            output = self._output_flag(schedule_ref, block_idx, b_idx)
            _, q_size = schedule_ref.get_dma_q(block_idx, b_idx)
            total_size += jnp.where(output, q_size, 0)

        flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
        pltpu.make_async_copy(
            flat_ref.at[pl.ds(0, total_size * o_hbm.shape[0])],
            flat_ref.at[pl.ds(0, total_size * o_hbm.shape[0])],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingQRef(pltpu.BufferedRef):
    """Fetch query blocks using precomputed decode metadata."""

    cfgs: decode_config.DecodeConfig = dataclasses.field(
        default=None, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: decode_config.DecodeConfig,
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
        src_ref: tuple[jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        q_hbm, schedule_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = grid_indices[0]
        vmem_dst = self.window_ref.at[slot]

        dma_list = []
        for b_idx in range(self.cfgs.batch_size):
            q_src, q_size = schedule_ref.get_dma_q(block_idx, b_idx)
            dma_list.append((q_src, q_size, b_idx))

        for q_src, q_size, b_idx in dma_list:
            pltpu.make_async_copy(
                q_hbm.at[:, pl.ds(q_src, q_size)],
                vmem_dst.at[b_idx, :, pl.ds(0, q_size)],
                sem,
            ).start()

    @jax.named_scope("q_wait_in")
    def wait_in(
        self,
        src_ref: tuple[jax.Ref, schedule.DecodeSchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, schedule_ref = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        block_idx = grid_indices[0]
        vmem_dst = self.window_ref.at[slot]

        total_size = 0
        for b_idx in range(self.cfgs.batch_size):
            _, q_size = schedule_ref.get_dma_q(block_idx, b_idx)
            total_size += q_size

        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[3:]))
        pltpu.make_async_copy(
            flat_vmem.at[pl.ds(0, total_size * vmem_dst.shape[1])],
            flat_vmem.at[pl.ds(0, total_size * vmem_dst.shape[1])],
            sem,
        ).wait()
