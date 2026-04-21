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

from tpu_inference.kernels.experimental.batched_rpa import configs, schedule


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
class KVBufferedRef(_BypassRef):
    """Handles fetching and updating KV cache using precomputed metadata."""

    cfgs: configs.RPAConfig = dataclasses.field(default=None,
                                                metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type,  # pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RPAConfig,
    ):
        # TODO(kyuyeunk): Uncomment this out after jax version update.
        # assert buffer_type == pltpu.BufferType.INPUT_OUTPUT

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

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.RPASchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # src_ref: (kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref)
        kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot, :, :, :self.cfgs.kv_hbm_stride]
        block_idx = jnp.maximum(grid_indices[0], 0)

        kv_cache_hbm_flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])

        dma_list_cache = []
        dma_list_new = []

        for b in range(self.cfgs.batch_size):
            for i in range(self.cfgs.bkv_p_cache):
                p_idx, dst_off, sz = schedule_ref.get_dma_kv_cache(
                    block_idx, b, i)
                src_off = page_indices_ref[p_idx] * self.cfgs.serve.page_size
                dma_list_cache.append((src_off, dst_off, sz, b))

            # Contiguous fetch for new KV
            _, src_new_off, dst_vmem_off, _ = schedule_ref.get_dma_kv_new(
                block_idx, b, 0)
            total_new_sz = 0
            for i in range(self.cfgs.bkv_p_new):
                _, _, _, sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
                total_new_sz += sz
            dma_list_new.append((src_new_off, dst_vmem_off, total_new_sz, b))

        for i in range(len(dma_list_cache)):
            src_off, dst_off, sz, b = dma_list_cache[i]
            pltpu.make_async_copy(
                kv_cache_hbm_flat.at[pl.ds(src_off, sz)],
                vmem_dst.at[b, pl.ds(dst_off, sz)],
                sem,
            ).start()

        for i in range(len(dma_list_new)):
            src_off, dst_off, sz, b = dma_list_new[i]
            pltpu.make_async_copy(
                new_kv_hbm.at[pl.ds(src_off, sz)],
                vmem_dst.at[b, pl.ds(dst_off, sz)],
                sem,
            ).start()

    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.RPASchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref, page_indices_ref = dst_ref
        kv_out_ref_flat = kv_out_ref.reshape(-1, *kv_out_ref.shape[2:])
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot, :, :, :self.cfgs.kv_hbm_stride]
        block_idx = grid_indices[0]

        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            for i in range(self.cfgs.bkv_p_new):
                encoded_dst_hbm_off, _, src_vmem_off, new_sz = (
                    schedule_ref.get_dma_kv_new(block_idx, b, i))
                global_p_idx = encoded_dst_hbm_off >> self.cfgs.serve.page_size_log2
                p_off = encoded_dst_hbm_off & self.cfgs.serve.page_size_mask
                dst_hbm_off = (page_indices_ref[global_p_idx] <<
                               self.cfgs.serve.page_size_log2) | p_off
                sz = jnp.where(do_writeback, new_sz, 0)
                pltpu.make_async_copy(
                    vmem_src.at[b, pl.ds(src_vmem_off, sz)],
                    kv_out_ref_flat.at[pl.ds(dst_hbm_off, sz)],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.RPASchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        _, _, schedule_ref, _ = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            for i in range(self.cfgs.bkv_p_cache):
                _, _, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
                total_sz += sz

            # Contiguous wait for new KV
            for i in range(self.cfgs.bkv_p_new):
                _, _, _, sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
                total_sz += sz

        # Flatten the first two dimensions (Batch, Seq) to create a 1D view.
        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[2:]))
        pltpu.make_async_copy(
            flat_vmem.at[pl.ds(0, total_sz), :self.cfgs.kv_hbm_stride],
            flat_vmem.at[pl.ds(0, total_sz), :self.cfgs.kv_hbm_stride],
            sem,
        ).wait()

    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, jax.Ref, schedule.RPASchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        kv_out_ref, _, schedule_ref, _ = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
            for i in range(self.cfgs.bkv_p_new):
                _, _, _, new_sz = schedule_ref.get_dma_kv_new(block_idx, b, i)
                sz = jnp.where(do_writeback, new_sz, 0)
                total_sz += sz

        # Flatten to 2D: (Total_Rows, Head_Dim)
        flat_ref = kv_out_ref.reshape((-1, *kv_out_ref.shape[2:]))
        pltpu.make_async_copy(
            flat_ref.at[pl.ds(0, total_sz)],
            flat_ref.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingORef(pltpu.BufferedRef):
    """Handles normalizing and storing the final attention output."""

    cfgs: configs.RPAConfig = dataclasses.field(default=None,
                                                metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type,  # pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RPAConfig,
    ):
        # TODO(kyuyeunk): Uncomment this out after jax version update.
        # assert buffer_type == pltpu.BufferType.OUTPUT

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

    def copy_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.RPASchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # dst_ref: (o_hbm, schedule_ref)
        o_hbm, schedule_ref = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        # is_last_k stride: batch size
        dma_list = []
        for b in range(self.cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
            q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
            q_sz = jnp.where(is_last_k, q_sz, 0)
            dma_list.append((q_src, q_sz, b))

        for i in range(len(dma_list)):
            q_src, q_sz, b = dma_list[i]
            pltpu.make_async_copy(
                vmem_src.at[b, :, pl.ds(0, q_sz)],
                o_hbm.at[:, pl.ds(q_src, q_sz)],
                sem,
            ).start()

    def wait_out(
        self,
        dst_ref: tuple[jax.Ref, schedule.RPASchedule],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        # dst_ref: (o_hbm, schedule_ref)
        o_hbm, schedule_ref = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
            _, q_sz = schedule_ref.get_dma_q(block_idx, b)
            q_sz = jnp.where(is_last_k, q_sz, 0)
            total_sz += q_sz

        flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
        pltpu.make_async_copy(
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingQRef(pltpu.BufferedRef):
    """Handles fetching Q blocks using precomputed metadata."""

    cfgs: configs.RPAConfig = dataclasses.field(default=None,
                                                metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type,  # pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: configs.RPAConfig,
    ):
        # TODO(kyuyeunk): Uncomment this out after jax version update.
        # assert buffer_type == pltpu.BufferType.INPUT

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

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RPASchedule],
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

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, schedule.RPASchedule],
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
