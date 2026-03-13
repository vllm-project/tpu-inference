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
from typing import Any

import jax
from jax import tree_util
from jax._src.pallas.mosaic import pipeline
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax.experimental import pallas as pl

from tpu_inference.kernels.ragged_paged_attention.v4 import \
    schedule as rpa_schedule


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pipeline.BufferedRef):
    """Helper class to safely bypass buffer_count checks during creation."""

    def __post_init__(self):
        pass


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KVBufferedRef(pipeline.BufferedRef):
    """Handles fetching and updating KV cache using precomputed metadata."""

    bkv_p_cache: int = dataclasses.field(metadata={"static": True})
    bkv_p_new: int = dataclasses.field(metadata={"static": True})
    page_size: int = dataclasses.field(metadata={"static": True})
    batch_size: int = dataclasses.field(metadata={"static": True})
    hbm_stride: int = dataclasses.field(metadata={"static": True})
    page_size_log2: int = dataclasses.field(metadata={"static": True})
    page_size_mask: int = dataclasses.field(metadata={"static": True})

    def __post_init__(self):
        # Override to bypass the buffer_count > 2 check for output refs
        pass

    @classmethod
    def from_ref(
        cls,
        ref: pipeline.BufferedRef,
        bkv_p_cache: int,
        bkv_p_new: int,
        page_size: int,
        batch_size: int,
        hbm_stride: int,
        page_size_log2: int,
        page_size_mask: int,
    ):
        return cls(
            bkv_p_cache=bkv_p_cache,
            bkv_p_new=bkv_p_new,
            page_size=page_size,
            batch_size=batch_size,
            hbm_stride=hbm_stride,
            page_size_log2=page_size_log2,
            page_size_mask=page_size_mask,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pipeline.BufferedRef)
            },
        )

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        source_memory_space: jax.Array,
        bkv_p_cache: int,
        bkv_p_new: int,
        page_size: int,
        batch_size: int,
        hbm_stride: int,
        page_size_log2: int,
        page_size_mask: int,
        buffer_count: int = 2,
        use_lookahead: bool = True,
    ):
        standard_ref = _BypassRef.create(
            spec=spec,
            dtype_or_type=pipeline._ref_to_value_aval(source_memory_space),
            buffer_type=pipeline.BufferType.INPUT_OUTPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
            source_memory_space=source_memory_space,
        )
        return cls.from_ref(
            standard_ref,
            bkv_p_cache=bkv_p_cache,
            bkv_p_new=bkv_p_new,
            page_size=page_size,
            batch_size=batch_size,
            hbm_stride=hbm_stride,
            page_size_log2=page_size_log2,
            page_size_mask=page_size_mask,
        )

    # def init_slots(self):
    #   super().init_slots()
    #   # Initialize all slots at once using a vectorized operation
    #   kv_ref_vec = self.window_ref.bitcast(jnp.uint32)  # type: ignore
    #   kv_ref_vec[...] = jnp.zeros_like(kv_ref_vec)

    def copy_in(
        self,
        src_ref: tuple[jax.Array, jax.Array, rpa_schedule.RPASchedule,
                       jax.Array],
        grid_indices: tuple[int, ...],
    ):
        # src_ref: (kv_cache_hbm, new_kv_hbm, schedule, page_indices_ref)
        kv_cache_hbm, new_kv_hbm, schedule, page_indices_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        kv_cache_hbm_flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])

        dma_list_cache = []
        dma_list_new = []

        for b in range(self.batch_size):
            for i in range(self.bkv_p_cache):
                p_idx, dst_off, sz = schedule.get_dma_kv_cache(block_idx, b, i)
                src_off = page_indices_ref[p_idx] * self.page_size
                dma_list_cache.append((src_off, dst_off, sz, b))

            # Contiguous fetch for new KV
            _, src_new_off, dst_vmem_off, _ = schedule.get_dma_kv_new(
                block_idx, b, 0)
            total_new_sz = 0
            for i in range(self.bkv_p_new):
                _, _, _, sz = schedule.get_dma_kv_new(block_idx, b, i)
                total_new_sz += sz
            dma_list_new.append((src_new_off, dst_vmem_off, total_new_sz, b))

        for i in range(len(dma_list_cache)):
            src_off, dst_off, sz, b = dma_list_cache[i]
            tpu_primitives.make_async_copy(
                kv_cache_hbm_flat.at[pl.ds(src_off, sz)],
                vmem_dst.at[b, pl.ds(dst_off, sz), :self.hbm_stride],
                sem,
            ).start()

        for i in range(len(dma_list_new)):
            src_off, dst_off, sz, b = dma_list_new[i]
            tpu_primitives.make_async_copy(
                new_kv_hbm.at[pl.ds(src_off, sz)],
                vmem_dst.at[b, pl.ds(dst_off, sz), :self.hbm_stride],
                sem,
            ).start()

    def copy_out(
        self,
        dst_ref: tuple[jax.Array, Any, rpa_schedule.RPASchedule, jax.Array],
        grid_indices: tuple[int, ...],
    ):
        kv_out_ref, _, schedule, page_indices_ref = dst_ref
        kv_out_ref_flat = kv_out_ref.reshape(-1, *kv_out_ref.shape[2:])
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        @pl.loop(0, self.batch_size, unroll=True)
        def _for_each_seq(b):
            idx = block_idx * self.batch_size + b
            do_writeback = schedule.do_writeback[idx]
            for i in range(self.bkv_p_new):
                encoded_dst_hbm_off, _, src_vmem_off, new_sz = schedule.get_dma_kv_new(
                    block_idx, b, i)
                global_p_idx = encoded_dst_hbm_off >> self.page_size_log2
                p_off = encoded_dst_hbm_off & self.page_size_mask
                dst_hbm_off = (page_indices_ref[global_p_idx] <<
                               self.page_size_log2) | p_off
                sz = jax.lax.select(do_writeback == 1, new_sz, 0)
                tpu_primitives.make_async_copy(
                    vmem_src.at[b,
                                pl.ds(src_vmem_off, sz), :self.hbm_stride],
                    kv_out_ref_flat.at[pl.ds(dst_hbm_off, sz)],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Array, jax.Array, rpa_schedule.RPASchedule,
                       jax.Array],
        grid_indices: tuple[int, ...],
    ):
        _, _, schedule, _ = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.batch_size):
            for i in range(self.bkv_p_cache):
                _, _, sz = schedule.get_dma_kv_cache(block_idx, b, i)
                total_sz += sz

            # Contiguous wait for new KV
            for i in range(self.bkv_p_new):
                _, _, _, sz = schedule.get_dma_kv_new(block_idx, b, i)
                total_sz += sz

        # Flatten the first two dimensions (Batch, Seq) to create a 1D view for waiting.
        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[2:]))
        tpu_primitives.make_async_copy(
            flat_vmem.at[pl.ds(0, total_sz), :self.hbm_stride],
            flat_vmem.at[pl.ds(0, total_sz), :self.hbm_stride],
            sem,
        ).wait()

    def wait_out(
        self,
        dst_ref: tuple[jax.Array, Any, rpa_schedule.RPASchedule, jax.Array],
        grid_indices: tuple[int, ...],
    ):
        kv_out_ref, _, schedule, _ = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.batch_size):
            idx = block_idx * self.batch_size + b
            do_writeback = schedule.do_writeback[idx]
            for i in range(self.bkv_p_new):
                _, _, _, new_sz = schedule.get_dma_kv_new(block_idx, b, i)
                sz = jax.lax.select(do_writeback == 1, new_sz, 0)
                total_sz += sz

        # Flatten to 2D: (Total_Rows, Head_Dim)
        flat_ref = kv_out_ref.reshape((-1, *kv_out_ref.shape[2:]))
        tpu_primitives.make_async_copy(
            flat_ref.at[pl.ds(0, total_sz)],
            flat_ref.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingORef(pipeline.BufferedRef):
    """Handles normalizing and storing the final attention output."""

    batch_size: int = dataclasses.field(metadata={"static": True})

    @classmethod
    def from_ref(cls, ref: pipeline.BufferedRef, batch_size: int):
        return cls(
            batch_size=batch_size,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pipeline.BufferedRef)
            },
        )

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        source_memory_space: jax.Array,
        batch_size: int,
        buffer_count: int = 2,
        use_lookahead: bool = False,
    ):
        standard_ref = pipeline.BufferedRef.create(
            spec=spec,
            dtype_or_type=pipeline._ref_to_value_aval(source_memory_space),
            buffer_type=pipeline.BufferType.OUTPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
            source_memory_space=source_memory_space,
        )
        return cls.from_ref(standard_ref, batch_size=batch_size)

    def copy_out(
        self,
        dst_ref: tuple[jax.Array, rpa_schedule.RPASchedule],
        grid_indices: tuple[int, ...],
    ):
        # dst_ref: (o_hbm, schedule)
        o_hbm, schedule = dst_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        # is_last_k stride: batch size
        dma_list = []
        for b in range(self.batch_size):
            idx = block_idx * self.batch_size + b
            is_last_k = schedule.is_last_k[idx]
            q_src, q_sz = schedule.get_dma_q(block_idx, b)
            q_sz = jax.lax.select(is_last_k == 1, q_sz, 0)
            dma_list.append((q_src, q_sz, b))

        for i in range(len(dma_list)):
            q_src, q_sz, b = dma_list[i]
            tpu_primitives.make_async_copy(
                vmem_src.at[b, :, pl.ds(0, q_sz)],
                o_hbm.at[:, pl.ds(q_src, q_sz)],
                sem,
            ).start()

    def wait_out(
        self,
        dst_ref: tuple[jax.Array, rpa_schedule.RPASchedule],
        grid_indices: tuple[int, ...],
    ):
        # dst_ref: (o_hbm, schedule)
        o_hbm, schedule = dst_ref
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.batch_size):
            idx = block_idx * self.batch_size + b
            is_last_k = schedule.is_last_k[idx]
            _, q_sz = schedule.get_dma_q(block_idx, b)
            q_sz = jax.lax.select(is_last_k == 1, q_sz, 0)
            total_sz += q_sz

        flat_ref = o_hbm.reshape((-1, *o_hbm.shape[2:]))
        tpu_primitives.make_async_copy(
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            flat_ref.at[pl.ds(0, total_sz * o_hbm.shape[0])],
            sem,
        ).wait()


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BatchingQRef(pipeline.BufferedRef):
    """Handles fetching Q blocks using precomputed metadata."""

    bq_sz: int = dataclasses.field(metadata={"static": True})
    batch_size: int = dataclasses.field(metadata={"static": True})

    @classmethod
    def from_ref(
        cls,
        ref: pipeline.BufferedRef,
        bq_sz: int,
        batch_size: int,
    ):
        return cls(
            bq_sz=bq_sz,
            batch_size=batch_size,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pipeline.BufferedRef)
            },
        )

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        source_memory_space: jax.Array,
        bq_sz: int,
        batch_size: int,
        buffer_count: int = 2,
        use_lookahead: bool = True,
    ):
        standard_ref = pipeline.BufferedRef.create(
            spec=spec,
            dtype_or_type=pipeline._ref_to_value_aval(source_memory_space),
            buffer_type=pipeline.BufferType.INPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
            source_memory_space=source_memory_space,
        )
        return cls.from_ref(standard_ref, bq_sz=bq_sz, batch_size=batch_size)

    def copy_in(
        self,
        src_ref: tuple[jax.Array, rpa_schedule.RPASchedule],
        grid_indices: tuple[int, ...],
    ):
        # src_ref: (q_hbm, schedule)
        q_hbm, schedule = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        dma_list = []
        for b in range(self.batch_size):
            q_src, q_sz = schedule.get_dma_q(block_idx, b)
            dma_list.append((q_src, q_sz, b))

        for i in range(len(dma_list)):
            q_src, q_sz, b = dma_list[i]
            tpu_primitives.make_async_copy(
                q_hbm.at[:, pl.ds(q_src, q_sz)],
                vmem_dst.at[b, :, pl.ds(0, q_sz)],
                sem,
            ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Array, rpa_schedule.RPASchedule],
        grid_indices: tuple[int, ...],
    ):
        _, schedule = src_ref
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot]
        block_idx = grid_indices[0]

        total_sz = 0
        for b in range(self.batch_size):
            _, q_sz = schedule.get_dma_q(block_idx, b)
            total_sz += q_sz

        # Flatten to 2D: (Total_Rows, Head_Dim)
        # vmem_dst is (Batch, Heads, Q, Head_Dim). We copy Heads * q_sz rows.
        flat_vmem = vmem_dst.reshape((-1, *vmem_dst.shape[3:]))
        tpu_primitives.make_async_copy(
            flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
            flat_vmem.at[pl.ds(0, total_sz * vmem_dst.shape[1])],
            sem,
        ).wait()
