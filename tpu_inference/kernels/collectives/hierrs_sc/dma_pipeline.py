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
"""DMA Pipeline implementations for Reduce-Scatter."""

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.collectives.hierrs_sc.config import Config
from tpu_inference.kernels.collectives.hierrs_sc.topology import (ChunkLocator,
                                                                  Topology)


def _accumulate(
    *,
    config: Config,
    target,
    addend,
    num_rows,
    num_cols,
    col_step=16,
):
    """Performs in-place element-wise addition: `target += addend`.

  Both `target` and `addend` are 2D references of shape (`num_rows`,
  `num_cols`).
  """
    packing = config.packing_factor

    @plsc.parallel_loop(0, num_cols, step=col_step)
    def _loop(c_in):
        c_slice = pl.ds(c_in, col_step)

        num_iters = num_rows // packing
        for i in range(num_iters):
            r = i * packing
            r_idx = r if packing == 1 else pl.ds(r, packing)
            val = target[r_idx, c_slice] + addend[r_idx, c_slice]
            target[r_idx, c_slice] = val


@dataclasses.dataclass(frozen=True)
class RemoteDmaManager:
    """Handles remote DMA (ICI) copies on SCS."""

    config: Config
    topo: Topology
    core_idx: jax.Array
    p1_send_sem: jax.Ref = dataclasses.field(kw_only=True)
    p2_send_sem: jax.Ref = dataclasses.field(kw_only=True)
    p1_recv_sem: jax.Ref = dataclasses.field(kw_only=True)
    p2_recv_sem: jax.Ref = dataclasses.field(kw_only=True)
    locator: ChunkLocator = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "locator",
            ChunkLocator(self.config,
                         self.topo,
                         self.core_idx,
                         subcore_idx=None),
        )

    @jax.named_scope("start_phase1_d2d_copies")
    def start_phase1_d2d_copies(
        self,
        *,
        mb_idx,
        src,
        dst,
    ):
        """Triggers remote D2D copies from `src` to `dst` for the micro-batch index `mb_idx`."""
        recv_slot = mb_idx % 2
        for c, send_chunk_idx in enumerate(
                self.locator.get_phase1_chunk_idxes(self.topo.partner_id)):
            row_slice, col_slice = self.locator.get_phase1_slice(
                send_chunk_idx, mb_idx)
            slice_src = src.at[row_slice, col_slice]
            slice_dst = dst.at[row_slice, col_slice]
            pltpu.async_remote_copy(
                slice_src,
                slice_dst,
                self.p1_send_sem.at[recv_slot, c],
                self.p1_recv_sem.at[recv_slot, c],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

    @jax.named_scope("start_phase2_c2c_copies")
    def start_phase2_c2c_copies(
        self,
        *,
        mb_idx,
        step_idx,
        src,
        dst,
    ):
        """Triggers remote C2C copies from `src` to `dst` for `mb_idx` and `step_idx`."""
        slot_idx = mb_idx % 2
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)

        @pl.loop(0, num_chunk_groups)
        def _(chunk_group_idx):

            @pl.loop(0, num_hcube_dims)
            def _(hcube_dim_idx):
                dim = (hcube_dim_idx + step_idx) % num_hcube_dims
                neighbor_device_id = self.topo.get_neighbor_device_id(dim)
                neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neighbor_device_id,
                    step_idx,
                    chunk_group_idx=chunk_group_idx,
                    hcube_dim_idx=hcube_dim_idx,
                )

                row_slice, col_slice = self.locator.get_phase2_slice(
                    neighbor_chunk_idx, mb_idx, hcube_dim_idx)
                slice_src = src.at[step_idx, row_slice, col_slice]
                slice_dst = dst.at[row_slice, col_slice]
                pltpu.async_remote_copy(
                    slice_src,
                    slice_dst,
                    self.p2_send_sem.at[slot_idx, step_idx, chunk_group_idx,
                                        hcube_dim_idx],
                    self.p2_recv_sem.at[slot_idx, step_idx, chunk_group_idx,
                                        hcube_dim_idx],
                    device_id=neighbor_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

    @jax.named_scope("wait_phase1_d2d_copies")
    def wait_phase1_d2d_copies(self,
                               mb_idx,
                               src,
                               dst,
                               *,
                               wait_send: bool = False):
        """Waits for Phase 1 (D2D) remote copies from `src` to `dst` for the index `mb_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
        recv_slot = mb_idx % 2
        p1_recv_sem_slice = self.p1_recv_sem.at[recv_slot]
        p1_send_sem_slice = self.p1_send_sem.at[recv_slot]

        dummy_send = src.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.mb_size),
        ]
        dummy_recv = dst.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.mb_size),
        ]
        my_id = self.topo.cur_id
        for c, _ in enumerate(self.locator.get_phase1_chunk_idxes(my_id)):
            dma = pltpu.make_async_remote_copy(
                src_ref=dummy_send,
                dst_ref=dummy_recv,
                send_sem=p1_send_sem_slice.at[c],
                recv_sem=p1_recv_sem_slice.at[c],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            if wait_send:
                dma.wait_send()
            else:
                dma.wait_recv()

    @jax.named_scope("wait_phase2_c2c_copies")
    def wait_phase2_c2c_copies(self,
                               mb_idx,
                               step_idx,
                               src,
                               dst,
                               *,
                               wait_send: bool = False):
        """Waits for Phase 2 (C2C) remote copies from `src` to `dst` for `mb_idx` and `step_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
        slot_idx = mb_idx % 2
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
        p2_recv_sem_slice = self.p2_recv_sem.at[slot_idx, step_idx]
        p2_send_sem_slice = self.p2_send_sem.at[slot_idx, step_idx]

        dummy_send = src.at[
            step_idx,
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.hc_chunk_size),
        ]
        dummy_recv = dst.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.hc_chunk_size),
        ]

        @pl.loop(0, num_chunk_groups)
        def _(chunk_group_idx):

            @pl.loop(0, num_hcube_dims)
            def _(hcube_dim_idx):
                dim = (hcube_dim_idx + step_idx) % num_hcube_dims
                neighbor_device_id = self.topo.get_neighbor_device_id(dim)

                dma = pltpu.make_async_remote_copy(
                    src_ref=dummy_send,
                    dst_ref=dummy_recv,
                    send_sem=p2_send_sem_slice.at[chunk_group_idx,
                                                  hcube_dim_idx],
                    recv_sem=p2_recv_sem_slice.at[chunk_group_idx,
                                                  hcube_dim_idx],
                    device_id=neighbor_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                if wait_send:
                    dma.wait_send()
                else:
                    dma.wait_recv()


@dataclasses.dataclass(frozen=True)
class LocalDmaManager:
    """Handles local (HBM <-> VMEM) DMA copies and pipelines on TEC."""

    config: Config
    topo: Topology
    core_idx: jax.Array  # integer scalar.
    subcore_idx: jax.Array  # integer scalar.
    locator: ChunkLocator = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "locator",
            ChunkLocator(self.config, self.topo, self.core_idx,
                         self.subcore_idx),
        )

    def _local_subcore_copy(self, src_ref, dst_ref, sem):
        """Issues asynchronous DMAs to copy a subcore chunk between `src_ref` and `dst_ref` using `sem`."""
        if self.config.subcore_chunk_size % 8 == 0:
            pltpu.make_async_copy(src_ref, dst_ref, sem).start()
        else:
            # If the number of rows is not a multiple of 8, it iterates over
            # rows to satisfy layout alignment requirements.
            src_32b = src_ref.bitcast(jax.numpy.uint32)
            dst_32b = dst_ref.bitcast(jax.numpy.uint32)
            for i in range(src_32b.shape[0]):
                pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :],
                                      sem).start()

    def _wait_subcore_copies(self, src_dummy, dst_dummy, sem, num_copies=1):
        """Waits for `num_copies` subcore chunks to complete on `sem` using dummy async copies."""
        for _ in range(num_copies):
            if self.config.subcore_chunk_size % 8 == 0:
                pltpu.make_async_copy(src_dummy, dst_dummy, sem).wait()
            else:
                src_32b = src_dummy.bitcast(jax.numpy.uint32)
                dst_32b = dst_dummy.bitcast(jax.numpy.uint32)
                for i in range(src_32b.shape[0]):
                    pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :],
                                          sem).wait()

    # TODO: Remove this and use emit_pipeline instead once correctness bug is
    # fixed that triggers when there are two src inputs.
    def _accumulate_pipeline(
        self,
        num_iters: int,
        get_src1_fn,
        get_src2_fn,
        get_out_fn,
        dtype,
        max_col_size,
    ):
        """Double-buffered accumulation of chunks across `num_iters` iterations."""
        if num_iters == 0:
            return

        @functools.partial(
            pl.run_scoped,
            src1_vmem_ref=pltpu.VMEM(
                (2, max(2, self.config.subcore_chunk_size), max_col_size),
                dtype),
            src2_vmem_ref=pltpu.VMEM(
                (2, max(2, self.config.subcore_chunk_size), max_col_size),
                dtype),
            src_sems=pltpu.SemaphoreType.DMA((num_iters, )),
            out_sems=pltpu.SemaphoreType.DMA((num_iters, )),
        )
        def _run(src1_vmem_ref, src2_vmem_ref, src_sems, out_sems):

            def copy_in_fn(idx, slot):
                src1 = get_src1_fn(idx)
                src2 = get_src2_fn(idx)
                r, c = src1.shape
                self._local_subcore_copy(src1, src1_vmem_ref.at[slot, :r, :c],
                                         src_sems.at[idx])
                self._local_subcore_copy(src2, src2_vmem_ref.at[slot, :r, :c],
                                         src_sems.at[idx])

            def wait_in_fn(idx, slot):
                src1 = get_src1_fn(idx)
                r, c = src1.shape
                with jax.named_scope("wait"):
                    self._wait_subcore_copies(
                        src1,
                        src1_vmem_ref.at[0, :r, :c],
                        src_sems.at[idx],
                        num_copies=2,
                    )

            def compute_fn(idx, slot):
                src1 = get_src1_fn(idx)
                r, c = src1.shape
                with jax.named_scope("accumulate"):
                    _accumulate(
                        config=self.config,
                        target=src1_vmem_ref.at[slot],
                        addend=src2_vmem_ref.at[slot],
                        num_rows=r,
                        num_cols=c,
                    )

            def copy_out_fn(idx, slot):
                out = get_out_fn(idx)
                r, c = out.shape
                self._local_subcore_copy(src1_vmem_ref.at[slot, :r, :c], out,
                                         out_sems.at[idx])

            def wait_out_fn(idx, slot):
                out = get_out_fn(idx)
                r, c = out.shape
                self._wait_subcore_copies(
                    src1_vmem_ref.at[0, :r, :c],
                    out,
                    out_sems.at[idx],
                    num_copies=1,
                )

            def _slot(idx):
                return idx % 2

            copy_in_fn(0, 0)

            @pl.loop(0, num_iters - 1)
            def _(idx):

                @pl.when(idx >= 1)
                def _():
                    wait_out_fn(idx - 1, _slot(idx - 1))

                copy_in_fn(idx + 1, _slot(idx + 1))
                wait_in_fn(idx, _slot(idx))
                compute_fn(idx, _slot(idx))
                copy_out_fn(idx, _slot(idx))

            last_idx = num_iters - 1
            wait_in_fn(last_idx, _slot(last_idx))
            compute_fn(last_idx, _slot(last_idx))
            copy_out_fn(last_idx, _slot(last_idx))

            if last_idx >= 1:
                wait_out_fn(last_idx - 1, _slot(last_idx - 1))
            wait_out_fn(last_idx, _slot(last_idx))

    def run_phase1_accumulate_pipeline(
        self,
        *,
        mb_idx,
        src1_ref,
        src2_ref,
        out_ref,
    ):
        """Executes Phase 1 pipelined accumulation for `mb_idx`."""
        num_iters = self.config.num_chips

        def get_src1_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return src1_ref.at[rs, cs]

        def get_src2_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return src2_ref.at[rs, cs]

        def get_out_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return out_ref.at[0].at[rs, cs]

        self._accumulate_pipeline(
            num_iters,
            get_src1_fn,
            get_src2_fn,
            get_out_fn,
            src1_ref.dtype,
            self.config.subcore_col_chunk_size_p1,
        )

    def run_phase2_accumulate_pipeline(
        self,
        *,
        mb_idx,
        step_idx,
        src1_ref,
        src2_ref,
        final_out_ref,
    ):
        """Executes Phase 2 pipelined reduction for `mb_idx` and `step_idx`."""
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
        num_iters = num_chunk_groups * num_hcube_dims
        is_last_step = step_idx == num_hcube_dims - 1

        def get_chunk_idx(idx):
            chunk_group_idx = idx // num_hcube_dims
            hcube_dim_idx = idx % num_hcube_dims
            return self.locator.get_phase2_chunk_idx(
                self.topo.cur_id,
                step_idx,
                chunk_group_idx=chunk_group_idx,
                hcube_dim_idx=hcube_dim_idx,
            )

        def get_src1_fn(idx):
            chunk_idx = get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            return src1_ref.at[step_idx, rs, cs]

        def get_src2_fn(idx):
            chunk_idx = get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            return src2_ref.at[step_idx + 1, rs, cs]

        def get_out_fn(idx):
            chunk_idx = 0 if is_last_step else get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            final_dst = final_out_ref if is_last_step else src1_ref.at[step_idx
                                                                       + 1]
            return final_dst.at[rs, cs]

        self._accumulate_pipeline(
            num_iters,
            get_src1_fn,
            get_src2_fn,
            get_out_fn,
            src1_ref.dtype,
            self.config.subcore_col_chunk_size_p2,
        )
