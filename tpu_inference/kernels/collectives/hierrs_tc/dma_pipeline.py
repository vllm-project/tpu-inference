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
"""DMA pipeline implementations for TensorCore Reduce-Scatter.

`RemoteWaitBufferedRef` is a BufferedRef whose copy_in is a cross-device remote
DMA, so the Pallas pipeline can consume peer data with the same double-buffered
machinery it uses for local copies. `DmaManager` owns both the explicit async
remote dispatch (phase 1 D2D, phase 2 C2C) and the emit_pipeline accumulation
passes that consume them.
"""

import dataclasses
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives.hierrs_tc.config import (
    FP8_E4M3_MAX, SCALE_LANE, Config, get_capped_bounds)
from tpu_inference.kernels.collectives.hierrs_tc.topology import (ChunkLocator,
                                                                  Topology)


def scoped(name: str):
    """Wraps a whole pipeline pass in one COARSE jax.named_scope.

  Deliberately coarse: one region per logical pass (phase-1 accumulate,
  quantize staging, phase-2 dequant+accumulate) rather than per DMA
  start/wait. Fine-grained scopes wrapped precisely the DMA issue/wait
  structure -- the region a scheduler would need to reorder -- and 9 of them
  were removed after measuring: bitwise-identical output, isolated A/B flat,
  e2e +0.20%, but +70 KB (8.3%) of Mosaic IR.

  Applied as a decorator so a call site is never re-indented: every caller of
  the wrapped method gets the region for free, and the diff stays reviewable.

  CAVEAT, measured (results/probe_scope_visibility.py): these regions are NOT
  currently visible to xprof. jax/_src/pallas/mosaic/lowering.py:1750 emits
  `tpu.trace_start(message=name, level=10)` with the level hardcoded, and 10
  is above the profiler's capture threshold. `trace_level` in
  ProfileOptions.advanced_configuration is inert (levels 2 and 15 produced
  byte-identical 94113-event traces) and `tpu_trace_mode` is rejected by this
  libtpu ("Invalid tpu_trace_mode (it is not supported)"), aborting collection
  entirely. So quant/dequant cost is obtained by DIFFERENTIAL measurement
  instead -- stub a pass out and diff whole-kernel device time. These scopes
  are kept for readability and for the day the capture threshold is settable.
  """

    def deco(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with jax.named_scope(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RemoteWaitBufferedRef(pltpu.BufferedRef):
    """Subclass of BufferedRef that implements semaphore-synchronized memory copies.

  Used to wait for remote device writes before initiating local HBM-to-VMEM
  copies.
  """

    index_fn_with_recv_sem: Callable[..., Any] | None = dataclasses.field(
        metadata={"static": True}, default=None)

    @classmethod
    def from_ref(
        cls,
        ref: pltpu.BufferedRef,
        *,
        index_fn_with_recv_sem: Callable | None = None,
    ):
        return cls(
            index_fn_with_recv_sem=index_fn_with_recv_sem,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    def copy_in(self, src_ref, grid_indices):
        if self.index_fn_with_recv_sem is None:
            super().copy_in(src_ref, grid_indices)
            return
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        chunk_slice, sem, size = self.index_fn_with_recv_sem(
            grid_indices, src_ref)

        window_ref_slice = (slot, slice(None), pl.ds(0, size))
        if sem is not None:
            pltpu.make_async_copy(
                self.window_ref.at[window_ref_slice],
                self.window_ref.at[window_ref_slice],
                sem,
            ).wait()

        hbm_array_ref = (src_ref[0] if isinstance(src_ref,
                                                  (tuple, list)) else src_ref)
        assert self.sem_recvs is not None
        pltpu.make_async_copy(
            hbm_array_ref.at[chunk_slice],
            self.window_ref.at[window_ref_slice],
            self.sem_recvs.at[slot],
        ).start()

    def wait_in(self, src_ref, grid_indices):
        if self.index_fn_with_recv_sem is None:
            super().wait_in(src_ref, grid_indices)
            return
        assert self.window_ref is not None
        wait_slot = self.current_wait_in_slot
        _, _, size = self.index_fn_with_recv_sem(grid_indices, src_ref)

        window_ref_slice = (wait_slot, slice(None), pl.ds(0, size))
        assert self.sem_recvs is not None
        pltpu.make_async_copy(
            self.window_ref.at[window_ref_slice],
            self.window_ref.at[window_ref_slice],
            self.sem_recvs.at[wait_slot],
        ).wait()


# ================================================================================
#                          CHUNK PARTITIONING MAP
# ================================================================================
#         |<----------------------- hidden_dim_size ------------------------>|
#         |<----------- mb_size ----------->|                                |
#         |<-- hc_chunk_size ->|            |                                |
#         +--------------------+------------+---------------+----------------+ ---
#       ^ |          |         |            |               |                |  ^
#       | |  Chunk   |  Chunk  |  MB1 Slice |   MB2 Slice   |   MB3 Slice    |  |
# seq_cs| |          |         |            |               |                |  | seqlen
#       v +--------------------+------------+---------------+----------------+  |
#       ^ |                                 |               |                |  |
# seq_cs| |      Device Slice 1             |               |                |  |
#       v +---------------------------------+---------------+----------------+  v
#                                                                              ---
# ================================================================================
# (seq_cs = seq_chunk_size, seqlen)


class DmaManager:
    """Handles Pallas pipeline emission and explicit async DMA dispatching."""

    def __init__(
        self,
        config: Config,
        topo: Topology,
        locator: ChunkLocator,
        recv_bref,
        run_bref,
        out_bref,
        phase1_send_sems,
        phase1_recv_sems,
        phase2_send_sems,
        phase2_recv_sems,
        # ── FP8 C2C params (None when fp8_comm=False) ──────────────────
        fp8_send_buf=None,
        fp8_recv_buf=None,
        scale_send_buf=None,
        scale_recv_buf=None,
        fp8_p2_send_sems=None,
        fp8_p2_recv_sems=None,
        scale_p2_send_sems=None,
        scale_p2_recv_sems=None,
        # send-side pipelined BufferedRefs (output: BF16->FP8 quantize)
        fp8_send_bref=None,
        scale_send_bref=None,
        # receive-side pipelined BufferedRefs
        fp8_recv_bref=None,
        scale_bref=None,
        # None -> per-chunk dynamic scale (max|x| / 448). A positive float ->
        # static: a fixed scale used by both sides, skipping the send-side
        # max-abs reduction and the scale transfer entirely.
        fp8_static_scale=None,
    ):
        self.config = config
        self.topo = topo
        self.locator = locator
        self.recv_bref = recv_bref
        self.run_bref = run_bref
        self.out_bref = out_bref
        self.phase1_send_sems = phase1_send_sems
        self.phase1_recv_sems = phase1_recv_sems
        self.phase2_send_sems = phase2_send_sems
        self.phase2_recv_sems = phase2_recv_sems
        self.fp8_send_buf = fp8_send_buf
        self.fp8_recv_buf = fp8_recv_buf
        self.scale_send_buf = scale_send_buf
        self.scale_recv_buf = scale_recv_buf
        self.fp8_p2_send_sems = fp8_p2_send_sems
        self.fp8_p2_recv_sems = fp8_p2_recv_sems
        self.scale_p2_send_sems = scale_p2_send_sems
        self.scale_p2_recv_sems = scale_p2_recv_sems
        self.fp8_send_bref = fp8_send_bref
        self.scale_send_bref = scale_send_bref
        self.fp8_recv_bref = fp8_recv_bref
        self.scale_bref = scale_bref
        self.fp8_static_scale = fp8_static_scale
        # Only static mode can drop the scale transfer: dynamic genuinely needs the
        # sender's per-chunk value on the receive side. In static mode the receiver
        # reconstructs the identical constant, so writing/sending/waiting on it is
        # pure overhead -- this is unconditional now, matching Config.skip_scale_dma.
        # Only static mode can drop the scale transfer.
        self.skip_scale_dma = fp8_static_scale is not None

    def start_phase1_d2d_copies(self, src, dst, mb_idx):
        ops = []
        mb_start = mb_idx * self.config.mb_size
        mb_start, mb_slice_size = get_capped_bounds(
            mb_start, self.config.mb_size, self.config.hidden_dim_size)
        partner_chunks = self.locator.get_phase1_chunk_idxes(
            self.topo.partner_id)
        for chip_idx, c_neigh in enumerate(partner_chunks):
            mb_slice = self.locator.get_slice(chunk_idx=c_neigh,
                                              start=mb_start,
                                              size=mb_slice_size)
            op = pltpu.make_async_remote_copy(
                src_ref=src.at[mb_slice],
                dst_ref=dst.at[mb_slice],
                send_sem=self.phase1_send_sems.at[chip_idx, mb_idx],
                recv_sem=self.phase1_recv_sems.at[chip_idx, mb_idx],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            op.start()
            ops.append(op)
        return ops

    def start_phase2_c2c_copies(self, src, dst, mb_idx, step_idx):
        mb_ops = []
        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0

        for op_idx in range(num_ops_in_step):
            for hcube_dim_idx in range(self.config.num_hcube_dims):
                dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims

                mb_start = mb_idx * self.locator.mb_stride
                chunk_start = mb_start + hcube_dim_idx * self.config.hc_chunk_size
                chunk_start, k_size = get_capped_bounds(
                    chunk_start, self.config.hc_chunk_size,
                    self.config.hidden_dim_size)

                neigh_device_id = self.topo.get_neighbor_device_id(dim)
                my_chunk_idx = self.locator.get_phase2_chunk_idx(
                    self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
                neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neigh_device_id, step_idx, op_idx, hcube_dim_idx)

                if k_size > 0:
                    mb_slice = self.locator.get_slice(neighbor_chunk_idx,
                                                      chunk_start, k_size)
                    op = pltpu.make_async_remote_copy(
                        src_ref=src.at[mb_slice],
                        dst_ref=dst.at[mb_slice],
                        send_sem=self.phase2_send_sems.at[step_idx, mb_idx,
                                                          hcube_dim_idx,
                                                          op_idx],
                        recv_sem=self.phase2_recv_sems.at[step_idx, mb_idx,
                                                          hcube_dim_idx,
                                                          op_idx],
                        device_id=neigh_device_id,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    op.start()
                    mb_ops.append((
                        op,
                        step_idx,
                        mb_idx,
                        hcube_dim_idx,
                        op_idx,
                        my_chunk_idx,
                        chunk_start,
                        k_size,
                    ))
        return mb_ops

    @scoped("p1_accum")
    def run_phase1_accumulate_pipeline(
        self,
        src1,
        src2,
        dst,
        in_index_fn,
        out_index_fn,
        hbm_index_fn,
        block_size,
        mb_idx,
    ):
        """Orchestrates a D2D accumulation pipeline on a 1D chip grid."""

        def accum_body(s1_ref, s2_ref, d_ref):
            d_ref[...] = s1_ref[...] + s2_ref[...]

        grid = (self.config.num_chips, )

        def in_index_fn_with_recv_sem(grid_indices, ref):
            hbm_index, size = hbm_index_fn(grid_indices, ref)
            (chip_idx, ) = grid_indices
            sem = self.phase1_recv_sems.at[chip_idx, mb_idx]
            return hbm_index, sem, size

        in_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=out_index_fn,
        )

        s1_bref = RemoteWaitBufferedRef.from_ref(
            self.recv_bref.with_spec(in_spec),
            index_fn_with_recv_sem=in_index_fn_with_recv_sem,
        )
        s2_bref = RemoteWaitBufferedRef.from_ref(
            self.run_bref.with_spec(in_spec))
        d_bref = RemoteWaitBufferedRef.from_ref(
            self.out_bref.with_spec(out_spec))

        pltpu.emit_pipeline(
            accum_body,
            grid=grid,
            in_specs=[in_spec, in_spec],
            out_specs=[out_spec],
        )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])

    @scoped("p2_accum")
    def run_phase2_accumulate_pipeline(
        self,
        src1,
        src2,
        dst,
        in_index_fn,
        out_index_fn,
        hbm_index_fn,
        block_size,
        mb_idx,
        step_idx,
    ):
        """Orchestrates a C2C accumulation pipeline."""

        def accum_body(s1_ref, s2_ref, d_ref):
            d_ref[...] = s1_ref[...] + s2_ref[...]

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def in_index_fn_with_recv_sem(grid_indices, ref):
            hbm_index, size = hbm_index_fn(grid_indices, ref)
            op_idx, hcube_dim_idx = grid_indices
            sem = self.phase2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx,
                                           op_idx]
            return hbm_index, sem, size

        in_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=out_index_fn,
        )

        s1_bref = RemoteWaitBufferedRef.from_ref(
            self.recv_bref.with_spec(in_spec),
            index_fn_with_recv_sem=in_index_fn_with_recv_sem,
        )
        s2_bref = RemoteWaitBufferedRef.from_ref(
            self.run_bref.with_spec(in_spec))
        d_bref = RemoteWaitBufferedRef.from_ref(
            self.out_bref.with_spec(out_spec))

        pltpu.emit_pipeline(
            accum_body,
            grid=grid,
            in_specs=[in_spec, in_spec],
            out_specs=[out_spec],
        )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])

    def _scale_slot(self, step_idx, mb_idx, hcube_dim_idx, op_idx):
        """Flatten (step, mb, hcube_dim, op) → a single scale buffer column."""
        max_ops = 2**(self.config.num_hcube_dims - 1)
        return (step_idx * self.config.num_micro_batches *
                self.config.num_hcube_dims * max_ops +
                mb_idx * self.config.num_hcube_dims * max_ops +
                hcube_dim_idx * max_ops + op_idx)

    @scoped("quant_stage")
    def quantize_chunks_to_fp8_staging(self, src_hbm, mb_idx, step_idx):
        """Pipelined quantize of every Phase-2 chunk this device SENDs this step.

    Send-side mirror of run_phase2_dequant_accumulate_pipeline: emit_pipeline
    double-buffers the BF16 source chunk (HBM->VMEM load), the FP8 staging
    chunk and the scale (both VMEM->HBM stores), so the DMA engine prefetches
    chunk s+1 while the VPU quantizes chunk s. This replaces the old serial
    load/wait → quantize → store/wait → scale-store/wait chain (6 blocking
    HBM round-trips per chunk).

    Reads BF16 from src_hbm (running_sum_ref); writes FP8 to fp8_send_buf and
    the per-chunk scale to scale_send_buf.  Must complete before
    start_phase2_c2c_copies_fp8 is called.
    """
        assert self.run_bref is not None
        assert self.fp8_send_bref is not None
        assert self.scale_send_bref is not None
        assert self.fp8_send_buf is not None
        assert self.scale_send_buf is not None

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        if num_ops_in_step == 0:
            return
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def quant_body(bf16_ref, fp8_ref, scale_ref=None):
            # Per-tensor quantize at a fixed static scale. There is no
            # cross-lane max-abs reduction to do -- that is the send-side win of
            # static mode -- and the round trip stays consistent because the
            # recv side dequantizes with the identical constant.
            data_f32 = bf16_ref[...].astype(jnp.float32)
            if self.fp8_static_scale is not None:
                # scale is "units per fp8 step": quant divides by it, dequant
                # multiplies. 1/fp8_static_scale mirrors the dynamic convention
                # so both paths share the identical clip/cast below.
                #
                # NO non-finite guard on this path -- deliberate, and measured.
                # There used to be a `where(isfinite(x), x, 0.0)` here. It cost
                # 0.587 us, 0.59% of the kernel and 7.9% of the quant stage
                # (EXP-013), and it never had anything to catch: 0 NaN reached
                # this kernel's input across 192,352 unpermute calls with the
                # guard disabled, even though gmm2_res itself carried up to
                # 45,101 NaN elements per call (EXP-014). Quality unchanged
                # (EXP-015, mmlu_pro 0.8204 +/- 0.0071 vs 0.8232 stock).
                #
                # WHY IT IS SAFE, and the condition that keeps it safe:
                # the NaN lives in rows of gmm2_res that this EP shard never
                # writes (gmm_wrapper passes zero_initialize=False). Only the
                # ONE-HOT unpermute reads them -- it contracts over the full
                # batch axis, and 0 * NaN = NaN, so a single poisoned row makes
                # 100% of the output NaN (EXP-010). ragged_gather_reduce never
                # loads an unowned row, so nothing reaches us.
                #
                # ==> This depends on ONEHOT_MOE_PERMUTE_THRESHOLD=0, the
                #     library default (envs.py), which makes the gather path
                #     unconditional. Raising it re-arms the one-hot matmul and
                #     removes the only reason this path can drop the guard.
                #     If that threshold is ever raised, restore the guard.
                scale = 1.0 / self.fp8_static_scale
            else:
                # Cross-lane max-abs reduction. This is what static mode buys
                # its send-side win by skipping -- and it is why the guard is
                # KEPT here and only here: one non-finite value makes this
                # scale NaN and takes the whole chunk with it, so dynamic is
                # structurally far more exposed than static, where a NaN would
                # stay local to its own element.
                data_f32 = jnp.where(jnp.isfinite(data_f32), data_f32, 0.0)
                scale = jnp.max(jnp.abs(data_f32)) / FP8_E4M3_MAX
                scale = jnp.where(scale == 0.0, 1.0, scale)
            fp8_ref[...] = jnp.clip(data_f32 / scale, -FP8_E4M3_MAX,
                                    FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
            # Dynamic must stage the scale for transfer; static does not --
            # the receiver reconstructs the identical constant, so under
            # skip_scale_dma the buffer, the DMA and the wait all disappear.
            if scale_ref is not None:
                scale_ref[...] = jnp.full((1, SCALE_LANE), scale, jnp.float32)

        # Data + scale land at neigh_chunk_idx (the chunk destined for the
        # neighbor) — same index the serial path wrote, and the same index
        # start_phase2_c2c_copies_fp8 reads back.
        def send_data_index_fn(op_idx, hcube_dim_idx):
            dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
            neigh_device_id = self.topo.get_neighbor_device_id(dim)
            neigh_chunk_idx = self.locator.get_phase2_chunk_idx(
                neigh_device_id, step_idx, op_idx, hcube_dim_idx)
            mb_col_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (neigh_chunk_idx, mb_col_idx)

        def send_scale_index_fn(op_idx, hcube_dim_idx):
            dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
            neigh_device_id = self.topo.get_neighbor_device_id(dim)
            neigh_chunk_idx = self.locator.get_phase2_chunk_idx(
                neigh_device_id, step_idx, op_idx, hcube_dim_idx)
            slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx)
            # block_shape (1, SCALE_LANE) -> element (neigh_chunk_idx, slot*128).
            # Return the BLOCK index (neigh_chunk_idx, slot), NOT slot*SCALE_LANE.
            return (neigh_chunk_idx, slot)

        bf16_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size,
                         self.config.hc_chunk_size),
            index_map=send_data_index_fn,
        )
        fp8_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size,
                         self.config.hc_chunk_size),
            index_map=send_data_index_fn,
        )
        scale_spec = pl.BlockSpec(block_shape=(1, SCALE_LANE),
                                  index_map=send_scale_index_fn)

        src_bref = RemoteWaitBufferedRef.from_ref(
            self.run_bref.with_spec(bf16_spec))
        fp8_out_bref = RemoteWaitBufferedRef.from_ref(
            self.fp8_send_bref.with_spec(fp8_spec))
        scale_out_bref = RemoteWaitBufferedRef.from_ref(
            self.scale_send_bref.with_spec(scale_spec))

        if self.skip_scale_dma:
            pltpu.emit_pipeline(
                quant_body,
                grid=grid,
                in_specs=[bf16_spec],
                out_specs=[fp8_spec],
            )(src_hbm, self.fp8_send_buf, allocations=[src_bref, fp8_out_bref])
        else:
            pltpu.emit_pipeline(
                quant_body,
                grid=grid,
                in_specs=[bf16_spec],
                out_specs=[fp8_spec, scale_spec],
            )(
                src_hbm,
                self.fp8_send_buf,
                self.scale_send_buf,
                allocations=[src_bref, fp8_out_bref, scale_out_bref],
            )

    def start_phase2_c2c_copies_fp8(self, mb_idx, step_idx):
        """Start Phase-2 inter-chip copies using FP8 buffers (8-bit wire transfer).

    Assumes quantize_chunks_to_fp8_staging has already been called for this
    (mb_idx, step_idx).  Returns list of op tuples for later accounting.
    """
        assert self.fp8_send_buf is not None
        assert self.fp8_recv_buf is not None
        assert self.fp8_p2_send_sems is not None
        assert self.fp8_p2_recv_sems is not None
        assert self.scale_send_buf is not None
        assert self.scale_recv_buf is not None
        assert self.scale_p2_send_sems is not None
        assert self.scale_p2_recv_sems is not None

        mb_ops = []
        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0

        for op_idx in range(num_ops_in_step):
            for hcube_dim_idx in range(self.config.num_hcube_dims):
                dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims

                mb_start = mb_idx * self.locator.mb_stride
                chunk_start = mb_start + hcube_dim_idx * self.config.hc_chunk_size
                chunk_start, k_size = get_capped_bounds(
                    chunk_start, self.config.hc_chunk_size,
                    self.config.hidden_dim_size)
                if k_size <= 0:
                    continue

                neigh_device_id = self.topo.get_neighbor_device_id(dim)
                my_chunk_idx = self.locator.get_phase2_chunk_idx(
                    self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
                neigh_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neigh_device_id, step_idx, op_idx, hcube_dim_idx)
                slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx,
                                        op_idx)

                data_slice = self.locator.get_slice(neigh_chunk_idx,
                                                    chunk_start, k_size)
                scale_slice = (
                    pl.ds(neigh_chunk_idx, 1),
                    pl.ds(slot * SCALE_LANE, SCALE_LANE),
                )

                # ── 8-bit FP8 data transfer ───────────────────────────────────
                data_op = pltpu.make_async_remote_copy(
                    src_ref=self.fp8_send_buf.at[data_slice],
                    dst_ref=self.fp8_recv_buf.at[data_slice],
                    send_sem=self.fp8_p2_send_sems.at[step_idx, mb_idx,
                                                      hcube_dim_idx, op_idx],
                    recv_sem=self.fp8_p2_recv_sems.at[step_idx, mb_idx,
                                                      hcube_dim_idx, op_idx],
                    device_id=neigh_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                data_op.start()

                # ── scale transfer ───────────────────────────────────────────
                # Negligible in BYTES (512 B) but not in cost: it is a second DMA issue
                # plus two more semaphore arrays per chunk, which is what doubles fp8's
                # fixed cost against bf16. Static mode does not need it at all.
                if self.skip_scale_dma:
                    scale_op = None
                else:
                    scale_op = pltpu.make_async_remote_copy(
                        src_ref=self.scale_send_buf.at[scale_slice],
                        dst_ref=self.scale_recv_buf.at[scale_slice],
                        send_sem=self.scale_p2_send_sems.at[step_idx, mb_idx,
                                                            hcube_dim_idx,
                                                            op_idx],
                        recv_sem=self.scale_p2_recv_sems.at[step_idx, mb_idx,
                                                            hcube_dim_idx,
                                                            op_idx],
                        device_id=neigh_device_id,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    scale_op.start()

                mb_ops.append((
                    data_op,
                    scale_op,
                    step_idx,
                    mb_idx,
                    hcube_dim_idx,
                    op_idx,
                    my_chunk_idx,
                    chunk_start,
                    k_size,
                ))
        return mb_ops

    @scoped("p2_dequant_accum")
    def run_phase2_dequant_accumulate_pipeline(self, running_sum_ref, dst_ref,
                                               mb_idx, step_idx, is_last_step):
        """Pipelined FP8 dequant + BF16 accumulate for Phase 2.

    Replaces the serial wait-loop. emit_pipeline double-buffers the FP8 recv
    chunk, the running-sum chunk, and the scale, so the ICI engine prefetches
    chunk s+1 while the VPU dequantizes/accumulates chunk s.
    """
        assert self.fp8_recv_bref is not None
        assert self.scale_bref is not None
        assert self.fp8_p2_recv_sems is not None
        assert self.scale_p2_recv_sems is not None
        assert self.fp8_recv_buf is not None
        assert self.scale_recv_buf is not None
        assert self.run_bref is not None
        assert self.out_bref is not None

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        if num_ops_in_step == 0:
            return
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def accum_body(fp8_ref, run_ref, *rest):
            # rest is (scale_ref, d_ref), or just (d_ref,) when the scale transfer
            # has been elided -- static mode reconstructs the identical constant, so
            # the sender and receiver still agree by construction.
            fp8_out_ref = None
            # Under skip_scale_dma (static) `rest` carries the destination ref
            # alone and the receiver reconstructs the sender's constant, so the
            # two agree by construction. Dynamic reads the transferred value.
            if self.skip_scale_dma:
                (d_ref, ) = rest
                scale_ref = None
            else:
                scale_ref, d_ref = rest
            if self.fp8_static_scale is not None:
                scale = 1.0 / self.fp8_static_scale
            else:
                scale = scale_ref[0, 0]
            recv_dq = (fp8_ref[...].astype(jnp.float32) * scale).astype(
                jnp.bfloat16)
            acc = recv_dq + run_ref[...]
            d_ref[...] = acc
            # Fused producer store: this block IS a whole transfer chunk (both sides
            # use block_shape (seq_chunk_size, hc_chunk_size)), and the next step
            # sends a subset of the chunks written here. Quantizing now saves the
            # staging pass its 2 B/elem re-read of exactly this data.
            if fp8_out_ref is not None:
                fp8_out_ref[...] = jnp.clip(
                    acc.astype(jnp.float32) / scale, -FP8_E4M3_MAX,
                    FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)

        data_index_fn = self.locator.make_phase2_index_fn(step_idx, mb_idx)
        out_index_fn = (self.locator.make_phase2_out_index_fn(
            step_idx, mb_idx) if is_last_step else
                        self.locator.make_phase2_index_fn(step_idx, mb_idx))

        def scale_index_fn(op_idx, hcube_dim_idx):
            my_chunk_idx = self.locator.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
            slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx)
            return (
                my_chunk_idx,
                slot,
            )  # block_shape (1, SCALE_LANE) -> element (my_chunk_idx, slot*128)

        # --- recv-sem index fns: wait for remote arrival before HBM->VMEM load ---
        def fp8_recv_sem_fn(grid_indices, ref):
            op_idx, hcube_dim_idx = grid_indices
            my_chunk_idx = self.locator.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
            mb_start = mb_idx * self.locator.mb_stride
            mb_start_idx = mb_start + hcube_dim_idx * self.config.hc_chunk_size
            chunk_slice = self.locator.get_slice(my_chunk_idx, mb_start_idx,
                                                 self.config.hc_chunk_size)
            sem = self.fp8_p2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx,
                                           op_idx]
            return chunk_slice, sem, self.config.hc_chunk_size

        def scale_recv_sem_fn(grid_indices, ref):
            op_idx, hcube_dim_idx = grid_indices
            my_chunk_idx = self.locator.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
            slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx)
            scale_slice = (
                pl.ds(my_chunk_idx, 1),
                pl.ds(slot * SCALE_LANE, SCALE_LANE),
            )
            sem = self.scale_p2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx,
                                             op_idx]
            return scale_slice, sem, SCALE_LANE

        fp8_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size,
                         self.config.hc_chunk_size),
            index_map=data_index_fn,
        )
        run_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size,
                         self.config.hc_chunk_size),
            index_map=data_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size,
                         self.config.hc_chunk_size),
            index_map=out_index_fn,
        )
        scale_spec = pl.BlockSpec(block_shape=(1, SCALE_LANE),
                                  index_map=scale_index_fn)

        fp8_bref = RemoteWaitBufferedRef.from_ref(
            self.fp8_recv_bref.with_spec(fp8_spec),
            index_fn_with_recv_sem=fp8_recv_sem_fn,
        )
        run_bref = RemoteWaitBufferedRef.from_ref(
            self.run_bref.with_spec(run_spec))
        scale_bref = RemoteWaitBufferedRef.from_ref(
            self.scale_bref.with_spec(scale_spec),
            index_fn_with_recv_sem=scale_recv_sem_fn,
        )
        out_bref = RemoteWaitBufferedRef.from_ref(
            self.out_bref.with_spec(out_spec))

        dst = dst_ref if is_last_step else running_sum_ref

        if self.skip_scale_dma:
            pltpu.emit_pipeline(
                accum_body,
                grid=grid,
                in_specs=[fp8_spec, run_spec],
                out_specs=[out_spec],
            )(
                self.fp8_recv_buf,
                running_sum_ref,
                dst,
                allocations=[fp8_bref, run_bref, out_bref],
            )
        else:
            pltpu.emit_pipeline(
                accum_body,
                grid=grid,
                in_specs=[fp8_spec, run_spec, scale_spec],
                out_specs=[out_spec],
            )(
                self.fp8_recv_buf,
                running_sum_ref,
                self.scale_recv_buf,
                dst,
                allocations=[fp8_bref, run_bref, scale_bref, out_bref],
            )
