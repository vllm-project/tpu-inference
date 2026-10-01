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
    """Wraps a whole pipeline pass in one coarse jax.named_scope.

  There is one region per logical pass (phase-1 accumulate, quantize staging,
  phase-2 dequant+accumulate) rather than one per DMA start/wait; per-DMA
  scopes inflate the Mosaic IR without adding useful structure.

  Applied as a decorator so every caller of the wrapped method gets the
  region without re-indenting the call site.

  Limitation: these regions are not currently visible to xprof. The Pallas
  Mosaic lowering emits `tpu.trace_start` with a hardcoded level of 10, which
  is above the profiler's capture threshold, and neither `trace_level` in
  ProfileOptions.advanced_configuration nor `tpu_trace_mode` changes that.
  The cost of an individual pass therefore has to be measured
  differentially: stub the pass out and compare whole-kernel device time.
  The scopes are kept for readability and for when the capture threshold
  becomes configurable.
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
    """BufferedRef whose copy_in is synchronized on a remote-write semaphore.

  Waits for a remote device's write to land in HBM before starting the local
  HBM-to-VMEM copy of that data.
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
        # Only static mode can drop the scale transfer: dynamic scaling needs
        # the sender's per-chunk value on the receive side, whereas in static
        # mode the receiver reconstructs the identical constant, so writing,
        # sending and waiting on it would be pure overhead. Matches
        # Config.skip_scale_dma.
        self.skip_scale_dma = fp8_static_scale is not None

    def start_phase1_d2d_copies(self, src, dst, mb_idx):
        """Push this device's partner-parity chunks into the partner's recv_buf.

    src is the full input operand; dst is the partner's packed recv_buf, so
    the two slices differ in their row offset. pack(c_neigh) == chip_idx for
    either parity, and c_neigh carries the partner's chiplet bit, which is the
    receiver's parity, so the packed destination row matches where the
    receiver's accumulate pipeline reads (also chip_idx).
    """
        ops = []
        mb_start = mb_idx * self.config.mb_size
        mb_start, mb_slice_size = get_capped_bounds(
            mb_start, self.config.mb_size, self.config.hidden_dim_size)
        partner_chunks = self.locator.get_phase1_chunk_idxes(
            self.topo.partner_id)
        for chip_idx, c_neigh in enumerate(partner_chunks):
            src_slice = self.locator.get_slice(chunk_idx=c_neigh,
                                               start=mb_start,
                                               size=mb_slice_size)
            dst_slice = self.locator.get_packed_slice(chunk_idx=c_neigh,
                                                      start=mb_start,
                                                      size=mb_slice_size)
            op = pltpu.make_async_remote_copy(
                src_ref=src.at[src_slice],
                dst_ref=dst.at[dst_slice],
                send_sem=self.phase1_send_sems.at[chip_idx, mb_idx],
                recv_sem=self.phase1_recv_sems.at[chip_idx, mb_idx],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            op.start()
            ops.append(op)
        return ops

    def start_phase2_c2c_copies(self,
                                mb_idx,
                                step_idx,
                                src=None,
                                dst=None,
                                fp8=False):
        """Start Phase-2 inter-chip (C2C) copies for one micro-batch and step.

    Serves both wires. The hypercube walk (which neighbour, which chunk,
    which hidden-dim slice) is identical for bf16 and fp8; only three things
    differ, and all three are parameters rather than structure:

      * buffers    bf16 takes `src`/`dst` explicitly (running_sum -> the
                   separate phase-2 landing buffer); fp8 always moves
                   fp8_send_buf -> fp8_recv_buf, which the caller has already
                   filled via quantize_chunks_to_fp8_staging.
      * semaphores each wire owns its own send/recv DMA semaphore arrays, so
                   in-flight bf16 and fp8 transfers cannot alias.
      * scale DMA  fp8 dynamic scaling sends a second, small (512 B) buffer
                   alongside the payload. It is cheap in bytes but not in
                   fixed cost: another DMA issue plus two more semaphore
                   arrays per chunk. Static scaling skips it entirely
                   (`skip_scale_dma`).

    Returns a list of tuples whose first two slots are always
    `(data_op, scale_op)`, with `scale_op` None on every path that sends no
    scale (bf16 always, fp8 under static scaling). Callers rely on that fixed
    shape to drain both wires with one code path.
    """
        if fp8:
            assert self.fp8_send_buf is not None
            assert self.fp8_recv_buf is not None
            assert self.fp8_p2_send_sems is not None
            assert self.fp8_p2_recv_sems is not None
            src = self.fp8_send_buf
            dst = self.fp8_recv_buf
            send_sems = self.fp8_p2_send_sems
            recv_sems = self.fp8_p2_recv_sems
            send_scale = not self.skip_scale_dma
            if send_scale:
                assert self.scale_send_buf is not None
                assert self.scale_recv_buf is not None
                assert self.scale_p2_send_sems is not None
                assert self.scale_p2_recv_sems is not None
        else:
            assert src is not None and dst is not None, (
                "the bf16 wire must be given explicit src/dst refs")
            send_sems = self.phase2_send_sems
            recv_sems = self.phase2_recv_sems
            send_scale = False

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
                neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neigh_device_id, step_idx, op_idx, hcube_dim_idx)

                # src and dst are both packed working buffers (bf16:
                # running_sum -> landing buffer; fp8: fp8_send -> fp8_recv),
                # and sender and receiver share chunk parity, so one packed
                # slice serves both ends.
                mb_slice = self.locator.get_packed_slice(
                    neighbor_chunk_idx, chunk_start, k_size)
                data_op = pltpu.make_async_remote_copy(
                    src_ref=src.at[mb_slice],
                    dst_ref=dst.at[mb_slice],
                    send_sem=send_sems.at[step_idx, mb_idx, hcube_dim_idx,
                                          op_idx],
                    recv_sem=recv_sems.at[step_idx, mb_idx, hcube_dim_idx,
                                          op_idx],
                    device_id=neigh_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                data_op.start()

                scale_op = None
                if send_scale:
                    slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx,
                                            op_idx)
                    scale_slice = (
                        pl.ds(neighbor_chunk_idx, 1),
                        pl.ds(slot * SCALE_LANE, SCALE_LANE),
                    )
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

    @scoped("p1_accum")
    def run_phase1_accumulate_pipeline(
        self,
        src1,
        src2,
        dst,
        in1_index_fn,
        in2_index_fn,
        out_index_fn,
        hbm_index_fn,
        block_size,
        mb_idx,
    ):
        """Orchestrates a D2D accumulation pipeline on a 1D chip grid.

    src1 (recv_buf) and dst (running_sum) are packed working buffers; src2 is
    the full input operand, so the two inputs take separate index fns. This is
    the only pipeline where packed and full indexing meet.
    """

        def accum_body(s1_ref, s2_ref, d_ref):
            d_ref[...] = s1_ref[...] + s2_ref[...]

        grid = (self.config.num_chips, )

        def in_index_fn_with_recv_sem(grid_indices, ref):
            hbm_index, size = hbm_index_fn(grid_indices, ref)
            (chip_idx, ) = grid_indices
            sem = self.phase1_recv_sems.at[chip_idx, mb_idx]
            return hbm_index, sem, size

        in1_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in1_index_fn,
        )
        in2_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in2_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=out_index_fn,
        )

        s1_bref = RemoteWaitBufferedRef.from_ref(
            self.recv_bref.with_spec(in1_spec),
            index_fn_with_recv_sem=in_index_fn_with_recv_sem,
        )
        s2_bref = RemoteWaitBufferedRef.from_ref(
            self.run_bref.with_spec(in2_spec))
        d_bref = RemoteWaitBufferedRef.from_ref(
            self.out_bref.with_spec(out_spec))

        pltpu.emit_pipeline(
            accum_body,
            grid=grid,
            in_specs=[in1_spec, in2_spec],
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
        """Pipelined quantize of every Phase-2 chunk sent this step.

    Send-side mirror of run_phase2_dequant_accumulate_pipeline: emit_pipeline
    double-buffers the BF16 source chunk (HBM->VMEM load), the FP8 staging
    chunk and the scale (both VMEM->HBM stores), so the DMA engine prefetches
    chunk s+1 while the VPU quantizes chunk s.

    Reads BF16 from src_hbm (running_sum_ref); writes FP8 to fp8_send_buf and
    the per-chunk scale to scale_send_buf. Must complete before
    start_phase2_c2c_copies(..., fp8=True) is called.
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
            # Static mode quantizes with a fixed per-tensor scale and skips the
            # cross-lane max-abs reduction; the round trip stays consistent
            # because the receiver dequantizes with the identical constant.
            # Dynamic mode computes a per-chunk scale and ships it.
            data_f32 = bf16_ref[...].astype(jnp.float32)
            if self.fp8_static_scale is not None:
                # scale is "units per fp8 step": quant divides by it, dequant
                # multiplies. 1/fp8_static_scale mirrors the dynamic convention
                # so both paths share the identical clip/cast below.
                #
                # This path has no non-finite guard, which keeps the quantize
                # stage cheap. Like psum_scatter, it passes a non-finite input
                # through. The MoE combine does not produce one: rows this EP
                # shard never writes are uninitialized (gmm_v2 runs with
                # zero_initialize=False), but ragged_gather_reduce never loads
                # them, and the one-hot unpermute zeroes them before its
                # contraction (fused_moe_gmm.moe_gmm_local).
                scale = 1.0 / self.fp8_static_scale
            else:
                # Per-chunk scale from a cross-lane max-abs reduction. The
                # non-finite guard is required here: one non-finite value would
                # make the scale NaN and corrupt the whole chunk, whereas under
                # a static scale a NaN stays local to its own element.
                data_f32 = jnp.where(jnp.isfinite(data_f32), data_f32, 0.0)
                scale = jnp.max(jnp.abs(data_f32)) / FP8_E4M3_MAX
                scale = jnp.where(scale == 0.0, 1.0, scale)
            fp8_ref[...] = jnp.clip(data_f32 / scale, -FP8_E4M3_MAX,
                                    FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
            # Dynamic mode must stage the scale for transfer. Static mode does
            # not: the receiver reconstructs the identical constant, so under
            # skip_scale_dma the buffer, the DMA and the wait are all omitted.
            if scale_ref is not None:
                scale_ref[...] = jnp.full((1, SCALE_LANE), scale, jnp.float32)

        # Data and scale are written at neigh_chunk_idx (the chunk destined
        # for the neighbor), the same index that
        # start_phase2_c2c_copies(..., fp8=True) reads back.
        def send_data_index_fn(op_idx, hcube_dim_idx):
            dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
            neigh_device_id = self.topo.get_neighbor_device_id(dim)
            neigh_chunk_idx = self.locator.get_phase2_chunk_idx(
                neigh_device_id, step_idx, op_idx, hcube_dim_idx)
            mb_col_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            # running_sum (source) and fp8_send (dest) are both packed.
            return (self.locator.pack(neigh_chunk_idx), mb_col_idx)

        def send_scale_index_fn(op_idx, hcube_dim_idx):
            dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
            neigh_device_id = self.topo.get_neighbor_device_id(dim)
            neigh_chunk_idx = self.locator.get_phase2_chunk_idx(
                neigh_device_id, step_idx, op_idx, hcube_dim_idx)
            slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx)
            # block_shape (1, SCALE_LANE): element (neigh_chunk_idx, slot*128).
            # Return the block index (neigh_chunk_idx, slot), not
            # slot * SCALE_LANE.
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

    @scoped("p2_dequant_accum")
    def run_phase2_dequant_accumulate_pipeline(self, running_sum_ref, dst_ref,
                                               mb_idx, step_idx, is_last_step):
        """Pipelined FP8 dequant + BF16 accumulate for Phase 2.

    emit_pipeline double-buffers the FP8 recv chunk, the running-sum chunk
    and the scale, so the DMA engine prefetches chunk s+1 while the VPU
    dequantizes and accumulates chunk s.
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
            # rest is (scale_ref, d_ref), or just (d_ref,) under skip_scale_dma.
            # Static mode reconstructs the sender's constant, so both sides
            # agree by construction; dynamic mode reads the transferred value.
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

        # Recv-sem index fns: wait for remote arrival before the HBM->VMEM load.
        def fp8_recv_sem_fn(grid_indices, ref):
            op_idx, hcube_dim_idx = grid_indices
            my_chunk_idx = self.locator.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, op_idx, hcube_dim_idx)
            mb_start = mb_idx * self.locator.mb_stride
            mb_start_idx = mb_start + hcube_dim_idx * self.config.hc_chunk_size
            # fp8_recv is packed.
            chunk_slice = self.locator.get_packed_slice(
                my_chunk_idx, mb_start_idx, self.config.hc_chunk_size)
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
