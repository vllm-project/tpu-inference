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
"""Top-level dispatcher for TensorCore Reduce-Scatter.

Phase 1 is intra-chip D2D (always BF16). Phase 2 is the inter-chip hypercube
(recursive doubling) over ICI, optionally FP8 on the wire at a fixed static
scale. Accumulation stays BF16 throughout; only the phase 2 transfer is FP8.
"""

import functools
import logging
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives.hierrs_tc.config import (
    FP8_COMM_MIN_ROWS, SCALE_LANE, Config, next_multiple_of,
    pick_num_micro_batches)
from tpu_inference.kernels.collectives.hierrs_tc.kernel import (
    hier_rs_kernel, make_unified_scratch_shapes)
# stdlib logging on purpose: every other import in this package stays inside
# hierrs_tc, which is what lets the kernel be imported (and unit-tested)
# without dragging in the vLLM graph. tpu_inference.logger imports vllm.
logger = logging.getLogger(__name__)

# RS_VMEM_WORK: place the kernel's WORKING buffers (running_sum, recv_buf and
# the wire's staging buffers) in VMEM scratch instead of HBM.
#
# Those buffers are pure scratch -- nothing downstream reads them. They are
# declared as `pl.ANY` OUTPUTS only because Pallas cannot allocate HBM scratch,
# so output-ness is the mechanism that forces them into HBM. Declaring them as
# real VMEM scratch instead removes the HBM round trip that made the kernel's
# reduce-scatter input and output spill where XLA's psum_scatter keeps both in
# alternate memory.
#
# Note this is NOT the same as `BlockSpec(memory_space=VMEM)` on an output,
# which was measured to colour nothing and merely add a copy-out.
#
# DEFAULT ON, but it is SHAPE-GATED and does not engage everywhere. The working
# set is O(local_seq_len * hidden_dim) and VMEM is 64 MiB total, so
# _plan_work_scratch falls back to the pl.ANY/HBM form once it stops fitting.
#
# The working set is `local_seq_len * hidden_dim * 3 bytes` on BOTH wires.
# The working buffers are PACKED to half the input's rows -- every chunk index
# that touches them carries this device's chiplet parity (ChunkLocator.pack),
# so the full-size allocation they used to get was half dead rows. Unpacked
# the cost was 6 B/elem: running_sum and recv_buf are bf16 (2 B/elem each)
# either way, and the fp8 wire replaces the bf16 phase-2 landing buffer with
# two 1-byte staging buffers. FP8 halves what crosses the wire; it never
# shrank the working set, and packing shrinks both wires equally.
#
# At hidden 4096 on 8 devices, against 0.92 * 64 = 58.9 MiB usable:
#
#   local_seq_len | operand | working set | scoped | total | VMEM scratch?
#            128  |   1.0   |     1.6     |   1.3  |   3.9 | yes
#            256  |   2.0   |     3.1     |   2.6  |   7.7 | yes
#            512  |   4.0   |     6.1     |   5.2  |  15.3 | yes
#           1024  |   8.0   |    12.1     |  10.4  |  30.5 | yes
#           2048  |  16.0   |    24.3     |  10.4  |  50.7 | yes (was NO unpacked)
#           4096  |  32.0   |    48.5     |  10.4  |  90.9 | NO -- over 58.9
#
# Packing is what brought the production-dominant 2048 shape inside the
# budget. 4096+ is still excluded by arithmetic rather than by a tuning
# constant; there the operand and primary output are still VMEM-pinned by
# _pick_vmem_plan / _RS_VMEM_OUT and only the working set stays in HBM.
# `RS_VMEM_WORK=1 ignored` is logged whenever that happens, once per shape,
# so the fallback is never silent.
#
# `local_seq_len` here is the PER-DEVICE, PRE-scatter row count, i.e. the
# operand's first dim -- 8x the post-scatter row count that appears in the HLO.
# A decode-heavy server sweeps this whole range in one run rather than running
# a single shape, so both branches of the table are live in production; with
# packing, every shape up to and including 2048 is covered.
#
# RS_VMEM_WORK=0 restores the pl.ANY output form unconditionally.
_RS_VMEM_WORK = os.environ.get("RS_VMEM_WORK", "1") not in ("0", "")
# Optional hard ceiling on the scoped claim, as a fraction of total VMEM.
# UNSET BY DEFAULT: _plan_work_scratch claims exactly what the shape needs and
# refuses the shape outright when that does not leave room for the operand, so
# a blanket fraction can only reject shapes that genuinely fit.
# Set it to re-impose a ceiling if a future module hits
# "Too many buffers are colored in the alternate memory ... size: 67108864" --
# that failure is what the ceiling was originally guarding against.
_RS_VMEM_WORK_FRAC = (float(os.environ["RS_VMEM_WORK_FRAC"])
                      if "RS_VMEM_WORK_FRAC" in os.environ else None)


def _work_set_bytes(local_seq_len, hidden_dim_size, itemsize, fp8_comm,
                    num_devices, num_scale_slots):
    """Total bytes of the buffers that move from pl.ANY outputs to VMEM scratch.

  The working buffers are PACKED to half the input's rows (see
  ChunkLocator.pack: only this device's chunk parity is ever touched), so the
  per-element cost against local_seq_len * hidden is 3 bytes on either wire:
  bf16 = (2 + 2 + 2)/2, fp8 = (2 + 2 + 1 + 1)/2. FP8 halves what crosses the
  wire; it never shrank the working set, and packing shrinks both wires
  equally.
  """
    packed = (local_seq_len // 2) * hidden_dim_size
    total = 2 * packed * itemsize  # running_sum + recv_buf
    if fp8_comm:
        total += 2 * packed  # fp8_send + fp8_recv, 1 byte/elem
        total += 2 * num_devices * num_scale_slots * SCALE_LANE * 4
    else:
        total += packed * itemsize  # the bf16 wire's phase-2 landing buffer
    return total


@functools.lru_cache(maxsize=None)
def _warn_work_scratch_off(work_bytes: int, capacity: int) -> None:
    """Log the RS_VMEM_WORK fallback once per distinct shape.

  lru_cache keyed on the sizes, so a server that sweeps many shapes logs one
  line per shape rather than one per call.
  """
    logger.info(
        "hierrs_tc: RS_VMEM_WORK=1 ignored at this shape -- the working set "
        "(%.2f MiB) plus the operand does not fit in %.0f MiB of VMEM. "
        "Falling back to pl.ANY outputs (HBM).", work_bytes / 2**20,
        capacity / 2**20)


def _plan_work_scratch(local_seq_len, hidden_dim_size, itemsize, fp8_comm,
                       num_devices, num_scale_slots, num_micro_batches,
                       vmem_frac):
    """Can the working set live in VMEM scratch, and at what scoped claim?

  Moving these buffers out of `pl.ANY` outputs and into scratch makes them
  genuinely VMEM-resident -- unlike a `BlockSpec(memory_space=VMEM)` on the
  output, which colours nothing and merely adds a copy-out.
  The cost is that they now come out of the SCOPED claim, so `vmem_frac` has to
  grow to cover them or the kernel dies at compile time with
  `CompileTimeScopedVmemOom`.

  The operand still has to be colourable out of what the scoped claim leaves
  behind, so this returns (False, unchanged_frac) whenever the two do not fit
  together, which is what happens at the largest shapes.

  Returns (enabled, vmem_frac).
  """
    if not _RS_VMEM_WORK:
        return False, vmem_frac
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    work = _work_set_bytes(local_seq_len, hidden_dim_size, itemsize, fp8_comm,
                           num_devices, num_scale_slots)
    # Existing BufferedRef/semaphore scratch, same model as _pick_vmem_plan.
    scoped = int(operand / max(1, num_micro_batches) * _VMEM_SCOPED_SLACK)
    need = scoped + work
    # THE decision: the scoped claim and the operand must both fit, with
    # _VMEM_TOTAL_SAFETY held back for everything else XLA colours. This is a
    # property of the shape -- no tuning knob can make a shape fit that does
    # not, and none should reject one that does.
    if need + operand > capacity * _VMEM_TOTAL_SAFETY:
        _warn_work_scratch_off(work, capacity)
        # HBM fallback for the WORK SET -- but do not inherit the fat 0.95
        # claim for the scoped scratch that remains. The claim is what starves
        # XLA's memory-space assignment: with 0.95 (60.8 of 64 MiB) MSA has
        # ~3 MiB and colours nothing (the fusion-barrier 0/240), while a
        # need-sized claim leaves it room to promote the operand and output to
        # S(1) on its own -- measured: S(1) copies appear at every shape whose
        # claim stays small, vanish at the 0.95 claim, and no annotation API
        # is involved (with_memory_space_constraint compiles to identical HLO).
        # 1.5x headroom on the scoped estimate; an underestimate fails loudly
        # at compile time (CompileTimeScopedVmemOom), never silently.
        scoped_frac = min(vmem_frac, (scoped * 1.5) / capacity)
        return False, scoped_frac
    # Claim exactly what this shape needs, not a fixed fraction. Claiming more
    # steals alternate memory MSA needs to colour the operand; claiming less
    # raises CompileTimeScopedVmemOom.
    frac = need / capacity
    if _RS_VMEM_WORK_FRAC is not None and frac > _RS_VMEM_WORK_FRAC:
        _warn_work_scratch_off(work, capacity)
        return False, vmem_frac
    return True, frac


# Scoped claim used when the operand cannot share VMEM with the kernel's
# scratch. It is a ceiling, not a reservation: the kernel only uses what its
# buffers need.
_VMEM_FRAC_DEFAULT = 0.95
# Headroom multiplier on the scoped requirement, so a shape whose footprint is
# slightly off the operand/num_micro_batches model still compiles.
_VMEM_SCOPED_SLACK = 1.30
# Fraction of VMEM the scoped claim and the operand may use together. The rest
# is left for XLA to place other small buffers.
_VMEM_TOTAL_SAFETY = 0.92


def _pick_scoped_claim(local_seq_len, hidden_dim_size, itemsize,
                       num_micro_batches):
    """Returns the scoped-VMEM claim for this call, as a fraction of VMEM.

  The claim reserves VMEM for the kernel's staging buffers and semaphores.
  Whatever VMEM it leaves free is what XLA's memory-space assignment can use
  to keep the kernel's operand and output in VMEM instead of HBM. So the claim
  is sized to what the kernel needs and capped so the operand still fits
  beside it. When the two cannot fit together, the default ceiling is
  returned and the operand stays in HBM.

  Too small a claim fails at compile time with CompileTimeScopedVmemOom, so
  the estimate carries `_VMEM_SCOPED_SLACK` headroom.
  """
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    # Staging buffers scale as operand / num_micro_batches: each BufferedRef
    # block is (seq_chunk_size, hidden / num_micro_batches).
    scoped_need = int(operand / max(1, num_micro_batches) * _VMEM_SCOPED_SLACK)

    if operand + scoped_need > capacity * _VMEM_TOTAL_SAFETY:
        return _VMEM_FRAC_DEFAULT

    lo = scoped_need / capacity
    hi = (capacity - operand) / capacity
    if lo > hi:
        return _VMEM_FRAC_DEFAULT
    return min(max(lo, 0.05), hi)


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
    fp8_min_rows: int | None = None,
) -> jax.Array:
    # Shape-specialized wire: local_x.shape is a static Python value at trace
    # time, so each token bucket compiles its own wire format -- plain BF16
    # kernel for small buckets (zero FP8 overhead), FP8 wire for large ones.
    # Zero runtime cost; callers and flags are unchanged.
    #
    # fp8_static_scale: None -> per-chunk dynamic FP8 scale; a positive float ->
    #   fixed static scale (skips the send-side max-abs reduction).
    # fp8_min_rows overrides the env-derived FP8_COMM_MIN_ROWS gate per call.
    #   FP8_COMM_MIN_ROWS is read from os.environ ONCE at import, so a process
    #   that imports this module before setting the env freezes the default for
    #   everyone. A caller that must force the FP8 wire regardless of size (e.g.
    #   a quality/perf harness comparing fp8 vs bf16 at every tested shape, or a
    #   unit test) should pass fp8_min_rows=0 rather than rely on the env var.
    min_rows = FP8_COMM_MIN_ROWS if fp8_min_rows is None else fp8_min_rows
    if fp8_comm and local_x.shape[0] < min_rows:
        fp8_comm = False
    num_chips = num_devices // 2
    num_hcube_dims = int(math.log2(num_chips))
    local_seq_len, hidden_dim_size = local_x.shape

    seq_chunk_size_orig = local_seq_len // num_devices
    # Row-dim (seq) DMA slices and BlockSpec row sizes must be aligned to the TPU
    # sublane tile. On newer chips (tpu7x) that tile is 16 sublanes for bf16 and
    # 32 for fp8 (vs 8 on v6e), and seq_chunk_size = local_seq_len // num_devices
    # is used directly as a block/slice size. Pad each per-device chunk up to a
    # multiple of 32 (covers fp8's worst case; also satisfies bf16/f32) so the
    # kernel compiles and stays correct for any seq length, including small
    # decode batches.
    _SEQ_TILE = 32
    seq_chunk_size_padded = next_multiple_of(max(seq_chunk_size_orig, 1),
                                             _SEQ_TILE)
    needs_padding = seq_chunk_size_padded != seq_chunk_size_orig

    if needs_padding:
        # Pad each device's seq chunk up to the tile multiple; trimmed after.
        reshaped_x = local_x.reshape(num_devices, -1, hidden_dim_size)
        padded_x = jnp.pad(
            reshaped_x,
            ((0, 0), (0, seq_chunk_size_padded - seq_chunk_size_orig), (0, 0)),
        )
        local_x = padded_x.reshape(-1, hidden_dim_size)
        local_seq_len = local_x.shape[0]

    if num_micro_batches is None:
        # Chosen from bytes per micro-batch, with a different target per wire.
        # Note `fp8_comm` is the RESOLVED wire: the FP8_COMM_MIN_ROWS downgrade
        # above may already have turned it off, and a downgraded call must use
        # the bf16 target (4x smaller stages) or it runs badly mis-tuned. This
        # is exactly why the choice lives here and not at the call site.
        num_micro_batches = pick_num_micro_batches(local_seq_len,
                                                   hidden_dim_size,
                                                   local_x.dtype.itemsize,
                                                   fp8_comm)

    vmem_frac = _pick_scoped_claim(local_seq_len, hidden_dim_size,
                                   local_x.dtype.itemsize, num_micro_batches)

    vector_width = pltpu.get_tpu_info().num_lanes
    mb_size = next_multiple_of(hidden_dim_size // num_micro_batches,
                               vector_width)
    assert (num_micro_batches - 1) * mb_size < hidden_dim_size, (
        f"Unsupported micro-batches config: num_micro_batches={num_micro_batches}"
        f" is too large for hidden_dim_size={hidden_dim_size} with"
        f" mb_size={mb_size} (due to padding).")
    hc_chunk_size = next_multiple_of(mb_size // max(1, num_hcube_dims),
                                     vector_width)
    seq_chunk_size = local_seq_len // num_devices

    num_scale_slots = max(
        1,
        num_hcube_dims * num_micro_batches * num_hcube_dims *
        (1 << max(0, num_hcube_dims - 1)),
    )

    out_shape = jax.ShapeDtypeStruct((seq_chunk_size, hidden_dim_size),
                                     local_x.dtype)
    # Working buffers are PACKED: every chunk index that touches them carries
    # this device's chiplet parity (see ChunkLocator.pack), so a full
    # (local_seq_len, hidden) allocation is half dead rows. num_chips chunks
    # of seq_chunk_size rows each == local_seq_len // 2.
    packed_seq_len = local_seq_len // 2
    running_sum_shape = jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                             local_x.dtype)
    recv_buf_shape = jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                          local_x.dtype)

    work_scratch_on, vmem_frac = _plan_work_scratch(
        local_seq_len, hidden_dim_size, local_x.dtype.itemsize, fp8_comm,
        num_devices, num_scale_slots, num_micro_batches, vmem_frac)

    # The working set -- running_sum, recv_buf, and the wire's staging buffers.
    # These are pure scratch: nothing downstream reads them. They are declared
    # as `pl.ANY` OUTPUTS only because Pallas cannot allocate HBM scratch, so
    # output-ness is what forces them into HBM. Where they fit, RS_VMEM_WORK=1
    # declares them as real VMEM scratch instead and the HBM buffers disappear.
    # A `BlockSpec(memory_space=VMEM)` on the OUTPUT is not an alternative: it
    # colours nothing and merely adds a copy-out.
    #
    # ORDER IS LOAD-BEARING. Pallas passes the kernel inputs, then outputs, then
    # scratch. Emitting this group in the same relative order either as trailing
    # outputs or as leading scratch leaves hier_rs_kernel's positional unpacking
    # byte-identical, which is why that file needs no change. Never promote a
    # subset -- that interleaves the two groups and silently permutes the args.
    out_shapes = [out_shape]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY)]
    work_scratch = []

    def _emit_work(shape_struct, memref):
        if work_scratch_on:
            work_scratch.append(memref)
        else:
            out_shapes.append(shape_struct)
            out_specs.append(pl.BlockSpec(memory_space=pl.ANY))

    _emit_work(running_sum_shape,
               pltpu.VMEM((packed_seq_len, hidden_dim_size), local_x.dtype))
    _emit_work(recv_buf_shape,
               pltpu.VMEM((packed_seq_len, hidden_dim_size), local_x.dtype))

    # Separate phase-2 landing buffer for the bf16 wire. Without it an incoming
    # phase-2 chunk can overwrite phase-1 bytes the receiver has not drained
    # yet -- a cross-device WAR that gave wrong answers in 157/200 runs at
    # 512 rows / mb=4. Not optional.
    # The fp8 wire already lands phase 2 in fp8_recv_buf, so it needs nothing.
    if not fp8_comm:
        _emit_work(
            jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                 local_x.dtype),
            pltpu.VMEM((packed_seq_len, hidden_dim_size), local_x.dtype))

    if fp8_comm:
        fp8_shape = jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                         jnp.float8_e4m3fn)
        scale_shape = jax.ShapeDtypeStruct(
            (num_devices, num_scale_slots * SCALE_LANE), jnp.float32)
        # Order must match hier_rs_kernel's unpacking:
        # fp8_send_ref, fp8_recv_ref, scale_send_ref, scale_recv_ref.
        for _ in range(2):
            _emit_work(
                fp8_shape,
                pltpu.VMEM((packed_seq_len, hidden_dim_size),
                           jnp.float8_e4m3fn))
        for _ in range(2):
            _emit_work(
                scale_shape,
                pltpu.VMEM((num_devices, num_scale_slots * SCALE_LANE),
                           jnp.float32))

    config = Config(
        num_devices=num_devices,
        hidden_dim_size=hidden_dim_size,
        local_seq_len=local_seq_len,
        dtype=local_x.dtype,
        fp8_comm=fp8_comm,
        fp8_static_scale=fp8_static_scale,
        _num_micro_batches=num_micro_batches,
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # pl.ANY lets XLA choose where the operand lives. A VMEM BlockSpec
        # would not place the operand in VMEM; it would copy it into the
        # kernel's scoped scratch, which costs more VMEM for no benefit.
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=tuple(out_specs),
        # work_scratch FIRST: it occupies exactly the positions these buffers
        # held as trailing outputs, so the kernel's unpacking is unchanged.
        scratch_shapes=tuple(work_scratch) + tuple(
            make_unified_scratch_shapes(
                seq_chunk_size,
                mb_size,
                local_x.dtype,
                num_chips,
                num_hcube_dims,
                num_micro_batches,
                fp8_comm=fp8_comm,
            )),
        grid=(1, ),
    )

    hier_rs = pl.pallas_call(
        jax.tree_util.Partial(
            hier_rs_kernel,
            config=config,
            axis_name=axis_name,
            fp8_comm=fp8_comm,
            fp8_static_scale=fp8_static_scale,
        ),
        out_shape=tuple(out_shapes),
        grid_spec=grid_spec,
        name=
        f"hier_rs_kernel.mb{num_micro_batches}{'_fp8' if fp8_comm else ''}",
        compiler_params=pltpu.CompilerParams(
            # Becomes scoped_memory_configs on the custom call: the VMEM the
            # kernel reserves for itself. Whatever it leaves free is what XLA
            # can use to keep the operand and output in VMEM, so it is sized to
            # need rather than fixed (see _pick_scoped_claim and
            # _plan_work_scratch).
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes *
                                 vmem_frac),
            disable_bounds_checks=True,
        ),
    )
    out = hier_rs(local_x)[0]
    if needs_padding:
        out = out[:seq_chunk_size_orig, :]
    return out


def hierarchical_reduce_scatter(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
        "x", None),
    num_micro_batches: int | None = None,
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
    fp8_min_rows: int | None = None,
) -> jax.Array:
    return shard_map.shard_map(
        lambda local_x: hierarchical_reduce_scatter_local(
            local_x,
            num_devices=mesh.devices.size,
            num_micro_batches=num_micro_batches,
            fp8_comm=fp8_comm,
            fp8_static_scale=fp8_static_scale,
            fp8_min_rows=fp8_min_rows,
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_rep=False,
    )(x)
