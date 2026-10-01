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

# RS_VMEM_WORK: place the kernel's working buffers (running_sum, recv_buf and
# the wire's staging buffers) in VMEM scratch instead of HBM.
#
# These buffers are pure scratch; nothing downstream reads them. Pallas cannot
# allocate HBM scratch, so the HBM form declares them as `pl.ANY` outputs.
# Declaring them as VMEM scratch instead removes that HBM round trip.
# `BlockSpec(memory_space=VMEM)` on an output is not equivalent: it does not
# place the buffer in VMEM and only adds a copy-out.
#
# On by default, but gated on shape. The working set is
# local_seq_len * hidden_dim * 3 bytes on both wires: the buffers are packed
# to half the input's rows (ChunkLocator.pack), running_sum and recv_buf are
# bf16, and the wire adds either one bf16 phase-2 landing buffer or two 1-byte
# fp8 staging buffers. VMEM is 64 MiB, so _plan_work_scratch falls back to the
# pl.ANY/HBM form once the working set no longer fits.
#
# At hidden 4096 on 8 devices, against 0.92 * 64 = 58.9 MiB usable (MiB):
#
#   local_seq_len | operand | working set | scoped | total | VMEM scratch?
#            128  |   1.0   |     1.6     |   1.3  |   3.9 | yes
#            256  |   2.0   |     3.1     |   2.6  |   7.7 | yes
#            512  |   4.0   |     6.1     |   5.2  |  15.3 | yes
#           1024  |   8.0   |    12.1     |  10.4  |  30.5 | yes
#           2048  |  16.0   |    24.3     |  10.4  |  50.7 | yes
#           4096  |  32.0   |    48.5     |  10.4  |  90.9 | no (over 58.9)
#
# 4096+ rows are excluded by arithmetic, not by a tuning constant. There only
# the working set falls back to HBM; the scoped claim stays need-sized, so XLA
# can still keep the operand and output in VMEM. The fallback is logged once
# per shape.
#
# `local_seq_len` is the per-device, pre-scatter row count (the operand's
# first dim), 8x the post-scatter row count seen in the HLO.
#
# RS_VMEM_WORK=0 restores the pl.ANY output form unconditionally.
_RS_VMEM_WORK = os.environ.get("RS_VMEM_WORK", "1") not in ("0", "")
# Optional ceiling on the scoped claim, as a fraction of total VMEM. Unset by
# default: _plan_work_scratch already claims exactly what the shape needs and
# rejects shapes whose claim leaves no room for the operand, so a blanket
# fraction can only reject shapes that fit. Set it to re-impose a ceiling if a
# module fails with "Too many buffers are colored in the alternate memory".
_RS_VMEM_WORK_FRAC = (float(os.environ["RS_VMEM_WORK_FRAC"])
                      if "RS_VMEM_WORK_FRAC" in os.environ else None)


def _work_set_bytes(local_seq_len, hidden_dim_size, itemsize, fp8_comm,
                    num_devices, num_scale_slots):
    """Total bytes of the working buffers when placed in VMEM scratch.

  The buffers are packed to half the input's rows (ChunkLocator.pack: only
  this device's chunk parity is touched), so the cost per element of
  local_seq_len * hidden is 3 bytes on either wire: bf16 = (2 + 2 + 2) / 2,
  fp8 = (2 + 2 + 1 + 1) / 2. FP8 halves the bytes on the wire, not the
  working set.
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
    """Decides whether the working set lives in VMEM scratch, and the claim.

  As scratch, the working buffers count against the scoped claim, so the
  claim grows to cover them; too small a claim fails at compile time with
  CompileTimeScopedVmemOom. The operand still has to fit in the VMEM the
  claim leaves free, so when the two do not fit together (the largest
  shapes) the working set stays in HBM as pl.ANY outputs and the claim is
  sized to the remaining scoped scratch only.

  Returns (enabled, vmem_frac).
  """
    if not _RS_VMEM_WORK:
        return False, vmem_frac
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    work = _work_set_bytes(local_seq_len, hidden_dim_size, itemsize, fp8_comm,
                           num_devices, num_scale_slots)
    # BufferedRef/semaphore scratch, same model as _pick_scoped_claim.
    scoped = int(operand / max(1, num_micro_batches) * _VMEM_SCOPED_SLACK)
    need = scoped + work
    # The scoped claim and the operand must both fit, with _VMEM_TOTAL_SAFETY
    # held back for everything else XLA places in VMEM. This depends only on
    # the shape.
    if need + operand > capacity * _VMEM_TOTAL_SAFETY:
        _warn_work_scratch_off(work, capacity)
        # The working set goes to HBM, but the claim for the remaining scoped
        # scratch stays need-sized rather than the 0.95 default. The claim is
        # what limits XLA's memory-space assignment: a small claim leaves it
        # room to keep the operand and output in VMEM. 1.5x headroom on the
        # scoped estimate; an underestimate fails at compile time with
        # CompileTimeScopedVmemOom, never silently.
        scoped_frac = min(vmem_frac, (scoped * 1.5) / capacity)
        return False, scoped_frac
    # Claim exactly what this shape needs, not a fixed fraction. Claiming more
    # takes VMEM that memory-space assignment needs for the operand; claiming
    # less raises CompileTimeScopedVmemOom.
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
    # local_x.shape is static at trace time, so each token bucket compiles its
    # own wire: the plain BF16 kernel below the FP8 row threshold, the FP8
    # wire at or above it.
    #
    # fp8_static_scale: None -> per-chunk dynamic FP8 scale; a positive float ->
    #   fixed static scale (skips the send-side max-abs reduction).
    # fp8_min_rows overrides FP8_COMM_MIN_ROWS for this call. FP8_COMM_MIN_ROWS
    #   is read from the environment once at import, so a caller that must
    #   force the FP8 wire at every shape (a test or an fp8-vs-bf16 harness)
    #   should pass fp8_min_rows=0 rather than set the env var.
    min_rows = FP8_COMM_MIN_ROWS if fp8_min_rows is None else fp8_min_rows
    if fp8_comm and local_x.shape[0] < min_rows:
        fp8_comm = False
    num_chips = num_devices // 2
    num_hcube_dims = int(math.log2(num_chips))
    local_seq_len, hidden_dim_size = local_x.shape

    seq_chunk_size_orig = local_seq_len // num_devices
    # Row-dim (seq) DMA slices and BlockSpec row sizes must be aligned to the
    # TPU sublane tile. On newer chips (tpu7x) that tile is 16 sublanes for
    # bf16 and 32 for fp8 (vs 8 on v6e), and seq_chunk_size =
    # local_seq_len // num_devices is used directly as a block/slice size. Pad
    # each per-device chunk up to a multiple of 32 (covers fp8's worst case;
    # also satisfies bf16/f32) so the kernel compiles and stays correct for any
    # seq length, including small decode batches.
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
        # `fp8_comm` here is the resolved wire after the FP8_COMM_MIN_ROWS
        # downgrade above, so a downgraded call gets the BF16 target. That is
        # why the choice is made here rather than at the call site.
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
    # Working buffers are packed: every chunk index that touches them carries
    # this device's chiplet parity (see ChunkLocator.pack), so a full
    # (local_seq_len, hidden) allocation would be half unused rows. num_chips
    # chunks of seq_chunk_size rows each == local_seq_len // 2.
    packed_seq_len = local_seq_len // 2
    running_sum_shape = jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                             local_x.dtype)
    recv_buf_shape = jax.ShapeDtypeStruct((packed_seq_len, hidden_dim_size),
                                          local_x.dtype)

    work_scratch_on, vmem_frac = _plan_work_scratch(
        local_seq_len, hidden_dim_size, local_x.dtype.itemsize, fp8_comm,
        num_devices, num_scale_slots, num_micro_batches, vmem_frac)

    # The working set: running_sum, recv_buf and the wire's staging buffers.
    # Nothing downstream reads them. Pallas cannot allocate HBM scratch, so the
    # HBM form declares them as `pl.ANY` outputs; where they fit
    # (_plan_work_scratch) they are declared as VMEM scratch instead.
    #
    # Order is load-bearing. Pallas passes the kernel its inputs, then outputs,
    # then scratch. Emitting this group in the same relative order, either as
    # trailing outputs or as leading scratch, keeps hier_rs_kernel's positional
    # unpacking the same. Promoting only a subset would interleave the two
    # groups and silently permute the arguments.
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
    # phase-2 chunk can overwrite phase-1 data the receiver has not consumed
    # yet (a cross-device write-after-read hazard). The fp8 wire already lands
    # phase 2 in fp8_recv_buf, so it needs none.
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
        # work_scratch first: it takes the positions these buffers hold as
        # trailing outputs in the HBM form, so the kernel unpacks either form
        # the same way.
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
