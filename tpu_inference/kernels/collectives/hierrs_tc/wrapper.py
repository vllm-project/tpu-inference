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

# RS_VMEM_PIN=0 disables VMEM operand pinning entirely and restores the old
# behaviour (pl.ANY operands, 0.95 scoped claim). RS_VMEM_FRAC / RS_VMEM_INPUT
# force a specific plan and exist for sweeps; unset, the plan is chosen by
# _pick_vmem_plan below.
_RS_VMEM_PIN = os.environ.get("RS_VMEM_PIN", "1") != "0"
# Also request VMEM for the PRIMARY output (out_shape, seq_chunk x hidden --
# 2 MiB at local_seq_len 2048). Takes effect only when the operand is pinned
# too (see `_out0_space` below). The working buffers are handled separately by
# RS_VMEM_WORK.
_RS_VMEM_OUT = os.environ.get("RS_VMEM_OUT", "0") == "1"
_RS_VMEM_INPUT_OVERRIDE = os.environ.get("RS_VMEM_INPUT")
_RS_VMEM_FRAC_OVERRIDE = os.environ.get("RS_VMEM_FRAC")

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
# The working set is `local_seq_len * hidden_dim * 6 bytes` on BOTH wires:
# running_sum and recv_buf are bf16 (2 B/elem each) either way, and the fp8
# wire replaces the bf16 phase-2 landing buffer with two 1-byte staging
# buffers. FP8 halves what crosses the wire; it does not shrink this.
#
# Measured at hidden 4096 on 8 devices, against 0.92 * 64 = 58.9 MiB usable:
#
#   local_seq_len | operand | working set | scoped | total | VMEM scratch?
#            128  |   1.0   |     3.1     |   1.3  |   5.4 | yes
#            256  |   2.0   |     6.1     |   2.6  |  10.7 | yes
#            512  |   4.0   |    12.1     |   5.2  |  21.3 | yes
#           1024  |   8.0   |    24.1     |  10.4  |  42.5 | yes
#           2048  |  16.0   |    48.1     |  10.4  |  74.5 | NO -- over 58.9
#
# Only 2048 is excluded, and it is excluded by arithmetic rather than by a
# tuning constant. There the operand and primary output are still VMEM-pinned
# by _pick_vmem_plan / _RS_VMEM_OUT; it is only the working set that stays in
# HBM. `RS_VMEM_WORK=1 ignored` is logged whenever that happens, once per
# shape, so the fallback is never silent.
#
# `local_seq_len` here is the PER-DEVICE, PRE-scatter row count, i.e. the
# operand's first dim -- 8x the post-scatter row count that appears in the HLO.
# A decode-heavy server sweeps this whole range in one run rather than running
# a single shape, so both branches of the table are live in production and the
# large shapes are NOT covered by this optimisation.
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
    """Total bytes of the buffers that move from pl.ANY outputs to VMEM scratch."""
    big = local_seq_len * hidden_dim_size
    total = 2 * big * itemsize  # running_sum + recv_buf
    if fp8_comm:
        total += 2 * big  # fp8_send + fp8_recv, 1 byte/elem
        total += 2 * num_devices * num_scale_slots * SCALE_LANE * 4
    else:
        total += big * itemsize  # the bf16 wire's phase-2 landing buffer
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
        return False, vmem_frac
    # Claim exactly what this shape needs, not a fixed fraction. Claiming more
    # steals alternate memory MSA needs to colour the operand; claiming less
    # raises CompileTimeScopedVmemOom.
    frac = need / capacity
    if _RS_VMEM_WORK_FRAC is not None and frac > _RS_VMEM_WORK_FRAC:
        _warn_work_scratch_off(work, capacity)
        return False, vmem_frac
    return True, frac


@functools.lru_cache(maxsize=1)
def _memory_space_constraint_survives() -> bool:
    """Will a with_memory_space_constraint aval survive an ordinary primitive?

    jax 0.11 gave jax.core its own MemorySpace enum and made
    `check_avals_context_mesh` assert `isinstance(aval.memory_space,
    MemorySpace)` for every primitive traced under a mesh. The space that
    `with_memory_space_constraint` attaches is Pallas's MemorySpace -- a
    different enum -- so once the annotated aval reaches a non-Pallas primitive
    under shard_map, abstract eval raises

        TypeError: Primitive broadcast_in_dim got aval bfloat16<vmem>[...]
                   with unknown memory_space type: <enum 'MemorySpace'>

    On jax 0.10 there was no such check and the identical code pinned the
    operand fine -- every EXP-017/021/024 pinned arm was measured that way.

    This evaluates jax's own predicate rather than tracing a probe function: the
    check is skipped unless BOTH the context mesh and the aval's mesh are
    non-empty, which only holds inside shard_map, so a standalone eval_shape
    probe reports "survives" no matter what and is worse than useless. If a
    later JAX teaches core about Pallas spaces, pinning switches itself back on
    with no code change here.
    """
    try:
        from jax._src import core as jax_core
    except Exception:  # noqa: BLE001 - unknown JAX layout, assume the old one
        return True
    core_space = getattr(jax_core, "MemorySpace", None)
    if core_space is None:
        # No core-level memory space enum -> no type check to trip over.
        return True
    return isinstance(pltpu.VMEM, core_space)


_PIN_UNSUPPORTED_WARNED = False


def _pin_unsupported_once() -> None:
    global _PIN_UNSUPPORTED_WARNED
    if not _PIN_UNSUPPORTED_WARNED:
        _PIN_UNSUPPORTED_WARNED = True
        print(
            "hierrs_tc: VMEM operand pinning disabled -- this JAX "
            f"({jax.__version__}) rejects a Pallas memory-space annotation on "
            "an aval that escapes into an ordinary primitive. The kernel runs "
            "unpinned (correct, and the measured cost is ~0.3% on the fp8 wire "
            "and ~2.2% on bf16). Set RS_VMEM_INPUT=2 to force the old path.",
            flush=True)


# Default scoped claim when we are NOT pinning. Deliberately generous: it is a
# ceiling, not a reservation (EXP-016 measured 60.80 MiB claimed vs 8.00 MiB
# used), and shrinking it buys nothing unless we are also pinning.
_VMEM_FRAC_UNPINNED = 0.95
# Headroom multiplier on the measured scoped requirement, so a shape whose
# footprint is slightly off the operand/2 rule still compiles.
_VMEM_SCOPED_SLACK = 1.30
# Leave a little of VMEM unclaimed so MSA has somewhere to put small buffers
# besides our operand.
_VMEM_TOTAL_SAFETY = 0.92


def _pick_vmem_plan(local_seq_len, hidden_dim_size, itemsize,
                    num_micro_batches):
    """Decide whether to pin the operand into VMEM, and how much to claim.

  Two facts drive this, both measured in EXP-017:

  1. `pl.ANY` resolves to input color 0 (HBM), so the 16 MiB operand
     round-trips through HBM. `with_memory_space_constraint` is the ONLY way to
     get color 1 -- a `BlockSpec(memory_space=VMEM)` leaves the color at 0 and
     merely stages the operand into scoped VMEM, which is strictly worse.
  2. The color is useless on its own. XLA reserves a colored operand out of
     "alternate memory", which is whatever VMEM our scoped claim leaves behind.
     At the old 0.95 claim that is ~3.2 MiB, and a 16 MiB operand fails to
     compile outright ("Too many buffers are colored in the alternate memory").

  So pinning needs the operand AND the kernel's own scratch to fit in VMEM
  together. Measured scratch is almost exactly half the operand (512 rows ->
  2 MiB of 4, 2048 -> 8 of 16, 4096 -> 16 of 32), so the requirement is ~1.5x
  the operand. At 8192 rows the operand alone is 64 MiB -- the entire VMEM --
  and no claim can make it fit, which is why this must be conditional rather
  than a global constant. Getting that wrong does not merely lose the speedup:
  an over-small claim raises CompileTimeScopedVmemOom and the server never
  boots.

  Returns (pin, frac).
  """
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    # Scratch scales as operand / num_micro_batches: the BufferedRef block is
    # (seq_chunk_size, hidden/mb), so halving mb doubles every staging buffer.
    # Verified exactly against measured `used_scoped_memory_configs`:
    #   512/mb2 -> 2.00 MiB, 2048/mb2 -> 8.00, 4096/mb2 -> 16.00,
    #   8192/mb2 -> 32.00, and 512/mb1 -> 4.00 MiB.
    # An earlier version of this used a flat operand/2 -- correct at mb=2 and
    # 2x too small at mb=1, which crashed a real server at startup with
    # "Scoped allocation with size 4.00M and limit 3.20M". mb MUST be an input.
    scoped_need = int(operand / max(1, num_micro_batches) * _VMEM_SCOPED_SLACK)

    if not _RS_VMEM_PIN or operand + scoped_need > capacity * _VMEM_TOTAL_SAFETY:
        return False, _VMEM_FRAC_UNPINNED

    # Claim enough for our scratch, but never so much that MSA cannot reserve
    # the operand out of what is left.
    lo = scoped_need / capacity
    hi = (capacity - operand) / capacity
    if lo > hi:
        return False, _VMEM_FRAC_UNPINNED
    return True, min(max(lo, 0.05), hi)


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

    vmem_pin, vmem_frac = _pick_vmem_plan(local_seq_len, hidden_dim_size,
                                          local_x.dtype.itemsize,
                                          num_micro_batches)
    if _RS_VMEM_INPUT_OVERRIDE is not None:
        vmem_pin = int(_RS_VMEM_INPUT_OVERRIDE) == 2
    if _RS_VMEM_FRAC_OVERRIDE is not None:
        vmem_frac = float(_RS_VMEM_FRAC_OVERRIDE)
    # Pinning needs a memory-space annotation that survives shard_map tracing.
    # Where it does not, fall back to the unpinned plan rather than crashing;
    # an explicit RS_VMEM_INPUT=2 is still honoured so the failure stays
    # reproducible.
    if (vmem_pin and _RS_VMEM_INPUT_OVERRIDE is None
            and not _memory_space_constraint_survives()):
        _pin_unsupported_once()
        vmem_pin = False

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
    running_sum_shape = jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
                                             local_x.dtype)
    recv_buf_shape = jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
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
    _out0_space = pltpu.VMEM if (_RS_VMEM_OUT and vmem_pin) else pl.ANY
    out_shapes = [out_shape]
    out_specs = [pl.BlockSpec(memory_space=_out0_space)]
    work_scratch = []

    def _emit_work(shape_struct, memref):
        if work_scratch_on:
            work_scratch.append(memref)
        else:
            out_shapes.append(shape_struct)
            out_specs.append(pl.BlockSpec(memory_space=pl.ANY))

    _emit_work(running_sum_shape,
               pltpu.VMEM((local_seq_len, hidden_dim_size), local_x.dtype))
    _emit_work(recv_buf_shape,
               pltpu.VMEM((local_seq_len, hidden_dim_size), local_x.dtype))

    # Separate phase-2 landing buffer for the bf16 wire. Without it an incoming
    # phase-2 chunk can overwrite phase-1 bytes the receiver has not drained
    # yet -- a cross-device WAR that gave wrong answers in 157/200 runs at
    # 512 rows / mb=4. Not optional.
    # The fp8 wire already lands phase 2 in fp8_recv_buf, so it needs nothing.
    if not fp8_comm:
        _emit_work(
            jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
                                 local_x.dtype),
            pltpu.VMEM((local_seq_len, hidden_dim_size), local_x.dtype))

    if fp8_comm:
        fp8_shape = jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
                                         jnp.float8_e4m3fn)
        scale_shape = jax.ShapeDtypeStruct(
            (num_devices, num_scale_slots * SCALE_LANE), jnp.float32)
        # Order must match hier_rs_kernel's unpacking:
        # fp8_send_ref, fp8_recv_ref, scale_send_ref, scale_recv_ref.
        for _ in range(2):
            _emit_work(
                fp8_shape,
                pltpu.VMEM((local_seq_len, hidden_dim_size),
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
        # pl.ANY here on purpose. A BlockSpec of pltpu.VMEM does NOT change the
        # operand color (EXP-017 arm 1: color stayed 0 and scoped usage jumped
        # 8 -> 24 MiB as the operand was staged into scratch). The color is set
        # by with_memory_space_constraint at the call site below instead.
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
            # This becomes scoped_memory_configs {"memory_space":1, "size":...} on
            # the custom-call, i.e. "reserve this much VMEM for me".
            #
            # 0.95 claims ~60.8 of 64 MiB and leaves XLA's memory-space
            # assignment ~3 MiB to work with -- which may be exactly why it never
            # promotes our 16 MiB operand to S(1) while it promotes the identical
            # buffer for its own reduce-scatter. `pl.ANY` does not pin us to HBM;
            # over-claiming scoped VMEM may leave MSA nowhere to put us.
            #
            # Actual need is ~8 MiB of BufferedRefs and semaphores, or ~24 MiB
            # with RS_VMEM_LANDING, so 0.95 over-claims by 3-7x.
            #
            # Default unchanged at 0.95 until this is measured.
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes *
                                 vmem_frac),
            disable_bounds_checks=True,
        ),
    )
    # NOTE: no pltpu.with_memory_space_constraint here. It does not survive
    # shard_map tracing -- the vmem-annotated aval reaches broadcast_in_dim,
    # which rejects it ("unknown memory_space type"), and that failed all 24
    # kernel tests. The in_specs BlockSpec above is sufficient on its own:
    # results/probe_vmem_which_knob.py shows the producer is annotated S(1) with
    # the BlockSpec alone and the constraint call adds nothing.
    if vmem_pin:
        # EXP-017 arm 2. Only `with_memory_space_constraint` sets the memory
        # space on the INPUT AVAL, which is what _resolve_memory_spaces reads
        # (jax/_src/pallas/mosaic/pallas_call_registration.py). The BlockSpec
        # alone does not (arm 1 leaves the color at 0 and merely stages the
        # operand into scoped VMEM). Historically this broke under shard_map --
        # the vmem aval reached broadcast_in_dim -- so this arm re-tests that
        # against the current JAX rather than trusting an old note.
        local_x = pltpu.with_memory_space_constraint(local_x, pltpu.VMEM)
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
