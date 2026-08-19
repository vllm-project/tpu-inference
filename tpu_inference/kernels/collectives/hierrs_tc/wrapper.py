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

# RS_VMEM_PIN=0 disables VMEM operand pinning entirely and restores the old
# behaviour (pl.ANY operands, 0.95 scoped claim). RS_VMEM_FRAC / RS_VMEM_INPUT
# force a specific plan and exist for sweeps; unset, the plan is chosen by
# _pick_vmem_plan below.
_RS_VMEM_PIN = os.environ.get("RS_VMEM_PIN", "1") != "0"
# EXP-021 probe: also request VMEM for the PRIMARY output (out_shape,
# seq_chunk x hidden -- 2 MiB at 2048 rows). The other outputs stay pl.ANY:
# running_sum / recv_buf / the fp8 payloads are ~16 MiB each and exist as
# outputs only because Pallas cannot allocate HBM scratch, so they cannot move.
_RS_VMEM_OUT = os.environ.get("RS_VMEM_OUT", "0") == "1"
_RS_VMEM_INPUT_OVERRIDE = os.environ.get("RS_VMEM_INPUT")
_RS_VMEM_FRAC_OVERRIDE = os.environ.get("RS_VMEM_FRAC")

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

    out_shapes = [out_shape, running_sum_shape, recv_buf_shape]
    _out0_space = pltpu.VMEM if (_RS_VMEM_OUT and vmem_pin) else pl.ANY
    out_specs = [pl.BlockSpec(memory_space=_out0_space)
                 ] + [pl.BlockSpec(memory_space=pl.ANY)] * 2

    # Separate phase-2 landing buffer for the bf16 wire. Without it an incoming
    # phase-2 chunk can overwrite phase-1 bytes the receiver has not drained
    # yet -- a cross-device WAR that gave wrong answers in 157/200 runs at
    # 512 rows / mb=4. Not optional.
    # The fp8 wire already lands phase 2 in fp8_recv_buf, so it needs nothing.
    if not fp8_comm:
        out_shapes.append(
            jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
                                 local_x.dtype))
        out_specs.append(pl.BlockSpec(memory_space=pl.ANY))

    if fp8_comm:
        fp8_shape = jax.ShapeDtypeStruct((local_seq_len, hidden_dim_size),
                                         jnp.float8_e4m3fn)
        SCALE_LANE = 128
        scale_shape = jax.ShapeDtypeStruct(
            (num_devices, num_scale_slots * SCALE_LANE), jnp.float32)
        out_shapes += [fp8_shape, fp8_shape, scale_shape, scale_shape]
        out_specs += [pl.BlockSpec(memory_space=pl.ANY)] * 4

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
        scratch_shapes=tuple(
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
