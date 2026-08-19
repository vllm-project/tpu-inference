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
"""Static config objects for TensorCore Reduce-Scatter."""

import dataclasses
import math
import os

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

# jnp.finfo(jnp.float8_e4m3fn).max
FP8_E4M3_MAX = 448.0

# f32 lanes per scale slot -> 128 * 4 = 512 B, the minimum DMA inner slice.
SCALE_LANE = 128

# Minimum local rows (tokens per chip) for the FP8 wire. Below this the transfer
# is latency-bound and FP8's quant/dequant cannot pay for the halved bytes
# (measured crossover ~2048 rows on tpu7x-8). Shapes are static at trace time,
# so this is a compile-time choice per token bucket.
#
# One of only TWO environment knobs in this kernel (the other is
# VLLM_TPU_FP8_RS_STATIC_SCALE below). Every other flag the monolithic version
# carried was experiment scaffolding and has been collapsed to its shipping
# behaviour; see the module history for what was removed and why.
FP8_COMM_MIN_ROWS = int(os.environ.get("VLLM_TPU_FP8_RS_MIN_TOKENS", "2048"))

# Selects STATIC FP8 scaling and sets its value. Unset/empty -> per-chunk
# DYNAMIC scaling (the default). A positive float -> static at that factor;
# 16 is the calibrated value for this model, within 0.08 dB of the dynamic
# ceiling on real activations.
#
# Static is the faster arm: it drops the send-side max-abs reduction AND the
# scale transfer, taking the fp8 phase-2 wire from 2 remote copies + 4
# semaphore arrays per chunk down to 1 + 2. Dynamic is kept as the
# calibration-free fallback for shapes/models this scale was not tuned on.
_static_scale_env = os.environ.get("VLLM_TPU_FP8_RS_STATIC_SCALE", "").strip()
FP8_STATIC_SCALE: float | None = (float(_static_scale_env)
                                  if _static_scale_env else None)


def next_multiple_of(val: int, multiple: int) -> int:
    """Rounds `val` up to the next multiple of `multiple`."""
    return ((val + multiple - 1) // multiple) * multiple


# Target bytes of local input per micro-batch, per wire. The two differ by 4x
# for a mechanical reason: the FP8 wire re-pays the quantize staging pass once
# per micro-batch, and that pass is ~7.4 us of near-pure FIXED cost (measured by
# removal: 8.4 us at 512 local rows against 7.3 us at 2048 -- four times the
# data for less time). So splitting finer is expensive for FP8. BF16 has no such
# pass; extra micro-batches cost it only DMA issues while buying more
# DMA/compute overlap, so it wants small stages.
#
# Keyed on BYTES PER MICRO-BATCH, never on mb or on row count. The FP8 penalty
# tracks STAGE SIZE, not stage count: mb=8 is optimal at a 64 MiB payload
# (8 MiB/stage) and costs +48% at a 16 MiB payload (2 MiB/stage) -- same mb,
# opposite verdict. A rule keyed on rows cannot express that.
_MB_STAGE_TARGET_BYTES = {False: 2 << 20, True: 8 << 20}  # bf16 : fp8

# Ceiling on the micro-batch count. Above this hc_chunk_size degenerates: at
# hidden 4096, mb=16 gives mb_size=256 and hc_chunk_size=128, a single vector
# width. It is also the largest value measured.
_MAX_MICRO_BATCHES = 8


def pick_num_micro_batches(local_seq_len: int, hidden_dim_size: int,
                           itemsize: int, fp8_comm: bool) -> int:
    """Chooses `num_micro_batches` from bytes per micro-batch, per wire.

  Fitted to a 12-cell device-time grid (mb {1,2,4,8} x local rows
  {512,2048,8192} x {bf16, fp8-static16}, hidden 4096) with XLA psum_scatter as
  a per-cell control at <=0.11% spread. Reproduces 6 of 6 measured optima:

      payload   bf16 opt   fp8 opt
       4 MiB       2          1
      16 MiB       8          2
      64 MiB       8          8      

  `fp8_comm` must be the wire ACTUALLY used, i.e. resolved after the
  FP8_COMM_MIN_ROWS downgrade -- picking the fp8 target for a call that runs
  bf16 selects stages 4x too large.

  See results/EXPERIMENTS.md, EXP-007 (grid) and EXP-008 (why the targets
  differ).
  """
    target = _MB_STAGE_TARGET_BYTES[bool(fp8_comm)]
    n = (local_seq_len * hidden_dim_size * itemsize) // target
    mb = 1 << max(0, n.bit_length() - 1)  # floor to a power of two
    # Keep hc_chunk_size >= 2 vector widths.
    mb = min(mb, max(1, hidden_dim_size // 512))
    return int(min(max(mb, 1), _MAX_MICRO_BATCHES))


def get_capped_bounds(start, length, max_size):
    """Calculates capped start and size for a slice to prevent out-of-bounds."""
    capped_start = min(start, max_size)
    capped_end = min(start + length, max_size)
    return capped_start, capped_end - capped_start


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Config:
    # yapf: disable
    """Dimensions and sharding sizes for the TensorCore hierarchical RS.

  Phase 1 is intra-chip D2D (always BF16). Phase 2 is the inter-chip hypercube
  (recursive doubling) over ICI, optionally FP8 on the wire.

  ================================================================================================
              FIGURE 1: PHASE 1 TENSOR PARTITIONING (PIPELINE MICRO-BATCHES)
  ================================================================================================
              |<--------------------------------- hidden_dim_size -------------------------------->|
              |                                         |                                          |
  Iteration:  |<---------------- mb_size -------------->|<---------------- mb_size --------------->| (x num_micro_batches)
              +-----------------------------------------+------------------------------------------+ ---
  seq_chunk_^ |                                         |                                          |  ^
  size      | |         Device 0, Microbatch 0          |          Device 0, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  | local_
  seq_chunk_^ |                                         |                                          |  | seq_len
  size      | |         Device 1, Microbatch 0          |          Device 1, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  |
              |                  ...                    |                    ...                   |  v
              +-----------------------------------------+------------------------------------------+ ---
  ================================================================================================

  ================================================================================================
            FIGURE 2: PHASE 2 TENSOR PARTITIONING (ICI CROSS-CHIP SCATTER)
  ================================================================================================
  Zooming into a SINGLE micro-batch column (`mb_size`) to show how Phase 2 slices
  the chunks further for Reduce-Scatter across the i-th hypercube dimension.

              |<-------------------------------- mb_size ---------------------------------------->|
              |                                         |                                         |
  ICI Phase 2:|<------------ hc_chunk_size ------------>|<------------ hc_chunk_size ------------>| (x num_hcube_dims)
              +-----------------------------------------+-----------------------------------------+ ---
  seq_chunk_^ |                                         |                                         |  ^
  size      | |    Device 0, RS through 0th axis        |    Device 0, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  | local_
  seq_chunk_^ |                                         |                                         |  | seq_len
  size      | |    Device 1, RS through 0th axis        |    Device 1, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  |
              |                  ...                    |                    ...                  |  v
              +-----------------------------------------+-----------------------------------------+ ---
  ================================================================================================
  LEGEND:
  - seq_chunk_size = local_seq_len // num_devices
  - mb_size        = round_up(hidden_dim_size // num_micro_batches, num_lanes)
  - hc_chunk_size  = round_up(mb_size // num_hcube_dims, num_lanes)

  NOTE both mb_size and hc_chunk_size are rounded UP to a vector width rather
  than divided exactly. A ragged tail is expected and is clamped at the slice
  sites via get_capped_bounds; do not "simplify" these to plain division.
  ================================================================================================
  """
    # yapf: enable

    # Total number of devices executing this kernel.
    num_devices: int
    # Total hidden size dimension (e.g., 4096).
    hidden_dim_size: int
    # Local sequence length on this device, post-padding.
    local_seq_len: int
    # Input data type (e.g., bfloat16).
    dtype: jnp.dtype
    # FP8 chip-to-chip wire for Phase 2. Phase 1 stays BF16 either way.
    fp8_comm: bool = False
    # Scale for the FP8 wire. None -> per-chunk dynamic scale (max|x| / 448),
    # which needs no calibration but costs a cross-lane reduction on the send
    # side and a scale transfer on the wire. A positive float -> STATIC scaling
    # at that factor, which elides both (see skip_scale_dma).
    #
    # Defaults to FP8_STATIC_SCALE, i.e. dynamic unless
    # VLLM_TPU_FP8_RS_STATIC_SCALE is set.
    #
    # 16 is the calibrated static value: measured against 200 captured real
    # pre-RS activations (1.68e9 elements) it gives 31.54 dB aggregate SNR
    # against a 31.62 dB dynamic ceiling -- dynamic buys 0.08 dB. Its
    # representable range is 448/16 = 28.0 against an observed |x| max of
    # 13.88, so ~2x clipping headroom. Do not raise past 32 without
    # recalibrating: range 14.0 would sit 1% under the observed max.
    fp8_static_scale: float | None = None
    # Pipelining unrolling factor for overlapping ALU/DMA. If None, determined
    # by heuristic.
    _num_micro_batches: int | None = None

    def __post_init__(self):
        assert self.cores_per_chip == 2, (
            "This kernel architecture strictly supports 2 cores per chip, but"
            f" found {self.cores_per_chip}.")
        assert (self.num_chips & (self.num_chips - 1)
                ) == 0, f"num_chips {self.num_chips} must be a power of 2"
        assert (self.num_hcube_dims
                >= 1), f"num_hcube_dims {self.num_hcube_dims} must be >= 1"
        assert (self.num_micro_batches -
                1) * self.mb_size < self.hidden_dim_size, (
                    "Unsupported micro-batches config: num_micro_batches="
                    f"{self.num_micro_batches} mb_size={self.mb_size} exceeds "
                    f"hidden_dim_size={self.hidden_dim_size}")

    @property
    def cores_per_chip(self) -> int:
        """Number of physical tensor cores per chip on this TPU architecture."""
        return pltpu.get_tpu_info(
        ).chip_version.num_physical_tensor_cores_per_chip

    @property
    def num_chips(self) -> int:
        """Number of physical TPU chips (num_devices // cores_per_chip).

    NOTE this is chips, not devices: tpu7x-8 is 4 chips x 2 TensorCores = 8 JAX
    devices, and the startup banner's `num_chips=8` refers to devices.
    """
        return self.num_devices // self.cores_per_chip

    @property
    def num_hcube_dims(self) -> int:
        """ICI hypercube logical network dimensions (log2(num_chips))."""
        return int(math.log2(self.num_chips))

    @property
    def num_micro_batches(self) -> int:
        """Pipelining unrolling factor for overlapping ALU/DMA.

    If not set explicitly, chosen from bytes per micro-batch with a per-wire
    target -- see pick_num_micro_batches. `fp8_comm` here is already the
    resolved wire: the caller applies the FP8_COMM_MIN_ROWS downgrade before
    building the Config.
    """
        if self._num_micro_batches is not None:
            return self._num_micro_batches
        return pick_num_micro_batches(self.local_seq_len, self.hidden_dim_size,
                                      jnp.dtype(self.dtype).itemsize,
                                      self.fp8_comm)

    @property
    def vector_width(self) -> int:
        """Lane count that mb_size and hc_chunk_size are aligned up to."""
        return pltpu.get_tpu_info().num_lanes

    @property
    def seq_chunk_size(self) -> int:
        """Local sequence slice per device (= local_seq_len // num_devices)."""
        return self.local_seq_len // self.num_devices

    @property
    def mb_size(self) -> int:
        """Micro batch slice size, rounded up to a vector width."""
        return next_multiple_of(self.hidden_dim_size // self.num_micro_batches,
                                self.vector_width)

    @property
    def hc_chunk_size(self) -> int:
        """Phase 2 (C2C) hypercube chunk slice size, rounded up."""
        return next_multiple_of(self.mb_size // max(1, self.num_hcube_dims),
                                self.vector_width)

    @property
    def packing_factor(self) -> int:
        """Number of array elements packed into a single 32-bit word.

    jnp.dtype() normalises both forms: callers pass `local_x.dtype` (a dtype
    instance, which has .itemsize) but a bare `jnp.bfloat16` type does not.
    """
        return 4 // jnp.dtype(self.dtype).itemsize

    @property
    def skip_scale_dma(self) -> bool:
        """Whether the Phase 2 scale transfer is elided.

    Static only. In static mode the scale is a compile-time constant on both
    sides and the receiver reconstructs it, so writing, sending and waiting on
    it is pure overhead. Eliding it takes the fp8 phase-2 wire from 2 remote
    copies + 4 semaphore arrays per chunk down to 1 + 2, the same fixed-cost
    profile as bf16 -- and that fixed cost, not payload bytes, is what sets
    FP8_COMM_MIN_ROWS.

    Dynamic genuinely needs the sender's per-chunk value on the receive side,
    so it keeps both the buffer and the transfer.
    """
        return self.fp8_comm and self.fp8_static_scale is not None
