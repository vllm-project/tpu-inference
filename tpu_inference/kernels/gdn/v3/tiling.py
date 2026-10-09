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

"""Head-group and sequence-chunk tiling selection logic for GDN v3 kernels."""

import math
from collections.abc import Sequence

# T=32 is excluded: at n_v=8 it measured +59.6% (N=64) and +15.8% (N=256)
# slower than T=16 because the per-slot f32 state copy and vector temporaries
# spill (18.25 MiB at T=32 vs 9.55 MiB at T=16 in the post-RA LLO ledger).
DECODE_TILE_SIZES = (16, 8, 4, 2, 1)
MIXED_TILE_SIZES = (128, 64, 32, 16, 8, 4, 2, 1)

# Decode tile target balances fixed per-grid-step overhead against per-slot
# work; the VMEM estimate acts as a feasibility ceiling. Constants are
# device-time fits from gdn_attention_benchmark_test on the Qwen3.5 shapes.
_STEP_OVERHEAD_US = 0.31
_DECODE_BASE_US_PER_SEQ = 0.26
_DECODE_US_PER_SEQ_PER_HEAD = 0.0168

_F32_BYTES = 4


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return _cdiv(x, alignment) * alignment


def _nearest_candidate(value: float, candidates: Sequence[int]) -> int:
    """Returns the candidate closest to value in log2 space."""
    return min(
        candidates,
        key=lambda c: abs(math.log2(c) - math.log2(max(value, 1.0))),
    )


def decode_tile_target(num_seqs: int, n_v: int) -> int:
    """Returns the smallest T within 2% of the measured decode optimum.

    Model: t(T) = O_step * ceil(N / T) + w_seq(n_v) * ceil(N / T) * T. The
    square-root balance point reproduces the measured knee for N <= 256 and
    n_v in {8, 32, 64}; DECODE_TILE_SIZES bounds it where the model does not
    encode spill growth.
    """
    if num_seqs <= 0 or n_v <= 0:
        return 1
    w_seq = _DECODE_BASE_US_PER_SEQ + _DECODE_US_PER_SEQ_PER_HEAD * n_v
    return _nearest_candidate(
        math.sqrt(2 * num_seqs * _STEP_OVERHEAD_US / w_seq), DECODE_TILE_SIZES
    )


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
    window_size: int = 1,
    rec_region_bytes: int | None = None,
    conv_region_bytes: int | None = None,
) -> int:
    """Estimates the on-chip VMEM footprint in bytes of one GDN grid step.

    Args:
      tile_b: Sequences per tile (decode T; 1 for PER_SEQ).
      chunk_sz: Tokens per sequence in the tile (decode: window_size).
      rec_state_bytes: Itemsize of a dense recurrent state buffer.
      window_size: State checkpoints per sequence (num_spec_tokens + 1).
      rec_region_bytes / conv_region_bytes: Exact VMEM bytes of one
        checkpoint's recurrent / conv tile when states stream from a pool
        (the raw block layout, padding included). None means dense states.
    """
    aligned_num_v_heads = align_to(n_v, num_lanes)
    aligned_d_k = align_to(d_k, num_lanes)
    aligned_d_v = align_to(d_v, num_lanes)
    dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
    aligned_out_dim = align_to(n_v * d_v, num_lanes)
    state_elems = n_v * d_v * d_k

    # 1. Double-buffered input activations: wrapper.py casts qkv, b and a
    # to f32 before the pipeline.
    qkv_bytes = 2 * tile_b * chunk_sz * dim_size * _F32_BYTES
    b_bytes = 2 * tile_b * chunk_sz * aligned_num_v_heads * _F32_BYTES
    a_bytes = 2 * tile_b * chunk_sz * aligned_num_v_heads * _F32_BYTES

    # 2. Double-buffered state tiles, one region per window checkpoint.
    if conv_region_bytes is None:
        # Dense conv state is cast to f32 before tiling.
        conv_region_bytes = max(0, kernel_size - 1) * conv_state_dim_size * _F32_BYTES
    if rec_region_bytes is None:
        rec_region_bytes = state_elems * rec_state_bytes
    # A windowed tile also holds the incoming state in its own extra position
    # (GDNConfig.state_window_alloc / state_in_pos), so it allocates
    # window_size + 1 regions per sequence.
    state_positions = window_size + (1 if window_size > 1 else 0)
    conv_state_buffer_bytes = 2 * tile_b * state_positions * conv_region_bytes
    recurrent_state_buffer_bytes = 2 * tile_b * state_positions * rec_region_bytes

    # 3. Double-buffered output activation.
    out_bytes = 2 * tile_b * chunk_sz * aligned_out_dim * act_out_bytes

    # 4. f32 working recurrent state: the previous state plus one per
    # emitted checkpoint (window_size; a single final state when not
    # windowed). PER_SEQ adds its carry scratch (conv and recurrent).
    live_state_copies = 1 + window_size
    scratch_recurrent_bytes = tile_b * live_state_copies * state_elems * _F32_BYTES
    if is_decode:
        scratch_conv_bytes = 0
    else:
        scratch_recurrent_bytes += tile_b * state_elems * _F32_BYTES
        scratch_conv_bytes = tile_b * max(0, kernel_size - 1) * dim_size * _F32_BYTES

    # 5. Weights resident in VMEM: conv weight [K, dim] + bias [dim] in f32;
    # a_log and dt_bias.
    weights_bytes = (kernel_size + 1) * dim_size * _F32_BYTES + (
        aligned_num_v_heads * 8
    )

    # 6. Intra-chunk recurrence, per-head projections and per-sequence
    # vector temporaries (8 arrays of shape [8, num_lanes] per slot).
    intermediate_bytes = (
        tile_b
        * (
            n_v * (5 * chunk_sz * chunk_sz + 3 * chunk_sz * (aligned_d_v + aligned_d_k))
            + 8 * 8 * num_lanes
        )
        * _F32_BYTES
    )

    return (
        qkv_bytes
        + b_bytes
        + a_bytes
        + conv_state_buffer_bytes
        + recurrent_state_buffer_bytes
        + out_bytes
        + scratch_conv_bytes
        + scratch_recurrent_bytes
        + weights_bytes
        + intermediate_bytes
    )


def calculate_decode_tile_size(
    batch_size: int,
    *,
    vmem_capacity_limit_bytes: int,
    window_size: int = 1,
    **shape,
) -> int:
    """Largest DECODE_TILE_SIZES entry <= batch_size whose estimate fits.

    ``shape`` carries the get_vmem_estimate_bytes shape/dtype arguments.
    Returns 1 for an empty batch or when nothing fits.
    """
    if batch_size <= 0:
        return 1
    candidates = [c for c in DECODE_TILE_SIZES if c <= batch_size] or [1]
    for cand in candidates:
        est = get_vmem_estimate_bytes(
            tile_b=cand,
            chunk_sz=window_size,
            is_decode=True,
            window_size=window_size,
            **shape,
        )
        if est <= vmem_capacity_limit_bytes:
            return cand
    return candidates[-1]


def calculate_mixed_tile_size(
    seq_len: int,
    *,
    vmem_capacity_limit_bytes: int,
    **shape,
) -> int:
    """Largest MIXED_TILE_SIZES chunk <= seq_len whose estimate fits.

    MIXED_TILE_SIZES stops at C=128, the largest chunk measured: C=128 beat
    C=64 on every swept shape where both fit (n_v=8 and 32), and the VMEM
    estimate rejects C=128 at n_v=64. C > 128 is unmeasured.
    """
    if seq_len <= 0:
        return 1
    candidates = [c for c in MIXED_TILE_SIZES if c <= seq_len] or [1]
    for cand in candidates:
        est = get_vmem_estimate_bytes(
            tile_b=1, chunk_sz=cand, is_decode=False, window_size=1, **shape
        )
        if est <= vmem_capacity_limit_bytes:
            return cand
    return candidates[-1]


def get_tile_sizes(
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    decode_vmem_limit_bytes: int,
    mixed_vmem_limit_bytes: int,
    window_size: int = 1,
    rec_region_bytes: int | None = None,
    conv_region_bytes: int | None = None,
) -> tuple[int, int]:
    """Derives (decode_tile_size, mixed_tile_cap).

    decode_tile_size is min(decode_tile_target, largest VMEM-feasible T).
    mixed_tile_cap is the largest VMEM-feasible PER_SEQ chunk; the caller
    applies it as a ceiling on its own mixed tile choice (this tree tunes
    mixed tiles to 128 rounded to multiples of 16, see wrapper.py), so it
    only binds for shapes where C=128 does not fit (e.g. n_v=64).

    Each limit is the vmem_limit_bytes its pallas_call compiles with
    (GDNConfig.get_vmem_limit_bytes: the decode call is windowed when
    window_size > 1, PER_SEQ never is), so sizing and compilation share one
    number. window_size scales the per-slot state buffers in decode
    (num_spec_tokens + 1).
    """
    shape = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_out_bytes=act_out_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        rec_region_bytes=rec_region_bytes,
        conv_region_bytes=conv_region_bytes,
    )
    vmem_fit = calculate_decode_tile_size(
        # BATCHED tiles over sequences; a tile never needs more slots than
        # there are sequences.
        min(padded_batch_size, num_seqs),
        vmem_capacity_limit_bytes=decode_vmem_limit_bytes,
        window_size=window_size,
        **shape,
    )
    decode_tile_size = min(decode_tile_target(num_seqs, n_v), vmem_fit)
    decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))

    mixed_tile_cap = calculate_mixed_tile_size(
        MIXED_TILE_SIZES[0],
        vmem_capacity_limit_bytes=mixed_vmem_limit_bytes,
        **shape,
    )
    return decode_tile_size, mixed_tile_cap


def spec_window_tile_cap(
    *,
    window: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    dim: int,
    aligned_num_v_heads: int,
    num_buffers: int,
    vmem_budget: int,
    rec_region_bytes: int | None = None,
    conv_region_bytes: int | None = None,
    state_elem_bytes: int = 4,
) -> int:
    """Sequences per verify-window tile that fit half the scoped VMEM.

    A verify window holds one state checkpoint per window position per
    sequence in VMEM, which multiplies the per-sequence footprint by the
    window size. Shrink the tile so the double-buffered windows fit in
    roughly half the scoped-VMEM budget (the rest goes to weights,
    activations scratch and compiler temporaries).

    When ``rec_region_bytes`` / ``conv_region_bytes`` (or ``state_elem_bytes``
    for a BF16 pool) are provided, sizes the double-buffered DMA windows using
    the actual pooled region footprint rather than assuming fp32 DMA buffers.
    """
    rec_bytes = (
        window * rec_region_bytes
        if rec_region_bytes is not None
        else window * n_v * d_k * d_v * state_elem_bytes
    )
    conv_bytes = (
        window * conv_region_bytes
        if conv_region_bytes is not None
        else window * (kernel_size - 1) * dim * state_elem_bytes
    )
    bytes_per_seq = (
        rec_bytes
        + conv_bytes
        # qkv (fp32), b/a (fp32), out (act_out).
        + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
    )
    return ((vmem_budget // 2) // num_buffers) // bytes_per_seq


def legacy_spec_batch_tile(batch_size: int) -> int:
    """Select token-count decode tile size for speculative verify windows."""
    return 1 if batch_size <= 8 else (2 if batch_size <= 32 else 4)


def select_decode_tile_size(
    *,
    default_tile: int,
    dyn_tile: int | None,
    batch_size: int,
    num_spec_tokens: int,
    spec_tile_cap: int | None,
    is_kda: bool = False,
    override: int = 0,
    num_seqs: int | None = None,
) -> int:
    """Final BATCHED-mode tile T for fused_conv1d_gdn.

    Args:
      default_tile: The caller's decode_tile_size (legacy path).
      dyn_tile: get_tile_sizes' decode tile, or None when dynamic tiling is
        off. Replaces both default_tile and, for verify windows, the legacy
        token-count heuristic.
      batch_size: Tokens in the call (qkv rows).
      spec_tile_cap: spec_window_tile_cap(...) when num_spec_tokens > 0.
      is_kda: Whether this is KDA mode.
      override: TPU_GDN_DECODE_TILE_SIZE / TPU_GDN_DECODE_TILE_OVERRIDE; > 0
        forces T (A/B, benches), except for KDA verify windows, which must
        stay at one sequence.
      num_seqs: Optional upper bound on the number of sequences. Because
        compute_batched_seq_metadata pads its SMEM arrays to a multiple of T,
        T does not need to divide num_seqs.
    """
    tile = default_tile if dyn_tile is None else dyn_tile
    tile = min(tile, batch_size)
    if num_seqs is not None and num_seqs > 0:
        tile = min(tile, num_seqs)
    if num_spec_tokens > 0:
        assert spec_tile_cap is not None
        heuristic = tile if dyn_tile is not None else legacy_spec_batch_tile(batch_size)
        tile = max(1, min(tile, heuristic, spec_tile_cap))
        if is_kda:
            tile = 1
    if override > 0 and not (is_kda and num_spec_tokens > 0):
        cap = (
            batch_size
            if (num_seqs is None or num_seqs <= 0)
            else min(batch_size, num_seqs)
        )
        tile = max(1, min(override, cap))
    return max(1, tile)
