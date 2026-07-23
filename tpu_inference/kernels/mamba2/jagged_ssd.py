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
"""Single Pallas kernel for jagged (variable-length) Mamba2 SSD prefill on TPU.

`ssd_candidate(x, dt, A_log, B, C, cu_seqlens) -> y` computes the chunked
state-space-dual (SSD) scan for a packed batch of sequences in one pallas_call,
fusing the three SSD phases (intra-chunk y, cross-chunk state carry, inter-chunk
y from the carried state) so no (NumChunks, H, P, N) intermediates hit HBM.

Algebra inside chunk c (h_init carries from the previous chunk):
    M[i,j,h]   = exp(cum_log_a[i,h]-cum_log_a[j,h]) * dt[j,h]
                 * 1[same sub-seq(i,j)] * 1[j<=i]
    score[i,j] = sum_n C[i,h,n] * B[j,h,n]
    y_intra    = sum_j M * score * x[j]
    h_local    = sum_j M[CHUNK-1,j,h] * x[j] (x) B[j]
    y_inter[t] = exp(cum_log_a[t]) * inter_valid[t] * sum_n C[t,h,n]*h_init[h,n]
    h_init    <- carry_factor[c] * h_init + h_local

Shapes: x (T,H,P) bf16, dt (T,H) bf16, A_log (H,) f32 (log(a)=A_log*dt),
B/C (T,H,N) bf16, cu_seqlens (num_seqs+1,) i32; y (T,H,P) bf16. Hardcoded
H=128, CHUNK=256, I_TILE=64; T is padded to a CHUNK multiple internally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


CHUNK = 256
I_TILE = 64
NUM_I_TILES = CHUNK // I_TILE   # 4
HBG_SIZE = 128

# Bound exp arg in masked / outside-causal positions so exp doesn't
# overflow. exp(-50) ≈ 2e-22 is well below the 5e-2 correctness gate.
MASK_FILL = -50.0


def _build_outer_kernel(NumChunks, NC_padded, H, P, N):

    def _inner_body(
        x_ref,         # bf16[1, HBG, P, CHUNK]
        dt_v_ref,      # bf16[1, HBG, CHUNK]
        clog_ref,      # f32 [1, HBG, CHUNK]
        B_ref,         # bf16[1, HBG, CHUNK, N]
        C_ref,         # bf16[1, HBG, CHUNK, N]
        y_ref,         # bf16[1, HBG, P, CHUNK]    (combined y, written)
        *,
        local_seg_vmem,        # i32[NC_padded, CHUNK]
        inter_valid_vmem,      # f32[NC_padded, CHUNK]
        carry_factor_vmem,     # f32[NC_padded, HBG]
        h_init_scratch,        # f32[HBG, P, N]    persists across iterations
    ):
        """One-chunk body: y_intra + y_inter + h_init update."""
        c = pl.program_id(0)

        # ----- per-chunk metadata (closed over VMEM refs) -----
        ls_chunk          = local_seg_vmem[c, :]         # (CHUNK,) i32
        inter_valid_chunk = inter_valid_vmem[c, :]       # (CHUNK,) f32
        carry_factor_h    = carry_factor_vmem[c, :]      # (H,) f32

        x_h_pj = x_ref[0]                                 # (H, P, CHUNK) bf16
        B_h    = B_ref[0]                                 # (H, CHUNK, N) bf16
        C_h    = C_ref[0]                                 # (H, CHUNK, N) bf16
        dt_v_h = dt_v_ref[0].astype(jnp.float32)          # (H, CHUNK) f32
        clog_h = clog_ref[0]                              # (H, CHUNK) f32

        # ----- prepare h_init for y_inter; precompute decay·inter_valid -----
        # y_inter is computed *per i-tile* (next loop) to keep the live VMEM
        # footprint small. h_init is hoisted once.
        h_init_f32 = h_init_scratch[...]                                   # (H, P, N) f32
        h_init_bf  = h_init_f32.astype(jnp.bfloat16)                       # (H, P, N) bf16

        decay_h = jnp.exp(clog_h)                                          # (H, CHUNK) f32
        factor_h_t = decay_h * inter_valid_chunk[None, :]                  # (H, CHUNK) f32

        # ----- Phase A part 1: h_local (state at end of chunk) -----
        last_clog = clog_h[:, CHUNK - 1]                                  # (H,)
        last_ls   = ls_chunk[CHUNK - 1]                                   # scalar i32

        same_seg_end_bool = (ls_chunk == last_ls)                         # (CHUNK,) bool
        log_diff_end = last_clog[:, None] - clog_h                        # (H, CHUNK) f32
        log_diff_end_safe = jnp.where(
            same_seg_end_bool[None, :], log_diff_end, jnp.float32(MASK_FILL)
        )
        w_end_h = jnp.exp(log_diff_end_safe) * dt_v_h                     # (H, CHUNK) f32

        weighted_x_bf = (
            w_end_h[:, None, :] * x_h_pj.astype(jnp.float32)
        ).astype(jnp.bfloat16)                                            # (H, P, CHUNK) bf16

        h_local = jnp.matmul(
            weighted_x_bf,                                                # (H, P, CHUNK)
            B_h,                                                          # (H, CHUNK, N)
            preferred_element_type=jnp.float32,
        )                                                                  # (H, P, N) f32

        # ----- Phase A part 2: y_intra per i-tile, summed with y_inter -----
        j_idx = jnp.arange(CHUNK, dtype=jnp.int32)

        for it in range(NUM_I_TILES):
            i_start  = it * I_TILE
            C_i      = C_h[:, i_start:i_start + I_TILE, :]                # (H, I_TILE, N)
            clog_i   = clog_h[:, i_start:i_start + I_TILE]                # (H, I_TILE)
            ls_i     = ls_chunk[i_start:i_start + I_TILE]                 # (I_TILE,)
            i_global = i_start + jnp.arange(I_TILE, dtype=jnp.int32)

            score_bf = jnp.matmul(
                C_i,                                                      # (H, I_TILE, N)
                B_h.swapaxes(-1, -2),                                     # (H, N, CHUNK)
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)                                        # (H, I_TILE, CHUNK)

            mask_ij_bool = (
                (ls_i[:, None] == ls_chunk[None, :])
                & (j_idx[None, :] <= i_global[:, None])
            )                                                              # (I_TILE, CHUNK) bool

            log_diff_raw = clog_i[:, :, None] - clog_h[:, None, :]        # (H, I_TILE, CHUNK)
            log_diff_safe = jnp.where(
                mask_ij_bool[None, :, :], log_diff_raw, jnp.float32(MASK_FILL)
            )

            M_bf = (
                jnp.exp(log_diff_safe) * dt_v_h[:, None, :]
            ).astype(jnp.bfloat16)                                        # (H, I_TILE, CHUNK)

            W_bf = M_bf * score_bf                                        # (H, I_TILE, CHUNK)

            y_tile_intra = jax.lax.dot_general(
                x_h_pj,                                                   # (H, P, CHUNK) bf16
                W_bf,                                                     # (H, I_TILE, CHUNK) bf16
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )                                                              # (H, P, I_TILE) f32

            # y_inter for this i-tile: (H, P, I_TILE) = matmul(h_init, C_i^T)
            # Computed per-tile to keep live VMEM smaller than materialising
            # the full (H, P, CHUNK) y_inter buffer upfront.
            y_inter_tile = jnp.matmul(
                h_init_bf,                                                # (H, P, N) bf16
                C_i.swapaxes(-1, -2),                                     # (H, N, I_TILE) bf16
                preferred_element_type=jnp.float32,
            )                                                              # (H, P, I_TILE) f32
            factor_tile = factor_h_t[:, i_start:i_start + I_TILE]         # (H, I_TILE) f32
            y_inter_tile = y_inter_tile * factor_tile[:, None, :]         # (H, P, I_TILE)

            y_tile = y_tile_intra + y_inter_tile
            y_ref[0, :, :, i_start:i_start + I_TILE] = y_tile.astype(jnp.bfloat16)

        # ----- Update h_init scratch for next chunk -----
        # h_init ← carry_factor[c] * h_init + h_local
        h_init_new = carry_factor_h[:, None, None] * h_init_f32 + h_local
        h_init_scratch[...] = h_init_new

    def _outer_kernel(
        x_hbm, dt_v_hbm, clog_hbm, B_hbm, C_hbm,
        local_seg_vmem, inter_valid_vmem, carry_factor_vmem,
        y_hbm,
        h_init_scratch,
    ):
        # Zero-initialize the cross-chunk h_init accumulator.
        h_init_scratch[...] = jnp.zeros((HBG_SIZE, P, N), jnp.float32)

        from functools import partial
        body = partial(
            _inner_body,
            local_seg_vmem=local_seg_vmem,
            inter_valid_vmem=inter_valid_vmem,
            carry_factor_vmem=carry_factor_vmem,
            h_init_scratch=h_init_scratch,
        )

        pltpu.emit_pipeline(
            body,
            grid=(NumChunks,),
            in_specs=[
                pl.BlockSpec((1, HBG_SIZE, P, CHUNK), lambda c: (c, 0, 0, 0)),    # x
                pl.BlockSpec((1, HBG_SIZE, CHUNK),    lambda c: (c, 0, 0)),       # dt_v
                pl.BlockSpec((1, HBG_SIZE, CHUNK),    lambda c: (c, 0, 0)),       # clog
                pl.BlockSpec((1, HBG_SIZE, CHUNK, N), lambda c: (c, 0, 0, 0)),    # B
                pl.BlockSpec((1, HBG_SIZE, CHUNK, N), lambda c: (c, 0, 0, 0)),    # C
            ],
            out_specs=[
                pl.BlockSpec((1, HBG_SIZE, P, CHUNK), lambda c: (c, 0, 0, 0)),    # y
            ],
        )(
            x_hbm, dt_v_hbm, clog_hbm, B_hbm, C_hbm, y_hbm,
        )

    return _outer_kernel


def _single_pallas_kernel(
    x_h, dt_v_h, cum_log_a_h, B_h, C_h,
    local_seg_per_chunk, inter_valid_per_chunk, carry_factor_per_chunk,
):
    NumChunks, H, P, _CHUNK = x_h.shape
    N = B_h.shape[-1]
    NC_padded = local_seg_per_chunk.shape[0]
    assert _CHUNK == CHUNK
    assert H == HBG_SIZE, f"v6 expects full H={HBG_SIZE} per program; got H={H}"

    out_shape = [jax.ShapeDtypeStruct((NumChunks, H, P, CHUNK), jnp.bfloat16)]

    outer_kernel = _build_outer_kernel(NumChunks, NC_padded, H, P, N)

    HBM  = pltpu.MemorySpace.HBM
    VMEM = pltpu.MemorySpace.VMEM

    return pl.pallas_call(
        outer_kernel,
        in_specs=[
            pl.BlockSpec(memory_space=HBM),   # x
            pl.BlockSpec(memory_space=HBM),   # dt_v
            pl.BlockSpec(memory_space=HBM),   # cum_log_a
            pl.BlockSpec(memory_space=HBM),   # B
            pl.BlockSpec(memory_space=HBM),   # C
            pl.BlockSpec(memory_space=VMEM),  # local_seg
            pl.BlockSpec(memory_space=VMEM),  # inter_valid
            pl.BlockSpec(memory_space=VMEM),  # carry_factor
        ],
        out_specs=[pl.BlockSpec(memory_space=HBM)],
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM((HBG_SIZE, P, N), jnp.float32)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
    )(
        x_h, dt_v_h, cum_log_a_h, B_h, C_h,
        local_seg_per_chunk, inter_valid_per_chunk, carry_factor_per_chunk,
    )


def _compute_chunk_metadata(cu_seqlens: jax.Array, T_pad: int, T_real: int):
    """Per-token metadata. Compact (NumChunks, CHUNK) layout; leading dim
    padded to multiple of 8 for VMEM dynamic-slice."""
    NumChunks = T_pad // CHUNK
    NC_padded = ((NumChunks + 7) // 8) * 8

    idx = jnp.arange(T_pad, dtype=cu_seqlens.dtype)
    valid_full = idx < jnp.asarray(T_real, dtype=cu_seqlens.dtype)
    seg_real = jnp.searchsorted(cu_seqlens[1:], idx, side="right").astype(jnp.int32)
    seg_pad_offset = (
        jnp.arange(T_pad, dtype=jnp.int32) + jnp.int32(cu_seqlens.shape[0] + 1000)
    )
    seg_id = jnp.where(valid_full, seg_real, seg_pad_offset)

    seg_id_c = seg_id.reshape(NumChunks, CHUNK)
    valid_c  = valid_full.reshape(NumChunks, CHUNK).astype(jnp.int32)

    seg_prev = jnp.concatenate([seg_id_c[:, :1], seg_id_c[:, :-1]], axis=1)
    is_new = (seg_id_c != seg_prev).astype(jnp.int32)
    is_new = is_new.at[:, 0].set(0)
    local_seg = jnp.cumsum(is_new, axis=1).astype(jnp.int32)

    chunk_first_seg = seg_id_c[:, 0]
    chunk_last_seg  = seg_id_c[:, -1]
    prev_last = jnp.concatenate(
        [jnp.full((1,), -1, dtype=chunk_last_seg.dtype), chunk_last_seg[:-1]]
    )
    first_b = (chunk_first_seg != prev_last).astype(jnp.int32)

    # Sublane-padded layouts.
    local_seg_pad = jnp.zeros((NC_padded, CHUNK), dtype=jnp.int32).at[:NumChunks].set(local_seg)
    return local_seg, valid_c, first_b, local_seg_pad, NumChunks, NC_padded


def ssd_candidate(x, dt, A_log, B, C, cu_seqlens):
    T, H, P = x.shape
    _, _, N = B.shape

    pad = (-T) % CHUNK
    T_pad = T + pad
    if pad > 0:
        x  = jnp.pad(x,  ((0, pad), (0, 0), (0, 0)))
        dt = jnp.pad(dt, ((0, pad), (0, 0)))
        B  = jnp.pad(B,  ((0, pad), (0, 0), (0, 0)))
        C  = jnp.pad(C,  ((0, pad), (0, 0), (0, 0)))

    local_seg, valid_c, first_b, local_seg_pad, NumChunks, NC_padded = (
        _compute_chunk_metadata(cu_seqlens, T_pad, T)
    )

    # JAX-side cum_log_a (cumsum unimplemented in Pallas-TPU lowering).
    dt_full = dt.astype(jnp.float32)
    valid_full_f32 = valid_c.reshape(T_pad).astype(jnp.float32)
    log_a   = (A_log[None, :] * dt_full) * valid_full_f32[:, None]                # (T_pad, H) f32
    log_a_c = log_a.reshape(NumChunks, CHUNK, H)
    cum_log_a_c = jnp.cumsum(log_a_c, axis=1)                                     # (NC, CHUNK, H) f32

    # ----- Precompute small VMEM-resident tensors -----
    # carry_factor[c, h] = exp(cum_log_a[c, -1, h]) * (1 if chunk has no reset else 0)
    last_local_seg  = local_seg[:, -1]                                             # (NC,) i32
    has_internal    = (last_local_seg > 0)
    chunk_has_reset = (first_b != 0) | has_internal
    no_reset_f32    = jnp.logical_not(chunk_has_reset).astype(jnp.float32)         # (NC,)
    full_decay_end  = jnp.exp(cum_log_a_c[:, CHUNK - 1, :])                        # (NC, H)
    carry_factor    = full_decay_end * no_reset_f32[:, None]                       # (NC, H) f32

    # inter_valid[c, t] = (local_seg[c, t] == 0) AND (first_b[c] == 0) AND (valid[c, t] != 0)
    inter_valid = (
        (local_seg == 0) & (first_b[:, None] == 0) & (valid_c != 0)
    ).astype(jnp.float32)                                                          # (NC, CHUNK) f32

    # Pad leading axis to NC_padded for sublane-tile-aligned VMEM dynamic-slice.
    carry_factor_pad = jnp.zeros((NC_padded, H), dtype=jnp.float32).at[:NumChunks].set(carry_factor)
    inter_valid_pad  = jnp.zeros((NC_padded, CHUNK), dtype=jnp.float32).at[:NumChunks].set(inter_valid)

    # Fold valid_j into dt; the kernel reads dt_v and never re-multiplies by valid.
    dt_v_full = (dt_full * valid_full_f32[:, None]).astype(jnp.bfloat16)           # (T_pad, H)

    # H-leading layout; x and y carry (NC, H, P, CHUNK).
    x_h         = x.reshape(NumChunks, CHUNK, H, P).transpose(0, 2, 3, 1)          # (NC, H, P, CHUNK)
    B_h         = B.reshape(NumChunks, CHUNK, H, N).transpose(0, 2, 1, 3)          # (NC, H, CHUNK, N)
    C_h         = C.reshape(NumChunks, CHUNK, H, N).transpose(0, 2, 1, 3)          # (NC, H, CHUNK, N)
    dt_v_h      = dt_v_full.reshape(NumChunks, CHUNK, H).transpose(0, 2, 1)        # (NC, H, CHUNK)
    cum_log_a_h = cum_log_a_c.transpose(0, 2, 1)                                    # (NC, H, CHUNK)

    (y_h_pj,) = _single_pallas_kernel(
        x_h, dt_v_h, cum_log_a_h, B_h, C_h,
        local_seg_pad, inter_valid_pad, carry_factor_pad,
    )                                                                               # (NC, H, P, CHUNK) bf16

    # Repack to (T_pad, H, P) then slice off padding.
    y_full = y_h_pj.transpose(0, 3, 1, 2).reshape(T_pad, H, P)                      # (T_pad, H, P) bf16
    return y_full[:T]


ssd_candidate_jit = jax.jit(ssd_candidate)

