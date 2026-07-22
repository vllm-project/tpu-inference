# Copyright 2026 Google LLC
"""Generalized C=N MoE gemv kernels (N = num real tokens, e.g. 3 or 4).

Extends `c2_moe_kernels` to an arbitrary small number of decode tokens.
There are `N * TOP_K` routed expert slots; slot j belongs to token
`t = j // TOP_K` and uses that token's LHS row. The N tokens' rows are
packed into a `[N * M_PAD, K]` buffer (token t real row at `t * M_PAD`).
Per-(token,expert) outputs land at row `j * M_PAD`.

Fused single-kernel variant (`_cn_w1w2_fused_mb_kernel_fp8`): per
(token,expert) slot it does gate+up+SwiGLU then down in one kernel,
keeping the intermediate in VMEM (no activation HBM round-trip) and
multi-buffering BOTH w1 and w2 weight DMAs (4 concurrent DMA streams).
Per-token sum-across-experts is accumulated in-kernel.
"""
from __future__ import annotations
import functools
import os as _os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M_PAD = 8
TOP_K = 10

_GEMV_NBUF = max(2, int(_os.getenv("MOE_GEMV_NBUF", "2")))
_FUSE_NBUF = max(2, int(_os.getenv("MOE_FUSE_NBUF", str(_GEMV_NBUF))))
_FUSE_CN_MIN_WIDTH = int(_os.getenv("MOE_FUSE_CN_MIN_WIDTH", "4"))
_K_TILE = int(_os.getenv("MOE_CN_K_TILE", "256"))
_I_TILE = int(_os.getenv("MOE_CN_I_TILE", "1792"))


# ============================================================
# Fused w1+w2 single-kernel variant
# ============================================================

def _cn_w1w2_fused_mb_kernel_fp8(
    ids_ref,            # [TOPK_TOTAL] int32 — flattened (token,expert) ids
    lhs_ref,            # [N*M_PAD, K] bf16 — VMEM, real token at row t*M_PAD
    topk_weights_ref,   # [TOPK_TOTAL] fp32 — VMEM
    w1_ref,             # [G, K, 2*I] fp8 — HBM
    w1_scale_ref,       # [G, K_BLOCKS, 1, 2*I] fp32 — HBM
    w2_ref,             # [G, I, H] fp8 — HBM
    w2_scale_ref,       # [G, I_BLOCKS, 1, H] fp32 — HBM
    o_ref,              # [N*M_PAD, H] bf16 — HBM output
    w1_bufs_ref,        # VMEM scratch [NBUF, K_TILE, 2*I] fp8
    w1_s_bufs_ref,      # VMEM scratch [NBUF, KB_TILE, 1, 2*I] fp32
    w2_bufs_ref,        # VMEM scratch [NBUF, I_TILE, H] fp8
    w2_s_bufs_ref,      # VMEM scratch [NBUF, IB_TILE, 1, H] fp32
    out_acc_ref,        # VMEM scratch [N*M_PAD, H] fp32 — per-token accumulator
    o_scratch_ref,      # VMEM scratch [N*M_PAD, H] bf16 — final cast
    sem_ref,            # DMA semaphores, 4*NBUF slots
    *,
    K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, TOPK_TOTAL_, TOP_K_, NBUF_,
    DTYPE_LHS, DTYPE_OUT,
):
    K_TILE = w1_bufs_ref.shape[1]
    KB_TILE = w1_s_bufs_ref.shape[1]
    NUM_K = K // K_TILE
    I_TILE = w2_bufs_ref.shape[1]
    IB_TILE = w2_s_bufs_ref.shape[1]
    NUM_I = I // I_TILE

    out_acc_ref[...] = jnp.zeros_like(out_acc_ref)

    def slot_body(i, _):
        """Process one (token, expert) slot — replaces the unrolled outer loop."""
        gj = ids_ref[i]       # expert id (runtime value now, not trace-time)
        t = i // TOP_K_       # which token

        # ---- Phase 1: K-tiled gate+up matmul + SwiGLU ----
        pltpu.make_async_copy(
            w1_ref.at[pl.ds(gj, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
            w1_bufs_ref.at[pl.ds(0, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
            sem_ref.at[0 * NBUF_ + 0]
        ).start()
        pltpu.make_async_copy(
            w1_scale_ref.at[pl.ds(gj, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
            w1_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
            sem_ref.at[1 * NBUF_ + 0]
        ).start()

        gate_up_acc = jnp.zeros((M_PAD, 2 * I), dtype=jnp.float32)

        for k in range(NUM_K):
            buf = k % NBUF_
            nxt_k = k + 1
            if nxt_k < NUM_K:
                nxt_buf = nxt_k % NBUF_
                pltpu.make_async_copy(
                    w1_ref.at[pl.ds(gj, 1), pl.ds(nxt_k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
                    w1_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                    sem_ref.at[0 * NBUF_ + nxt_buf]
                ).start()
                pltpu.make_async_copy(
                    w1_scale_ref.at[pl.ds(gj, 1), pl.ds(nxt_k * KB_TILE, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    w1_s_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    sem_ref.at[1 * NBUF_ + nxt_buf]
                ).start()

            pltpu.make_async_copy(
                w1_ref.at[pl.ds(gj, 1), pl.ds(k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
                w1_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                sem_ref.at[0 * NBUF_ + buf]
            ).wait()
            pltpu.make_async_copy(
                w1_scale_ref.at[pl.ds(gj, 1), pl.ds(k * KB_TILE, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                w1_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                sem_ref.at[1 * NBUF_ + buf]
            ).wait()

            w1_fp8 = w1_bufs_ref[buf]
            s1 = w1_s_bufs_ref[buf]
            w1_fp32 = w1_fp8.astype(jnp.float32).reshape(KB_TILE, QB, 2 * I)
            w1_dequant = (w1_fp32 * s1).reshape(K_TILE, 2 * I).astype(DTYPE_LHS)

            lhs_tile = lhs_ref[pl.ds(t * M_PAD, M_PAD), pl.ds(k * K_TILE, K_TILE)]
            gate_up_acc = gate_up_acc + jnp.matmul(lhs_tile, w1_dequant, preferred_element_type=jnp.float32)

        gate_up_bf16 = gate_up_acc.astype(DTYPE_OUT)
        gate = gate_up_bf16[:, :I].astype(jnp.float32)
        up = gate_up_bf16[:, I:].astype(jnp.float32)
        silu_gate = gate * jax.nn.sigmoid(gate)
        intermediate = (silu_gate * up).astype(DTYPE_LHS)  # [M_PAD, I]

        # ---- Phase 2: I-tiled down matmul ----
        pltpu.make_async_copy(
            w2_ref.at[pl.ds(gj, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
            w2_bufs_ref.at[pl.ds(0, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
            sem_ref.at[2 * NBUF_ + 0]
        ).start()
        pltpu.make_async_copy(
            w2_scale_ref.at[pl.ds(gj, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            w2_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            sem_ref.at[3 * NBUF_ + 0]
        ).start()

        down_acc = jnp.zeros((M_PAD, H), dtype=jnp.float32)

        for m in range(NUM_I):
            buf = m % NBUF_
            nxt_m = m + 1
            if nxt_m < NUM_I:
                nxt_buf = nxt_m % NBUF_
                pltpu.make_async_copy(
                    w2_ref.at[pl.ds(gj, 1), pl.ds(nxt_m * I_TILE, I_TILE), pl.ds(0, H)],
                    w2_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                    sem_ref.at[2 * NBUF_ + nxt_buf]
                ).start()
                pltpu.make_async_copy(
                    w2_scale_ref.at[pl.ds(gj, 1), pl.ds(nxt_m * IB_TILE, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    w2_s_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    sem_ref.at[3 * NBUF_ + nxt_buf]
                ).start()

            pltpu.make_async_copy(
                w2_ref.at[pl.ds(gj, 1), pl.ds(m * I_TILE, I_TILE), pl.ds(0, H)],
                w2_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                sem_ref.at[2 * NBUF_ + buf]
            ).wait()
            pltpu.make_async_copy(
                w2_scale_ref.at[pl.ds(gj, 1), pl.ds(m * IB_TILE, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                w2_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                sem_ref.at[3 * NBUF_ + buf]
            ).wait()

            w2_fp8 = w2_bufs_ref[buf]
            s2 = w2_s_bufs_ref[buf]
            w2_fp32 = w2_fp8.astype(jnp.float32).reshape(IB_TILE, IB, H)
            w2_dequant = (w2_fp32 * s2).reshape(I_TILE, H).astype(DTYPE_LHS)

            inter_tile = intermediate[:, m * I_TILE : (m + 1) * I_TILE]
            down_acc = down_acc + jnp.matmul(inter_tile, w2_dequant, preferred_element_type=jnp.float32)

        sl = pl.ds(t * M_PAD, M_PAD)
        out_acc_ref[sl, :] = out_acc_ref[sl, :] + down_acc * topk_weights_ref[i]
        return None

    jax.lax.fori_loop(0, TOPK_TOTAL_, slot_body, None)

    o_scratch_ref[...] = out_acc_ref[...].astype(DTYPE_OUT)
    out_copy = pltpu.make_async_copy(o_scratch_ref, o_ref, sem_ref.at[0])
    out_copy.start()
    out_copy.wait()


def cn_gemv_w1w2_fused_mb_fp8(lhs, w1, w1_scale, w2, w2_scale,
                              active_ids, topk_weights, *,
                              n_tokens, interpret=False):
    """Fused gate+up+SwiGLU+down for the C=N MoE block, multi-buffered at
    depth ``_FUSE_NBUF``. lhs [N*M_PAD, K]; active_ids/topk_weights
    [N*TOP_K] flattened. Returns bf16[N*M_PAD, H] with token t's
    sum-across-experts at row t*M_PAD (rows t*M_PAD+1.. are zero)."""
    G, K, N1 = w1.shape
    assert N1 % 2 == 0, f"w1 last dim must be 2*I; got {N1}"
    I = N1 // 2
    G2, I2, H = w2.shape
    assert I == I2 and G == G2
    K_BLOCKS = w1_scale.shape[1]
    I_BLOCKS = w2_scale.shape[1]
    QB = K // K_BLOCKS
    IB = I // I_BLOCKS
    TOPK_TOTAL = n_tokens * TOP_K
    NBUF = min(_FUSE_NBUF, TOPK_TOTAL)
    topk_weights = topk_weights.astype(jnp.float32)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    K_TILE = min(_K_TILE, K)
    assert K % K_TILE == 0, f"K={K} not divisible by K_TILE={K_TILE}"
    KB_TILE = K_BLOCKS // (K // K_TILE)
    I_TILE = min(_I_TILE, I)
    assert I % I_TILE == 0, f"I={I} not divisible by I_TILE={I_TILE}"
    IB_TILE = I_BLOCKS // (I // I_TILE)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[vmem_spec, vmem_spec, any_spec, any_spec, any_spec,
                  any_spec],
        out_specs=any_spec, grid=(1,),
        scratch_shapes=[
            pltpu.VMEM((NBUF, K_TILE, N1), w1.dtype),
            pltpu.VMEM((NBUF, KB_TILE, 1, N1), w1_scale.dtype),
            pltpu.VMEM((NBUF, I_TILE, H), w2.dtype),
            pltpu.VMEM((NBUF, IB_TILE, 1, H), w2_scale.dtype),
            pltpu.VMEM((n_tokens * M_PAD, H), jnp.float32),
            pltpu.VMEM((n_tokens * M_PAD, H), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((4 * NBUF,)),
        ])
    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))
    return pl.pallas_call(
        functools.partial(_cn_w1w2_fused_mb_kernel_fp8, K=K, I=I, H=H,
                          K_BLOCKS=K_BLOCKS, QB=QB, I_BLOCKS=I_BLOCKS, IB=IB,
                          TOPK_TOTAL_=TOPK_TOTAL, TOP_K_=TOP_K, NBUF_=NBUF,
                          DTYPE_LHS=lhs.dtype, DTYPE_OUT=jnp.bfloat16),
        out_shape=jax.ShapeDtypeStruct((n_tokens * M_PAD, H), jnp.bfloat16),
        grid_spec=grid_spec, compiler_params=compiler_params,
        interpret=interpret, name="cn_gemv_w1w2_fused_mb_fp8",
    )(active_ids, lhs, topk_weights, w1, w1_scale, w2, w2_scale)


def cn_moe_full(hidden_state, w1, w1_scale, w2, w2_scale,
                active_ids, topk_weights, *, interpret=False):
    """hidden_state [C, K]; active_ids/topk_weights [C, TOP_K].
    Returns [C, K] — token t's MoE output at row t (bit-identical to running
    each token's experts independently).

    Expert weights are gathered by direct HBM indexing on active_ids with no
    bounds clamping, so callers must pass a valid expert id for every slot,
    using id 0 + weight 0.0 for any padded/inactive (token, expert) slot.
    """
    C = hidden_state.shape[0]
    K = hidden_state.shape[1]
    lhs = jnp.zeros((C * M_PAD, K), dtype=hidden_state.dtype)
    for t in range(C):
        lhs = lhs.at[t * M_PAD].set(hidden_state[t])
    ids_flat = active_ids.reshape(C * TOP_K)
    weights_flat = topk_weights.reshape(C * TOP_K)
    fused_out = cn_gemv_w1w2_fused_mb_fp8(
        lhs, w1, w1_scale, w2, w2_scale, ids_flat, weights_flat,
        n_tokens=C, interpret=interpret)
    return fused_out[::M_PAD][:C].astype(jnp.bfloat16)

