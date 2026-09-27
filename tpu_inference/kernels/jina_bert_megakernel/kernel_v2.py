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
"""JinaBert v2 encoder megakernel, version 2 (`JINA_BERT_MEGAKERNEL_VERSION`).

Same contract, packed weights (`JinaBertPackedWeights`), MXU operand
rounding and float32 numerics as version 1 (`kernel.py`), and the same
structure (every layer in one grid-less call, residual stream resident in
VMEM, next layer's weights streamed in by double-buffered DMAs). The
attention inner loop, most of the kernel's time, does less work per
(query tile, KV tile) step, and input-independent work moves out of it:

* **ALiBi bias tables once per step.** v1 recomputes the token distance,
  the segment mask and `slope_h * -|i - j|` in every (layer, query tile, KV
  tile) step. None of that depends on the layer, and the bias does not
  depend on the input at all: v2 builds per-head bias tables for every tile
  offset once per kernel call, while the input DMAs are in flight, and a KV
  step just adds the table for its offset.
* **No mask work inside one request.** Tile pairs that both lie in the same
  request (every pair of a 1x1024 or 1x2048 step, and most pairs of packed
  steps) skip the segment mask. Other pairs build a 0 / -2e30 mask once and
  share it between all heads.
* **Head pairs stacked along rows.** The two heads that share a 128-lane
  head pair are stacked into one [2 * tile, 128] query operand, so one K^T
  and one V weight load on the MXU serve both heads (half of v1's weight
  loads), and each head's online-softmax state lives in its own rows (no
  per-lane head selects in the loop).
* **Softmax in base 2.** Scores are scaled by log2(e) in float32 after the
  MXU (the MXU operands are exactly v1's) and the bias tables carry
  `slope_h * log2(e)`, so `exp2` replaces `exp` (which lowers to a multiply
  plus exp2) for the probabilities and the running-max correction.
* **Cheaper GELU.** Same Abramowitz & Stegun 7.1.26 erf as v1 with the
  constants folded and `x * Phi(x)` written as `max(x, 0) - |x| * h`.

The running max, the denominator (sum of the unrounded probabilities) and
the accumulator stay float32; the probabilities are rounded to the MXU
operand dtype for P.V as in v1 and the XLA path.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.jina_bert_megakernel import kernel as v1

LOG2E = math.log2(math.e)
# Initial running max (finite: avoids inf - inf in the first correction).
_M_INIT = -1e30
# Added to the scores of token pairs in different requests. More negative
# than _M_INIT, so exp2(masked - running max) is exactly 0 even for a row
# that has only seen masked keys so far (its running max is still _M_INIT).
_MASK_NEG = -2e30
SUPPORTED_TILES = (128, 256)


def supports_tile(tile: int) -> bool:
    return tile in SUPPORTED_TILES


def gelu_erf(x: jax.Array) -> jax.Array:
    """Exact (erf) GELU in float32, as `kernel.gelu_erf` with the constants
    folded: x Phi(x) = max(x, 0) - |x| h with h = 0.5 erfc(|x| / sqrt(2))
    from A&S 7.1.26 (|error| <= 1.5e-7)."""
    ax = jnp.abs(x)
    t = 1.0 / (1.0 + (0.3275911 / math.sqrt(2.0)) * ax)
    h = t * (0.5 * 0.254829592 + t * (0.5 * -0.284496736 + t *
                                      (0.5 * 1.421413741 + t *
                                       (0.5 * -1.453152027 + t *
                                        (0.5 * 1.061405429)))))
    h = h * jnp.exp2(x * x * (-0.5 * LOG2E))  # * exp(-x^2 / 2)
    return jnp.maximum(x, 0.0) - ax * h


def _megakernel_v2(
        meta_ref,  # SMEM int32 [1 + 4 * n_tiles]
        seg_q_ref,  # VMEM int32 [n_tiles, tile, 128]
        seg_k_ref,  # VMEM int32 [n_tiles, 8, tile]
        vecs_ref,  # VMEM f32 [L, 16, D]
        x_hbm,  # ANY f32 [n_tiles, tile, D]
        wqkv_hbm,  # ANY [L, D, 3D]
        wo_hbm,  # ANY [L, D, D]
        wg_hbm,  # ANY [L, D, 2F]
        wd_hbm,  # ANY [L, F, D]
        out_hbm,  # ANY f32 [n_tiles, tile, D]
        x_res,  # f32 [n_tiles, tile, D]: residual stream, VMEM-resident.
        q_scr,  # cdt [n_tiles, n_pairs, 2 * tile, 128]: stacked masked Q.
        k_scr,  # cdt [n_tiles, tile, D]
        v_scr,  # cdt [n_tiles, tile, D]
        wqkv_buf,  # wdt [2, D, 3D]  (double-buffered over layers)
        wo_buf,  # wdt [2, D, D]
        wg_buf,  # wdt [2, D, 2F]
        wd_buf,  # wdt [2, F, D]
        bias_scr,  # f32 [n_pairs, 2 * n_tiles - 1, 2 * tile, tile]
        mask_scr,  # f32 [2 * tile, tile]: 0 / _MASK_NEG.
        m_scr,  # f32 [n_pairs, 2 * tile, 128]: running max (base 2).
        l_scr,  # f32 [n_pairs, 2 * tile, 128]: running denominator.
        acc_scr,  # f32 [n_pairs, 2 * tile, 128]: running P.V.
        ctx_scr,  # cdt [tile, D]: normalized attention context.
        h1_scr,  # f32 [tile, D]: post-attention LayerNorm output.
        h1c_scr,  # cdt [tile, D]: same, as MXU operand.
        y_scr,  # f32 [tile, D]: MLP accumulator.
        sems,  # DMA semaphores [10].
        *,
        num_layers: int,
        alibi_slopes: tuple[float, ...],
        eps: float,
        highest: bool):
    """Kernel body; see the module docstring.

    Stacked layouts [2 * tile, ...] of head pair p hold head 2p in rows
    [0, tile) and head 2p + 1 in rows [tile, 2 * tile). Q rows of head h
    are zero outside h's 64 lanes of the pair (as in v1), so contracting
    over the pair's 128 lanes gives head h's scores, and lanes [0, 64) /
    [64, 128) of a P.V row are the context of head 2p / 2p + 1.
    """
    n_tiles, tile, d = x_res.shape
    n_pairs = q_scr.shape[1]
    ffn = wd_buf.shape[1]
    n_chunks = ffn // v1.MLP_CHUNK
    cdt = q_scr.dtype
    reps = tile // v1.NUM_LANES
    f32 = jnp.float32
    precision = lax.Precision.HIGHEST if highest else None
    lanes128 = v1.NUM_LANES
    slopes2 = [s * LOG2E for s in alibi_slopes]  # Base-2 ALiBi slopes.

    def mm(a, b):  # [M, K] @ [K, N]
        return lax.dot_general(a,
                               b, (((1, ), (0, )), ((), ())),
                               precision=precision,
                               preferred_element_type=f32)

    def mm_nt(a, b):  # [M, K] @ [N, K]^T
        return lax.dot_general(a,
                               b, (((1, ), (1, )), ((), ())),
                               precision=precision,
                               preferred_element_type=f32)

    def lane_tile(x):  # [M, 128] lane-replicated -> [M, tile]
        return x if reps == 1 else jnp.tile(x, (1, reps))

    def even_head_lanes():  # Lanes of the first head of a 128-lane pair.
        lane = lax.broadcasted_iota(jnp.int32, (tile, lanes128), 1)
        return lane < v1.HEAD_DIM

    def weight_copies(layer, slot):
        pairs = ((wqkv_hbm, wqkv_buf), (wo_hbm, wo_buf), (wg_hbm, wg_buf),
                 (wd_hbm, wd_buf))
        copies = []
        for i, (src, dst) in enumerate(pairs):
            sem = sems.at[2 * i + slot]  # Semaphores 0-7: 4 weights x 2 slots.
            copies.append(
                pltpu.make_async_copy(src.at[layer], dst.at[slot], sem))
        return tuple(copies)

    # Per-tile metadata offsets in `meta_ref` (see encoder_megakernel_v2).
    meta_kv_lo = 1
    meta_kv_hi = 1 + n_tiles
    meta_seg_first = 1 + 2 * n_tiles
    meta_seg_last = 1 + 3 * n_tiles

    # Prologue: residual stream and layer-0 weights HBM -> VMEM, overlapped
    # with building the ALiBi bias tables.
    x_in = pltpu.make_async_copy(x_hbm, x_res, sems.at[8])
    x_in.start()
    for cp in weight_copies(0, 0):
        cp.start()
    n_active = meta_ref[0]

    # bias_scr[p, o] is the base-2 ALiBi bias of head pair p for query tile
    # qi and KV tile kj with o = qi - kj + n_tiles - 1: slope2_h * -|i - j|
    # for h = 2p (rows [0, tile)) and 2p + 1 (rows [tile, 2 * tile)). Only
    # offsets between active tiles are ever read, so only those are built.
    def bias_table(o, carry):
        dist = (lax.broadcasted_iota(jnp.int32, (tile, tile), 0) -
                lax.broadcasted_iota(jnp.int32, (tile, tile), 1) +
                (o - (n_tiles - 1)) * tile)  # i - j
        neg_dist = jnp.minimum(dist, -dist).astype(f32)  # -|i - j|
        for p in range(n_pairs):
            bias_scr[p, o, 0:tile] = slopes2[2 * p] * neg_dist
            bias_scr[p, o, tile:2 * tile] = slopes2[2 * p + 1] * neg_dist
        return carry

    lax.fori_loop(n_tiles - n_active, n_tiles + n_active - 1, bias_table, 0)
    x_in.wait()

    def qkv_projection(layer, slot):
        bq = vecs_ref[layer, v1.ROW_BQ:v1.ROW_BQ + 1, :]
        bk = vecs_ref[layer, v1.ROW_BK:v1.ROW_BK + 1, :]
        bv = vecs_ref[layer, v1.ROW_BV:v1.ROW_BV + 1, :]

        def body(t, carry):
            xt = x_res[t].astype(cdt)
            q = mm(xt, wqkv_buf[slot, :, 0:d]) + bq
            even = even_head_lanes()
            for p in range(n_pairs):
                q_pair = q[:, p * lanes128:(p + 1) * lanes128]
                q_scr[t, p, 0:tile] = jnp.where(even, q_pair, 0.0).astype(cdt)
                q_scr[t, p, tile:2 * tile] = jnp.where(even, 0.0,
                                                       q_pair).astype(cdt)
            k_scr[t] = (mm(xt, wqkv_buf[slot, :, d:2 * d]) + bk).astype(cdt)
            v_scr[t] = (mm(xt, wqkv_buf[slot, :, 2 * d:3 * d]) +
                        bv).astype(cdt)
            return carry

        lax.fori_loop(0, n_active, body, 0)

    def attention_and_mlp(layer, slot):
        bo = vecs_ref[layer, v1.ROW_BO:v1.ROW_BO + 1, :]
        ln1_g = vecs_ref[layer, v1.ROW_LN1_G:v1.ROW_LN1_G + 1, :]
        ln1_b = vecs_ref[layer, v1.ROW_LN1_B:v1.ROW_LN1_B + 1, :]
        bd = vecs_ref[layer, v1.ROW_BD:v1.ROW_BD + 1, :]
        ln2_g = vecs_ref[layer, v1.ROW_LN2_G:v1.ROW_LN2_G + 1, :]
        ln2_b = vecs_ref[layer, v1.ROW_LN2_B:v1.ROW_LN2_B + 1, :]

        def q_tile(qi, carry):
            m_scr[...] = jnp.full(m_scr.shape, _M_INIT, f32)
            l_scr[...] = jnp.zeros(l_scr.shape, f32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, f32)
            q_first = meta_ref[meta_seg_first + qi]
            q_pure = q_first == meta_ref[meta_seg_last + qi]

            def update(kj, masked):
                o = qi - kj + (n_tiles - 1)
                for p in range(n_pairs):
                    lanes = slice(p * lanes128, (p + 1) * lanes128)
                    # Scores of both heads of pair p, base 2: [2 tile, tile].
                    s = mm_nt(q_scr[qi, p], k_scr[kj, :, lanes])
                    s = s * LOG2E + bias_scr[p, o]
                    if masked:
                        s = s + mask_scr[...]
                    m_prev = m_scr[p]
                    m_new = jnp.maximum(m_prev,
                                        jnp.max(s, axis=1, keepdims=True))
                    probs = jnp.exp2(s - lane_tile(m_new))
                    alpha = jnp.exp2(m_prev - m_new)
                    l_scr[p] = alpha * l_scr[p] + jnp.sum(
                        probs, axis=1, keepdims=True)
                    m_scr[p] = m_new
                    acc_scr[p] = alpha * acc_scr[p] + mm(
                        probs.astype(cdt), v_scr[kj, :, lanes])

            def kv_step(kj, carry):
                # A tile is "pure" if its first and last rows are in the
                # same segment (segments are contiguous).
                k_first = meta_ref[meta_seg_first + kj]
                unmasked = (q_pure & (k_first == meta_ref[meta_seg_last + kj])
                            & (q_first == k_first))

                @pl.when(unmasked)
                def _same_request():
                    update(kj, masked=False)

                @pl.when(jnp.logical_not(unmasked))
                def _mixed():
                    k_seg = seg_k_ref[kj, 0:1, :]  # [1, tile]
                    same_seq = lane_tile(seg_q_ref[qi]) == k_seg
                    mask = jnp.where(same_seq, 0.0, _MASK_NEG)
                    mask_scr[0:tile] = mask
                    mask_scr[tile:2 * tile] = mask
                    update(kj, masked=True)

                return carry

            lax.fori_loop(meta_ref[meta_kv_lo + qi], meta_ref[meta_kv_hi + qi],
                          kv_step, 0)

            # Normalize and un-stack: lanes [0, 64) of a pair come from head
            # 2p's rows, [64, 128) from head 2p + 1's.
            even = even_head_lanes()
            for p in range(n_pairs):
                lanes = slice(p * lanes128, (p + 1) * lanes128)
                acc = acc_scr[p]
                denom = l_scr[p]
                num = jnp.where(even, acc[0:tile], acc[tile:2 * tile])
                den = jnp.where(even, denom[0:tile], denom[tile:2 * tile])
                ctx_scr[:, lanes] = (num / den).astype(cdt)

            attn = mm(ctx_scr[...], wo_buf[slot]) + bo
            h1 = v1._layer_norm(attn + x_res[qi], ln1_g, ln1_b, eps)
            h1_scr[...] = h1
            h1c_scr[...] = h1.astype(cdt)
            for c in range(n_chunks):
                gate_cols = slice(c * v1.MLP_CHUNK, (c + 1) * v1.MLP_CHUNK)
                up_cols = slice(ffn + c * v1.MLP_CHUNK,
                                ffn + (c + 1) * v1.MLP_CHUNK)
                gate = mm(h1c_scr[...], wg_buf[slot, :, gate_cols])
                up = mm(h1c_scr[...], wg_buf[slot, :, up_cols])
                act = (gelu_erf(gate) * up).astype(cdt)
                part = mm(act, wd_buf[slot, gate_cols, :])
                if c == 0:
                    y_scr[...] = part
                else:
                    y_scr[...] += part
            x_res[qi] = v1._layer_norm(y_scr[...] + bd + h1_scr[...], ln2_g,
                                       ln2_b, eps)
            return carry

        lax.fori_loop(0, n_active, q_tile, 0)

    def layer_body(layer, carry):
        slot = layer % 2

        # The other slot was last read by layer - 1, which has finished:
        # stream layer + 1's weights into it while this layer computes.
        @pl.when(layer + 1 < num_layers)
        def _prefetch_next_layer():
            for cp in weight_copies(layer + 1, 1 - slot):
                cp.start()

        wqkv_cp, wo_cp, wg_cp, wd_cp = weight_copies(layer, slot)
        wqkv_cp.wait()
        qkv_projection(layer, slot)
        wo_cp.wait()
        wg_cp.wait()
        wd_cp.wait()
        attention_and_mlp(layer, slot)
        return carry

    lax.fori_loop(0, num_layers, layer_body, 0)

    x_out = pltpu.make_async_copy(x_res, out_hbm, sems.at[9])
    x_out.start()
    x_out.wait()


def _scratch_shapes(n_tiles, tile, d, num_heads, ffn, cdt, wdt):
    n_pairs = num_heads // 2
    f32 = jnp.float32
    return [
        pltpu.VMEM((n_tiles, tile, d), f32),  # x_res
        pltpu.VMEM((n_tiles, n_pairs, 2 * tile, v1.NUM_LANES), cdt),  # q_scr
        pltpu.VMEM((n_tiles, tile, d), cdt),  # k_scr
        pltpu.VMEM((n_tiles, tile, d), cdt),  # v_scr
        pltpu.VMEM((2, d, 3 * d), wdt),  # wqkv_buf
        pltpu.VMEM((2, d, d), wdt),  # wo_buf
        pltpu.VMEM((2, d, 2 * ffn), wdt),  # wg_buf
        pltpu.VMEM((2, ffn, d), wdt),  # wd_buf
        pltpu.VMEM((n_pairs, 2 * n_tiles - 1, 2 * tile, tile),
                   f32),  # bias_scr
        pltpu.VMEM((2 * tile, tile), f32),  # mask_scr
        pltpu.VMEM((n_pairs, 2 * tile, v1.NUM_LANES), f32),  # m_scr
        pltpu.VMEM((n_pairs, 2 * tile, v1.NUM_LANES), f32),  # l_scr
        pltpu.VMEM((n_pairs, 2 * tile, v1.NUM_LANES), f32),  # acc_scr
        pltpu.VMEM((tile, d), cdt),  # ctx_scr
        pltpu.VMEM((tile, d), f32),  # h1_scr
        pltpu.VMEM((tile, d), cdt),  # h1c_scr
        pltpu.VMEM((tile, d), f32),  # y_scr
        pltpu.SemaphoreType.DMA((10, )),
    ]


def estimate_vmem_bytes(n_tiles: int, tile: int, *, hidden_size: int,
                        num_heads: int, intermediate_size: int,
                        num_layers: int, precision: str) -> int:
    """VMEM held by the v2 kernel's buffers (excludes Mosaic scratch)."""
    cdt = wdt = v1.mxu_dtype_for(precision)
    scratch = _scratch_shapes(n_tiles, tile, hidden_size, num_heads,
                              intermediate_size, cdt, wdt)
    # Skip the DMA semaphores, which do not live in VMEM.
    total = sum(
        v1._nbytes(s.shape, s.dtype) for s in scratch
        if s.memory_space == pltpu.VMEM)
    # Inputs Pallas copies into VMEM: segment ids and the vector table.
    total += v1._nbytes((n_tiles, tile, v1.NUM_LANES), jnp.int32)  # seg_q
    total += v1._nbytes((n_tiles, 8, tile), jnp.int32)  # seg_k
    total += v1._nbytes((num_layers, v1.NUM_VEC_ROWS, hidden_size),
                        jnp.float32)  # vecs
    return total


@functools.partial(jax.jit,
                   static_argnames=("alibi_slopes", "eps", "precision", "tile",
                                    "interpret"))
def encoder_megakernel_v2(x, seq_lens, weights: v1.JinaBertPackedWeights, *,
                          alibi_slopes: tuple[float, ...], eps: float,
                          precision: str, tile: int, interpret: Any):
    """jit'ed v2 implementation behind `kernel.jina_bert_encoder_megakernel`
    (which validates the inputs)."""
    num_tokens, d = x.shape
    num_layers = weights.wqkv.shape[0]
    ffn = weights.wd.shape[1]
    num_heads = len(alibi_slopes)
    padded = -(-num_tokens // tile) * tile
    n_tiles = padded // tile
    cdt = wdt = v1.mxu_dtype_for(precision)

    meta, seg_q, seg_k = v1.build_tile_metadata(seq_lens, num_tokens, padded,
                                                tile)
    # Append the segment of each tile's first and last row: a tile pair
    # needs no mask iff both tiles lie in one and the same segment.
    meta = jnp.concatenate([meta, seg_q[:, 0, 0], seg_q[:, tile - 1, 0]])
    x_tiles = jnp.pad(x, ((0, padded - num_tokens), (0, 0)))
    x_tiles = x_tiles.reshape(n_tiles, tile, d)

    needed = estimate_vmem_bytes(n_tiles,
                                 tile,
                                 hidden_size=d,
                                 num_heads=num_heads,
                                 intermediate_size=ffn,
                                 num_layers=num_layers,
                                 precision=precision)
    vmem_limit = v1.vmem_limit_bytes(needed)
    if needed + 4 * v1._MIB > vmem_limit:
        raise ValueError(
            f"The JinaBert encoder megakernel (v2) needs ~"
            f"{needed / v1._MIB:.1f} MiB of VMEM for {num_tokens} tokens "
            f"(precision={precision!r}), more than the "
            f"{vmem_limit / v1._MIB:.0f} MiB available. Set "
            "JINA_BERT_MEGAKERNEL_VERSION=v1, or USE_JINA_BERT_MEGAKERNEL=0 "
            "to use the XLA path.")

    w_bytes = sum(
        v1._nbytes(w.shape, w.dtype)
        for w in (weights.wqkv, weights.wo, weights.wg, weights.wd))
    flops = 2 * num_layers * padded * (d * 3 * d + d * d + d * 2 * ffn +
                                       ffn * d + 2 * padded * d)
    cost = pl.CostEstimate(
        flops=int(flops),
        transcendentals=int(num_layers * padded * (num_heads * padded + ffn)),
        bytes_accessed=int(w_bytes + 2 * v1._nbytes((padded, d), jnp.float32)))

    kernel = functools.partial(_megakernel_v2,
                               num_layers=num_layers,
                               alibi_slopes=alibi_slopes,
                               eps=eps,
                               highest=precision == "highest")
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_tiles, tile, d), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),  # meta
            vmem,  # seg_q
            vmem,  # seg_k
            vmem,  # vecs
            hbm,  # x
            hbm,  # wqkv
            hbm,  # wo
            hbm,  # wg
            hbm,  # wd
        ],
        out_specs=hbm,
        scratch_shapes=_scratch_shapes(n_tiles, tile, d, num_heads, ffn, cdt,
                                       wdt),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
        cost_estimate=cost,
        interpret=interpret,
        name="jina_bert_encoder_megakernel_v2",
    )(meta, seg_q, seg_k, weights.vecs, x_tiles, weights.wqkv, weights.wo,
      weights.wg, weights.wd)
    return out.reshape(padded, d)[:num_tokens]
