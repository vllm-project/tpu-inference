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
"""JinaBert v2 encoder megakernel: the whole encoder stack in one Pallas call.

Megakernel in the sense of https://inferact.ai/blog/tpu-megakernels (design
inspiration only; no code is taken from github.com/Inferact/tpu-megakernels):
a single grid-less TPU program runs every encoder layer back to back while
the residual stream stays resident in VMEM, and the next layer's weights are
streamed HBM->VMEM with async DMAs (double-buffered) while the current layer
computes.

Per layer (post-LN BERT block with symmetric ALiBi and a GeGLU MLP)::

    q, k, v = x Wq + bq, x Wk + bk, x Wv + bv
    ctx     = softmax(q k^T / sqrt(64) - slope_h * |i - j|, same-sequence) v
    h1      = LayerNorm(ctx Wo + bo + x)
    x       = LayerNorm(gelu(h1 Wg[:, :F]) * (h1 Wg[:, F:]) Wd + bd + h1)

Two passes over token tiles per layer:

* Pass 1 (QKV): for each token tile, one K=D matmul per projection. Q is
  written per head, masked to that head's 64 lanes inside its 128-lane head
  pair, so that attention can contract over an aligned 128-lane slice (the
  zero lanes are free on the MXU) without unaligned lane slicing.
* Pass 2 (per query tile): flash-style online softmax over only the KV tiles
  that can hold tokens of the same sequences (per-tile [kv_lo, kv_hi) bounds
  in SMEM), ALiBi and the segment mask computed in-kernel (no O(T^2) bias in
  HBM), then out-projection + residual + LayerNorm, then the GeGLU MLP in
  512-wide chunks + residual + LayerNorm, written back in place into the
  VMEM-resident residual stream.

Matmul precision (explicit, see `PRECISIONS`):

* ``"default"`` (default): MXU operands are rounded to bfloat16, one MXU
  pass, float32 accumulation. This is what `lax.Precision.DEFAULT` does to
  float32 operands on TPU, i.e. the same matmul precision the XLA path uses
  for its projections. Weights are stored in bf16 for the kernel (a derived
  copy; model params stay float32).
* ``"highest"``: float32 operands with `lax.Precision.HIGHEST`
  (``#tpu.contract_precision<fp32>``), i.e. full-fp32 matmuls.

In both modes parameters, activations, the residual stream, biases,
LayerNorm, softmax, GELU and all accumulations are float32.
"""

import functools
import math
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Hard upper bound on tokens per step (per the serving config:
# --max-num-batched-tokens 2048). Longer inputs raise instead of degrading.
MAX_TOKENS = 2048

PRECISIONS = ("default", "highest")

NUM_LANES = 128
HEAD_DIM = 64  # Heads are processed in 128-lane pairs; only 64 supported.
MLP_CHUNK = 512  # Intermediate columns processed per MLP step.

# Rows of the per-layer vector table `JinaBertPackedWeights.vecs`.
ROW_BQ, ROW_BK, ROW_BV, ROW_BO, ROW_LN1_G, ROW_LN1_B, ROW_BD, ROW_LN2_G, \
    ROW_LN2_B = range(9)
NUM_VEC_ROWS = 16  # 9 used rows, padded to a multiple of 8 sublanes.

_MIB = 1024 * 1024
# Finite initial running max: avoids inf - inf in the first rescale.
_M_INIT = -1e30
# "-distance" used for pairs in different sequences. After scaling by the
# smallest slope (2^-8) this is still ~-4e6, far below any real logit, so
# exp() underflows to exactly 0.
_MASKED_NEG_DIST = -(2**30)


class JinaBertPackedWeights(NamedTuple):
    """Encoder weights in the megakernel layout, stacked over layers.

    Built once after checkpoint loading (see `pack_jina_bert_weights`).
    """
    wqkv: jax.Array  # [L, D, 3D]; Q columns pre-scaled by 1/sqrt(head_dim).
    wo: jax.Array  # [L, D, D] attention output projection.
    wg: jax.Array  # [L, D, 2F] gated_layers (gate cols first, then up).
    wd: jax.Array  # [L, F, D] MLP output projection (`wo` in the checkpoint).
    vecs: jax.Array  # [L, 16, D] float32 biases / LayerNorm params (ROW_*).


def mxu_dtype_for(precision: str) -> jnp.dtype:
    """dtype of the matmul operands (and packed weights) for `precision`."""
    if precision == "default":
        return jnp.dtype(jnp.bfloat16)
    if precision == "highest":
        return jnp.dtype(jnp.float32)
    raise ValueError(
        f"Unknown megakernel precision {precision!r}; expected one of "
        f"{PRECISIONS}.")


def pack_jina_bert_weights(layers: Sequence[Mapping[str, jax.Array]],
                           *,
                           precision: str = "default",
                           head_dim: int = HEAD_DIM) -> JinaBertPackedWeights:
    """Re-lays out per-layer JinaBert weights for the megakernel.

    Args:
      layers: one mapping per encoder layer with the model's (post-load)
        weights, in the layouts used by `jina_bert.py`:
        ``q_w/k_w/v_w`` [D, N, H] (or [D, D]), ``q_b/k_b/v_b`` [N, H] (or
        [D]), ``o_w`` [N, H, D] (or [D, D]), ``o_b`` [D], ``ln1_g/ln1_b``
        [D], ``gate_w`` [D, 2F], ``down_w`` [F, D], ``down_b`` [D],
        ``ln2_g/ln2_b`` [D].
      precision: one of `PRECISIONS`; selects the matmul weight dtype.
      head_dim: attention head size (the softmax scale is folded into Q).

    Returns:
      `JinaBertPackedWeights`. Matmul weights are cast to `mxu_dtype_for(
      precision)`; vectors stay float32. Folding 1/sqrt(64) = 2^-3 into Q is
      exact in floating point.
    """
    wdt = mxu_dtype_for(precision)
    sm_scale = head_dim**-0.5
    f32 = jnp.float32
    wqkv, wo, wg, wd, vecs = [], [], [], [], []
    for p in layers:
        d = p["q_w"].shape[0]

        def mat(w, d=d):
            return jnp.reshape(w, (d, d)).astype(f32)

        def vec(b, d=d):
            return jnp.reshape(b, (d, )).astype(f32)

        qkv = [mat(p["q_w"]) * sm_scale, mat(p["k_w"]), mat(p["v_w"])]
        wqkv.append(jnp.concatenate(qkv, axis=1).astype(wdt))
        wo.append(mat(p["o_w"]).astype(wdt))
        wg.append(p["gate_w"].astype(wdt))
        wd.append(p["down_w"].astype(wdt))
        rows = [
            vec(p["q_b"]) * sm_scale,
            vec(p["k_b"]),
            vec(p["v_b"]),
            vec(p["o_b"]),
            vec(p["ln1_g"]),
            vec(p["ln1_b"]),
            vec(p["down_b"]),
            vec(p["ln2_g"]),
            vec(p["ln2_b"]),
        ]
        rows += [jnp.zeros((d, ), f32)] * (NUM_VEC_ROWS - len(rows))
        vecs.append(jnp.stack(rows))
    return JinaBertPackedWeights(wqkv=jnp.stack(wqkv),
                                 wo=jnp.stack(wo),
                                 wg=jnp.stack(wg),
                                 wd=jnp.stack(wd),
                                 vecs=jnp.stack(vecs))


def unsupported_geometry_reason(*, hidden_size: int, num_heads: int,
                                intermediate_size: int) -> str | None:
    """Why the megakernel can't run this encoder geometry (None if it can)."""
    if num_heads % 2 or num_heads * HEAD_DIM != hidden_size:
        return (f"it needs an even number of {HEAD_DIM}-wide attention heads, "
                f"got hidden_size={hidden_size} with num_heads={num_heads}")
    if intermediate_size % MLP_CHUNK:
        return (f"it needs intermediate_size % {MLP_CHUNK} == 0, got "
                f"{intermediate_size}")
    return None


def default_tile_size(num_tokens: int) -> int:
    """Token-tile size (rows per MXU step and attention block)."""
    return NUM_LANES if num_tokens <= NUM_LANES else 2 * NUM_LANES


def build_tile_metadata(seq_lens: jax.Array, num_tokens: int,
                        padded_tokens: int, tile: int):
    """Per-step metadata derived from `seq_lens` (runs in XLA, not in-kernel).

    Mirrors `encoder_only_flash_attention`'s segment ids: request r owns
    tokens [cu[r], cu[r+1]) with cu = [0, cumsum(seq_lens)]; tokens at or
    after sum(seq_lens) (including the kernel's own padding up to
    `padded_tokens`) form one extra "padding" segment. Tokens attend only
    within their own segment.

    Returns:
      meta: int32 [1 + 2 * n_tiles] (SMEM): meta[0] = number of active tiles
        (tiles holding at least one real token); meta[1 + i] / meta[1 +
        n_tiles + i] = first / one-past-last KV tile query tile i must visit.
      seg_q: int32 [n_tiles, tile, 128], segment id per row, lane-replicated.
      seg_k: int32 [n_tiles, 8, tile], segment id per column,
        sublane-replicated.
    """
    n_tiles = padded_tokens // tile
    num_seqs = seq_lens.shape[0]
    lens = jnp.maximum(seq_lens.astype(jnp.int32), 0)
    cu = jnp.concatenate([jnp.zeros((1, ), jnp.int32), jnp.cumsum(lens)])
    cu = jnp.minimum(cu, num_tokens)  # Never let a request cover padding.
    total = cu[num_seqs]
    n_active = (total + tile - 1) // tile
    tok = jnp.arange(padded_tokens, dtype=jnp.int32)
    seg = jnp.sum(tok[:, None] >= cu[None, 1:], axis=1, dtype=jnp.int32)
    seg_start = cu[seg]
    # The padding segment is truncated at the last active tile: inactive
    # tiles are never computed, so nothing may attend to them.
    seg_end = jnp.where(seg < num_seqs, cu[jnp.minimum(seg + 1, num_seqs)],
                        n_active * tile)
    kv_lo = seg_start[0::tile] // tile
    kv_hi = jnp.minimum((seg_end[tile - 1::tile] + tile - 1) // tile, n_tiles)
    meta = jnp.concatenate([n_active[None], kv_lo, kv_hi]).astype(jnp.int32)
    seg_q = jnp.broadcast_to(seg[:, None], (padded_tokens, NUM_LANES))
    seg_q = seg_q.reshape(n_tiles, tile, NUM_LANES)
    seg_k = jnp.broadcast_to(seg[None, :], (8, padded_tokens))
    seg_k = seg_k.reshape(8, n_tiles, tile).transpose(1, 0, 2)
    return meta, seg_q, seg_k


def gelu_erf(x: jax.Array) -> jax.Array:
    """Exact (erf) GELU, 0.5 x (1 + erf(x / sqrt(2))), in float32.

    erf uses Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7), which only needs
    exp and a division, both supported by Mosaic.
    """
    z = jnp.abs(x) * (1.0 / math.sqrt(2.0))
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t *
                                   (1.421413741 + t *
                                    (-1.453152027 + t * 1.061405429))))
    erfc = poly * jnp.exp(-z * z)  # erfc(|x| / sqrt(2))
    return 0.5 * x * jnp.where(x >= 0, 2.0 - erfc, erfc)


def _layer_norm(x, gamma, beta, eps):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    xc = x - mean
    var = jnp.mean(xc * xc, axis=-1, keepdims=True)
    return xc * lax.rsqrt(var + eps) * gamma + beta


def _megakernel(
        meta_ref,  # SMEM int32 [1 + 2 * n_tiles]
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
        q_scr,  # cdt [n_tiles, H, tile, 128]: per-head masked Q.
        k_scr,  # cdt [n_tiles, tile, D]
        v_scr,  # cdt [n_tiles, tile, D]
        wqkv_buf,  # wdt [2, D, 3D]  (double-buffered over layers)
        wo_buf,  # wdt [2, D, D]
        wg_buf,  # wdt [2, D, 2F]
        wd_buf,  # wdt [2, F, D]
        rc_scr,  # i32 [tile, tile]: row - col.
        nd_scr,  # f32 [tile, tile]: -|i - j| or masked.
        m_scr,  # f32 [H, tile, 128]: running max (lane-replicated).
        l_scr,  # f32 [H, tile, 128]: running denominator.
        acc_scr,  # f32 [tile, D]: attention accumulator / context.
        h1_scr,  # f32 [tile, D]: post-attention LayerNorm output.
        h1c_scr,  # cdt [tile, D]: same, as MXU operand.
        y_scr,  # f32 [tile, D]: MLP accumulator.
        sems,  # DMA semaphores [10].
        *,
        num_layers: int,
        alibi_slopes: tuple[float, ...],
        eps: float,
        highest: bool):
    """Kernel body. Refs: inputs (`meta_ref` .. `wd_hbm`), the output
    (`out_hbm`), then scratch (`x_res` .. `sems`); `cdt`/`wdt` are the MXU
    operand dtype (`mxu_dtype_for(precision)`)."""
    n_tiles, tile, d = x_res.shape
    num_heads = q_scr.shape[1]
    n_pairs = num_heads // 2
    ffn = wd_buf.shape[1]
    n_chunks = ffn // MLP_CHUNK
    cdt = q_scr.dtype
    reps = tile // NUM_LANES
    f32 = jnp.float32
    precision = lax.Precision.HIGHEST if highest else None

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

    def lane_tile(x):  # [tile, 128] lane-replicated -> [tile, tile]
        return x if reps == 1 else jnp.tile(x, (1, reps))

    def even_head_lanes():  # Lanes of the first head of a 128-lane pair.
        return lax.broadcasted_iota(jnp.int32, (tile, NUM_LANES), 1) < HEAD_DIM

    def weight_copies(layer, slot):
        pairs = ((wqkv_hbm, wqkv_buf), (wo_hbm, wo_buf), (wg_hbm, wg_buf),
                 (wd_hbm, wd_buf))
        copies = []
        for i, (src, dst) in enumerate(pairs):
            sem = sems.at[2 * i + slot]  # Semaphores 0-7: 4 weights x 2 slots.
            copies.append(
                pltpu.make_async_copy(src.at[layer], dst.at[slot], sem))
        return tuple(copies)

    # Prologue: residual stream and layer-0 weights HBM -> VMEM.
    x_in = pltpu.make_async_copy(x_hbm, x_res, sems.at[8])
    x_in.start()
    for cp in weight_copies(0, 0):
        cp.start()
    rc_scr[...] = (lax.broadcasted_iota(jnp.int32, (tile, tile), 0) -
                   lax.broadcasted_iota(jnp.int32, (tile, tile), 1))
    n_active = meta_ref[0]
    x_in.wait()

    def qkv_projection(layer, slot):
        bq = vecs_ref[layer, ROW_BQ:ROW_BQ + 1, :]
        bk = vecs_ref[layer, ROW_BK:ROW_BK + 1, :]
        bv = vecs_ref[layer, ROW_BV:ROW_BV + 1, :]

        def body(t, carry):
            xt = x_res[t].astype(cdt)
            q = mm(xt, wqkv_buf[slot, :, 0:d]) + bq
            even = even_head_lanes()
            for p in range(n_pairs):
                q_pair = q[:, p * NUM_LANES:(p + 1) * NUM_LANES]
                q_scr[t, 2 * p] = jnp.where(even, q_pair, 0.0).astype(cdt)
                q_scr[t, 2 * p + 1] = jnp.where(even, 0.0, q_pair).astype(cdt)
            k_scr[t] = (mm(xt, wqkv_buf[slot, :, d:2 * d]) + bk).astype(cdt)
            v_scr[t] = (mm(xt, wqkv_buf[slot, :, 2 * d:3 * d]) +
                        bv).astype(cdt)
            return carry

        lax.fori_loop(0, n_active, body, 0)

    def attention_and_mlp(layer, slot):
        bo = vecs_ref[layer, ROW_BO:ROW_BO + 1, :]
        ln1_g = vecs_ref[layer, ROW_LN1_G:ROW_LN1_G + 1, :]
        ln1_b = vecs_ref[layer, ROW_LN1_B:ROW_LN1_B + 1, :]
        bd = vecs_ref[layer, ROW_BD:ROW_BD + 1, :]
        ln2_g = vecs_ref[layer, ROW_LN2_G:ROW_LN2_G + 1, :]
        ln2_b = vecs_ref[layer, ROW_LN2_B:ROW_LN2_B + 1, :]

        def q_tile(qi, carry):
            m_scr[...] = jnp.full(m_scr.shape, _M_INIT, f32)
            l_scr[...] = jnp.zeros(l_scr.shape, f32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, f32)

            def kv_step(kj, carry):
                # ALiBi distance and segment mask for this (q, kv) tile pair,
                # shared by all heads.
                same_seq = lane_tile(seg_q_ref[qi]) == seg_k_ref[kj, 0:1, :]
                dist = rc_scr[...] + (qi - kj) * tile  # i - j
                neg_dist = jnp.minimum(dist, -dist)  # -|i - j|
                nd_scr[...] = jnp.where(same_seq, neg_dist,
                                        _MASKED_NEG_DIST).astype(f32)
                even = even_head_lanes()
                for p in range(n_pairs):
                    lanes = slice(p * NUM_LANES, (p + 1) * NUM_LANES)
                    k_pair = k_scr[kj, :, lanes]
                    v_pair = v_scr[kj, :, lanes]
                    alphas, pvs = [], []
                    for h in (2 * p, 2 * p + 1):
                        # Q is pre-scaled by 1/sqrt(64) and zero outside
                        # head h's lanes, so this is head h's q.k only.
                        s = mm_nt(q_scr[qi, h], k_pair)
                        s = s + alibi_slopes[h] * nd_scr[...]
                        m_prev = m_scr[h]
                        m_new = jnp.maximum(m_prev,
                                            jnp.max(s, axis=1, keepdims=True))
                        probs = jnp.exp(s - lane_tile(m_new))
                        alpha = jnp.exp(m_prev - m_new)
                        l_scr[h] = alpha * l_scr[h] + jnp.sum(
                            probs, axis=1, keepdims=True)
                        m_scr[h] = m_new
                        alphas.append(alpha)
                        pvs.append(mm(probs.astype(cdt), v_pair))
                    # Lanes [0, 64) belong to head 2p, [64, 128) to 2p + 1.
                    scale = jnp.where(even, alphas[0], alphas[1])
                    pv = jnp.where(even, pvs[0], pvs[1])
                    acc_scr[:, lanes] = scale * acc_scr[:, lanes] + pv
                return carry

            lax.fori_loop(meta_ref[1 + qi], meta_ref[1 + n_tiles + qi],
                          kv_step, 0)

            even = even_head_lanes()
            for p in range(n_pairs):
                lanes = slice(p * NUM_LANES, (p + 1) * NUM_LANES)
                denom = jnp.where(even, l_scr[2 * p], l_scr[2 * p + 1])
                acc_scr[:, lanes] = acc_scr[:, lanes] / denom

            attn = mm(acc_scr[...].astype(cdt), wo_buf[slot]) + bo
            h1 = _layer_norm(attn + x_res[qi], ln1_g, ln1_b, eps)
            h1_scr[...] = h1
            h1c_scr[...] = h1.astype(cdt)
            for c in range(n_chunks):
                gate_cols = slice(c * MLP_CHUNK, (c + 1) * MLP_CHUNK)
                up_cols = slice(ffn + c * MLP_CHUNK, ffn + (c + 1) * MLP_CHUNK)
                gate = mm(h1c_scr[...], wg_buf[slot, :, gate_cols])
                up = mm(h1c_scr[...], wg_buf[slot, :, up_cols])
                act = (gelu_erf(gate) * up).astype(cdt)
                part = mm(act, wd_buf[slot, gate_cols, :])
                if c == 0:
                    y_scr[...] = part
                else:
                    y_scr[...] += part
            x_res[qi] = _layer_norm(y_scr[...] + bd + h1_scr[...], ln2_g,
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


def _nbytes(shape, dtype) -> int:
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def _vmem_capacity_bytes() -> int:
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # Not on TPU (e.g. interpret mode on CPU).
        return 128 * _MIB


def _scratch_shapes(n_tiles, tile, d, num_heads, ffn, cdt, wdt):
    return [
        pltpu.VMEM((n_tiles, tile, d), jnp.float32),  # x_res
        pltpu.VMEM((n_tiles, num_heads, tile, NUM_LANES), cdt),  # q_scr
        pltpu.VMEM((n_tiles, tile, d), cdt),  # k_scr
        pltpu.VMEM((n_tiles, tile, d), cdt),  # v_scr
        pltpu.VMEM((2, d, 3 * d), wdt),  # wqkv_buf
        pltpu.VMEM((2, d, d), wdt),  # wo_buf
        pltpu.VMEM((2, d, 2 * ffn), wdt),  # wg_buf
        pltpu.VMEM((2, ffn, d), wdt),  # wd_buf
        pltpu.VMEM((tile, tile), jnp.int32),  # rc_scr
        pltpu.VMEM((tile, tile), jnp.float32),  # nd_scr
        pltpu.VMEM((num_heads, tile, NUM_LANES), jnp.float32),  # m_scr
        pltpu.VMEM((num_heads, tile, NUM_LANES), jnp.float32),  # l_scr
        pltpu.VMEM((tile, d), jnp.float32),  # acc_scr
        pltpu.VMEM((tile, d), jnp.float32),  # h1_scr
        pltpu.VMEM((tile, d), cdt),  # h1c_scr
        pltpu.VMEM((tile, d), jnp.float32),  # y_scr
        pltpu.SemaphoreType.DMA((10, )),
    ]


def estimate_vmem_bytes(num_tokens: int,
                        *,
                        hidden_size: int = 512,
                        num_heads: int = 8,
                        intermediate_size: int = 2048,
                        num_layers: int = 4,
                        precision: str = "default",
                        tile: int | None = None) -> int:
    """VMEM held by the kernel's buffers (excludes Mosaic-internal scratch)."""
    tile = tile or default_tile_size(num_tokens)
    padded = -(-num_tokens // tile) * tile
    n_tiles = padded // tile
    cdt = wdt = mxu_dtype_for(precision)
    scratch = _scratch_shapes(n_tiles, tile, hidden_size, num_heads,
                              intermediate_size, cdt, wdt)
    # Skip the DMA semaphores, which do not live in VMEM.
    total = sum(
        _nbytes(s.shape, s.dtype) for s in scratch
        if s.memory_space == pltpu.VMEM)
    # Inputs Pallas copies into VMEM: segment ids and the vector table.
    total += _nbytes((n_tiles, tile, NUM_LANES), jnp.int32)
    total += _nbytes((n_tiles, 8, tile), jnp.int32)
    total += _nbytes((num_layers, NUM_VEC_ROWS, hidden_size), jnp.float32)
    return total


def vmem_limit_bytes(vmem_needed: int) -> int:
    """Scoped-VMEM limit requested from Mosaic for `vmem_needed` of buffers.

    Leaves room for Mosaic-internal scratch, capped at 85% of the chip's VMEM.
    """
    return min(vmem_needed + 32 * _MIB, int(_vmem_capacity_bytes() * 0.85))


def unsupported_vmem_reason(num_tokens: int,
                            *,
                            hidden_size: int = 512,
                            num_heads: int = 8,
                            intermediate_size: int = 2048,
                            num_layers: int = 4,
                            precision: str = "default",
                            tile: int | None = None) -> str | None:
    """Why the kernel's buffers don't fit this chip's VMEM (None if they do)."""
    needed = estimate_vmem_bytes(num_tokens,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 intermediate_size=intermediate_size,
                                 num_layers=num_layers,
                                 precision=precision,
                                 tile=tile)
    limit = vmem_limit_bytes(needed)
    if needed + 4 * _MIB > limit:
        return (f"it needs ~{needed / _MIB:.1f} MiB of VMEM for {num_tokens} "
                f"tokens (precision={precision!r}), more than the "
                f"{limit / _MIB:.0f} MiB available on this chip")
    return None


def _check_inputs(x, seq_lens, weights: JinaBertPackedWeights, alibi_slopes,
                  precision, tile):
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown megakernel precision {precision!r}; "
                         f"expected one of {PRECISIONS}.")
    if x.ndim != 2:
        raise ValueError(f"x must be [num_tokens, hidden], got {x.shape}.")
    num_tokens, d = x.shape
    if num_tokens > MAX_TOKENS:
        raise ValueError(
            f"The JinaBert encoder megakernel supports at most {MAX_TOKENS} "
            f"tokens per step, got {num_tokens}. Serve with "
            f"--max-num-batched-tokens {MAX_TOKENS} (and --max-model-len <= "
            f"{MAX_TOKENS}), or set USE_JINA_BERT_MEGAKERNEL=0 to use the "
            "XLA path.")
    if num_tokens < 1:
        raise ValueError("x must contain at least one token.")
    if x.dtype != jnp.float32:
        raise TypeError(
            f"The megakernel expects float32 activations, got {x.dtype}.")
    if seq_lens.ndim != 1 or not jnp.issubdtype(seq_lens.dtype, jnp.integer):
        raise ValueError(
            f"seq_lens must be a 1-D integer array, got {seq_lens.dtype}"
            f"{list(seq_lens.shape)}.")
    num_heads = len(alibi_slopes)
    num_layers = weights.wqkv.shape[0]
    ffn = weights.wd.shape[1]
    reason = unsupported_geometry_reason(hidden_size=d,
                                         num_heads=num_heads,
                                         intermediate_size=ffn)
    if reason is not None:
        raise NotImplementedError(
            f"Unsupported geometry for the JinaBert megakernel: {reason}.")
    expected = JinaBertPackedWeights(wqkv=(num_layers, d, 3 * d),
                                     wo=(num_layers, d, d),
                                     wg=(num_layers, d, 2 * ffn),
                                     wd=(num_layers, ffn, d),
                                     vecs=(num_layers, NUM_VEC_ROWS, d))
    for name, want in expected._asdict().items():
        got = getattr(weights, name).shape
        if tuple(got) != want:
            raise ValueError(
                f"Packed weight {name!r} has shape {got}, expected {want}.")
    wdt = mxu_dtype_for(precision)
    for name in ("wqkv", "wo", "wg", "wd"):
        if getattr(weights, name).dtype != wdt:
            raise TypeError(
                f"Packed weight {name!r} is {getattr(weights, name).dtype} "
                f"but precision={precision!r} needs {wdt}; re-pack with "
                "pack_jina_bert_weights(..., precision=...).")
    if weights.vecs.dtype != jnp.float32:
        raise TypeError("Packed vectors must be float32.")
    if tile is not None and (tile <= 0 or tile % NUM_LANES):
        raise ValueError(f"tile={tile} must be a positive multiple of "
                         f"{NUM_LANES}.")


@functools.partial(jax.jit,
                   static_argnames=("alibi_slopes", "eps", "precision", "tile",
                                    "interpret"))
def _encoder_megakernel(x, seq_lens, weights: JinaBertPackedWeights, *,
                        alibi_slopes: tuple[float, ...], eps: float,
                        precision: str, tile: int | None, interpret: Any):
    num_tokens, d = x.shape
    num_layers = weights.wqkv.shape[0]
    ffn = weights.wd.shape[1]
    num_heads = len(alibi_slopes)
    tile = tile or default_tile_size(num_tokens)
    padded = -(-num_tokens // tile) * tile
    n_tiles = padded // tile
    cdt = wdt = mxu_dtype_for(precision)

    meta, seg_q, seg_k = build_tile_metadata(seq_lens, num_tokens, padded,
                                             tile)
    x_tiles = jnp.pad(x, ((0, padded - num_tokens), (0, 0)))
    x_tiles = x_tiles.reshape(n_tiles, tile, d)

    geometry = dict(hidden_size=d,
                    num_heads=num_heads,
                    intermediate_size=ffn,
                    num_layers=num_layers,
                    precision=precision,
                    tile=tile)
    reason = unsupported_vmem_reason(num_tokens, **geometry)
    if reason is not None:
        raise ValueError(
            f"The JinaBert encoder megakernel can't run: {reason}. "
            "Set USE_JINA_BERT_MEGAKERNEL=0 to use the XLA path.")
    vmem_limit = vmem_limit_bytes(estimate_vmem_bytes(num_tokens, **geometry))

    w_bytes = sum(
        _nbytes(w.shape, w.dtype)
        for w in (weights.wqkv, weights.wo, weights.wg, weights.wd))
    flops = 2 * num_layers * padded * (d * 3 * d + d * d + d * 2 * ffn +
                                       ffn * d + 2 * padded * d)
    cost = pl.CostEstimate(
        flops=int(flops),
        transcendentals=int(num_layers * padded * (num_heads * padded + ffn)),
        bytes_accessed=int(w_bytes + 2 * _nbytes((padded, d), jnp.float32)))

    kernel = functools.partial(_megakernel,
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
        name="jina_bert_encoder_megakernel",
    )(meta, seg_q, seg_k, weights.vecs, x_tiles, weights.wqkv, weights.wo,
      weights.wg, weights.wd)
    return out.reshape(padded, d)[:num_tokens]


def jina_bert_encoder_megakernel(x: jax.Array,
                                 seq_lens: jax.Array,
                                 weights: JinaBertPackedWeights,
                                 *,
                                 alibi_slopes: Sequence[float],
                                 eps: float = 1e-12,
                                 precision: str = "default",
                                 mesh: Mesh | None = None,
                                 tile: int | None = None,
                                 interpret: Any = False) -> jax.Array:
    """Runs all JinaBert encoder layers in one Pallas TPU kernel.

    Drop-in for `JinaBertEncoder.__call__`'s XLA path.

    Args:
      x: float32 [T, D] encoder input (output of `JinaBertEmbeddings`),
        T <= `MAX_TOKENS`.
      seq_lens: int [max_num_seqs] per-request token counts; requests are
        packed back to back from token 0, tokens past sum(seq_lens) are
        padding, zero entries are unused slots.
      weights: output of `pack_jina_bert_weights` for the same `precision`.
      alibi_slopes: per-head ALiBi slopes (`get_alibi_slopes(num_heads)`).
      eps: LayerNorm epsilon.
      precision: matmul precision, one of `PRECISIONS` (see module doc).
      mesh: if given, the kernel runs under `shard_map` on this mesh, which
        must have exactly one device (single-chip, TP=1).
      tile: token-tile size override (multiple of 128); None = heuristic.
      interpret: forwarded to `pallas_call` (e.g. `pltpu.InterpretParams()`
        to run on CPU).

    Returns:
      float32 [T, D] final hidden states. Rows at padding positions are
      unspecified.

    Raises:
      ValueError: if T > `MAX_TOKENS` (at trace time), or on malformed inputs.
    """
    alibi_slopes = tuple(float(s) for s in alibi_slopes)
    _check_inputs(x, seq_lens, weights, alibi_slopes, precision, tile)
    if interpret is True:
        interpret = pltpu.InterpretParams()
    impl = functools.partial(_encoder_megakernel,
                             alibi_slopes=alibi_slopes,
                             eps=float(eps),
                             precision=precision,
                             tile=tile,
                             interpret=interpret)
    if mesh is None:
        return impl(x, seq_lens, weights)
    if mesh.size != 1:
        raise NotImplementedError(
            "The JinaBert encoder megakernel runs on a single TPU chip "
            f"(TP=1); got a mesh with {mesh.size} devices. Set "
            "USE_JINA_BERT_MEGAKERNEL=0 to use the XLA path.")
    return jax.shard_map(impl,
                         mesh=mesh,
                         in_specs=(P(), P(), P()),
                         out_specs=P(),
                         check_vma=False)(x, seq_lens, weights)
