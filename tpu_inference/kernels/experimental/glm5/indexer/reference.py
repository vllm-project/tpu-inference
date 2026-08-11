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
"""Pure-JAX reference for GLM-5.2's Dynamic Sparse Attention "lightning
indexer" (DeepSeek-V3.2-style, upstream vLLM ``Indexer``/``sparse_attn_indexer``).

This module implements the *math* of the indexer -- interleaved (GPT-J-style)
RoPE on a leading slice of each head, FP8 (``ue8m0``) quantization, the
``sum_h weight[t, h] * relu(q[t, h] . k[s])`` scoring formula, causal top-k
selection, and the shared-indexer layer-skip pattern -- starting from
already-projected (``wq_b`` / fused ``wk_weights_proj`` + ``k_norm``) Q/K/raw
weight tensors. The linear projections themselves, and the raw per-token FP8
K-cache write path, are model-wiring / Pallas-kernel concerns handled in later
phases (see ``tpu_inference/kernels/experimental/glm5/indexer`` Pallas kernels,
not yet implemented).

Reference sources (see the GLM-5.2 DSA bring-up plan for exact line numbers):
  - ``vllm/model_executor/models/deepseek_v2.py::Indexer.forward`` for the
    RoPE-split / FP8-quant / weight-folding pipeline.
  - ``vllm/model_executor/models/deepseek_v2.py`` (the ``_skip_topk`` block
    building each layer's indexer) for the layer-skip formula.
  - ``tpu_inference/kernels/experimental/deepseek_v4/indexer/streamindex_topk.py``
    (and its test's ``streamindex_topk_ref``) for the
    ``relu(q.k) * weight`` scoring/top-k math, degraded here to
    ``compression_ratio=1`` (raw, uncompressed KV) with GLM-5.2's real
    ``index_n_heads=32`` / ``index_head_dim=128`` / interleaved-RoPE config.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# GLM-5.2 config (from the real ``nvidia/GLM-5.2-NVFP4`` checkpoint, not
# vanilla DeepSeek-V3.2 defaults -- see the bring-up plan/state.md).
INDEX_TOPK = 2048
INDEX_TOPK_FREQ = 4
INDEX_SKIP_TOPK_OFFSET = 3
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEXER_ROPE_DIM = 64  # config.qk_rope_head_dim, the roped slice of head_dim.
INDEXER_QUANT_BLOCK_SIZE = 128  # == INDEX_HEAD_DIM: one FP8 scale per token.

_FP8_DTYPE = jnp.float8_e4m3fn


def quantize_fp8_ue8m0(
    x: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Per-``block_size``-group FP8 (``e4m3fn``) quantization with a
    power-of-two (``ue8m0``-style) block scale.

    Args:
      x: ``[..., D]``, ``D % block_size == 0``.
      block_size: group width along the trailing axis.

    Returns:
      ``(q, scale)``: ``q`` is ``x`` quantized to ``float8_e4m3fn``, same
      shape as ``x``. ``scale`` is ``float32`` with shape
      ``[..., D // block_size]`` such that
      ``q.astype(float32) * jnp.repeat(scale, block_size, axis=-1) ~= x``.
      The scale is rounded to a power of two (matching the ``ue8m0`` scale
      format used by the production FP8 indexer cache), kept as ``float32``
      here purely for host-side readability -- numerically identical to the
      bit-packed ``float8_e8m0fnu`` format the Pallas kernel will use.
    """
    fp8_max = float(jnp.finfo(_FP8_DTYPE).max)
    *lead, dim = x.shape
    if dim % block_size != 0:
        raise ValueError(f"dim ({dim}) must be a multiple of block_size "
                         f"({block_size})")
    blocked = x.reshape(*lead, dim // block_size,
                        block_size).astype(jnp.float32)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(_FP8_DTYPE).reshape(x.shape)
    scale = jnp.squeeze(scale, -1)
    return q, scale


def dequantize_fp8_ue8m0(q: jax.Array, scale: jax.Array,
                         block_size: int) -> jax.Array:
    """Inverse of ``quantize_fp8_ue8m0``: ``q * repeat(scale, block_size)``.

    Args:
      q: ``[..., D]`` (any dtype, typically ``float8_e4m3fn``).
      scale: ``[..., D // block_size]``, as returned by
        ``quantize_fp8_ue8m0``.
      block_size: must match the ``block_size`` used to produce ``scale``.

    Returns:
      ``[..., D]`` float32.
    """
    *lead, dim = q.shape
    num_groups = dim // block_size
    q_blocked = q.astype(jnp.float32).reshape(*lead, num_groups, block_size)
    return (q_blocked * scale[..., None]).reshape(q.shape)


def compute_rope_cos_sin(
    positions: jax.Array,  # [T] int32
    rotary_dim: int,
    *,
    base: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Standard RoPE frequency table.

    Returns ``(cos, sin)``, each ``[T, rotary_dim // 2]`` float32.
    """
    inv_freq = 1.0 / (base**(jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) /
                             rotary_dim))
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    return jnp.cos(freqs), jnp.sin(freqs)


def _apply_interleaved_rope(x_rot: jax.Array, cos: jax.Array,
                            sin: jax.Array) -> jax.Array:
    """GPT-J-style (interleaved) rotation, matching vLLM's
    ``ApplyRotaryEmb.forward_static(..., is_neox_style=False)``:
    ``x1 = x[..., ::2]; x2 = x[..., 1::2]``,
    ``o1 = x1*cos - x2*sin; o2 = x2*cos + x1*sin``, re-interleaved.

    Args:
      x_rot: ``[..., rotary_dim]``, the slice of the head to rotate.
      cos, sin: broadcastable to ``[..., rotary_dim // 2]``.
    """
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.stack([o1, o2], axis=-1).reshape(x_rot.shape)


def apply_indexer_rope(
    x: jax.Array,  # [T, D] or [T, H, D]
    positions: jax.Array,  # [T] int32
    *,
    rope_dim: int = INDEXER_ROPE_DIM,
    base: float = 10000.0,
) -> jax.Array:
    """Applies GLM-5.2's interleaved indexer RoPE to the *leading*
    ``rope_dim`` channels of ``x`` (``indexer_rope_interleave=True``); the
    trailing ``D - rope_dim`` (NoPE) channels pass through unchanged.

    This mirrors ``Indexer.forward``'s
    ``q_pe, q_nope = split(q, [rope_dim, head_dim - rope_dim])`` /
    ``rotary_emb(positions, q_pe, ...)`` / ``cat([q_pe, q_nope])`` sequence,
    but is NeoX-vs-GPT-J-style-parameterized to GPT-J (interleaved) per
    GLM-5.2's ``indexer_rope_interleave=True`` (vs. NeoX for vanilla
    DeepSeek-V3.2).

    Args:
      x: ``[T, D]`` (single MQA K vector per token) or ``[T, H, D]`` (Q, one
        vector per indexer head).
      positions: ``[T]`` int32 absolute token positions.
      rope_dim: width of the leading roped slice (``config.qk_rope_head_dim``,
        64 for GLM-5.2's indexer).
      base: RoPE theta.

    Returns:
      Same shape/dtype as ``x``.
    """
    if x.shape[-1] < rope_dim:
        raise ValueError(f"head_dim ({x.shape[-1]}) must be >= rope_dim "
                         f"({rope_dim})")
    orig_dtype = x.dtype
    x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    cos, sin = compute_rope_cos_sin(positions, rope_dim, base=base)
    if x.ndim == 3:  # [T, H, D]: broadcast the per-token frequencies over H.
        cos, sin = cos[:, None, :], sin[:, None, :]
    x_rot = _apply_interleaved_rope(x_rot.astype(jnp.float32), cos, sin)
    return jnp.concatenate([x_rot, x_pass.astype(jnp.float32)],
                           axis=-1).astype(orig_dtype)


def lightning_indexer_scores(
    q: jax.Array,  # [T, H, Dh] pre-RoPE, post wq_b projection
    k: jax.Array,  # [S, Dh] pre-RoPE, post wk_weights_proj + k_norm, MQA
    weights: jax.Array,  # [T, H] raw per-head indexer weight (pre-scale-fold)
    q_positions: jax.Array,  # [T] int32 absolute positions
    k_positions: jax.Array,  # [S] int32 absolute positions
    *,
    rope_dim: int = INDEXER_ROPE_DIM,
    rope_base: float = 10000.0,
    quant_block_size: int = INDEXER_QUANT_BLOCK_SIZE,
    softmax_scale: float | None = None,
    n_head_scale: float | None = None,
    use_fp8_quant: bool = True,
) -> jax.Array:
    """Computes the raw-KV lightning-indexer causal score matrix.

    ``logit[t, s] = sum_h weight[t, h] * relu(q_rope[t, h] . k_rope[s])``,
    ``weight[t, h] := raw_weight[t, h] * softmax_scale * n_head_scale``
    (matches ``Indexer.forward``'s ``weights * q_scale * softmax_scale *
    n_head_scale`` after folding the FP8 quantization scales back into the
    dot product -- see the bring-up plan for the algebra showing this is
    exactly equivalent to quantizing then dequantizing Q/K before the dot
    product, since per-token/per-head FP8 scales are positive scalars that
    commute with both the dot product and ``relu``).

    Args:
      q: ``[T, num_heads, head_dim]``.
      k: ``[S, head_dim]``, single shared (MQA) K vector per raw KV token.
      weights: ``[T, num_heads]``, the indexer's raw per-head weight
        (pre softmax_scale/n_head_scale folding).
      q_positions, k_positions: absolute token positions, used both for RoPE
        and for the causal mask.
      rope_dim: width of the roped slice of each ``head_dim``-wide vector.
      quant_block_size: FP8 quantization block width (``128`` == head_dim for
        GLM-5.2, i.e. one scale per token/head).
      softmax_scale: defaults to ``head_dim ** -0.5``.
      n_head_scale: defaults to ``num_heads ** -0.5``.
      use_fp8_quant: if ``True`` (production numerics), quantize Q/K to FP8
        (``ue8m0`` block scale) before the dot product, matching the real
        indexer's ``DeepseekV32IndexerCache``/``per_token_group_quant_fp8``
        path. If ``False``, compute in full precision (useful as a
        high-precision ground truth to isolate FP8 rounding error).

    Returns:
      ``[T, S]`` float32 scores, ``-inf`` at causally-invalid (``s`` position
      after ``t``'s position) entries.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_kv, k_head_dim = k.shape
    if k_head_dim != head_dim:
        raise ValueError(f"q/k head_dim mismatch: {head_dim} vs {k_head_dim}")
    if weights.shape != (num_tokens, num_heads):
        raise ValueError(f"weights shape {weights.shape} != "
                         f"{(num_tokens, num_heads)}")

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if n_head_scale is None:
        n_head_scale = num_heads**-0.5

    q_rope = apply_indexer_rope(q,
                                q_positions,
                                rope_dim=rope_dim,
                                base=rope_base)
    k_rope = apply_indexer_rope(k,
                                k_positions,
                                rope_dim=rope_dim,
                                base=rope_base)

    if use_fp8_quant:
        q_fp8, q_scale = quantize_fp8_ue8m0(q_rope, quant_block_size)
        k_fp8, k_scale = quantize_fp8_ue8m0(k_rope, quant_block_size)
        # quant_block_size == head_dim (128) for GLM-5.2, so this is one
        # scale per (token, head) / per kv-token, but handle the general
        # multi-group case via dequantize_fp8_ue8m0 regardless.
        q_dequant = dequantize_fp8_ue8m0(q_fp8, q_scale, quant_block_size)
        k_dequant = dequantize_fp8_ue8m0(k_fp8, k_scale, quant_block_size)
    else:
        q_dequant = q_rope.astype(jnp.float32)
        k_dequant = k_rope.astype(jnp.float32)

    # [T, H, S]. This is a correctness reference (used to bound production
    # kernel error), so force full float32 matmul precision -- TPU's default
    # matmul precision silently truncates float32 inputs to bf16-ish passes,
    # which would otherwise mask real numerical bugs under `use_fp8_quant`'s
    # already-tight rtol/atol.
    logits = jnp.einsum("thd,sd->ths",
                        q_dequant,
                        k_dequant,
                        preferred_element_type=jnp.float32,
                        precision=jax.lax.Precision.HIGHEST)
    relu_logits = jnp.maximum(logits, 0.0)
    folded_weights = weights.astype(jnp.float32) * softmax_scale * n_head_scale
    scores = jnp.einsum("ths,th->ts",
                        relu_logits,
                        folded_weights,
                        precision=jax.lax.Precision.HIGHEST)

    causal = k_positions[None, :] <= q_positions[:, None]
    return jnp.where(causal, scores, -jnp.inf)


def select_topk_indices(scores: jax.Array, topk: int) -> jax.Array:
    """Selects the top-``topk`` KV indices per query token from a score
    matrix, ``-1``-padding rows with fewer than ``topk`` valid (non ``-inf``)
    entries.

    Args:
      scores: ``[T, S]`` float32, ``-inf`` at masked/invalid entries.
      topk: number of indices to select per row.

    Returns:
      ``[T, topk]`` int32, ``-1`` padded (padding always trails valid
      indices within a row, matching the production convention).
    """
    num_kv = scores.shape[-1]
    if num_kv < topk:
        scores = jnp.pad(scores, ((0, 0), (0, topk - num_kv)),
                         constant_values=-jnp.inf)
    top_vals, top_idxs = jax.lax.top_k(scores, topk)
    return jnp.where(top_vals == -jnp.inf, -1, top_idxs).astype(jnp.int32)


def should_recompute_indexer_layer(
    layer_id: int,
    *,
    index_topk_freq: int = INDEX_TOPK_FREQ,
    index_skip_topk_offset: int = INDEX_SKIP_TOPK_OFFSET,
) -> bool:
    """Whether decoder layer ``layer_id`` runs a full lightning-indexer
    recompute, vs. reusing the most-recently-computed shared top-k indices.

    Exactly mirrors upstream vLLM's ``deepseek_v2.py`` layer-construction
    logic (``_skip_topk = (max(layer_id - index_skip_topk_offset + 1, 0) %
    index_topk_freq) != 0``), inverted to a "should recompute" boolean.
    ``layer_id`` is a static (compile-time) model-structure constant, not a
    traced value -- this is a plain Python function, not a jitted one.
    """
    if index_topk_freq <= 0:
        raise ValueError(f"index_topk_freq must be positive, got "
                         f"{index_topk_freq}")
    return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq == 0


def lightning_indexer_layer(
    layer_id: int,
    q: jax.Array,
    k: jax.Array,
    weights: jax.Array,
    q_positions: jax.Array,
    k_positions: jax.Array,
    previous_topk_indices: jax.Array,
    *,
    topk: int = INDEX_TOPK,
    index_topk_freq: int = INDEX_TOPK_FREQ,
    index_skip_topk_offset: int = INDEX_SKIP_TOPK_OFFSET,
    **score_kwargs,
) -> jax.Array:
    """Full per-layer indexer step: either recompute top-k indices from
    scratch, or (on a "shared" layer) reuse ``previous_topk_indices``
    unmodified -- the ``index_topk_freq`` / ``index_skip_topk_offset``
    shared-indexer pattern.

    Args:
      layer_id: static decoder layer index.
      q, k, weights, q_positions, k_positions: see
        ``lightning_indexer_scores``. Ignored (not even traced through the
        scoring math) on skip layers.
      previous_topk_indices: ``[T, topk]`` int32, the most-recently-computed
        shared top-k buffer; returned as-is on skip layers.
      topk: number of indices to select on recompute layers.
      score_kwargs: forwarded to ``lightning_indexer_scores``.

    Returns:
      ``[T, topk]`` int32 top-k indices for this layer.
    """
    if not should_recompute_indexer_layer(
            layer_id,
            index_topk_freq=index_topk_freq,
            index_skip_topk_offset=index_skip_topk_offset,
    ):
        return previous_topk_indices
    scores = lightning_indexer_scores(q, k, weights, q_positions, k_positions,
                                      **score_kwargs)
    return select_topk_indices(scores, topk)
