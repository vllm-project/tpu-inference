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
"""GLM-5.2 indexer top-k Pallas kernel: a thin, GLM-5.2-parameterized wrapper
around ``tpu_inference/kernels/experimental/deepseek_v4/indexer/streamindex_topk.py``
(``compression_ratio=1``, i.e. exact raw-per-token indexing -- DSv4's own
docstring already notes this degrades cleanly to the raw-KV case), plus the
byte-packing glue to turn ``kv_writer.index_qk_rope_quant``'s per-token
``(k_fp8, k_scale)`` output into the page-blocked FP8 cache-with-scale format
``streamindex_topk`` expects (the same layout as
``tests/kernels/deepseek_v4/test_streamindex_topk.py``'s
``test_streamindex_topk_numerical_correctness``/``test_streamindex_topk_quantized``
build by hand: one ``(head_dim + head_dim // 128)``-byte record per token,
padded to a 128-byte boundary, 4 tokens packed per physical 128-lane row).

Numerically: ``streamindex_topk``'s scoring kernel dot-products its `q` input
(kept at full precision -- it does *not* re-quantize Q) against the FP8 raw
bytes of the packed KV cache, then multiplies the ReLU-weighted, head-summed
result by the KV token's per-token scale (see
``streamindex_topk.py::_scores_kernel.compute_scores``). Since a per-token
scale is a positive scalar that commutes with both the dot product and
``ReLU``, this is exactly equivalent to dequantizing K *before* the dot
product -- the same algebraic identity documented in
``indexer/reference.py::lightning_indexer_scores``. This wrapper therefore
feeds it the *dequantized* ``q = q_fp8.astype(f32) * q_scale`` (from
``kv_writer.index_qk_rope_quant``) and folds only ``softmax_scale *
n_head_scale`` (not ``q_scale``, already applied) into the per-head weight,
reproducing ``lightning_indexer_scores(..., use_fp8_quant=True)`` exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.glm5.indexer.reference import (
    INDEX_HEAD_DIM, INDEX_TOPK)

_FP8_DTYPE = jnp.float8_e4m3fn
_KV_PACKING = 4  # uint8 cache -> 32 / 8 = 4 tokens packed per physical row.


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_to(a: int, b: int) -> int:
    return cdiv(a, b) * b


def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterprets each element of ``x``'s trailing dim as raw bytes (same
    helper as ``tests/kernels/deepseek_v4/test_streamindex_topk.py``)."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def indexer_kv_cache_width(head_dim: int = INDEX_HEAD_DIM) -> int:
    """Physical per-token record width (bytes), 128-byte aligned: ``head_dim``
    FP8 bytes + ``head_dim // 128`` ``ue8m0`` scale bytes, padded."""
    q_lkv_dim = align_to(head_dim, 128)
    record_width = q_lkv_dim + q_lkv_dim // 128
    return align_to(record_width, 128)


def pack_indexer_kv_cache(
    k_fp8: jax.Array,  # [num_tokens, head_dim] float8_e4m3fn
    k_scale: jax.Array,  # [num_tokens] float32 (power-of-two)
    *,
    page_size: int,
) -> jax.Array:
    """Packs per-token FP8 K + power-of-two scale (``kv_writer.index_qk_rope_quant``'s
    output) into ``streamindex_topk``'s page-blocked cache format:
    ``[num_pages, page_size // 4, 4, width]`` uint8, one
    ``head_dim``-FP8-bytes-then-scale-byte record per token, tokens assigned
    to pages/slots in contiguous order (token ``t`` -> page ``t //
    page_size``, slot ``t % page_size``).

    Args:
      k_fp8: ``[num_tokens, head_dim]``.
      k_scale: ``[num_tokens]``, must be exactly representable as a
        power-of-two ``ue8m0`` scale (as produced by
        ``kv_writer.index_qk_rope_quant``/``indexer.reference.quantize_fp8_ue8m0``).
      page_size: tokens per physical page; must be a multiple of 4.

    Returns:
      ``[cdiv(num_tokens, page_size), page_size // 4, 4, width]`` uint8.
    """
    if page_size % _KV_PACKING != 0:
        raise ValueError(f"page_size ({page_size}) must be a multiple of "
                         f"{_KV_PACKING}")
    num_tokens, head_dim = k_fp8.shape
    width = indexer_kv_cache_width(head_dim)

    num_pages = cdiv(num_tokens, page_size)
    padded_tokens = num_pages * page_size
    pad = padded_tokens - num_tokens
    if pad:
        k_fp8 = jnp.pad(k_fp8, ((0, pad), (0, 0)))
        # Padding-region scale value is irrelevant: `streamindex_topk` masks
        # padding slots via `seq_lens`/causal bounds before they can affect
        # any real query's top-k.
        k_scale = jnp.pad(k_scale, (0, pad), constant_values=1.0)

    k_scale_e8m0 = k_scale.astype(jnp.float8_e8m0fnu)
    q_bytes = _to_byte_lane(k_fp8)  # [padded_tokens, head_dim] uint8
    scale_bytes = _to_byte_lane(k_scale_e8m0)[:, None]  # [padded_tokens, 1]

    record = jnp.concatenate([q_bytes, scale_bytes], axis=-1)
    record = jnp.pad(record, ((0, 0), (0, width - record.shape[-1])))

    return record.reshape(num_pages, page_size // _KV_PACKING, _KV_PACKING,
                          width)


def glm5_indexer_topk(
    q_fp8: jax.Array,  # [T, H, head_dim] float8_e4m3fn
    q_scale: jax.Array,  # [T, H] float32
    raw_weights: jax.Array,  # [T, H] float32, pre-scale-fold indexer weight
    k_fp8: jax.Array,  # [num_kv_tokens, head_dim] float8_e4m3fn
    k_scale: jax.Array,  # [num_kv_tokens] float32
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3], see streamindex_topk's docstring
    *,
    topk: int = INDEX_TOPK,
    page_size: int,
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    softmax_scale: float | None = None,
    n_head_scale: float | None = None,
    **kwargs,
) -> jax.Array:
    """GLM-5.2-parameterized ``streamindex_topk`` (``compression_ratio=1``,
    i.e. exact raw/uncompressed per-token top-k indexing).

    Args:
      q_fp8, q_scale: the query side of ``kv_writer.index_qk_rope_quant``'s
        output.
      raw_weights: ``[T, num_heads]``, the indexer's raw per-head weight
        *before* the ``softmax_scale * n_head_scale`` fold (matching
        ``indexer.reference.lightning_indexer_scores``'s ``weights`` arg).
      k_fp8, k_scale: the K side of ``kv_writer.index_qk_rope_quant``'s
        output (pre-packing; this function calls ``pack_indexer_kv_cache``
        internally).
      seq_lens, page_indices, cu_q_lens, distribution, num_kv_pages_per_block,
        num_queries_per_block: batching/paging metadata, forwarded to
        ``streamindex_topk`` -- see its docstring.
      topk: number of indices to select (``2048`` for GLM-5.2).
      page_size: must match the page size used to build ``page_indices``.
      softmax_scale: defaults to ``head_dim ** -0.5``.
      n_head_scale: defaults to ``num_heads ** -0.5``.
      kwargs: forwarded to ``streamindex_topk`` (e.g. ``vmem_limit_bytes``,
        ``decode_req_batch_size``).

    Returns:
      ``[T, topk]`` int32, ``-1``-padded, selected via
      ``jax.lax.approx_max_k(recall_target=1.0)`` (not exact
      ``jax.lax.top_k`` -- see ``streamindex_topk.py``). Compare against
      ``indexer.reference.lightning_indexer_scores``/``select_topk_indices``
      with a recall-aware tolerance, not exact index-set equality.
    """
    num_tokens, num_heads, head_dim = q_fp8.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if n_head_scale is None:
        n_head_scale = num_heads**-0.5

    q_dequant = q_fp8.astype(jnp.float32) * q_scale[..., None]
    folded_weights = raw_weights.astype(
        jnp.float32) * softmax_scale * n_head_scale

    cache_kv = pack_indexer_kv_cache(k_fp8, k_scale, page_size=page_size)

    return streamindex_topk(
        q_dequant,
        folded_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=topk,
        compression_ratio=1,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        **kwargs,
    )


def build_single_sequence_batch_metadata(
    num_q_tokens: int,
    num_kv_tokens: int,
    *,
    page_size: int,
    seq_len: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[int, int, int]]:
    """Builds the simplest possible single-sequence batching metadata for
    ``glm5_indexer_topk``/``streamindex_topk``: one sequence, contiguous
    page table ``[0, 1, ..., num_pages - 1]`` (matching
    ``pack_indexer_kv_cache``'s contiguous token -> page/slot assignment).

    IMPORTANT semantic note (a real gotcha the causal-masking test in
    ``tests/kernels/glm5/topk_test.py`` catches): ``streamindex_topk``
    derives each query token's *absolute causal position* purely from
    ``seq_lens``/``cu_q_lens`` -- it assumes a sequence's query tokens are
    exactly its **last** ``q_len`` positions (``abs_pos = seq_len - q_len +
    local_idx``), matching the standard ragged/paged-attention batching
    convention. It does **not** take an independent per-token positions
    array. ``seq_len`` here is therefore the true logical sequence length
    (how many of the physical ``num_kv_tokens`` cache rows are causally
    valid), which may be less than ``num_kv_tokens`` (the physical cache can
    hold more rows than are "live" yet); it defaults to ``num_kv_tokens``,
    i.e. "the query tokens are the tail of a fully-populated cache".

    Returns ``(seq_lens, page_indices, cu_q_lens, distribution)``.
    """
    if seq_len is None:
        seq_len = num_kv_tokens
    num_pages = cdiv(num_kv_tokens, page_size)
    seq_lens = jnp.array([seq_len], dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_q_tokens], dtype=jnp.int32)
    # A single decode (q_len == 1) sequence takes the DECODE path; anything
    # else (prefill/mixed q_len) takes the MIXED path -- mirrors
    # test_streamindex_topk_numerical_correctness's num_decodes bookkeeping.
    is_decode = num_q_tokens == 1
    distribution = (1, 1, 1) if is_decode else (0, 0, 1)
    return seq_lens, page_indices, cu_q_lens, distribution
