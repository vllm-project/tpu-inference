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
"""Pallas TPU kernel: GLM-5.2 DSA main sparse attention (gather top-k raw
KV + FlashAttention-style online-softmax attend).

Engineering template: this kernel's structure -- per-query-token/-block grid,
double-buffered VMEM scratch, explicit DMA semaphores, online-softmax
(running max/sum/accumulator) -- follows
``tpu_inference/kernels/experimental/deepseek_v4/core_attention/sparse_mla.py``'s
FlashAttention shape. The actual *gather* mechanism is a simplification of
``core_attention/csa_gather.py``'s SparseCore ``pl.Indirect`` vector gather:
instead of a batched hardware vector-gather (which needs the v7x SparseCore
mesh and a fair amount of bit-packing machinery not obviously portable to an
arbitrary raw KV row width), this kernel issues one ``pltpu.make_async_copy``
DMA per selected KV row, chunked and double-buffered across the ``topk`` axis
(mirroring ``streamindex_topk.py``'s ``_fetch_bkv`` prefetch-ahead pattern,
just gathering individual token rows instead of whole pages). This is a
correctness-first, not maximally bandwidth-efficient, design -- see the
Phase 2 bring-up report for the SparseCore-gather follow-up this leaves on
the table.

KV cache layout: a flat ``[total_slots, head_dim]`` view of
``tpu_inference/kernels/mla/v2/kernel.py``'s raw/uncompressed paged cache
convention (``[num_pages, page_size_per_kv_packing, kv_packing, head_dim]``,
collapsible to ``[num_pages * page_size, head_dim]`` by construction -- see
``flatten_paged_kv_cache``). V is the leading ``v_head_dim`` columns of the
same row as K (``qk_head_dim >= v_head_dim``), matching
``vllm/v1/attention/ops/xpu_mla_sparse.py``'s ``v = kv[..., :d_v]`` MLA-style
convention (single, MQA-shared KV "head").

Index resolution (``resolve_topk_to_physical_slots``) -- converting the
lightning indexer's per-request-relative top-k indices into physical flat
cache-row indices via a block table -- is kept as a separate, cheap
index-arithmetic pre-pass in plain JAX, mirroring how upstream vLLM itself
splits this out as its own Triton kernel
(``vllm/v1/attention/backends/mla/sparse_utils.py::triton_convert_req_index_to_global_index``),
independent of the (bandwidth-bound) attention compute kernel below.

Quantized (FP8) cache support (Phase 2 follow-up, added after the Phase 3
wiring pass surfaced it as a hard blocker: ``nvidia/GLM-5.2-NVFP4`` auto-
selects an FP8 KV cache regardless of CLI flags): ``cache_kv`` may be any
floating dtype, including ``float8_e4m3fn`` -- see ``_gather_row_leading_axis``
for how the per-row DMA gather is made to work for sub-32-bit dtypes (a
Mosaic VMEM/HBM tiling constraint, not a numerics one), and ``k_scale``/
``v_scale`` for the dequantization convention. GLM-5.2's *main* attention
cache uses this repo's standard **static per-tensor** FP8 quantization
(``tpu_inference/layers/common/quantization/__init__.py::quantize_kv``/
``static_per_tensor_quantize_tensor``: ``q = clip(x / scale, ...).astype(fp8)``,
one scalar ``k_scale``/``v_scale`` per layer) -- confirmed by reading the
existing dense backend, ``layers/vllm/backends/flash_attn_mla.py``, which
already quantizes this same cache this way. This is a different, and
considerably simpler, quantization scheme than the lightning *indexer*'s own
K-cache (``indexer/kv_writer.py``/``indexer/topk.py``: per-token ``ue8m0``
block scale, packed FP8 bytes) -- do not conflate the two; this kernel's
``k_scale``/``v_scale`` are plain per-call Python floats, not a per-row
scale array.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DEFAULT_MASK_VALUE = -1e30


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def flatten_paged_kv_cache(cache_kv: jax.Array) -> jax.Array:
    """Collapses ``mla/v2``-style
    ``[num_pages, page_size_per_kv_packing, kv_packing, head_dim]`` into the
    flat ``[num_pages * page_size, head_dim]`` view this kernel gathers rows
    from -- a free reshape (no data movement): the same collapse
    ``mla/v1/kernel.py::ref_mla_ragged_paged_attention`` performs
    (``cache_kv.reshape(total_num_pages, page_size, lkv_dim)``), one step
    further to drop the page axis too.
    """
    if cache_kv.ndim == 2:
        return cache_kv
    if cache_kv.ndim != 4:
        raise ValueError(f"expected a 2D (already flat) or 4D (paged) cache, "
                         f"got shape {cache_kv.shape}")
    num_pages, page_size_per_kv_packing, kv_packing, head_dim = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    return cache_kv.reshape(num_pages * page_size, head_dim)


def resolve_topk_to_physical_slots(
    topk_indices: jax.Array,  # i32[T, topk], -1 = invalid/padding
    block_table: jax.Array,  # i32[num_requests, max_blocks_per_req]
    req_ids: jax.Array,  # i32[T], per-query-token request id
    *,
    block_size: int,
) -> jax.Array:
    """Converts per-request-relative top-k KV indices into physical flat
    cache-row indices via a block table:
    ``out[t, i] = block_table[req_ids[t], topk_indices[t, i] // block_size]
    * block_size + topk_indices[t, i] % block_size``, ``-1`` preserved for
    padding. Pure index arithmetic -- mirrors
    ``vllm/v1/attention/backends/mla/sparse_utils.py::triton_convert_req_index_to_global_index``
    but in plain (vectorized) JAX rather than a dedicated Triton/Pallas
    kernel, since it is not on the attention kernel's bandwidth-critical
    path.
    """
    valid = topk_indices >= 0
    safe_indices = jnp.clip(topk_indices, 0, None)
    block_idx = safe_indices // block_size
    offset = safe_indices % block_size
    physical_block = block_table[req_ids[:, None], block_idx]
    physical_slot = physical_block * block_size + offset
    return jnp.where(valid, physical_slot, -1).astype(jnp.int32)


def _sparse_attention_kernel_body(
    # Scalar prefetch
    physical_slots_ref,  # i32[T, topk]
    # Inputs
    q_ref,  # VMEM [1, H, qk_head_dim]
    cache_kv_hbm_ref,  # HBM [total_slots, 1, qk_head_dim] (see the
    # module docstring / sparse_causal_attention_kernel: the singleton
    # middle axis moves the row-selection axis off the tiled (sublane, lane)
    # pair -- see _gather_row_leading_axis).
    # Output
    o_ref,  # VMEM [1, H, v_head_dim]
    # Scratch
    kv_x2_ref,  # VMEM [2, chunk_n, 1, qk_head_dim]
    sems,  # DMA semaphores [2, chunk_n]
    *,
    topk: int,
    chunk_n: int,
    v_head_dim: int,
    softmax_scale: float,
    mask_value: float,
    k_scale: float,
    v_scale: float,
    k_scale_rope: float,
):
    t = pl.program_id(0)
    num_chunks = topk // chunk_n
    num_heads = q_ref.shape[1]

    def start_chunk(chunk_idx, buf_idx):
        base = chunk_idx * chunk_n
        for r in range(chunk_n):
            slot = physical_slots_ref[t, base + r]
            safe_slot = jnp.maximum(slot, 0)
            pltpu.make_async_copy(
                cache_kv_hbm_ref.at[pl.ds(safe_slot, 1)],
                kv_x2_ref.at[buf_idx, pl.ds(r, 1)],
                sems.at[buf_idx, r],
            ).start()

    def wait_chunk(chunk_idx, buf_idx):
        base = chunk_idx * chunk_n
        for r in range(chunk_n):
            slot = physical_slots_ref[t, base + r]
            safe_slot = jnp.maximum(slot, 0)
            pltpu.make_async_copy(
                cache_kv_hbm_ref.at[pl.ds(safe_slot, 1)],
                kv_x2_ref.at[buf_idx, pl.ds(r, 1)],
                sems.at[buf_idx, r],
            ).wait()

    q = q_ref[0].astype(jnp.float32)  # [H, qk_head_dim]

    start_chunk(0, 0)

    def body(chunk_idx, carry):
        acc, m, l = carry
        buf_idx = jax.lax.rem(chunk_idx, 2)
        next_buf_idx = 1 - buf_idx

        @pl.when(chunk_idx + 1 < num_chunks)
        def _prefetch_next():
            start_chunk(chunk_idx + 1, next_buf_idx)

        wait_chunk(chunk_idx, buf_idx)

        # Each row lives at a *leading* (untiled) `(chunk_n, 1)` address --
        # `kv_x2_ref[buf_idx, r]` is a full, tile-aligned `(1, qk_head_dim)`
        # block regardless of `cache_kv`'s dtype width (see
        # `_gather_row_leading_axis`'s note); stack the chunk's rows into one
        # `(chunk_n, qk_head_dim)` array for the matmuls below. This is a
        # plain (non-DMA) VMEM read, not subject to the DMA tile-alignment
        # restriction that motivated the leading-axis layout in the first
        # place.
        kv_rows = jnp.concatenate(
            [kv_x2_ref[buf_idx, r] for r in range(chunk_n)],
            axis=0)  # [chunk_n, qk_head_dim], raw (possibly fp8) dtype

        base = chunk_idx * chunk_n
        # Scalar-prefetch (SMEM) refs only support scalar loads -- gather the
        # chunk's slots one at a time (static Python `r`, dynamic `base`),
        # same as `start_chunk`/`wait_chunk` above, then stack into a vector.
        slots_chunk = jnp.stack(
            [physical_slots_ref[t, base + r] for r in range(chunk_n)])
        valid = slots_chunk >= 0  # [chunk_n]

        # Dequantize immediately after the gather (static per-tensor scale;
        # `k_scale`/`v_scale` default to 1.0 for an already-float cache, a
        # no-op multiply). V is the leading `v_head_dim` columns of the
        # *same* physical row, dequantized with its own (generally
        # different) scale -- matching `layers/vllm/backends/
        # flash_attn_mla.py`'s dense backend, which applies `k_scale` to the
        # row when it's used as K and `v_scale` when (a copy of) the same
        # latent is used as V.
        #
        # K itself is *not* dequantized with one uniform scale across the
        # whole row: for MLA, a K row is `concat(latent, rope)`, and for
        # GLM-5.2 specifically these two halves live in wildly different
        # magnitude regimes (measured live: latent/`kv_c_normed`
        # mean|x|~0.003/max~0.08; rope/`k_pe` mean|x|~0.7/max~8, a
        # ~100-1000x difference) -- confirmed via a live bring-up debug
        # comparison against a full-precision (non-fp8) reference computed
        # from the exact same top-k-selected rows, which showed this
        # kernel's single-shared-`k_scale` output was still ~10-50x smaller
        # in magnitude than the unquantized reference, i.e. real
        # information loss beyond what an ideally-calibrated single scalar
        # scale could explain. fp8 e4m3's dynamic range is only ~200x
        # (~0.002 to 448), so no single static scale can represent both
        # halves without either underflowing latent or saturating/clipping
        # rope -- a single scale calibrated for the (much smaller) latent
        # half leaves rope severely clipped, corrupting the *positional*
        # component of the attention score. Apply `k_scale` to the leading
        # `v_head_dim` (latent) columns and a separate `k_scale_rope` to the
        # remaining (rope) columns instead.
        # Guarded by a Python-level (trace-time-static, not data-dependent)
        # check: some callers (e.g. dense-equivalent tests/models with no
        # separate rope slice, `qk_head_dim == v_head_dim`) have a
        # zero-width "rope" remainder -- Mosaic can't lower a
        # zero-size vector slice/concatenate, so skip the split entirely in
        # that case (equivalent anyway, since there's nothing to split).
        if kv_rows.shape[-1] > v_head_dim:
            k_latent = kv_rows[:, :v_head_dim].astype(jnp.float32) * k_scale
            k_rope = kv_rows[:, v_head_dim:].astype(
                jnp.float32) * k_scale_rope
            k_chunk = jnp.concatenate([k_latent, k_rope],
                                      axis=-1)  # [chunk_n, qk_dim]
        else:
            k_chunk = kv_rows.astype(jnp.float32) * k_scale
        v_chunk = kv_rows[:, :v_head_dim].astype(
            jnp.float32) * v_scale  # [chunk_n, v_head_dim]

        logits = jnp.dot(q,
                         k_chunk.T,
                         preferred_element_type=jnp.float32,
                         precision=jax.lax.Precision.HIGHEST
                         ) * softmax_scale  # [H, chunk_n]
        logits = jnp.where(valid[None, :], logits, mask_value)

        chunk_max = jnp.max(logits, axis=-1, keepdims=True)  # [H, 1]
        new_m = jnp.maximum(m, chunk_max)
        # Explicitly zero (not just `exp(mask_value - mask_value) == 1`) the
        # invalid lanes: if a whole chunk (or the whole row) is invalid,
        # `logits - new_m` degenerates to `mask_value - mask_value == 0` ->
        # `exp(0) == 1`, which would otherwise average in garbage/zero KV
        # rows instead of contributing nothing to `l`/`acc` (matching the
        # Phase 1 JAX reference's `row_has_valid` all-invalid guard).
        p = jnp.where(valid[None, :], jnp.exp(logits - new_m), 0.0)
        alpha = jnp.exp(m - new_m)  # [H, 1]
        new_l = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
        new_acc = acc * alpha + jnp.dot(p,
                                        v_chunk,
                                        preferred_element_type=jnp.float32,
                                        precision=jax.lax.Precision.HIGHEST)
        return new_acc, new_m, new_l

    acc0 = jnp.zeros((num_heads, v_head_dim), dtype=jnp.float32)
    m0 = jnp.full((num_heads, 1), mask_value, dtype=jnp.float32)
    l0 = jnp.zeros((num_heads, 1), dtype=jnp.float32)
    acc, _, l = jax.lax.fori_loop(0, num_chunks, body, (acc0, m0, l0))

    out = acc / jnp.where(l == 0.0, 1.0, l)
    o_ref[0] = out.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("v_head_dim", "topk", "chunk_n", "softmax_scale",
                     "mask_value", "k_scale", "v_scale", "k_scale_rope",
                     "vmem_limit_bytes", "name"),
)
def sparse_causal_attention_kernel(
    q: jax.Array,  # [T, H, qk_head_dim]
    cache_kv: jax.Array,  # [total_slots, qk_head_dim] (flat; see
    # flatten_paged_kv_cache for the paged -> flat adapter)
    physical_slots: jax.Array,  # i32[T, topk], -1 = invalid/padding
    *,
    v_head_dim: int,
    topk: int,
    chunk_n: int = 8,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    k_scale_rope: float | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    name: str = "glm5_sparse_causal_attention",
) -> jax.Array:
    """Gathers the ``topk`` selected raw KV cache rows per query token (via
    per-row DMA, double-buffered in chunks of ``chunk_n``) and computes
    FlashAttention-style online-softmax attention over just that gathered
    subset.

    Causal correctness relies entirely on ``physical_slots`` already
    encoding a causally-valid (or ``-1``-padded) selection -- as produced by
    the lightning indexer (``indexer.reference``/``indexer.topk``) -- this
    kernel does not independently re-check query/key positions (unlike the
    Phase 1 JAX reference's defensive re-check, which is affordable in plain
    JAX but not worth the extra scalar-prefetch plumbing in the hot Pallas
    path).

    Args:
      q: ``[T, num_heads, qk_head_dim]``.
      cache_kv: ``[total_slots, qk_head_dim]``, a single (MQA-shared) KV
        vector per physical cache row; V is the leading ``v_head_dim``
        columns of the same row. May be ``float32``, ``bfloat16``, or
        ``float8_e4m3fn`` (or any other floating dtype) -- see the module
        docstring's "Quantized (FP8) cache support" section.
      physical_slots: ``[T, topk]`` int32 row indices into ``cache_kv``
        (already resolved from the indexer's per-request-relative top-k via
        ``resolve_topk_to_physical_slots``), ``-1`` for padding.
      v_head_dim: width of the V slice of each cache row (``<= qk_head_dim``).
      topk: number of selected KV rows per query token; must be a multiple
        of ``chunk_n``.
      chunk_n: number of KV rows gathered per double-buffered DMA chunk.
      softmax_scale: defaults to ``qk_head_dim ** -0.5``.
      mask_value: logit value used for masked/invalid KV rows.
      k_scale, v_scale: static per-tensor FP8 dequantization scales (see the
        module docstring), applied as ``row.astype(f32) * scale`` immediately
        after each row's DMA lands. Leave at the default ``1.0`` (a no-op)
        for an already-float ``cache_kv``. ``k_scale`` applies to the
        leading ``v_head_dim`` (latent) columns of K; ``k_scale_rope``
        (defaults to ``k_scale`` if not given) applies to the remaining
        (rope) columns -- see the "K is *not* dequantized with one uniform
        scale" note in ``_sparse_attention_kernel_body`` for why a single
        scale is generally wrong for GLM-5.2's concatenated latent+rope K
        row (their magnitudes differ by ~100-1000x, well beyond fp8 e4m3's
        ~200x dynamic range).

    Returns:
      ``[T, num_heads, v_head_dim]``, same dtype as ``q``.
    """
    num_tokens, num_heads, qk_head_dim = q.shape
    total_slots, cache_head_dim = cache_kv.shape
    if cache_head_dim != qk_head_dim:
        raise ValueError(f"cache_kv head_dim {cache_head_dim} != q's "
                         f"{qk_head_dim}")
    if v_head_dim > qk_head_dim:
        raise ValueError(f"v_head_dim ({v_head_dim}) must be <= qk_head_dim "
                         f"({qk_head_dim})")
    if physical_slots.shape != (num_tokens, topk):
        raise ValueError(f"physical_slots shape {physical_slots.shape} != "
                         f"{(num_tokens, topk)}")
    if topk % chunk_n != 0:
        raise ValueError(f"topk ({topk}) must be a multiple of chunk_n "
                         f"({chunk_n})")
    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"cache_kv dtype {cache_kv.dtype} must be a "
                         "floating dtype (float32/bfloat16/float8_*).")

    if softmax_scale is None:
        softmax_scale = qk_head_dim**-0.5
    if k_scale_rope is None:
        k_scale_rope = k_scale

    # Give the row-selection axis its own leading (untiled) dimension: a
    # `(total_slots, 1, qk_head_dim)` HBM ref lets a single dynamic-offset
    # row DMA (`.at[pl.ds(slot, 1)]`) address an arbitrary row regardless of
    # `cache_kv`'s dtype width, since only the *last two* dims of a Pallas
    # TPU memref carry the (sublane, lane) tiling/packing constraints that a
    # plain `(total_slots, qk_head_dim)` 2D ref would apply to the
    # (sub-32-bit-packed) row axis -- see the module docstring's "Quantized
    # (FP8) cache support" note. Free reshape, no data movement. The VMEM
    # scratch (`kv_x2_ref`) mirrors this: `(2, chunk_n, 1, qk_head_dim)`, so
    # `.at[buf_idx, pl.ds(r, 1)]` is likewise a whole-leading-dim slice.
    cache_kv_3d = cache_kv.reshape(total_slots, 1, qk_head_dim)

    kernel = functools.partial(
        _sparse_attention_kernel_body,
        topk=topk,
        chunk_n=chunk_n,
        v_head_dim=v_head_dim,
        softmax_scale=softmax_scale,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        k_scale_rope=k_scale_rope,
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[
            pl.BlockSpec((1, num_heads, qk_head_dim), lambda t, _: (t, 0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((1, num_heads, v_head_dim), lambda t, _:
                               (t, 0, 0)),
        grid=(num_tokens, ),
        scratch_shapes=[
            pltpu.VMEM((2, chunk_n, 1, qk_head_dim), cache_kv.dtype),
            pltpu.SemaphoreType.DMA((2, chunk_n)),
        ],
    )

    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads, v_head_dim),
                                       q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=name,
    )(physical_slots, q, cache_kv_3d)

    return out
