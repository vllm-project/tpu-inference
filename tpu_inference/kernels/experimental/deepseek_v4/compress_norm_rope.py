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
"""Compress, norm, RoPE, store DeepSeek-V4 KV cache at compression boundaries.

Shared cache layout (sparse / head_dim=512 path)
-------------------------------------------------
The compressor *state cache* and the compressed *KV cache* are collapsed into a
single ``uint8`` ``jax.Array`` shaped like the MLA paged cache::

    [num_pages, page_size // PACKING, PACKING, width]   # PACKING=4, width=640

A page holds ``page_size`` (=64) KV row-slots of ``width`` (=640) bytes. Two
logical caches overlay the same bytes (addressed by independent block tables,
so their slot namespaces never collide):

* Compressed KV -- written here, boundary tokens only, one row-slot per token.
  The minor dim ``width`` = ``align_to(583, 128)`` = 640. The 583 used bytes are
  ``[ nope 448 fp8 | rope 128 bf16 | scale 7 e8m0 ]`` and ``[583:640]`` is pad.

* Compressor state -- read here for the window pool. One state token spans
  ``page_size // state_block_size`` (=16) row-slots; each row-slot stores the
  first 512 bytes (= 128 f32) of state and pads ``[512:640]``. So a state
  token's ``state_dim`` (=2048) f32 values tile 16 row-slots x 128 f32.

Shared cache layout (indexer / head_dim=128 path)
-------------------------------------------------
Same ``[num_pages, page_size // 4, 4, width]`` layout with ``width = 256``.
KV record: ``[ fp8 128 | scale 1 e8m0 ] + pad``.
"""

import jax
import jax.numpy as jnp


def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes.

    ``bitcast_convert_type`` appends a trailing itemsize dim for dtypes wider
    than 8 bits (bf16 -> ``[..., 2]``, f32 -> ``[..., 4]``).
    """
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def _from_byte_lane(b: jax.Array, dtype) -> jax.Array:
    """Inverse of ``_to_byte_lane``: read trailing bytes back as ``dtype``."""
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize == 1:
        return jax.lax.bitcast_convert_type(b, dtype)
    grouped = b.reshape(*b.shape[:-1], b.shape[-1] // itemsize, itemsize)
    return jax.lax.bitcast_convert_type(grouped, dtype)


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


def sparse_packed_width(nope_dim: int, rope_head_dim: int,
                        quant_block: int) -> int:
    """Bytes per token in the packed sparse (head_dim=512) KV cache."""
    # nope fp8 (1B) + rope bf16 (2B) + UE8M0 block scale (1B)
    return nope_dim + rope_head_dim * 2 + (nope_dim // quant_block)


def indexer_packed_width(head_dim: int, quant_block: int) -> int:
    """Bytes per token in the packed indexer (head_dim=128) KV cache."""
    # fp8 (1B) + UE8M0 block scale (1B)
    return head_dim + (head_dim // quant_block)


def unpack_sparse_kv_cache(kv_cache: jax.Array, nope_dim: int,
                           rope_head_dim: int, quant_block: int):
    """Split the packed sparse KV cache into native-dtype component views.

    The block scale is stored as UE8M0 (``float8_e8m0fnu``, one byte per
    block) and returned as the equivalent power-of-two ``float32``.
    """
    n_qb = nope_dim // quant_block
    a = nope_dim
    b = a + rope_head_dim * 2
    nope = _from_byte_lane(kv_cache[..., :a], jnp.float8_e4m3fn)
    rope = _from_byte_lane(kv_cache[..., a:b], jnp.bfloat16)
    scale = _from_byte_lane(kv_cache[..., b:b + n_qb],
                            jnp.float8_e8m0fnu).astype(jnp.float32)
    return nope, rope, scale


def unpack_indexer_kv_cache(kv_cache: jax.Array, head_dim: int,
                            quant_block: int):
    """Split the packed indexer KV cache into ``(fp8, scale)`` views."""
    n_qb = head_dim // quant_block
    fp8 = _from_byte_lane(kv_cache[..., :head_dim], jnp.float8_e4m3fn)
    scale = _from_byte_lane(kv_cache[..., head_dim:head_dim + n_qb],
                            jnp.float8_e8m0fnu).astype(jnp.float32)
    return fp8, scale


# uint8 lanes per 32-bit MLA word
PACKING = 4


def _align_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def shared_sparse_cache_shape(num_pages: int, page_size: int, nope_dim: int,
                              rope_head_dim: int, quant_block: int):
    """MLA-style shape of the shared state+KV ``uint8`` buffer (sparse path).

    Returns ``[num_pages, page_size // PACKING, PACKING, width]`` where
    ``width = align_to(sparse_packed_width(...), 128)`` (= 640 for DeepSeek-V4
    head_dim=512).
    """
    width = _align_to(
        sparse_packed_width(nope_dim, rope_head_dim, quant_block), 128)
    return (num_pages, _align_to(page_size, PACKING) // PACKING, PACKING,
            width)


def shared_indexer_cache_shape(num_pages: int, page_size: int, head_dim: int,
                               quant_block: int):
    """MLA-style shape of the shared state+KV ``uint8`` buffer (indexer path).

    Same layout as ``shared_sparse_cache_shape`` but sized for the indexer
    record: ``width = align_to(indexer_packed_width(...), 128)`` (= 256 for
    DeepSeek-V4 head_dim=128). 
    """
    width = _align_to(indexer_packed_width(head_dim, quant_block), 128)
    return (num_pages, _align_to(page_size, PACKING) // PACKING, PACKING,
            width)


def _state_chunk_dims(cache_shape, state_block_size: int, state_dim: int):
    """Geometry of state token in cache.

    Returns:
        kv_slots: row-slots per page (``rows * packing``, e.g. 64).
        rows_per_token: row-slots per state token (``page_size //
            state_block_size``, e.g. 16).
        f32_per_row: f32 values per row-slot (e.g. 128).
        bytes_per_row: bytes used per row-slot (``f32_per_row * 4``, e.g. 512).
    """
    num_pages, rows, packing, width = cache_shape
    kv_slots = rows * packing
    if kv_slots % state_block_size != 0:
        raise ValueError(f"page_size {kv_slots} not divisible by "
                         f"state_block_size {state_block_size}")
    rows_per_token = kv_slots // state_block_size
    if state_dim % rows_per_token != 0:
        raise ValueError(f"state_dim {state_dim} not divisible by "
                         f"rows_per_token {rows_per_token}")
    f32_per_row = state_dim // rows_per_token
    bytes_per_row = f32_per_row * 4
    if bytes_per_row > width:
        raise ValueError(
            f"state row needs {bytes_per_row}B but cache width is {width}B")
    return kv_slots, rows_per_token, f32_per_row, bytes_per_row


def unpack_state_cache(cache: jax.Array, state_block_size: int,
                       state_dim: int) -> jax.Array:
    """Read the shared buffer's f32 state view ``[num_pages, sb, state_dim]``.

    Inverse of ``pack_state_cache``. Only the leading ``bytes_per_row`` of each
    row-slot carry state; trailing pad bytes are ignored.
    """
    num_pages, rows, packing, width = cache.shape
    kv_slots, rpt, _, bpr = _state_chunk_dims(cache.shape, state_block_size,
                                              state_dim)
    slots = cache.reshape(num_pages, kv_slots, width)
    chunk = slots[:, :, :bpr].reshape(num_pages, state_block_size, rpt, bpr)
    f32 = _from_byte_lane(chunk, jnp.float32)  # [num_pages, sb, rpt, fpr]
    return f32.reshape(num_pages, state_block_size, state_dim)


def pack_state_cache(cache: jax.Array, state: jax.Array) -> jax.Array:
    """Write an f32 state view ``[num_pages, sb, state_dim]`` into ``cache``.

    Inverse of ``unpack_state_cache``; leaves each row-slot's pad bytes (and the
    KV-only row-slots) untouched.
    """
    num_pages, rows, packing, width = cache.shape
    sb, state_dim = state.shape[1], state.shape[2]
    kv_slots, rpt, fpr, bpr = _state_chunk_dims(cache.shape, sb, state_dim)
    chunk = state.reshape(num_pages, sb, rpt, fpr)
    chunk_bytes = _to_byte_lane(chunk).reshape(num_pages, kv_slots, bpr)
    slots = cache.reshape(num_pages, kv_slots, width)
    slots = slots.at[:, :, :bpr].set(chunk_bytes)
    return slots.reshape(num_pages, rows, packing, width)


def interleaved_rope(
    x: jax.Array,  # [..., head_dim] fp32
    cos_sin: jax.Array,  # [..., rope_head_dim] fp32 ([cos | sin])
    rope_head_dim: int,
) -> jax.Array:
    """Interleaved (GPT-J) RoPE on the trailing ``rope_head_dim`` elements."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even; got {head_dim}")
    if rope_head_dim % 2 != 0 or rope_head_dim > head_dim:
        raise ValueError(f"rope_head_dim must be even and <= head_dim; got "
                         f"rope_head_dim={rope_head_dim}, head_dim={head_dim}")

    half_rope = rope_head_dim // 2
    num_pairs = head_dim // 2
    nope_pairs = num_pairs - half_rope

    pairs = x.reshape(*x.shape[:-1], num_pairs, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]

    cos = cos_sin[..., :half_rope]
    sin = cos_sin[..., half_rope:rope_head_dim]

    pad_shape = (*cos.shape[:-1], nope_pairs)
    cos_full = jnp.concatenate([jnp.ones(pad_shape, x.dtype), cos], axis=-1)
    sin_full = jnp.concatenate([jnp.zeros(pad_shape, x.dtype), sin], axis=-1)

    new_even = even * cos_full - odd * sin_full
    new_odd = odd * cos_full + even * sin_full

    out = jnp.stack([new_even, new_odd], axis=-1)
    return out.reshape(x.shape)


def compress_norm_rope(
    kv_window: jax.Array,  # [num_tokens, window, head_dim] fp32
    score_window: jax.Array,  # [num_tokens, window, head_dim] fp32
    valid_mask: jax.Array,  # [num_tokens, window] bool
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    compressed_pos: jax.Array,  # [num_tokens] int
    rms_eps: float,
    rope_head_dim: int,
) -> jax.Array:
    """Window softmax-pool, RMSNorm, and interleaved RoPE."""
    neg_inf = jnp.array(-jnp.inf, dtype=score_window.dtype)
    masked_score = jnp.where(valid_mask[..., None], score_window, neg_inf)
    weights = jax.nn.softmax(masked_score, axis=1)

    compressed_kv = jnp.sum(weights * kv_window,
                            axis=1)  # [num_tokens, head_dim]

    # RMSNorm over head_dim. variance: [num_tokens, 1];
    # normed: [num_tokens, head_dim].
    variance = jnp.mean(jnp.square(compressed_kv), axis=-1, keepdims=True)
    normed = compressed_kv * jax.lax.rsqrt(variance + rms_eps) * rms_weight

    cos_sin = cos_sin_cache[compressed_pos]
    return interleaved_rope(normed, cos_sin, rope_head_dim)


def gather_state_windows(
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
):
    """
    Gather ``[kv_window, score_window, valid_mask]`` from the paged cache.

    Returns:
      kv_window: [token, window, head_dim] -- partial kv vectors to pool 
      score_window: [tokens, window, head_dim] -- scores for softmax weight
      valid_mask: [token, window] -- False where window goes before seq start.
    
    """
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio

    start = positions - window + 1
    w_idx = jnp.arange(window)
    pos = start[:, None] + w_idx[None, :]
    valid_mask = pos >= 0

    safe_pos = jnp.where(valid_mask, pos, 0)
    req = token_to_req_indices[:, None]
    # Gather page numbers (optimized to 1D indexing to avoid 2D index bitpacking)
    max_blocks = block_table.shape[-1]
    flat_index = req * max_blocks + (safe_pos // block_size)
    block_numbers = block_table.reshape(-1)[flat_index]
    block_offsets = safe_pos % block_size

    # C4 overlap: slots >= compress_ratio read the second head slice.
    # head_offset = 512 if w_idx >= compress_ratio else 0
    # this is because last dimension is 2 * state_width (2048)
    head_offset = (w_idx >= compress_ratio).astype(jnp.int32) * head_dim
    # cols = head_offset + [0..511]
    col = head_offset[None, :, None] + jnp.arange(head_dim)[None, None, :]

    bn = block_numbers[:, :, None]
    bo = block_offsets[:, :, None]
    kv_window = state_cache[bn, bo, col]
    score_window = state_cache[bn, bo, state_width + col]
    return kv_window, score_window, valid_mask


def _boundary_dest(
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int
    compress_ratio: int,
    num_slots: int,
) -> jax.Array:
    is_boundary = ((positions + 1) % compress_ratio) == 0
    store = is_boundary & (slot_mapping >= 0) & (kv_slot_mapping >= 0)
    return jnp.where(store, kv_slot_mapping, num_slots)


def compress_norm_rope_store(
    cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int (state-cache slots)
    block_table: jax.Array,  # [num_reqs, max_blocks] int (state pages)
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int (compressed-KV slots)
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """Compress, norm, RoPE, and write boundary KV into the shared cache."""
    state_view = state_cache

    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_view,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    compressed_pos = (positions // compress_ratio) * compress_ratio
    compressed = compress_norm_rope(
        kv_window=kv_window,
        score_window=score_window,
        valid_mask=valid_mask,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        compressed_pos=compressed_pos,
        rms_eps=rms_eps,
        rope_head_dim=rope_head_dim,
    )

    nope_dim = head_dim - rope_head_dim
    nope = compressed[..., :nope_dim]
    rope = compressed[..., nope_dim:]

    q, scale = quantize_fp8_ue8m0(nope, quant_block)
    rope_q = rope.astype(jnp.bfloat16)

    record = jnp.concatenate(
        [_to_byte_lane(q),
         _to_byte_lane(rope_q),
         _to_byte_lane(scale)],
        axis=-1)

    num_pages, rows, packing, width = cache.shape
    pad = width - record.shape[-1]
    if pad < 0:
        raise ValueError(
            f"packed record {record.shape[-1]}B exceeds cache width {width}B")
    record = jnp.pad(record, ((0, 0), (0, pad)))

    num_slots = num_pages * rows * packing
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)
    flat = cache.reshape(num_slots, width)
    flat = flat.at[dest].set(record, mode="drop")
    return flat.reshape(num_pages, rows, packing, width)


def compress_norm_rope_store_indexer(
    cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int (state-cache slots)
    block_table: jax.Array,  # [num_reqs, max_blocks] int (state pages)
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int (indexer-KV slots)
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """Indexer (head_dim=128) twin of ``compress_norm_rope_store``."""
    state_view = state_cache

    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_view,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    compressed_pos = (positions // compress_ratio) * compress_ratio
    compressed = compress_norm_rope(
        kv_window=kv_window,
        score_window=score_window,
        valid_mask=valid_mask,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        compressed_pos=compressed_pos,
        rms_eps=rms_eps,
        rope_head_dim=rope_head_dim,
    )

    q, scale = quantize_fp8_ue8m0(compressed, quant_block)

    record = jnp.concatenate([_to_byte_lane(q), _to_byte_lane(scale)], axis=-1)

    num_pages, rows, packing, width = cache.shape
    pad = width - record.shape[-1]
    if pad < 0:
        raise ValueError(
            f"packed record {record.shape[-1]}B exceeds cache width {width}B")
    record = jnp.pad(record, ((0, 0), (0, pad)))

    num_slots = num_pages * rows * packing
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)
    flat = cache.reshape(num_slots, width)
    flat = flat.at[dest].set(record, mode="drop")
    return flat.reshape(num_pages, rows, packing, width)
