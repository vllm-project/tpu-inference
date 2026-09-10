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

import jax
import jax.numpy as jnp


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
        raise ValueError("rope_head_dim must be even and <= head_dim; got "
                         f"rope_head_dim={rope_head_dim}, head_dim={head_dim}")

    half_rope = rope_head_dim // 2  # 32
    num_pairs = head_dim // 2  # 256
    nope_pairs = num_pairs - half_rope  # 224

    pairs = x.reshape(*x.shape[:-1], num_pairs, 2)  # [tokens, 256, 2]
    even = pairs[..., 0]
    odd = pairs[..., 1]

    cos = cos_sin[..., :half_rope]  # [tokens, 32]
    sin = cos_sin[..., half_rope:rope_head_dim]  # [tokens, 32]

    pad_shape = (*cos.shape[:-1], nope_pairs)
    cos_full = jnp.concatenate([jnp.ones(pad_shape, x.dtype), cos],
                               axis=-1)  # [tokens, 256]
    sin_full = jnp.concatenate([jnp.zeros(pad_shape, x.dtype), sin],
                               axis=-1)  # [tokens, 256]

    new_even = even * cos_full - odd * sin_full  # [tokens, 256]
    new_odd = odd * cos_full + even * sin_full  # [tokens, 256]

    out = jnp.stack([new_even, new_odd], axis=-1)  # [tokens, 256, 2]
    return out.reshape(x.shape)  # [tokens, head_dim]


def gather_state_windows(
    state_cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,  # [num_reqs * block_table_stride]
    block_table_stride: int,
    token_to_req_indices: jax.Array,
    block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
):
    """Gather ``[kv_window, score_window, valid_mask]`` from the paged cache."""
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio

    start = positions - window + 1
    w_idx = jnp.arange(window)
    pos = start[:, None] + w_idx[None, :]
    valid_mask = pos >= 0

    safe_pos = jnp.maximum(pos, 0)
    req = token_to_req_indices[:, None]
    block_numbers = block_table[req * block_table_stride +
                                safe_pos // block_size]
    block_offsets = safe_pos % block_size

    head_offset = (w_idx >= compress_ratio).astype(jnp.int32) * head_dim
    col = head_offset[None, :, None] + jnp.arange(head_dim)[None, None, :]

    bn = block_numbers[:, :, None]
    bo = block_offsets[:, :, None]
    kv_window = state_cache[bn, bo, col]
    score_window = state_cache[bn, bo, state_width + col]
    return kv_window, score_window, valid_mask


def compress_norm_rope(
    kv_window: jax.Array,  # [N, W, D]
    score_window: jax.Array,  # [N, W, D]
    valid_mask: jax.Array,  # [N, W]
    rms_weight: jax.Array,  # [D]
    cos_sin_cache: jax.Array,  # [M, R]
    compressed_pos: jax.Array,  # [N]
    rms_eps: float,
    rope_head_dim: int,
) -> jax.Array:  # [N, D]
    """Window softmax-pool, RMSNorm, and interleaved RoPE."""
    neg_inf = jnp.array(-jnp.inf, dtype=score_window.dtype)
    # valid_mask[..., None]: [N, W, 1]
    # masked_score: [N, W, D]
    masked_score = jnp.where(valid_mask[..., None], score_window, neg_inf)
    weights = jax.nn.softmax(masked_score, axis=1)

    compressed_kv = jnp.sum(weights * kv_window,
                            axis=1)  # [num_tokens, head_dim]

    # variance: [N, 1]
    variance = jnp.mean(jnp.square(compressed_kv), axis=-1, keepdims=True)

    # rms_weight: [1, D]
    # normed: [N, D]
    normed = (compressed_kv * jax.lax.rsqrt(variance + rms_eps) *
              rms_weight[None, :].astype(compressed_kv.dtype))

    # cos_sin: [N, R]
    cos_sin = cos_sin_cache[compressed_pos]

    res = interleaved_rope(normed, cos_sin, rope_head_dim)

    return res


def _boundary_dest(
    positions: jax.Array,
    slot_mapping: jax.Array,
    kv_slot_mapping: jax.Array,
    compress_ratio: int,
    num_slots: int,
) -> jax.Array:
    is_boundary = ((positions + 1) % compress_ratio) == 0
    store = is_boundary & (slot_mapping >= 0) & (kv_slot_mapping >= 0)
    return jnp.where(store, kv_slot_mapping, num_slots)


def ref_compress_norm_rope_store(
    cache: jax.Array,  # [num_pages, page_size, slots_per_part_hbm, 128] uint8
    rope_cache: jax.Array,  # [num_pages, rope_page_size, 4, 128] uint8
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int (state-cache slots)
    block_table: jax.Array,  # [num_reqs * stride] int (state pages)
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int (compressed-KV slots)
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    block_table_stride: int,
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    is_quantized: bool = True,
    has_rope: bool = True,
    has_rope_cache: bool | None = None,
    state_cache: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Golden JAX reference for Kernel 2 (compress + store).

    ``state_cache`` is the array holding the f32 states; ``None`` means it
    shares ``cache``'s buffer (CSA / indexer).
    """
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width

    if has_rope_cache is None:
        has_rope_cache = is_quantized and has_rope

    # Internally reshape packed layouts to unpacked shapes for easy processing
    is_indexer_mode = (cache.shape[-1] == 256
                       and cache.shape[-2] == 4) or (cache.shape[-1] == 128
                                                     and cache.shape[-2] == 8)
    if is_indexer_mode:
        pack_factor = cache.shape[-2]
        last_dim = cache.shape[-1]
        unpacked_page_size = cache.shape[1] * pack_factor
        cache_for_unpack = cache.reshape(cache.shape[0], unpacked_page_size, 1,
                                         last_dim)
    else:
        cache_for_unpack = cache

    rope_page_size = rope_cache.shape[
        1] * 4 if has_rope_cache else rope_cache.shape[1]
    if has_rope_cache:
        rope_cache_for_unpack = rope_cache.reshape(rope_cache.shape[0],
                                                   rope_page_size, 1, 128)
    else:
        rope_cache_for_unpack = rope_cache

    # 1. Geometry setup
    if is_indexer_mode:
        num_pages, page_size, dummy, d2 = cache_for_unpack.shape
        assert dummy == 1
        slots_per_part_hbm = 0
        slot_bytes = d2
    else:
        num_pages, page_size, slots_per_part_hbm, d2 = cache.shape
        assert d2 == 128
        slot_bytes = slots_per_part_hbm * d2

    state_source = cache if state_cache is None else state_cache
    state_page_size = page_size if state_cache is None else state_cache.shape[1]
    slots_per_token = state_page_size // state_block_size
    f32_per_slot = state_dim // slots_per_token
    bytes_per_slot = f32_per_slot * 4
    assert bytes_per_slot == slot_bytes

    rope_num_pages, rope_page_size_unpacked, rope_d1, rope_d2 = rope_cache_for_unpack.shape
    assert rope_num_pages == num_pages
    assert rope_page_size_unpacked == rope_page_size
    assert rope_d1 == 1 and rope_d2 == 128

    # 2. Unpack state cache inline.
    # The state is read from the *physical* cache, whatever the trailing dims are:
    # one token owns ``state_rows`` rows of ``(hbm_pack, last_dim)`` bytes, and
    # within a row the ``hbm_pack`` sub-slots hold the four bytes of an f32, so a
    # single byte-transpose inverts the packing for every mode. This is the exact
    # inverse of the pack in ``ref_wkv_proj_and_save_state``.
    phys_pages, phys_page_size, hbm_pack, last_dim = state_source.shape
    assert hbm_pack == 4, f"expected 4 bytes per f32 in a slot row, got {hbm_pack}"
    state_rows = phys_page_size // state_block_size
    assert state_rows * hbm_pack * last_dim == state_dim * 4
    state_bytes_t = state_source.reshape(phys_pages, state_block_size,
                                         state_rows, hbm_pack,
                                         last_dim).transpose(0, 1, 2, 4, 3)
    state_view = jax.lax.bitcast_convert_type(state_bytes_t,
                                              jnp.float32).reshape(
                                                  phys_pages, state_block_size,
                                                  state_dim)

    # 3. Gather state windows
    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_view,
        positions=positions,
        block_table=block_table,
        block_table_stride=block_table_stride,
        token_to_req_indices=token_to_req_indices,
        block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    # 4. Compress, norm, RoPE
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

    # 5. Prepare NOPE record
    nope_dim = head_dim - rope_head_dim
    nope_store_dim = head_dim - rope_head_dim if has_rope_cache else head_dim
    if is_quantized:
        # Quantized (CSA / Indexer)
        q_full, scale_full = quantize_fp8_ue8m0(compressed, quant_block)
        q_bytes = jax.lax.bitcast_convert_type(q_full, jnp.uint8)
        scale_bytes = jax.lax.bitcast_convert_type(scale_full, jnp.uint8)

        q_bytes_flat = q_bytes.reshape(q_bytes.shape[0], -1)
        scale_bytes_flat = scale_bytes.reshape(scale_bytes.shape[0], -1)

        q_nope = q_bytes_flat[..., :nope_store_dim]
        nope_blocks = (nope_store_dim + quant_block - 1) // quant_block
        scale_nope = scale_bytes_flat[..., :nope_blocks]

        nope_record = jnp.concatenate([q_nope, scale_nope], axis=-1)
    else:
        # No quantization (HCA)
        compressed_bf16 = compressed.astype(jnp.bfloat16)
        slots_per_part = head_dim // 128
        compressed_bf16_tiled = compressed_bf16.reshape(
            compressed_bf16.shape[0], slots_per_part, 128)
        compressed_bytes = jax.lax.bitcast_convert_type(
            compressed_bf16_tiled, jnp.uint8)
        compressed_bytes_split = compressed_bytes.reshape(
            compressed_bytes.shape[0], slots_per_part, 128, 2)
        compressed_bytes_t = compressed_bytes_split.transpose(0, 1, 3, 2)
        nope_record = compressed_bytes_t.reshape(compressed_bytes.shape[0], -1)

    # Calculate output slots
    if is_quantized:
        total_bytes_out = 256 if head_dim == 128 else 512
    else:
        total_bytes_out = head_dim * 2

    if is_indexer_mode:
        slots_per_part_out = total_bytes_out // slot_bytes
    else:
        slots_per_part_out = total_bytes_out // (slots_per_part_hbm * 128)

    # Pad NOPE record to HBM write size
    nope_width = slots_per_part_out * slot_bytes
    nope_pad = nope_width - nope_record.shape[-1]
    if nope_pad < 0:
        raise ValueError(
            f"packed NOPE record {nope_record.shape[-1]}B exceeds cache width"
            f" {nope_width}B")
    nope_record = jnp.pad(nope_record, ((0, 0), (0, nope_pad)))

    # Split nope_record for multi-slot write if slots_per_part_out > 1
    nope_record_split = nope_record.reshape(nope_record.shape[0],
                                            slots_per_part_out, slot_bytes)

    # 6. Scatter NOPE into cache
    logical_slots_factor = 1
    page_size_physical = cache_for_unpack.shape[1]
    page_size_logical = page_size_physical * logical_slots_factor
    nope_num_slots_logical = num_pages * page_size_logical
    nope_num_slots_physical = num_pages * page_size_physical

    dest_logical = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                                  compress_ratio, nope_num_slots_logical)
    dest_nope_physical = dest_logical // logical_slots_factor

    flat_nope = cache_for_unpack.reshape(nope_num_slots_physical,
                                         slot_size_bytes := slot_bytes)
    flat_nope_padded = jnp.concatenate(
        [
            flat_nope,
            jnp.zeros((slots_per_part_out, slot_size_bytes), dtype=jnp.uint8),
        ],
        axis=0,
    )

    for o in range(slots_per_part_out):
        dest_o = jnp.where(
            dest_nope_physical < nope_num_slots_physical,
            dest_nope_physical + o,
            nope_num_slots_physical + o,
        )
        flat_nope_padded = flat_nope_padded.at[dest_o].set(
            nope_record_split[:, o])

    new_cache_unpacked = flat_nope_padded[:-slots_per_part_out].reshape(
        cache_for_unpack.shape)
    new_cache = new_cache_unpacked.reshape(cache.shape)

    # 7. Scatter ROPE into rope_cache (only if has_rope_cache)
    if has_rope_cache:
        rope = compressed[..., nope_dim:]
        rope_bf16 = rope.astype(jnp.bfloat16)
        rope_hi = ((jax.lax.bitcast_convert_type(rope_bf16, jnp.uint16) >> 8)
                   & 0xFF).astype(jnp.uint8)
        rope_lo = (jax.lax.bitcast_convert_type(rope_bf16, jnp.uint16)
                   & 0xFF).astype(jnp.uint8)
        rope_bytes_flat = jnp.concatenate([rope_hi, rope_lo], axis=-1)

        rope_width_padded = 128
        rope_pad = rope_width_padded - rope_bytes_flat.shape[-1]
        if rope_pad < 0:
            raise ValueError(
                f"packed ROPE record {rope_bytes_flat.shape[-1]}B exceeds cache width"
                f" {rope_width_padded}B")
        rope_record = jnp.pad(rope_bytes_flat, ((0, 0), (0, rope_pad)))

        rope_logical_slots_factor = 1
        rope_num_slots = rope_num_pages * rope_page_size

        dest_logical_rope = _boundary_dest(positions, slot_mapping,
                                           kv_slot_mapping, compress_ratio,
                                           rope_num_slots)

        flat_rope = rope_cache_for_unpack.reshape(rope_num_slots, rope_d1, 128)
        flat_rope_padded = jnp.concatenate(
            [flat_rope,
             jnp.zeros((1, rope_d1, 128), dtype=jnp.uint8)], axis=0)
        rope_record_dup = jnp.broadcast_to(
            rope_record[:, None, :], (rope_record.shape[0], rope_d1, 128))
        flat_rope_padded = flat_rope_padded.at[
            dest_logical_rope *
            rope_logical_slots_factor, :, :].set(rope_record_dup)
        new_rope_cache_unpacked = flat_rope_padded[:-1].reshape(
            rope_cache_for_unpack.shape)
        new_rope_cache = new_rope_cache_unpacked.reshape(rope_cache.shape)
        return new_cache, new_rope_cache
    else:
        return new_cache, rope_cache
