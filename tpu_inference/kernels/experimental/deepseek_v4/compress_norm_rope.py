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
"""Compress, norm, RoPE, and store DeepSeek-V4 KV cache at compression boundaries."""

import jax
import jax.numpy as jnp

from tpu_inference.layers.common.quantization import quantize_tensor


def interleaved_rope(
    x: jax.Array,            # [..., head_dim] fp32
    cos_sin: jax.Array,        # [..., rope_head_dim] fp32 ([cos | sin])
    rope_head_dim: int,
) -> jax.Array:
    """Interleaved (GPT-J) RoPE on the trailing ``rope_head_dim`` elements."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even; got {head_dim}")
    if rope_head_dim % 2 != 0 or rope_head_dim > head_dim:
        raise ValueError(
            f"rope_head_dim must be even and <= head_dim; got "
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
    kv_window: jax.Array,       # [num_tokens, window, head_dim] fp32
    score_window: jax.Array,    # [num_tokens, window, head_dim] fp32
    valid_mask: jax.Array,      # [num_tokens, window] bool
    rms_weight: jax.Array,      # [head_dim] fp32
    cos_sin_cache: jax.Array,   # [max_pos, rope_head_dim] fp32
    compressed_pos: jax.Array,  # [num_tokens] int
    rms_eps: float,
    rope_head_dim: int,
) -> jax.Array:
    """Window softmax-pool, RMSNorm, and interleaved RoPE."""
    neg_inf = jnp.array(-jnp.inf, dtype=score_window.dtype)
    masked_score = jnp.where(valid_mask[..., None], score_window, neg_inf)
    weights = jax.nn.softmax(masked_score, axis=1)

    compressed_kv = jnp.sum(weights * kv_window, axis=1)

    variance = jnp.mean(jnp.square(compressed_kv), axis=-1, keepdims=True)
    normed = compressed_kv * jax.lax.rsqrt(variance + rms_eps) * rms_weight

    cos_sin = cos_sin_cache[compressed_pos]
    return interleaved_rope(normed, cos_sin, rope_head_dim)


def gather_state_windows(
    state_cache: jax.Array,         # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,           # [num_tokens] int
    block_table: jax.Array,         # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
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

    safe_pos = jnp.where(valid_mask, pos, 0)
    req = token_to_req_indices[:, None]
    block_numbers = block_table[req, safe_pos // block_size]
    block_offsets = safe_pos % block_size

    # C4 overlap: slots >= compress_ratio read the second head slice.
    head_offset = (w_idx >= compress_ratio).astype(jnp.int32) * head_dim
    col = head_offset[None, :, None] + jnp.arange(head_dim)[None, None, :]

    bn = block_numbers[:, :, None]
    bo = block_offsets[:, :, None]
    kv_window = state_cache[bn, bo, col]
    score_window = state_cache[bn, bo, state_width + col]
    return kv_window, score_window, valid_mask


def _boundary_dest(
    positions: jax.Array,       # [num_tokens] int
    slot_mapping: jax.Array,      # [num_tokens] int
    kv_slot_mapping: jax.Array,   # [num_tokens] int
    compress_ratio: int,
    num_slots: int,
) -> jax.Array:
    is_boundary = ((positions + 1) % compress_ratio) == 0
    store = is_boundary & (slot_mapping >= 0) & (kv_slot_mapping >= 0)
    return jnp.where(store, kv_slot_mapping, num_slots)


def compress_norm_rope_store(
    state_cache: jax.Array,         # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,           # [num_tokens] int
    slot_mapping: jax.Array,        # [num_tokens] int
    block_table: jax.Array,         # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,     # [num_tokens] int
    nope_cache: jax.Array,          # [num_blocks, block_size, nope_dim] fp8
    rope_cache: jax.Array,          # [num_blocks, block_size, rope_head_dim] bf16
    scale_cache: jax.Array,         # [num_blocks, block_size, nope_dim//qb] f32
    rms_weight: jax.Array,          # [head_dim] fp32
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32
    block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """head_dim=512: fp8 nope + bf16 rope at compression boundaries."""
    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_cache,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=block_size,
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

    q, scale = quantize_tensor(
        jnp.float8_e4m3fn, nope, axis=-1, block_size=quant_block)
    rope_q = rope.astype(jnp.bfloat16)

    num_blocks, blk, _ = nope_cache.shape
    num_slots = num_blocks * blk
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)

    n_qb = scale.shape[-1]
    flat_nope = nope_cache.reshape(num_slots, nope_dim)
    flat_rope = rope_cache.reshape(num_slots, rope_head_dim)
    flat_scale = scale_cache.reshape(num_slots, n_qb)
    flat_nope = flat_nope.at[dest].set(q, mode="drop")
    flat_rope = flat_rope.at[dest].set(rope_q, mode="drop")
    flat_scale = flat_scale.at[dest].set(scale, mode="drop")
    return (flat_nope.reshape(num_blocks, blk, nope_dim),
            flat_rope.reshape(num_blocks, blk, rope_head_dim),
            flat_scale.reshape(num_blocks, blk, n_qb))


def compress_norm_rope_store_indexer(
    state_cache: jax.Array,         # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,           # [num_tokens] int
    slot_mapping: jax.Array,        # [num_tokens] int
    block_table: jax.Array,         # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,     # [num_tokens] int
    fp8_cache: jax.Array,           # [num_blocks, block_size, head_dim] fp8
    scale_cache: jax.Array,         # [num_blocks, block_size, head_dim//qb] f32
    rms_weight: jax.Array,          # [head_dim] fp32
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32
    block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """head_dim=128: whole-head fp8 + one scale at boundaries."""
    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_cache,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=block_size,
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

    q, scale = quantize_tensor(
        jnp.float8_e4m3fn, compressed, axis=-1, block_size=quant_block)

    num_blocks, blk, _ = fp8_cache.shape
    num_slots = num_blocks * blk
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)

    n_qb = scale.shape[-1]
    flat_fp8 = fp8_cache.reshape(num_slots, head_dim)
    flat_scale = scale_cache.reshape(num_slots, n_qb)
    flat_fp8 = flat_fp8.at[dest].set(q, mode="drop")
    flat_scale = flat_scale.at[dest].set(scale, mode="drop")
    return (flat_fp8.reshape(num_blocks, blk, head_dim),
            flat_scale.reshape(num_blocks, blk, n_qb))
