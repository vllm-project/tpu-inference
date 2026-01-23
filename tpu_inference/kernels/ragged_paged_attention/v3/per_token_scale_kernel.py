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
"""TPU-Friendly Ragged Paged Attention kernel.
This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools

import jax
import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_q_len",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
    ),
)
def ref_ragged_paged_attention_pate(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    max_q_len: int | None = None,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    if max_q_len is None:
        max_q_len = queries.shape[0]

    # Initialization
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]

    pad_width = ((0, max_q_len), (0, 0), (0, 0))
    queries_padded = jnp.pad(queries, pad_width)

    merged_kv = merge_kv(keys, values)
    merged_kv_padded = jnp.pad(merged_kv,
                               ((0, max_q_len), (0, 0), (0, 0), (0, 0)))

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    output_buffer = jnp.zeros(
        (queries.shape[0] + max_q_len, actual_num_q_heads, actual_head_dim),
        dtype=queries.dtype,
    )

    # "Kernel" to run via fori loop
    def body_fn(i, val):
        curr_kv_cache, curr_outputs = val

        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        actual_q_len = q_end - q_start
        kv_len = kv_lens[i]

        q_chunk = jax.lax.dynamic_slice_in_dim(queries_padded,
                                               q_start,
                                               max_q_len,
                                               axis=0)
        q_chunk_sliced = q_chunk[:, :, :actual_head_dim]
        new_kv_chunk = jax.lax.dynamic_slice_in_dim(merged_kv_padded,
                                                    q_start,
                                                    max_q_len,
                                                    axis=0)

        indices_start = i * pages_per_seq
        seq_page_indices = jax.lax.dynamic_slice(page_indices,
                                                 (indices_start, ),
                                                 (pages_per_seq, ))
        gathered_kv = curr_kv_cache[seq_page_indices]
        gathered_shape = gathered_kv.shape
        gathered_kv_flat = gathered_kv.reshape(-1, *gathered_shape[-3:])

        chunk_mask = jnp.arange(max_q_len) < actual_q_len
        chunk_mask_expanded = chunk_mask[:, None, None, None]
        new_kv_chunk_clean = jnp.where(chunk_mask_expanded, new_kv_chunk, 0.0)

        update_start_idx = kv_len - actual_q_len
        scatter_indices = update_start_idx + jnp.arange(max_q_len)

        safe_scatter_indices = jnp.where(
            chunk_mask,
            scatter_indices,
            gathered_kv_flat.shape[0] + 1  # Force OOB
        )

        gathered_kv_flat = gathered_kv_flat.at[safe_scatter_indices].set(
            new_kv_chunk_clean, mode="drop")

        curr_kv_cache = curr_kv_cache.at[seq_page_indices].set(
            gathered_kv_flat.reshape(gathered_shape))

        # Prepare K/V
        kv_reshaped = gathered_kv_flat.reshape(-1, num_kv_heads_x2, head_dim)
        kv_reshaped = kv_reshaped[:, :actual_num_kv_heads * 2, :]
        kv_reshaped = kv_reshaped.reshape(-1, actual_num_kv_heads,
                                          head_dim * 2)
        k = kv_reshaped[:, :, :head_dim][:, :, :actual_head_dim]
        v = kv_reshaped[:, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.nan_to_num(k, nan=0.0)
        v = jnp.nan_to_num(v, nan=0.0)
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        # Prepare Q
        q_curr = q_chunk_sliced
        if q_scale is not None:
            q_curr = q_curr / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q_curr = jnp.clip(q_curr, min=minval, max=maxval)
            q_curr = q_curr.astype(k.dtype)

        # Attention
        attn = jnp.einsum("qhd,khd->hqk",
                          q_curr,
                          k,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        # Masking
        q_idxs = jnp.arange(max_q_len)
        kv_idxs = jnp.arange(k.shape[0])

        q_span = (kv_len - actual_q_len) + q_idxs[:, None]
        kv_span = kv_idxs[None, :]

        causal_mask = q_span >= kv_span
        valid_q_mask = q_idxs[:, None] < actual_q_len
        valid_k_mask = kv_idxs[None, :] < kv_len

        mask = causal_mask & valid_q_mask & valid_k_mask
        if sliding_window is not None:
            mask = mask & (q_span - sliding_window < kv_span)

        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out_chunk = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        if v_scale is not None:
            out_chunk *= v_scale

        # Zero out invalid q rows in output
        out_chunk = jnp.where(q_idxs[:, None, None] < actual_q_len, out_chunk,
                              0.0)

        # Update output buffer
        curr_outputs = jax.lax.dynamic_update_slice_in_dim(curr_outputs,
                                                           out_chunk,
                                                           q_start,
                                                           axis=0)

        return curr_kv_cache, curr_outputs

    num_seqs = distribution[-1]
    init_val = (kv_cache, output_buffer)
    final_kv_cache, final_outputs = jax.lax.fori_loop(0, num_seqs, body_fn,
                                                      init_val)

    # Remove padding before returning
    return final_outputs[:queries.shape[0]], final_kv_cache


def quantize_kv_per_token(key: jax.Array, value: jax.Array,
                          kv_cache_quantized_dtype: jnp.dtype):
    """
    Quantizes K and V dynamically per token.
    Returns the quantized tensors AND the scales needed to dequantize them.
    Args:
        key: The key tensor to quantize of shape [max_num_tokens, actual_num_kv_heads, actual_head_dim].
        value: The value tensor to quantize of shape [max_num_tokens, actual_num_kv_heads, actual_head_dim].
        kv_cache_quantized_dtype: The dtype to quantize the key and value tensors to.
    Returns:
        key_q: The quantized key tensor of shape [max_num_tokens, actual_num_kv_heads, actual_head_dim].
        value_q: The quantized value tensor of shape [max_num_tokens, actual_num_kv_heads, actual_head_dim].
        k_scale: The scale to quantize the key tensor by of shape [max_num_tokens, actual_num_kv_heads, 1]
        v_scale: The scale to quantize the value tensor by of shape [max_num_tokens, actual_num_kv_heads, 1]
    """
    dtype_info = jnp.finfo(kv_cache_quantized_dtype)
    minval, maxval = float(dtype_info.min), float(dtype_info.max)
    q_max = float(dtype_info.max)  # e.g., 127 for int8

    # 1. Calculate max absolute value per token (keepdims to broadcast later)
    # Shape becomes: [Batch, Seq, Heads, 1]
    k_abs_max = jnp.max(jnp.abs(key), axis=-1, keepdims=True)
    v_abs_max = jnp.max(jnp.abs(value), axis=-1, keepdims=True)

    # 2. Calculate Scale (add epsilon to avoid div by zero)
    # Formula: Scale = Max_Val / Q_Max
    epsilon = 1e-5
    k_scale = jnp.maximum(k_abs_max, epsilon) / q_max
    v_scale = jnp.maximum(v_abs_max, epsilon) / q_max

    # 3. Quantize
    # K_quant = Clip(Round(K_float / K_scale))
    key = key.astype(jnp.float32)
    key_q = key / k_scale
    key_q = jnp.clip(key_q, minval, maxval)
    key_q = key_q.astype(kv_cache_quantized_dtype)

    value = value.astype(jnp.float32)
    value_q = value / v_scale
    value_q = jnp.clip(value_q, minval, maxval)
    value_q = value_q.astype(kv_cache_quantized_dtype)

    return key_q, value_q, k_scale.astype(jnp.float32), v_scale.astype(
        jnp.float32)


def merge_kv(
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v],
                   axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2,
                                    actual_head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_q_len",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
    ),
)
def ref_ragged_paged_attention_pate_per_token(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    k_scale_cache: jax.Array,  # Added argument
    v_scale_cache: jax.Array,  # Added argument
    *,
    max_q_len: int | None = None,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    if max_q_len is None:
        max_q_len = queries.shape[0]

    # Initialization
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]

    pad_width = ((0, max_q_len), (0, 0), (0, 0))
    queries_padded = jnp.pad(queries, pad_width)

    # 1. Quantize Keys/Values and extract per-token scales
    # k_q, v_q: [max_num_tokens, actual_num_kv_heads, head_dim]
    # k_scale_new, v_scale_new: [max_num_tokens, actual_num_kv_heads, 1]
    k_q, v_q, k_scale_new, v_scale_new = quantize_kv_per_token(
        keys, values, kv_cache.dtype)

    merged_kv = merge_kv(k_q, v_q)

    # Pad merged KV
    merged_kv_padded = jnp.pad(merged_kv,
                               ((0, max_q_len), (0, 0), (0, 0), (0, 0)))

    # Pad Scales
    # Shape: [N, H_kv, 1] -> Pad N dimension
    scale_pad_width = ((0, max_q_len), (0, 0), (0, 0))
    k_scale_padded = jnp.pad(k_scale_new, scale_pad_width)
    v_scale_padded = jnp.pad(v_scale_new, scale_pad_width)

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    output_buffer = jnp.zeros(
        (queries.shape[0] + max_q_len, actual_num_q_heads, actual_head_dim),
        dtype=queries.dtype,
    )

    # "Kernel" to run via fori loop
    def body_fn(i, val):
        # Unpack state, now including scale caches
        curr_kv_cache, curr_outputs, curr_k_scale_cache, curr_v_scale_cache = val

        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        actual_q_len = q_end - q_start
        kv_len = kv_lens[i]

        # ---------------------------------------------------------
        # Slice Inputs (Q, KV, K_scale, V_scale)
        # ---------------------------------------------------------
        q_chunk = jax.lax.dynamic_slice_in_dim(queries_padded,
                                               q_start,
                                               max_q_len,
                                               axis=0)
        q_chunk_sliced = q_chunk[:, :, :actual_head_dim]

        new_kv_chunk = jax.lax.dynamic_slice_in_dim(merged_kv_padded,
                                                    q_start,
                                                    max_q_len,
                                                    axis=0)

        new_k_scale_chunk = jax.lax.dynamic_slice_in_dim(k_scale_padded,
                                                         q_start,
                                                         max_q_len,
                                                         axis=0)

        new_v_scale_chunk = jax.lax.dynamic_slice_in_dim(v_scale_padded,
                                                         q_start,
                                                         max_q_len,
                                                         axis=0)

        # ---------------------------------------------------------
        # Gather Caches (KV, K_scale, V_scale)
        # ---------------------------------------------------------
        indices_start = i * pages_per_seq
        seq_page_indices = jax.lax.dynamic_slice(page_indices,
                                                 (indices_start, ),
                                                 (pages_per_seq, ))

        # KV Cache Gather
        gathered_kv = curr_kv_cache[seq_page_indices]
        gathered_shape = gathered_kv.shape
        gathered_kv_flat = gathered_kv.reshape(-1, *gathered_shape[-3:])

        # Scale Cache Gather
        # Shape in cache: [pages, page_size, num_kv_heads, 1]
        gathered_k_scales = curr_k_scale_cache[seq_page_indices]
        g_scale_shape = gathered_k_scales.shape
        gathered_k_scales_flat = gathered_k_scales.reshape(
            -1, *g_scale_shape[-2:])

        gathered_v_scales = curr_v_scale_cache[seq_page_indices]
        gathered_v_scales_flat = gathered_v_scales.reshape(
            -1, *g_scale_shape[-2:])

        # ---------------------------------------------------------
        # Update Caches
        # ---------------------------------------------------------
        chunk_mask = jnp.arange(max_q_len) < actual_q_len

        # Prepare Mask for KV
        chunk_mask_expanded = chunk_mask[:, None, None, None]
        new_kv_chunk_clean = jnp.where(chunk_mask_expanded, new_kv_chunk, 0.0)

        # Prepare Mask for Scales
        chunk_mask_scales = chunk_mask[:, None, None]
        new_k_scale_chunk_clean = jnp.where(chunk_mask_scales,
                                            new_k_scale_chunk, 0.0)
        new_v_scale_chunk_clean = jnp.where(chunk_mask_scales,
                                            new_v_scale_chunk, 0.0)

        update_start_idx = kv_len - actual_q_len
        scatter_indices = update_start_idx + jnp.arange(max_q_len)

        # Determine safe indices
        safe_scatter_indices = jnp.where(
            chunk_mask,
            scatter_indices,
            gathered_kv_flat.shape[0] + 1  # Force OOB
        )

        # Apply Updates via Scatter
        gathered_kv_flat = gathered_kv_flat.at[safe_scatter_indices].set(
            new_kv_chunk_clean, mode="drop")

        gathered_k_scales_flat = gathered_k_scales_flat.at[
            safe_scatter_indices].set(new_k_scale_chunk_clean, mode="drop")

        gathered_v_scales_flat = gathered_v_scales_flat.at[
            safe_scatter_indices].set(new_v_scale_chunk_clean, mode="drop")

        # Write back to main caches
        curr_kv_cache = curr_kv_cache.at[seq_page_indices].set(
            gathered_kv_flat.reshape(gathered_shape))

        curr_k_scale_cache = curr_k_scale_cache.at[seq_page_indices].set(
            gathered_k_scales_flat.reshape(g_scale_shape))

        curr_v_scale_cache = curr_v_scale_cache.at[seq_page_indices].set(
            gathered_v_scales_flat.reshape(g_scale_shape))

        # ---------------------------------------------------------
        # Prepare K/V and Scales for Attention
        # ---------------------------------------------------------
        # 1. Extract K/V
        kv_reshaped = gathered_kv_flat.reshape(-1, num_kv_heads_x2, head_dim)
        kv_reshaped = kv_reshaped[:, :actual_num_kv_heads * 2, :]
        kv_reshaped = kv_reshaped.reshape(-1, actual_num_kv_heads,
                                          head_dim * 2)
        k = kv_reshaped[:, :, :head_dim][:, :, :actual_head_dim]
        v = kv_reshaped[:, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.nan_to_num(k, nan=0.0)
        v = jnp.nan_to_num(v, nan=0.0)

        # GQA Repeat K/V
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        # 2. Extract Scales
        # gathered_k_scales_flat is [tokens, num_kv_heads, 1]
        k_scales_active = gathered_k_scales_flat
        v_scales_active = gathered_v_scales_flat

        # GQA Repeat Scales
        k_scales_active = jnp.repeat(k_scales_active,
                                     actual_num_q_heads_per_kv_head,
                                     axis=1)
        v_scales_active = jnp.repeat(v_scales_active,
                                     actual_num_q_heads_per_kv_head,
                                     axis=1)

        # ---------------------------------------------------------
        # Prepare Q
        # ---------------------------------------------------------
        q_curr = q_chunk_sliced
        if q_scale is not None:
            q_curr = q_curr / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q_curr = jnp.clip(q_curr, min=minval, max=maxval)
            q_curr = q_curr.astype(k.dtype)

        # ---------------------------------------------------------
        # Attention Calculation
        # ---------------------------------------------------------
        # Q shape: [q_len, num_q_heads, head_dim]
        # K shape: [kv_len_padded, num_q_heads, head_dim]
        attn = jnp.einsum("qhd,khd->hqk",
                          q_curr,
                          k,
                          preferred_element_type=jnp.float32)

        # Apply K Per-Token Scales
        # attn: [num_q_heads, q_len, kv_len]
        # k_scales_active: [kv_len, num_q_heads, 1]
        # We need k_scale to be broadcastable to [num_q_heads, q_len, kv_len]
        # Transpose k_scales to [num_q_heads, 1, kv_len]
        k_scales_T = jnp.transpose(k_scales_active, (1, 2, 0))
        attn = attn * k_scales_T

        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        # Masking
        q_idxs = jnp.arange(max_q_len)
        kv_idxs = jnp.arange(k.shape[0])

        q_span = (kv_len - actual_q_len) + q_idxs[:, None]
        kv_span = kv_idxs[None, :]

        causal_mask = q_span >= kv_span
        valid_q_mask = q_idxs[:, None] < actual_q_len
        valid_k_mask = kv_idxs[None, :] < kv_len

        mask = causal_mask & valid_q_mask & valid_k_mask
        if sliding_window is not None:
            mask = mask & (q_span - sliding_window < kv_span)

        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply V Per-Token Scales
        # v_scales_active: [kv_len, num_q_heads, 1]
        # Transpose to [num_q_heads, 1, kv_len]
        v_scales_T = jnp.transpose(v_scales_active, (1, 2, 0))
        attn = attn * v_scales_T

        out_chunk = jnp.einsum("hqk,khd->qhd",
                               attn,
                               v,
                               preferred_element_type=jnp.float32).astype(
                                   queries.dtype)
        if v_scale is not None:
            out_chunk *= v_scale

        # Zero out invalid q rows in output
        out_chunk = jnp.where(q_idxs[:, None, None] < actual_q_len, out_chunk,
                              0.0)

        # Update output buffer
        curr_outputs = jax.lax.dynamic_update_slice_in_dim(curr_outputs,
                                                           out_chunk,
                                                           q_start,
                                                           axis=0)

        return curr_kv_cache, curr_outputs, curr_k_scale_cache, curr_v_scale_cache

    num_seqs = distribution[-1]
    # Initialize loop with scale caches
    init_val = (kv_cache, output_buffer, k_scale_cache, v_scale_cache)
    final_kv_cache, final_outputs, final_k_scale_cache, final_v_scale_cache = jax.lax.fori_loop(
        0, num_seqs, body_fn, init_val)

    # Remove padding before returning.
    # Returning updated scale caches to match the non-jit version's style.
    return final_outputs[:queries.shape[
        0]], final_kv_cache, final_k_scale_cache, final_v_scale_cache


def ref_ragged_paged_attention_per_token_non_jit(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    k_scale_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, 1]
    v_scale_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, 1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    # dynamic_validate_inputs(
    #     queries,
    #     keys,
    #     values,
    #     kv_cache,
    #     kv_lens,
    #     page_indices,
    #     cu_q_lens,
    #     distribution,
    #     sm_scale=sm_scale,
    #     sliding_window=sliding_window,
    #     soft_cap=soft_cap,
    #     mask_value=mask_value,
    #     # q_scale=q_scale,
    #     # k_scale=k_scale,
    #     # v_scale=v_scale,
    # )
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]

    # k/v_q have shape [max_num_tokens, actual_num_kv_heads, head_dim]
    # k_scale/v_scale have shape [max_num_tokens, actual_num_kv_heads, 1]
    k_q, v_q, k_scale, v_scale = \
        quantize_kv_per_token(keys, values, kv_cache.dtype)

    merged_kv = merge_kv(k_q, v_q)

    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    # outputs = []
    result = queries

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len:kv_len].set(
            merged_kv[q_start:q_end])
        kv_cache = kv_cache.at[indices].set(
            gathered_kv.reshape(gathered_shape))

        # 1. Update K Scales
        gathered_k_scales = k_scale_cache[indices]
        g_k_shape = gathered_k_scales.shape
        gathered_k_scales = gathered_k_scales.reshape(-1, *g_k_shape[-2:])

        gathered_k_scales = gathered_k_scales.at[kv_len - q_len:kv_len].set(
            k_scale[q_start:q_end])
        k_scale_cache = k_scale_cache.at[indices].set(
            gathered_k_scales.reshape(g_k_shape))

        # 2. Update V Scales (Similar logic)
        gathered_v_scales = v_scale_cache[indices]
        g_v_shape = gathered_v_scales.shape
        gathered_v_scales = gathered_v_scales.reshape(-1, *g_v_shape[-2:])
        gathered_v_scales = gathered_v_scales.at[kv_len - q_len:kv_len].set(
            v_scale[q_start:q_end])
        v_scale_cache = v_scale_cache.at[indices].set(
            gathered_v_scales.reshape(g_v_shape))

        kv = gathered_kv.reshape(
            -1, num_kv_heads_x2,
            head_dim)[:, :actual_num_kv_heads * 2, :].reshape(
                -1, actual_num_kv_heads, head_dim * 2)
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        k_scales_active = gathered_k_scales.reshape(-1, actual_num_kv_heads,
                                                    1)[:kv_len]
        v_scales_active = gathered_v_scales.reshape(-1, actual_num_kv_heads,
                                                    1)[:kv_len]

        k_scales_active = jnp.repeat(k_scales_active,
                                     actual_num_q_heads_per_kv_head,
                                     axis=1)
        v_scales_active = jnp.repeat(v_scales_active,
                                     actual_num_q_heads_per_kv_head,
                                     axis=1)

        # TODO: add q_scale
        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32)
        # the attn output will have shape [num_q_heads, q_len, kv_len]
        # but the k_scales_active will have shape [kv_len, num_q_heads, 1]
        # so transpose so that the scale will have shape [num_q_heads, 1, kv_len]
        k_scales_T = jnp.transpose(k_scales_active, (1, 2, 0))

        attn = attn * k_scales_T * sm_scale

        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1)

        v_scales_T = jnp.transpose(v_scales_active, (1, 2, 0))
        attn = attn * v_scales_T
        out = jnp.einsum("hqk,khd->qhd",
                         attn,
                         v,
                         preferred_element_type=jnp.float32).astype(
                             queries.dtype)
        result = result.at[q_start:q_end].set(out)

    return result, kv_cache, k_scale_cache, v_scale_cache


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32 +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq]
        align_to(max_num_seqs * pages_per_seq, 128) * 32 +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32 +
        # distribution_ref: i32[3]
        128 * 32 +
        # sem_ids_ref: i32[3]
        128 * 32 +
        # bo_ids_ref: i32[4]
        128 * 32 +
        # bkv_update_ids_ref: i32[6]
        128 * 32)
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * num_kv_heads_x2 * head_dim) * (32 // kv_packing) +
        # bq_x2_ref + bo_x2_ref
        2 * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head *
             head_dim) * (32 // q_packing) +
        # l_ref + m_ref
        2 *
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32 +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) *
        32)
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


@jax.jit(static_argnames=("mask_value", "soft_cap", "sliding_window"))
def _attn_math_block(
    q,  # [q_len, num_q_heads, head_dim]
    k,  # [kv_len, num_kv_heads, head_dim] (Quantized or Float)
    v,  # [kv_len, num_kv_heads, head_dim] (Quantized or Float)
    k_scale,  # [kv_len, num_kv_heads, 1] (Per-Token) OR Scalar (Per-Tensor)
    v_scale,  # [kv_len, num_kv_heads, 1] (Per-Token) OR Scalar (Per-Tensor)
    sm_scale,
    mask_value,
    soft_cap,
    sliding_window,
):
    # 3. QK Dot Product
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    k_scales_T = jnp.transpose(k_scale, (1, 2, 0))
    attn = attn * k_scales_T * sm_scale

    # 4. Masking
    q_len = q.shape[0]
    kv_len = k.shape[0]

    # Construct masks using relative positions
    # q_pos + (kv_len - q_len) aligns the query to the end of the context
    q_idx = jnp.arange(q_len)[:, None]
    k_idx = jnp.arange(kv_len)[None, :]
    q_pos = q_idx + (kv_len - q_len)

    mask = q_pos < k_idx  # Causal
    if sliding_window is not None:
        mask = jnp.logical_or(mask, q_pos - sliding_window >= k_idx)

    if soft_cap is not None:
        attn = soft_cap * jnp.tanh(attn / soft_cap)

    attn = jnp.where(mask, mask_value, attn)
    attn = jax.nn.softmax(attn, axis=-1)

    v_scales_T = jnp.transpose(v_scale, (1, 2, 0))
    attn = attn * v_scales_T

    out = jnp.einsum("hqk,khd->qhd",
                     attn,
                     v,
                     preferred_element_type=jnp.float32)

    return out.astype(q.dtype)
