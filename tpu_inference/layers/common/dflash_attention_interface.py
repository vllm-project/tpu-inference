# Copyright 2025 Google LLC
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
"""DFlash-specific attention helpers."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.layers.common.attention_metadata import AttentionMetadata


@functools.partial(jax.jit, static_argnames=("max_query_len", ))
def dflash_concat_attention(
    q: jax.Array,  # [T, N, H]
    k_ctx: jax.Array,  # [T, K, H]
    k_noise: jax.Array,  # [T, K, H]
    v_ctx: jax.Array,  # [T, K, H]
    v_noise: jax.Array,  # [T, K, H]
    attention_metadata: AttentionMetadata,
    *,
    max_query_len: int,
    sm_scale: float,
) -> jax.Array:
    """Computes DFlash concat attention outputs for query tokens.

    This path follows DFlash semantics by concatenating context/noise keys and
    values along token axis, while keeping query tokens as the noise stream.
    """
    if max_query_len <= 0:
        raise ValueError(f"{max_query_len=} must be positive.")
    if not (q.shape[0] == k_ctx.shape[0] == k_noise.shape[0] == v_ctx.shape[0]
            == v_noise.shape[0]):
        raise ValueError(
            "All DFlash attention streams must share the same token count.")

    num_tokens, num_heads, _ = q.shape
    num_kv_heads = k_ctx.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Expected num_heads divisible by num_kv_heads, got {num_heads=} {num_kv_heads=}"
        )

    kv_repeat = num_heads // num_kv_heads
    if kv_repeat > 1:
        k_ctx = jnp.repeat(k_ctx, kv_repeat, axis=1)
        k_noise = jnp.repeat(k_noise, kv_repeat, axis=1)
        v_ctx = jnp.repeat(v_ctx, kv_repeat, axis=1)
        v_noise = jnp.repeat(v_noise, kv_repeat, axis=1)

    pad_len = max_query_len
    q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
    k_ctx = jnp.pad(k_ctx, ((0, pad_len), (0, 0), (0, 0)))
    k_noise = jnp.pad(k_noise, ((0, pad_len), (0, 0), (0, 0)))
    v_ctx = jnp.pad(v_ctx, ((0, pad_len), (0, 0), (0, 0)))
    v_noise = jnp.pad(v_noise, ((0, pad_len), (0, 0), (0, 0)))

    query_start_loc = attention_metadata.query_start_loc
    req_lens = query_start_loc[1:] - query_start_loc[:-1]
    if attention_metadata.request_distribution is not None:
        num_reqs = jnp.minimum(attention_metadata.request_distribution[2],
                               req_lens.shape[0])
    else:
        num_reqs = req_lens.shape[0]

    arange_q = jnp.arange(max_query_len)
    arange_kv = jnp.arange(2 * max_query_len)

    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)
    outputs = jnp.zeros_like(q)

    def _body(i: int, current: jax.Array) -> jax.Array:
        start = query_start_loc[i]
        req_len = req_lens[i]
        req_len = jnp.clip(req_len, 0, max_query_len)

        q_blk = lax.dynamic_slice_in_dim(q, start, max_query_len, axis=0)
        k_ctx_blk = lax.dynamic_slice_in_dim(k_ctx,
                                             start,
                                             max_query_len,
                                             axis=0)
        k_noise_blk = lax.dynamic_slice_in_dim(k_noise,
                                               start,
                                               max_query_len,
                                               axis=0)
        v_ctx_blk = lax.dynamic_slice_in_dim(v_ctx,
                                             start,
                                             max_query_len,
                                             axis=0)
        v_noise_blk = lax.dynamic_slice_in_dim(v_noise,
                                               start,
                                               max_query_len,
                                               axis=0)

        k_blk = jnp.concatenate([k_ctx_blk, k_noise_blk], axis=0)
        v_blk = jnp.concatenate([v_ctx_blk, v_noise_blk], axis=0)

        q_valid = arange_q < req_len
        kv_valid_len = jnp.maximum(2 * req_len, 1)
        kv_valid = arange_kv < kv_valid_len

        logits = jnp.einsum("qnh,knh->nqk", q_blk.astype(jnp.float32),
                            k_blk.astype(jnp.float32))
        logits = logits * sm_scale
        logits = jnp.where(kv_valid[None, None, :], logits, mask_value)

        probs = jax.nn.softmax(logits, axis=-1).astype(v_blk.dtype)
        out_blk = jnp.einsum("nqk,knh->qnh", probs, v_blk)
        out_blk = jnp.where(q_valid[:, None, None], out_blk,
                            jnp.zeros_like(out_blk))

        return lax.dynamic_update_slice_in_dim(current, out_blk, start, axis=0)

    outputs = lax.fori_loop(0, num_reqs, _body, outputs)
    return outputs[:num_tokens]
