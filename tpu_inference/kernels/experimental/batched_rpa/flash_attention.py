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

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from tpu_inference.kernels.experimental.batched_rpa import \
    schedule as rpa_schedule


def align_to(a, b):
    return pl.cdiv(a, b) * b


def broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])],
                           axis=-1)[..., :shape[-1]]


def flash_attention(
    q,  # [B, KV, TQ, H]
    k,  # [B, KV, S, H]
    v,  # [B, KV, S, H]
    o_prev,  # [B, KV, TQ, H]
    m_prev,  # [B, KV, TQ, 128]
    l_prev,  # [B, KV, TQ, 128]
    *,
    processed_q_len,  # [B]
    processed_kv_len,  # [B]
    effective_kv_len,  # [B]
    config: rpa_schedule.RPAConfig,
):
    """Flash attention kernel."""
    b, k_heads, tq, h = q.shape
    s = k.shape[2]

    if config.q_scale is not None:
        q = q / config.q_scale
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    qk = lax.dot_general(
        q.reshape((b * k_heads, tq, h)),
        k.reshape((b * k_heads, s, h)),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    qk = qk.reshape((b, k_heads, tq, s))

    qk *= config.sm_scale
    if config.k_scale is not None:
        qk *= config.k_scale
    if config.q_scale is not None:
        qk *= config.q_scale

    if config.soft_cap is not None:
        qk = config.soft_cap * jnp.tanh(qk / config.soft_cap)

    qk_seq_masks: list[jnp.ndarray] = []
    for i in range(b):
        kv_idx_i = processed_kv_len[i] + lax.broadcasted_iota(
            jnp.int32, (k_heads, tq, s), 2)
        q_idx_i = (processed_q_len[i] +
                   lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                   config.num_q_heads_per_kv_head)
        mask_i = kv_idx_i < effective_kv_len[i]
        mask_i &= q_idx_i < effective_kv_len[i]
        mask_i &= q_idx_i >= kv_idx_i
        if config.sliding_window is not None:
            mask_i &= q_idx_i < kv_idx_i + config.sliding_window
        qk_seq_masks.append(mask_i)
    mask = jnp.stack(qk_seq_masks, axis=0)

    qk = jnp.where(mask, qk, config.mask_value)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - broadcast_minor(m_next, qk.shape))

    pv = lax.dot_general(
        p.reshape((b * k_heads, tq, s)),
        v.reshape((b * k_heads, s, h)),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    pv = pv.reshape((b, k_heads, tq, h))

    if config.v_scale is not None:
        pv *= config.v_scale

    p_rowsum = jnp.sum(p, axis=-1, keepdims=True)
    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    o_next = broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return m_next, l_next, o_next.astype(o_prev.dtype)
