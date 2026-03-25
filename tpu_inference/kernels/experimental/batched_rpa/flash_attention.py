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
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import \
    schedule as rpa_schedule

_NUM_LANES = 128


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
    qk = qk.reshape((b, k_heads, tq, s)).astype(config.out_dtype)

    qk *= config.sm_scale
    if config.k_scale is not None:
        qk *= config.k_scale
    if config.q_scale is not None:
        qk *= config.q_scale

    if config.soft_cap is not None:
        qk = config.soft_cap * jnp.tanh(qk / config.soft_cap)

    qk_masked = []
    v_masked = []

    int_ty = jnp.int32
    if (rpa_schedule.get_dtype_packing(config.q_dtype) != 1
            and pltpu.get_tpu_info().generation >= 6):
        int_ty = jnp.int16

    for b_idx in range(config.batch_size):
        kv_idx_b = (lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) +
                    processed_kv_len[b_idx])
        q_idx_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                   config.num_q_heads_per_kv_head
                   ).astype(int_ty) + processed_q_len[b_idx]

        eff_kv_len_b = effective_kv_len[b_idx]
        mask_b = q_idx_b < eff_kv_len_b
        mask_b &= q_idx_b >= kv_idx_b

        if config.sliding_window is not None:
            mask_b &= q_idx_b < kv_idx_b + config.sliding_window

        if not config.mask_v:
            mask_b &= kv_idx_b < eff_kv_len_b

        qk_masked.append(jnp.where(mask_b, qk[b_idx], config.mask_value))

        if config.mask_v:
            kv_idx_v = (lax.broadcasted_iota(int_ty, (k_heads, s, h), 1) +
                        processed_kv_len[b_idx])
            v_mask_b = kv_idx_v < eff_kv_len_b
            v_masked.append(jnp.where(v_mask_b, v[b_idx], 0))
        else:
            v_masked.append(v[b_idx])

    qk = jnp.stack(qk_masked, axis=0)
    v = jnp.stack(v_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - broadcast_minor(m_next, qk.shape))

    pv = lax.dot_general(
        p.reshape((b * k_heads, tq, s)),
        v.reshape((b * k_heads, s, h)),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    pv = pv.reshape((b, k_heads, tq, h)).astype(config.out_dtype)

    if config.v_scale is not None:
        pv *= config.v_scale

    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=config.out_dtype)
    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    o_next = broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return m_next, l_next, o_next
