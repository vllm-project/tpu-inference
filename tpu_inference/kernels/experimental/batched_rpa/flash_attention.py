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
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, utils


def flash_attention(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H]
    v: jax.Array,  # [B, KV, S, H]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RPAConfig,
):
    """Flash attention kernel."""
    b, k_heads, tq, h = q.shape
    s = k.shape[2]

    if cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    qk = lax.dot_general(
        pltpu.einshape("bkth->(bk)th", q, True),
        pltpu.einshape("bksh->(bk)sh", k, True),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.serve.dtype_out)
    qk = pltpu.einshape("(bk)ts->bkts", qk, True, b=b)

    qk *= cfgs.model.sm_scale
    if cfgs.serve.scale_k is not None:
        qk *= cfgs.serve.scale_k
    if cfgs.serve.scale_q is not None:
        qk *= cfgs.serve.scale_q

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    qk_masked = []
    v_masked = []

    int_ty = cfgs.serve.int_ty

    for b_idx in range(cfgs.block.batch_size):
        kv_idx_b = (lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) +
                    processed_kv_len[b_idx])
        q_idx_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                   cfgs.model.num_q_heads_per_kv_head
                   ).astype(int_ty) + processed_q_len[b_idx]

        eff_kv_len_b = effective_kv_len[b_idx]
        mask_b = q_idx_b < eff_kv_len_b
        mask_b = jnp.logical_and(mask_b, q_idx_b >= kv_idx_b)

        if (sliding_window := cfgs.model.sliding_window) is not None:
            mask_b = jnp.logical_and(mask_b, q_idx_b
                                     < kv_idx_b + sliding_window)

        if not cfgs.mask_v:
            mask_b = jnp.logical_and(mask_b, kv_idx_b < eff_kv_len_b)

        qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))

        if cfgs.mask_v:
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
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))

    pv = lax.dot_general(
        pltpu.einshape("bkts->(bk)ts", p, True),
        pltpu.einshape("bksh->(bk)sh", v, True),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.serve.dtype_out)
    pv = pltpu.einshape("(bk)th->bkth", pv, True, b=b)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)
    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return m_next, l_next, o_next
