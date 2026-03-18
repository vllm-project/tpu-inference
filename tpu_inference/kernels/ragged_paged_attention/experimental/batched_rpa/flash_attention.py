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

import einshape
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.ragged_paged_attention.experimental.rpa_batched import \
    schedule as rpa_schedule

_NUM_LANES = 128


def cdiv(a, b):
    return (a + b - 1) // b


def align_to(a, b):
    return cdiv(a, b) * b


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
    mask,
    config: rpa_schedule.RPAConfig,
):
    """Flash attention kernel."""
    b, k_heads, tq, h = q.shape

    if config.q_scale is not None:
        q = q / config.q_scale
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    qk = lax.dot(
        einshape.jax_einshape("bkth->(bk)th", q),
        einshape.jax_einshape("bksh->(bk)sh", k),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    qk = einshape.jax_einshape("(bk)ts->bkts", qk, b=b,
                               k=k_heads).astype(config.out_dtype)

    qk *= config.sm_scale
    if config.k_scale is not None:
        qk *= config.k_scale
    if config.q_scale is not None:
        qk *= config.q_scale

    if config.soft_cap is not None:
        qk = config.soft_cap * jnp.tanh(qk / config.soft_cap)

    t = config.bq_sz
    dim_q = tq // t
    qk_masked = []
    for i in range(b):
        qk_mask_one = mask[i].astype(jnp.int32)
        qk_mask_one = pltpu.repeat(qk_mask_one, dim_q, axis=0)
        qk_masked.append(
            jnp.where(qk_mask_one[None, ...], qk[i], config.mask_value))
    qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - broadcast_minor(m_next, qk.shape))

    pv = lax.dot(
        einshape.jax_einshape("bkts->(bk)ts", p),
        einshape.jax_einshape("bksh->(bk)sh", v),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    pv = einshape.jax_einshape("(bk)th->bkth", pv, b=b,
                               k=k_heads).astype(config.out_dtype)

    if config.v_scale is not None:
        pv *= config.v_scale

    p_rowsum = jnp.sum(p, axis=-1, keepdims=True)
    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    o_next = broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return m_next, l_next, o_next
