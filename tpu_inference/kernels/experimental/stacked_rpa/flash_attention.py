# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils

_BRANCHLESS_MASK_MAX_ELEMENTS = 64 * 1024


def _apply_attention_mask(qk, compute_mask, skip_mask, mask_value):
    """Apply a statically selected eager or lazy attention-mask path."""
    if skip_mask is None:
        return jnp.where(compute_mask(), qk, mask_value)

    # Small decode tiles can eagerly materialize the masked alternative. This
    # removes Mosaic's control-flow barrier and lets mask work fuse into the QK
    # and softmax pipeline. Large prefill tiles retain the lazy conditional:
    # eagerly materializing their full masked QK alternative spills VREGs.
    if qk.shape[-2] * qk.shape[-1] <= _BRANCHLESS_MASK_MAX_ELEMENTS:
        masked_qk = jnp.where(compute_mask(), qk, mask_value)
        return jnp.where(skip_mask != 0, qk, masked_qk)

    return lax.cond(
        skip_mask != 0,
        lambda _: qk,
        lambda _: jnp.where(compute_mask(), qk, mask_value),
        operand=None,
    )


@jax.named_scope("flash_qk_softmax")
def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array | None,  # [B, KV, TQ, 128]
    l_prev: jax.Array | None,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    skip_mask: list[jax.Array] | None = None,  # [B]
    cfgs: configs.AttentionConfig,
    bq_start: int,
    initialize: bool = False,
):
    """Flash attention kernel."""
    b, k_heads, tq, _ = q.shape

    if cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    s = k.shape[3]
    qk = lax.dot_general(
        q.reshape(b * k_heads, tq, q.shape[-1]),
        k.reshape(b * k_heads, k.shape[2], s),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(configs.accum_dtype(cfgs.serve.dtype_out))

    qk = qk.reshape(b, k_heads, tq, s)

    qk *= cfgs.model.sm_scale
    if cfgs.serve.scale_k is not None:
        qk *= cfgs.serve.scale_k
    if cfgs.serve.scale_q is not None:
        qk *= cfgs.serve.scale_q

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    qk_masked = []
    mask_value = jnp.asarray(cfgs.model.mask_value, dtype=qk.dtype)

    # Sliding-window addition can overflow int16 near the short-context limit,
    # and wide int16 comparisons can acquire incompatible Mosaic layouts on a
    # trimmed compute view. Keep metadata compact but use int32 mask coordinates.
    int_ty = jnp.int32 if cfgs.model.sliding_window is not None else cfgs.serve.int_ty

    for b_idx in range(b):
        kv_idx_b = (lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) +
                    processed_kv_len[b_idx])
        q_offset_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                      cfgs.aligned_num_q_heads_per_kv_head + bq_start)

        def compute_mask():
            q_idx_b = q_offset_b.astype(int_ty) + processed_q_len[b_idx]
            eff_kv_len_b = effective_kv_len[b_idx]
            mask = q_idx_b < eff_kv_len_b
            mask = jnp.logical_and(mask, q_idx_b >= kv_idx_b)

            if (sliding_window := cfgs.model.sliding_window) is not None:
                mask = jnp.logical_and(mask, q_idx_b
                                       < kv_idx_b + sliding_window)
            return mask

        skip_mask_b = None if skip_mask is None else skip_mask[b_idx]
        qk_masked_b = _apply_attention_mask(
            qk[b_idx],
            compute_mask,
            skip_mask_b,
            mask_value,
        )

        qk_masked.append(qk_masked_b)
    qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    if initialize:
        p = jnp.exp(qk - m_curr)
        l_next = jnp.sum(
            p,
            axis=-1,
            keepdims=True,
            dtype=configs.accum_dtype(cfgs.serve.dtype_out),
        )
        stats_shape = m_curr.shape[:-1] + (pltpu.get_tpu_info().num_lanes, )
        return (
            p,
            None,
            jnp.broadcast_to(m_curr, stats_shape),
            jnp.broadcast_to(l_next, stats_shape),
        )

    assert m_prev is not None and l_prev is not None
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
    p_rowsum = jnp.sum(p,
                       axis=-1,
                       keepdims=True,
                       dtype=configs.accum_dtype(cfgs.serve.dtype_out))

    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    return p, alpha, m_next, l_next


@jax.named_scope("flash_pv")
def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha: jax.Array | None,  # [B, KV, TQ, 128]
    o_prev: jax.Array | None,  # [B, KV, TQ, H]
    cfgs: configs.AttentionConfig,
    initialize: bool = False,
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape
    pv = lax.dot_general(
        p.reshape(b * k_heads, tq, s),
        v.reshape(b * k_heads, v.shape[2], v.shape[3]),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(configs.accum_dtype(cfgs.serve.dtype_out))
    pv = pv.reshape(b, k_heads, tq, v.shape[2])

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    if initialize:
        return pv

    assert alpha is not None and o_prev is not None
    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return o_next
