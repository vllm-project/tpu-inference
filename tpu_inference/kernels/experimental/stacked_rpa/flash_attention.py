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

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    visibility: list[jax.Array] | None = None,  # [B] x [bq_sz, 128]
    skip_mask: list[jax.Array] | None = None,  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
):
    """Flash attention kernel."""
    b, k_heads, tq, h_size = q.shape

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
        q.reshape(-1, tq, h_size),
        k.reshape(-1, h_size, s),
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

    int_ty = cfgs.serve.int_ty

    for b_idx in range(b):
        kv_idx_b = (lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) +
                    processed_kv_len[b_idx])
        q_offset_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1) //
                      cfgs.aligned_num_q_heads_per_kv_head + bq_start)

        def compute_mask(_):
            if visibility is not None:
                num_query_tokens = tq // cfgs.aligned_num_q_heads_per_kv_head
                vis = visibility[b_idx][bq_start:bq_start +
                                        num_query_tokens, :2]
                bq_shape = (num_query_tokens, s)
                k_span_bq = (lax.broadcasted_iota(jnp.int32, bq_shape, 1) +
                             processed_kv_len[b_idx])
                vis_start = lax.broadcast_in_dim(vis[:, 0], bq_shape, (0, ))
                vis_end = lax.broadcast_in_dim(vis[:, 1], bq_shape, (0, ))
                mask_bq = jnp.logical_and(k_span_bq >= vis_start, k_span_bq
                                          <= vis_end)
                mask_tq = jnp.repeat(mask_bq,
                                     cfgs.aligned_num_q_heads_per_kv_head,
                                     axis=0)
                return jnp.stack([mask_tq for _ in range(k_heads)], axis=0)

            q_idx_b = q_offset_b.astype(int_ty) + processed_q_len[b_idx]
            eff_kv_len_b = effective_kv_len[b_idx]
            mask = q_idx_b < eff_kv_len_b
            mask = jnp.logical_and(mask, q_idx_b >= kv_idx_b)

            if (sliding_window := cfgs.model.sliding_window) is not None:
                # Overflow guard: on v6e/v7x with int_ty=int16, `kv_idx_b +
                # sliding_window` can exceed INT16_MAX (32767) even when
                # max_model_len itself is <= 32767, because iota + processed_kv_len
                # already lands near max_model_len at the tail. Wrapping there
                # flips the predicate and silently drops valid keys.  Promote the
                # sum to int32 for the compare -- the compare's result is a bool,
                # so no downstream cost.
                mask = jnp.logical_and(
                    mask,
                    q_idx_b.astype(jnp.int32)
                    < kv_idx_b.astype(jnp.int32) + int(sliding_window))
            return mask

        # Small qk tiles (decode, tq == 1) fit the branchless path without VREG
        # spill; the always-materialised mask elides Mosaic's control-flow
        # barrier and lets the mask compute fuse into the QK+softmax pipeline.
        # For large tiles (prefill/mixed, tq >> 1) the always-materialised
        # ``masked_qk`` is a (K, TQ, S) intermediate that spills VREGs
        # catastrophically (>44 MB scratch on bq_sz=512 prefill measured on
        # v7x-8), so keep the original ``lax.cond`` there. Threshold picked as
        # tq == 1 -- exactly the single-token decode fast path.
        _branchless_ok = (skip_mask is not None
                          and qk.shape[2] * qk.shape[3] <= 64 * 1024)
        if skip_mask is not None and _branchless_ok:
            mask_pred = compute_mask(None)
            masked_qk = jnp.where(mask_pred, qk[b_idx], mask_value)
            qk_masked_b = jnp.where(skip_mask[b_idx] != 0, qk[b_idx],
                                    masked_qk)
        elif skip_mask is not None:
            qk_masked_b = lax.cond(
                skip_mask[b_idx] != 0,
                lambda _: qk[b_idx],
                lambda _: jnp.where(compute_mask(None), qk[b_idx], mask_value),
                operand=None,
            )
        else:
            qk_masked_b = jnp.where(compute_mask(None), qk[b_idx], mask_value)

        qk_masked.append(qk_masked_b)
    qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
    p_rowsum = jnp.sum(p,
                       axis=-1,
                       keepdims=True,
                       dtype=configs.accum_dtype(cfgs.serve.dtype_out))

    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    return p, alpha, m_next, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    cfgs: configs.RpaConfigs,
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape
    h_size = v.shape[2]
    pv = lax.dot_general(
        p.reshape(-1, tq, s),
        v.reshape(-1, h_size, s),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(configs.accum_dtype(cfgs.serve.dtype_out))
    pv = pv.reshape(b, k_heads, tq, h_size)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return o_next
