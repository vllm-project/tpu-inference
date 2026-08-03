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
"""Chunked-KV flash attention (SEQ_ALONG_LANE, 5-D VMEM)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils


def flash_attention_chunked(
    q: jax.Array,  # [B, KV, TQ, H]
    kv_in_vref: jax.Ref,  # 5-D VMEM: (batch, num_pages, kvh2, hd, page_size)
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    *,
    b_batch_idx: int,
    processed_q_len: jax.Array,
    processed_kv_len: jax.Array,
    effective_kv_len: jax.Array,
    bkv_sz_frm_cache: jax.Array,
    new_tok_offset: jax.Array,
    visibility: jax.Array | None,
    skip_mask: jax.Array | None,
    cfgs: configs.RpaConfigs,
    bq_start: int,
    num_cache_pages: int,
    num_new_pages: int,
):
    b, k_heads, tq, hd = q.shape
    page_size = cfgs.serve.page_size

    if cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q

    scale = float(cfgs.model.sm_scale)
    if cfgs.serve.scale_k is not None:
        scale *= float(cfgs.serve.scale_k)
    if cfgs.serve.scale_q is not None:
        scale *= float(cfgs.serve.scale_q)

    m_run = m_prev
    l_run = l_prev
    o_run = o_prev

    int_ty = cfgs.serve.int_ty
    accum_dtype = configs.accum_dtype(cfgs.serve.dtype_out)
    mask_value = jnp.asarray(cfgs.model.mask_value, dtype=accum_dtype)

    num_pages_to_process = num_cache_pages + num_new_pages
    for page_idx in range(num_pages_to_process):
        is_cache_page = page_idx < num_cache_pages
        if is_cache_page:
            kv_base = processed_kv_len + page_idx * page_size
        else:
            halo_j = page_idx - num_cache_pages
            kv_base = (processed_kv_len + bkv_sz_frm_cache +
                       halo_j * page_size - new_tok_offset)

        ks_p = []
        vs_p = []
        for kv_head in range(cfgs.model.num_kv_heads):
            k_head = kv_in_vref[b_batch_idx, page_idx, kv_head * 2, :, :]
            ks_p.append(k_head)
            v_head = kv_in_vref[b_batch_idx, page_idx, kv_head * 2 + 1, :, :]
            vs_p.append(v_head)
        k_p = jnp.stack(ks_p, axis=0)[None, ...]
        v_p = jnp.stack(vs_p, axis=0)[None, ...]

        if cfgs.serve.scale_q is not None:
            if jnp.issubdtype(k_p.dtype, jnp.floating):
                dtype_info = jnp.finfo(k_p.dtype)
                q_use = jnp.clip(q,
                                 min=float(dtype_info.min),
                                 max=float(dtype_info.max)).astype(k_p.dtype)
            else:
                q_use = q.astype(k_p.dtype)
        else:
            q_use = q

        qk_p = lax.dot_general(
            q_use.reshape(-1, tq, hd),
            k_p.reshape(-1, hd, page_size),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        ).astype(accum_dtype)
        qk_p = qk_p.reshape(b, k_heads, tq, page_size)
        qk_p *= scale

        if cfgs.model.soft_cap is not None:
            qk_p = cfgs.model.soft_cap * jnp.tanh(qk_p / cfgs.model.soft_cap)

        s_p = page_size
        kv_idx_b = (lax.broadcasted_iota(int_ty,
                                         (k_heads, tq, s_p), 2) + kv_base)
        q_offset_b = (lax.broadcasted_iota(jnp.int32, (k_heads, tq, s_p), 1) //
                      cfgs.aligned_num_q_heads_per_kv_head + bq_start)

        def compute_mask(_):
            if visibility is not None:
                num_query_tokens = tq // cfgs.aligned_num_q_heads_per_kv_head
                vis = visibility[bq_start:bq_start + num_query_tokens, :2]
                bq_shape = (num_query_tokens, s_p)
                k_span_bq = (lax.broadcasted_iota(jnp.int32, bq_shape, 1) +
                             kv_base)
                vis_start = lax.broadcast_in_dim(vis[:, 0], bq_shape, (0, ))
                vis_end = lax.broadcast_in_dim(vis[:, 1], bq_shape, (0, ))
                mask_bq = jnp.logical_and(k_span_bq >= vis_start, k_span_bq
                                          <= vis_end)
                mask_tq = jnp.repeat(mask_bq,
                                     cfgs.aligned_num_q_heads_per_kv_head,
                                     axis=0)
                return jnp.stack([mask_tq for _ in range(k_heads)], axis=0)
            q_idx_b = q_offset_b.astype(int_ty) + processed_q_len
            m0 = q_idx_b < effective_kv_len
            m0 = jnp.logical_and(m0, q_idx_b >= kv_idx_b)
            if (sliding_window := cfgs.model.sliding_window) is not None:
                m0 = jnp.logical_and(
                    m0,
                    q_idx_b.astype(jnp.int32)
                    < kv_idx_b.astype(jnp.int32) + int(sliding_window))
            return m0

        _branchless_ok = (skip_mask is not None and tq * s_p <= 64 * 1024)
        if skip_mask is not None and _branchless_ok:
            mask_pred = compute_mask(None)
            masked = jnp.where(mask_pred, qk_p[0], mask_value)
            qk_masked_b = jnp.where(skip_mask != 0, qk_p[0], masked)
        elif skip_mask is not None:
            qk_masked_b = lax.cond(
                skip_mask != 0,
                lambda _: qk_p[0],
                lambda _: jnp.where(compute_mask(None), qk_p[0], mask_value),
                operand=None,
            )
        else:
            qk_masked_b = jnp.where(compute_mask(None), qk_p[0], mask_value)
        qk_p = qk_masked_b[None, ...]

        m_curr = jnp.max(qk_p, axis=-1, keepdims=True)
        m_next = jnp.maximum(m_run, m_curr)
        alpha_page = jnp.exp(m_run - m_next)
        p_p = jnp.exp(qk_p - utils.broadcast_minor(m_next, qk_p.shape))
        p_rowsum = jnp.sum(p_p, axis=-1, keepdims=True, dtype=accum_dtype)
        l_run = alpha_page * l_run + p_rowsum
        pv_p = lax.dot_general(
            p_p.reshape(-1, tq, page_size),
            v_p.reshape(-1, hd, page_size),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        ).astype(accum_dtype)
        pv_p = pv_p.reshape(b, k_heads, tq, hd)
        if cfgs.serve.scale_v is not None:
            pv_p *= cfgs.serve.scale_v
        o_run = utils.broadcast_minor(alpha_page, o_run.shape) * o_run + pv_p
        m_run = m_next

    return m_run, l_run, o_run
