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

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils

# ---------------------------------------------------------------------------
# `pltpu.einshape` compatibility shim. Older JAX nightlies (e.g. older
# deployed images on libtpu==0.0.37.dev) don't ship einshape.
# Implement a focused fallback covering the merge/split patterns used in this
# file. Remove once the deployed image is rebuilt with jax==0.9.2.
# ---------------------------------------------------------------------------
if not hasattr(pltpu, "einshape"):

    def _tokens(spec: str):
        toks, i = [], 0
        while i < len(spec):
            if spec[i] == "(":
                j = spec.index(")", i)
                toks.append(spec[i + 1:j])
                i = j + 1
            else:
                toks.append(spec[i])
                i += 1
        return toks

    def _einshape_compat(spec, x, *_args, **kwargs):
        in_spec, out_spec = spec.split("->")
        in_toks, out_toks = _tokens(in_spec), _tokens(out_spec)
        in_has_groups = any(len(t) > 1 for t in in_toks)
        out_has_groups = any(len(t) > 1 for t in out_toks)

        if out_has_groups and not in_has_groups:
            # Merge case: e.g. "bkth->(bk)th"
            in_axis = {t: i for i, t in enumerate(in_toks)}
            new_shape = []
            for t in out_toks:
                if len(t) > 1:
                    size = 1
                    for ch in t:
                        size *= x.shape[in_axis[ch]]
                    new_shape.append(size)
                else:
                    new_shape.append(x.shape[in_axis[t]])
            return x.reshape(new_shape)

        if in_has_groups and not out_has_groups:
            # Split case: e.g. "(bk)ts->bkts" with b=B kwarg.
            new_shape = []
            for axis_idx, t in enumerate(in_toks):
                if len(t) > 1:
                    merged = x.shape[axis_idx]
                    running = 1
                    # All chars but the last get sizes from kwargs; last is the
                    # remainder.
                    for ch in t[:-1]:
                        sz = kwargs[ch]
                        new_shape.append(sz)
                        running *= sz
                    new_shape.append(merged // running)
                else:
                    new_shape.append(x.shape[axis_idx])
            return x.reshape(new_shape)

        # Pure permutation
        in_axis = {t: i for i, t in enumerate(in_toks)}
        return jnp.transpose(x, [in_axis[t] for t in out_toks])

    pltpu.einshape = _einshape_compat


@jax.named_scope("flash_qk_softmax")
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
        pltpu.einshape("bkth->(bk)th", q, True),
        pltpu.einshape("bkhs->(bk)hs", k, True),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(configs.accum_dtype(cfgs.serve.dtype_out))

    qk = pltpu.einshape("(bk)ts->bkts", qk, True, b=b)

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
                mask = jnp.logical_and(mask, q_idx_b
                                       < kv_idx_b + sliding_window)
            return mask

        if skip_mask is not None:
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


@jax.named_scope("flash_pv")
def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    cfgs: configs.RpaConfigs,
):
    """Flash attention kernel."""
    b = p.shape[0]
    pv = lax.dot_general(
        pltpu.einshape("bkts->(bk)ts", p, True),
        pltpu.einshape("bkhs->(bk)hs", v, True),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(configs.accum_dtype(cfgs.serve.dtype_out))
    pv = pltpu.einshape("(bk)th->bkth", pv, True, b=b)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return o_next
