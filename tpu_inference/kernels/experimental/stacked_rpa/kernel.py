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

import dataclasses
import functools
import inspect

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

# isort: off
from tpu_inference.kernels.experimental.stacked_rpa import (
    bref_override, configs, flash_attention, schedule, stitch_utils, utils)
# isort: on

# ---------------------------------------------------------------------------
# `pl.with_scoped` compatibility shim.
#
# Upstream introduced `pl.with_scoped` (a decorator form of `pl.run_scoped`) in
# the JAX nightly that ships with the current tpu-inference image. Older JAX
# nightlies (e.g. older deployed images on libtpu==0.0.37.dev)
# only have `pl.run_scoped`. Synthesize `with_scoped` from `run_scoped` on
# those older versions so the vendored kernel runs on both.
# Remove this shim once the deployed image is rebuilt with jax==0.9.2 (which
# ships `pl.with_scoped` natively).
# ---------------------------------------------------------------------------
if not hasattr(pl, "with_scoped"):

    def _with_scoped_compat(**scopes):

        def decorator(fn):
            param_names = list(inspect.signature(fn).parameters)
            ordered = tuple(scopes[name] for name in param_names
                            if name in scopes)

            def wrapper(*args, **kwargs):
                return pl.run_scoped(
                    lambda *res: fn(*res, *args, **kwargs),
                    *ordered,
                )

            return wrapper

        return decorator

    pl.with_scoped = _with_scoped_compat

# Define inner kernel.


@jax.named_scope("calculate_and_store_out")
def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):

    @jax.named_scope("accum")
    def _accum(b_idx: int):
        batch_acc = acc_scratch_ref[b_idx]
        batch_l = l_scratch_ref[b_idx]
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (cfgs.serve.dtype_out == jnp.float32
                or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc.astype(jnp.float32) * pl.reciprocal(
                batch_l.astype(jnp.float32), approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
            out = jnp.pad(
                out,
                (
                    (0, 0),
                    (0, 0),
                    (0, cfgs.aligned_q_head_dim - cfgs.aligned_kv_head_dim),
                ),
                constant_values=0,
            )
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    for b in range(cfgs.batch_size):
        # Adding a conditional causes a scheduling barrier. In prefill, we often
        # use small block sizes, so it's not worth executing the accumulation
        # on every block. In decode, because of the large block sizes / and or
        # batch sizes, we almost always use accumulation on every block. Please
        # tune `fuse_accum` for your use case.
        if not cfgs.fuse_accum:
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            jax.lax.cond(is_last_k, jax.named_call(_accum, name="accum"),
                         lambda _: None, b)
        else:
            _accum(b)


@jax.named_scope("stacked_combine_and_store")
def _stacked_combine_and_store(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    m_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stacked segmented combine.

    Several requests can complete in one step. Each is a contiguous group of
    cells ``[root, root+span)`` marked by ``combine_span[step, root] = span``
    (0 for non-roots). For every root we online-softmax-merge just that group's
    cells (excluded cells contribute alpha=0, so an adjacent request in the same
    step is not mixed in), write the normalized result to the root cell's output
    slot, and reset the group's cells to identity for reuse.
    """
    n = cfgs.batch_size

    for root in range(n):  # static root candidates
        span = schedule_ref.combine_span[step_idx, root]

        @pl.when(span > 0)
        def _combine(root=root, span=span):

            def included(m):  # cell m is in group [root, root+span)
                return (m - root) < span

            # 1) group max (excluded members forced to -inf)
            m_comb = m_scratch_ref[root]
            for m in range(root + 1, n):
                cand = jnp.where(included(m), m_scratch_ref[m], -jnp.inf)
                m_comb = jnp.maximum(m_comb, cand)

            # 2) rescaled sums; excluded members get alpha=0 -> contribute nothing
            l_comb = jnp.zeros_like(l_scratch_ref[root])
            acc_comb = jnp.zeros_like(acc_scratch_ref[root])
            for m in range(root, n):
                if m == root:
                    m_m = m_scratch_ref[root]
                else:
                    m_m = jnp.where(included(m), m_scratch_ref[m], -jnp.inf)
                alpha = jnp.exp(m_m - m_comb)  # 0 for excluded/identity
                l_comb = l_comb + alpha * l_scratch_ref[m]
                acc_comb = acc_comb + (utils.broadcast_minor(
                    alpha, acc_comb.shape) * acc_scratch_ref[m])

            # 3) normalize
            batch_l = utils.broadcast_minor(l_comb, acc_comb.shape)
            if (cfgs.serve.dtype_out == jnp.float32
                    or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
                result = lax.div(acc_comb, batch_l)
            else:
                result = acc_comb.astype(jnp.float32) * pl.reciprocal(
                    batch_l.astype(jnp.float32), approx=True)
            out = result.astype(cfgs.serve.dtype_out)
            if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
                out = jnp.pad(
                    out,
                    (
                        (0, 0),
                        (0, 0),
                        (0,
                         cfgs.aligned_q_head_dim - cfgs.aligned_kv_head_dim),
                    ),
                    constant_values=0,
                )

            # 4) write to the ROOT cell's output slot (copy_out DMAs roots).
            o_u32_vref = o_vref.at[root].bitcast(jnp.uint32)
            out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
            out_b = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out_b)

            # 5) reset the group's cells to identity for reuse.
            for m in range(root, n):
                reset = jnp.array(True) if m == root else included(m)
                m_scratch_ref[m] = jnp.where(reset, -jnp.inf, m_scratch_ref[m])
                l_scratch_ref[m] = jnp.where(reset, 0.0, l_scratch_ref[m])
                acc_scratch_ref[m] = jnp.where(reset, 0.0, acc_scratch_ref[m])


@jax.named_scope("dense_combine_and_store")
def _dense_combine_and_store(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    m_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    m_carry_ref: jax.Ref,
    l_carry_ref: jax.Ref,
    acc_carry_ref: jax.Ref,
    carry_valid_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Dense-packing combine with a single cross-step carry slot.

    Cells hold FRESH per-block partials (reset each step). For each contiguous
    group [root, root+span) in this step (left->right): reduce its cells; the
    group at cell 0 additionally merges the carry (the one request straddling in
    from the previous step); if is_final -> normalize + output; else -> this
    group straddles to the next step, so store it as the new carry.
    """
    n = cfgs.batch_size

    # --- Phase 1: reduce each multi-cell group in place into its root cell. ---
    # combine_span[step, root] gives the group range [root, root+span). span==1
    # groups need no reduction (the "empty reduce-list" fast path); only span>1
    # groups run a cross-cell softmax merge. Reducing in place lets Phase 4
    # normalize every output cell uniformly with a single vectorized store.
    for root in range(n - 1):  # the last cell can only ever be a span==1 group
        span = schedule_ref.combine_span[step_idx, root]

        @pl.when(span > 1)
        def _reduce(root=root, span=span):

            def included(m):
                return (m - root) < span

            m_g = m_scratch_ref[root]
            for m in range(root + 1, n):
                m_g = jnp.maximum(
                    m_g, jnp.where(included(m), m_scratch_ref[m], -jnp.inf))
            l_g = jnp.zeros_like(l_scratch_ref[root])
            acc_g = jnp.zeros_like(acc_scratch_ref[root])
            for m in range(root, n):
                m_m = (m_scratch_ref[root] if m == root else jnp.where(
                    included(m), m_scratch_ref[m], -jnp.inf))
                a = jnp.exp(m_m - m_g)
                l_g = l_g + a * l_scratch_ref[m]
                acc_g = acc_g + (utils.broadcast_minor(a, acc_g.shape) *
                                 acc_scratch_ref[m])
            m_scratch_ref[root] = m_g
            l_scratch_ref[root] = l_g
            acc_scratch_ref[root] = acc_g

    # --- Phase 2: merge the incoming cross-step carry into cell 0 (the one
    # straddling request from the previous step always continues at cell 0). ---
    @pl.when(carry_valid_ref[0] == 1)
    def _merge_carry():
        cm = m_carry_ref[0]
        m0 = m_scratch_ref[0]
        mx = jnp.maximum(cm, m0)
        ac = jnp.exp(cm - mx)
        ag = jnp.exp(m0 - mx)
        m_scratch_ref[0] = mx
        l_scratch_ref[0] = ac * l_carry_ref[0] + ag * l_scratch_ref[0]
        acc_scratch_ref[0] = (
            utils.broadcast_minor(ac, acc_scratch_ref[0].shape) *
            acc_carry_ref[0] +
            utils.broadcast_minor(ag, acc_scratch_ref[0].shape) *
            acc_scratch_ref[0])
        carry_valid_ref[0] = 0

    # --- Phase 3: the single non-final (straddling) group becomes the carry. ---
    for root in range(n):
        active = jnp.logical_and(
            schedule_ref.combine_span[step_idx, root] > 0,
            schedule_ref.is_final[step_idx, root] == 0,
        )

        @pl.when(active)
        def _save_carry(root=root):
            m_carry_ref[0] = m_scratch_ref[root]
            l_carry_ref[0] = l_scratch_ref[root]
            acc_carry_ref[0] = acc_scratch_ref[root]
            carry_valid_ref[0] = 1

    # --- Phase 4: vectorized normalize + store for ALL cells at once, only on
    # steps where at least one request completes (is_final==1 implies a
    # completing root). Skipping no-completion steps avoids wasted stores on the
    # many mid-stream steps of a long request. ---
    #
    # TODO(perf, adaptive-combine): this vectorized all-cell store wins when many
    # requests complete per step (short/high-concurrency: outputs ~= n) but wastes
    # ~(n-1)/n work when few complete (long context: one span==n group -> a single
    # output), a ~1-2% long-context regression vs a per-root store (measured
    # SEQ_ALONG_LANE bs=32 hd64: 1k 0.918x->0.976x good, 64k ~0.99x->0.977x). Make
    # this adaptive on a per-step completion count (see the matching TODO in
    # schedule.py): vectorize when the count is high, per-root store of only the
    # output cells when it is low, to get the best of both with no regression.
    has_output = schedule_ref.is_final[step_idx, 0] == 1
    for root in range(1, n):
        has_output = jnp.logical_or(has_output,
                                    schedule_ref.is_final[step_idx, root] == 1)

    @pl.when(has_output)
    def _emit_all():
        # BatchingORef DMAs only cells with combine_span>0 & is_final, so
        # normalized garbage in idle / non-final cells is never written out.
        l_all = l_scratch_ref[...]
        acc_all = acc_scratch_ref[...]
        denom = jnp.where(l_all == 0.0, 1.0, l_all)  # avoid 0/0 in idle cells
        batch_l = utils.broadcast_minor(denom, acc_all.shape)
        if (cfgs.serve.dtype_out == jnp.float32
                or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
            result = lax.div(acc_all, batch_l)
        else:
            result = acc_all.astype(jnp.float32) * pl.reciprocal(
                batch_l.astype(jnp.float32), approx=True)
        out = result.astype(cfgs.serve.dtype_out)
        if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
            out = jnp.pad(
                out,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, cfgs.aligned_q_head_dim - cfgs.aligned_kv_head_dim),
                ),
                constant_values=0,
            )
        o_u32_vref = o_vref.bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        out_b = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out_b)


@jax.named_scope("rpa_body")
def rpa_body(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    visibility_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    m_carry_ref: jax.Ref,
    l_carry_ref: jax.Ref,
    acc_carry_ref: jax.Ref,
    carry_valid_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    # Configs.
    cfgs: configs.RpaConfigs,
):
    step = pl.program_id(0)

    # Step 1: Fetch metadata.
    # TODO(perf, hoist-metadata): this phase is per-(step, b_idx) scalar math
    # derived from the schedule + kv_lens/cu_q_lens. It can be precomputed once
    # per call (vectorized over [num_steps, batch] in XLA / generate_rpa_metadata)
    # and passed as SMEM schedule fields, reducing this phase to plain loads. Best
    # candidates are the stitch pair (bkv_sz_frm_cache, new_kv_len_start) and
    # processed_q_len, which gate kv_stitch; effective_kv_len/processed_kv_len are
    # cheap gathers that can stay in-kernel. The derived fields depend on
    # kv_lens/cu_q_lens (per-call), so the precompute runs each call and cannot go
    # in a cached index-only schedule; also watch the SMEM budget (each field is
    # steps*batch*4B on top of the existing s_idx/q_idx/k_idx/skip_mask arrays).
    # MEASURE FIRST (LLO-off): this is scalar-core work that may already overlap
    # the DMA/VPU on the critical path -- like kv_zero_pad, it can look serial in
    # an LLO trace but cost ~0 wall-clock. Only hoist if a stub/precompute A/B
    # actually moves LLO-off latency.
    with jax.named_scope("rpa_metadata"):
        processed_q_len = []
        processed_kv_len = []
        effective_kv_len = []
        # Lists to hold the 2 variables needed for stitching
        bkv_sz_frm_cache_list = []
        new_kv_len_start_list = []
        visibility_list = []
        skip_mask_list = []
        int_ty = cfgs.serve.int_ty
        for b_idx in range(cfgs.batch_size):
            s_idx = schedule_ref.s_idx[step, b_idx]
            is_valid = s_idx != -1
            q_idx = schedule_ref.q_idx[step, b_idx]
            k_idx = schedule_ref.k_idx[step, b_idx]
            kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
            q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
            q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
            q_len = q_end - q_start
            offset = kv_len - q_len

            if cfgs.use_window_anchor:
                # Match schedule.py's anchored block base (page-aligned window
                # start); k_idx is the local block index (0) in the anchored range.
                anchor_tok = utils.window_anchor_tok(kv_len, q_len,
                                                     cfgs.model.sliding_window,
                                                     cfgs.serve.page_size_log2)
                k_id = jnp.where(is_valid, anchor_tok + k_idx * cfgs.bkv_sz, 0)
            else:
                k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)

            processed_q_len.append(
                (q_idx * cfgs.bq_sz + offset).astype(int_ty))
            processed_kv_len.append(k_id.astype(int_ty))
            effective_kv_len.append(kv_len.astype(int_ty))
            if cfgs.has_visibility:
                visibility_list.append(visibility_vref[b_idx, :, :2])
            skip_mask_list.append(schedule_ref.skip_mask[step, b_idx])

            # Stitching metadata
            kv_left = jnp.maximum(kv_len - k_id, 0)
            if cfgs.update_kv_cache:
                kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
            else:
                kv_left_frm_cache = kv_left
            kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)

            bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
            new_kv_len_start = q_end - kv_left_frm_new

            bkv_sz_frm_cache_list.append(bkv_sz_frm_cache.astype(int_ty))
            new_kv_len_start_list.append(new_kv_len_start.astype(int_ty))

            if not cfgs.is_stacked:
                start_k_idx = 0
                if (sliding_window := cfgs.model.sliding_window) is not None:
                    sw_start_idx = (kv_len - q_len + q_idx * cfgs.bq_sz -
                                    sliding_window + 1)
                    start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

                is_first_k_block = k_idx == start_k_idx
                reset_cond = jnp.logical_and(is_valid, is_first_k_block)
                m_scratch_ref[b_idx] = jnp.where(reset_cond, -jnp.inf,
                                                 m_scratch_ref[b_idx])
                l_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0,
                                                 l_scratch_ref[b_idx])
                acc_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0,
                                                   acc_scratch_ref[b_idx])

    # Step 2: Fetch inputs.
    with jax.named_scope("q_load"):
        q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
        q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
        q_loaded = utils.strided_load(
            q_ref,
            0,
            cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
            1,
            dtype=cfgs.serve.dtype_q,
        )
        q = q_loaded.reshape(
            cfgs.batch_size,
            cfgs.model.num_kv_heads,
            cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
            cfgs.aligned_q_head_dim,
        )
        if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
            q = q[..., :cfgs.aligned_kv_head_dim]

    # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
    # where bkv_stride ~= num_kv_heads * 2 // kv_packing
    # to 2x (batch, num_kv_heads, bkv_sz, d)
    # We use strided_load to avoid the expensive transpose.
    # Overlapped V-load: for single-token SEQ_ALONG_LANE decode, defer the V
    # VMEM->reg load until after the QK softmax (below) so the load units overlap
    # the VPU softmax. Only valid for bq_sz==1 (the only-final-pv flow).
    _defer_v = cfgs.bq_sz == 1
    k_b = []
    v_b = []

    with jax.named_scope("kv_stitch"):
        stitch_results = []
        for b_idx in range(cfgs.batch_size):
            res = stitch_utils.stitch_new_kv_lane(
                kv_in_vref,
                b_idx,
                bkv_sz_frm_cache_list[b_idx],
                new_kv_len_start_list[b_idx],
                cfgs=cfgs,
            )
            stitch_results.append(res)
        for b_idx in range(cfgs.batch_size):
            stitch_utils.store_new_kv_lane(
                kv_in_vref,
                b_idx,
                stitch_results[b_idx],
                cfgs=cfgs,
            )
    with jax.named_scope("k_load" if _defer_v else "kv_load"):
        for b_idx in range(cfgs.batch_size):
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                # 4D [.., head_dim, tokens]: slice is already [hd, bkv], no
                # reshape (the old [hd_sub, pk]->hd reshape forced a relayout).
                k_head = kv_in_vref[b_idx, kv_head * 2, :, 0:cfgs.bkv_sz]
                ks.append(k_head)
                if not _defer_v:
                    v_head = kv_in_vref[b_idx, kv_head * 2 + 1, :,
                                        0:cfgs.bkv_sz]
                    vs.append(v_head)
            k_b.append(jnp.stack(ks, axis=0))
            if not _defer_v:
                v_b.append(jnp.stack(vs, axis=0))
    # Stack to (batch, num_heads, bkv_sz, num_lanes)
    with jax.named_scope("kv_reshape"):
        k = jnp.stack(k_b, axis=0)
        if not _defer_v:
            v = jnp.stack(v_b, axis=0)

    # Zero V beyond the block's valid length before p@V: the QK score mask forces
    # p==0 on padding positions, but 0*NaN==NaN, so stale/uninitialized paged-cache
    # V in the padding slots must be made finite. Matches the original RPA kernel:
    # fold the V-mask into the point where V is consumed (per path below) instead
    # of a standalone zero-pad phase. K needs no masking -- the QK score mask
    # already sanitizes padding/NaN K.
    def _mask_v(vv):
        kv_valid = jnp.stack([
            jnp.clip(
                effective_kv_len[b].astype(jnp.int32) -
                processed_kv_len[b].astype(jnp.int32),
                0,
                cfgs.bkv_sz,
            ) for b in range(cfgs.batch_size)
        ])  # int32: Mosaic only supports i32 subi
        keep = lax.broadcasted_iota(
            jnp.int32, vv.shape, vv.ndim -
            1) < kv_valid.reshape((cfgs.batch_size, ) + (1, ) * (vv.ndim - 1))
        return jnp.where(keep, vv, 0)

    if not _defer_v:
        v = _mask_v(v)

    # Dense packing: each cell holds a NEW request's block this step, so reset
    # all cells to identity before the flash (fresh per-block partials).
    if cfgs.dense_pack:
        with jax.named_scope("dense_reset"):
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref[...], -jnp.inf)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref[...])
            acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref[...])

    # Step 3: Perform compute.
    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]

    prev_p = prev_alpha = prev_q_slice = None
    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
        bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
        q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
        q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
        q_slice = slice(q_start, q_end)

        p, alpha, m_next, l_next = flash_attention.flash_attention_qk_softmax(
            q[:, :, q_slice],
            k,
            m_val[:, :, q_slice],
            l_val[:, :, q_slice],
            processed_q_len=processed_q_len,
            processed_kv_len=processed_kv_len,
            effective_kv_len=effective_kv_len,
            visibility=visibility_list if cfgs.has_visibility else None,
            skip_mask=skip_mask_list,
            cfgs=cfgs,
            bq_start=bq_start,
        )
        m_scratch_ref[:, :, q_slice] = m_next
        l_scratch_ref[:, :, q_slice] = l_next

        if prev_p is not None:
            o_next = flash_attention.flash_attention_pv(
                prev_p,
                v,
                prev_alpha,
                acc_val[:, :, prev_q_slice],
                cfgs=cfgs,
            )
            acc_scratch_ref[:, :, prev_q_slice] = o_next

        prev_p = p
        prev_alpha = alpha
        prev_q_slice = q_slice

    # Overlapped V-load: load V now (after the QK softmax) so the VMEM->reg load
    # overlaps the softmax VPU work above. Single-token decode only (_defer_v).
    if _defer_v:
        with jax.named_scope("load_v"):
            v_b_deferred = []
            for b_idx in range(cfgs.batch_size):
                vs = []
                for kv_head in range(cfgs.model.num_kv_heads):
                    v_head = kv_in_vref[b_idx, kv_head * 2 + 1, :,
                                        0:cfgs.bkv_sz]
                    vs.append(v_head)
                v_b_deferred.append(jnp.stack(vs, axis=0))
            v = jnp.stack(v_b_deferred, axis=0)
            v = _mask_v(v)

    assert prev_p is not None
    o_next = flash_attention.flash_attention_pv(
        prev_p,
        v,
        prev_alpha,
        acc_val[:, :, prev_q_slice],
        cfgs=cfgs,
    )
    acc_scratch_ref[:, :, prev_q_slice] = o_next

    if cfgs.is_stacked:
        # Idle cells (s_idx == -1) ran a fully-masked flash which, for an
        # identity (m == -inf) cell, produces garbage (exp(0) == 1). Restore
        # every idle cell to its pre-step value so it either stays identity or
        # keeps its earlier partial (S>1 tail cells), keeping the cross-cell
        # combine correct.
        with jax.named_scope("stacked_idle_restore"):
            for b_idx in range(cfgs.batch_size):
                is_valid_b = schedule_ref.s_idx[step, b_idx] != -1
                m_scratch_ref[b_idx] = jnp.where(is_valid_b,
                                                 m_scratch_ref[b_idx],
                                                 m_val[b_idx])
                l_scratch_ref[b_idx] = jnp.where(is_valid_b,
                                                 l_scratch_ref[b_idx],
                                                 l_val[b_idx])
                acc_scratch_ref[b_idx] = jnp.where(is_valid_b,
                                                   acc_scratch_ref[b_idx],
                                                   acc_val[b_idx])

    # Step 4: Write back outputs.
    if cfgs.is_stacked and cfgs.dense_pack:
        _dense_combine_and_store(
            step,
            schedule_ref,
            acc_scratch_ref,
            l_scratch_ref,
            m_scratch_ref,
            o_vref,
            m_carry_ref,
            l_carry_ref,
            acc_carry_ref,
            carry_valid_ref,
            cfgs=cfgs,
        )
    elif cfgs.is_stacked:
        _stacked_combine_and_store(
            step,
            schedule_ref,
            acc_scratch_ref,
            l_scratch_ref,
            m_scratch_ref,
            o_vref,
            cfgs=cfgs,
        )
    else:
        calculate_and_store_out(
            step,
            schedule_ref,
            acc_scratch_ref,
            l_scratch_ref,
            o_vref,
            cfgs=cfgs,
        )


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    visibility_hbm_ref: jax.Ref,
    cfgs: configs.RpaConfigs,
):
    kv_cache_spec = pl.BlockSpec(
        block_shape=cfgs.kv_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer,
                                  use_lookahead=True),
    )
    q_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer,
                                  use_lookahead=True),
    )
    o_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
    )
    visibility_spec = pl.BlockSpec(
        block_shape=(cfgs.batch_size, cfgs.bq_sz, 128),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer,
                                  use_lookahead=True),
    )

    kv_cache_alloc_cls = bref_override.KVBufferedRefSeqAlongLane

    kv_cache_alloc = kv_cache_alloc_cls.input_output(
        spec=kv_cache_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    q_alloc = bref_override.BatchingQRef.input(
        spec=q_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    o_alloc = bref_override.BatchingORef.output(
        spec=o_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    visibility_alloc = bref_override.BatchingVisibilityRef.input(
        spec=visibility_spec,
        dtype_or_type=visibility_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )

    return q_alloc, kv_cache_alloc, visibility_alloc, o_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
    name = f"RPA{cfgs.mode.symbol}-p{cfgs.serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    if cfgs.model.sliding_window:
        name += f"-sw{cfgs.model.sliding_window}"
    return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs, ) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


def rpa_kernel(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    visibility_hbm: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention with scheduler data.

    Args:
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
        page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        schedule_hbm: Output of scheduler kernel. It informs which: 1. seqs 2. q
            block 3. kv block that should be processed at a given step.
        q_hbm: [max_num_tokens, num_q_heads_per_kv_heads, cdiv(num_kv_heads,
            q_packing), q_packing, head_dim]. Output of q projection that has been
            pre-processed to align with existing kv cache data layout.
        new_kv_hbm: [max_num_tokens, cdiv(num_kv_heads * 2, kv_packing), kv_packing,
            head_dim]. Output of k & v projection that has been pre-processed to align
            with existing kv cache data layout.
        kv_cache_hbm: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        cfgs: Configuration of the kernel.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
            head_dim]. Result of new kv cache.
    """

    def ragged_paged_attention_pipeline(
        # Scalar prefetch.
        cu_q_lens_ref: jax.Ref,
        kv_lens_ref: jax.Ref,
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        visibility_hbm_ref: jax.Ref,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
    ):

        del o_kv_cache_hbm_ref

        q_alloc, kv_cache_alloc, visibility_alloc, o_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, visibility_hbm_ref, cfgs)

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, visibility_alloc, o_alloc),
            # One schedule SMEM window if the whole schedule fits; otherwise TWO
            # (double-buffered: prefetch window w+1 while computing window w).
            schedule_ref=tuple(
                schedule.RpaSchedule.create_shape_dtype(
                    cfgs, steps=cfgs.sched_window).scratch_shapes()
                for _ in range(1 if cfgs.fits_one_window else 2)),
            dma_sem=pltpu.SemaphoreType.DMA((
                1, ) if cfgs.fits_one_window else (2, )),
            scratches=(
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # m
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # l
                pltpu.VMEM(
                    cfgs.acc_scratch_shape,
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # acc
                pltpu.VMEM(
                    (1, ) + tuple(cfgs.lm_scratch_shape[1:]),
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # m_carry
                pltpu.VMEM(
                    (1, ) + tuple(cfgs.lm_scratch_shape[1:]),
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # l_carry
                pltpu.VMEM(
                    (1, ) + tuple(cfgs.acc_scratch_shape[1:]),
                    dtype=configs.accum_dtype(cfgs.serve.dtype_out),
                ),  # acc_carry
                pltpu.SMEM((1, ), jnp.int32),  # carry_valid
            ),
        )
        def _run(final_allocs, schedule_ref, dma_sem, scratches):

            actual_steps = schedule_hbm_ref.actual_steps[0]
            w_size = cfgs.sched_window
            num_windows = cfgs.total_steps_ub // w_size  # static, >= 1

            if cfgs.is_stacked:
                # One-time identity init of the online-softmax scratch. Stacked
                # relies on reset-after-combine per request thereafter (no
                # per-first-block reset), so cells not yet touched by any request
                # start (and stay) at identity.
                m_s, l_s, acc_s, m_c, l_c, acc_c, cv = scratches
                m_s[...] = jnp.full_like(m_s[...], -jnp.inf)
                l_s[...] = jnp.zeros_like(l_s[...])
                acc_s[...] = jnp.zeros_like(acc_s[...])
                if cfgs.dense_pack:
                    m_c[...] = jnp.full_like(m_c[...], -jnp.inf)
                    l_c[...] = jnp.zeros_like(l_c[...])
                    acc_c[...] = jnp.zeros_like(acc_c[...])
                    cv[0] = 0

            flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)

            def _kv_cache_init():
                # Init KV-cache scratch to zeros. p*v causal-masks on lhs (p) by
                # zeroing masked columns, which is only safe if rhs (v) has no NaNs;
                # pre-zeroing the scratch guarantees that (stale non-NaN data is
                # also fine).
                with jax.named_scope("kv_cache_init"):
                    kv_alloc = final_allocs[1]
                    # Zero the KV scratch via the u32 view directly. Do NOT reshape
                    # to [-1, num_lanes]: under the collapsed [.., head_dim, v_len]
                    # SEQ layout that changes the minormost (tiled) dim, which
                    # Mosaic rejects ("minormost dimension must be unchanged").
                    kv_ref_u32 = kv_alloc.window_ref.bitcast(jnp.uint32)
                    kv_ref_u32[...] = jnp.zeros_like(kv_ref_u32)

            def _sched_copies(w, n_steps, buf, sem_idx):
                # Build (not start) the clamped HBM->SMEM copies for window w into
                # `buf`. Copy ONLY the n_steps this window runs (tile-aligned to the
                # 1024-int DMA tile), not the full SMEM window. copy_len is a
                # deterministic function of (w, actual_steps), so a descriptor
                # re-created to .wait() matches the one used to .start() (the manual
                # double-buffer pattern). Returns the descriptors.
                buf_leaves = jax.tree_util.tree_leaves(buf)
                descs = []
                for h, s in zip(flat_hbm, buf_leaves):
                    if h.shape[0] > 1:
                        s_len = s.shape[0]  # padded window flat
                        raw_stride = s_len // w_size  # int32s per step
                        copy_len = jnp.minimum(
                            pl.multiple_of(
                                pl.cdiv(n_steps * raw_stride, 1024) * 1024,
                                1024),
                            s_len,
                        )
                        descs.append(
                            pltpu.make_async_copy(
                                h.at[pl.ds(pl.multiple_of(w * s_len, s_len),
                                           copy_len)],
                                s.at[pl.ds(0, copy_len)],
                                dma_sem.at[sem_idx],
                            ))
                return descs

            def _run_pipeline(n_steps, buf):
                pipeline_func = pltpu.emit_pipeline(
                    body=functools.partial(
                        rpa_body,
                        cfgs=cfgs,
                        cu_q_lens_ref=cu_q_lens_ref,
                        kv_lens_ref=kv_lens_ref,
                    ),
                    grid=(n_steps, ),
                    in_specs=(
                        q_alloc.spec,
                        kv_cache_alloc.spec,
                        visibility_alloc.spec,
                    ),
                    out_specs=(o_alloc.spec, ),
                )
                pipeline_func(
                    (q_hbm_ref, buf),
                    (kv_cache_hbm_ref, new_kv_hbm_ref, buf),
                    (visibility_hbm_ref, buf),
                    (o_hbm_ref, buf),
                    scratches=(buf, ) + scratches,
                    allocations=final_allocs,
                )

            if num_windows == 1:
                # FITS path (static): the whole schedule fits one SMEM window.
                # Single clamped copy, started async and OVERLAPPED with
                # kv_cache_init, then one pipeline pass (upstream structure).
                buf = schedule_ref[0]
                safe_steps = jnp.minimum(actual_steps, w_size)
                with jax.named_scope("sched_dma_load"):
                    descs = _sched_copies(0, safe_steps, buf, 0)
                    for c in descs:
                        c.start()
                    _kv_cache_init()  # overlaps the schedule copy
                    jax.tree.map(lambda c: c.wait(), descs)
                _run_pipeline(safe_steps, buf)
            else:
                # MULTI-WINDOW path (static): the schedule exceeds one SMEM window.
                # Double-buffer two SMEM windows: prefetch window w+1 into the other
                # buffer while window w computes, so schedule streaming overlaps
                # compute. The m/l/acc scratches persist across windows so a
                # sequence spanning windows accumulates correctly.
                buf0, buf1 = schedule_ref
                num_windows_actual = jnp.minimum(pl.cdiv(actual_steps, w_size),
                                                 num_windows)

                # Prologue: start window 0's copy into buf0 (sem 0), overlapped
                # with kv_cache_init. Its wait happens in iteration 0.
                win0_steps = jnp.minimum(actual_steps, w_size)
                with jax.named_scope("sched_dma_load"):
                    for c in _sched_copies(0, win0_steps, buf0, 0):
                        c.start()
                    _kv_cache_init()

                def _window(w, _):
                    win_steps = jnp.clip(actual_steps - w * w_size, 0, w_size)
                    nxt = w + 1
                    nxt_steps = jnp.clip(actual_steps - nxt * w_size, 0,
                                         w_size)
                    even = (w % 2) == 0

                    # Prefetch window w+1 into the OTHER buffer (async, no wait) so
                    # it overlaps this window's compute.
                    @pl.when(jnp.logical_and(nxt < num_windows_actual, even))
                    def _prefetch_to_buf1():
                        with jax.named_scope("sched_dma_load"):
                            for c in _sched_copies(nxt, nxt_steps, buf1, 1):
                                c.start()

                    @pl.when(
                        jnp.logical_and(nxt < num_windows_actual,
                                        jnp.logical_not(even)))
                    def _prefetch_to_buf0():
                        with jax.named_scope("sched_dma_load"):
                            for c in _sched_copies(nxt, nxt_steps, buf0, 0):
                                c.start()

                    # Wait this window's copy (started in the previous iteration or
                    # the prologue) and run its pipeline, on the parity-selected
                    # buffer (static within each branch).
                    @pl.when(even)
                    def _compute_buf0():
                        jax.tree.map(
                            lambda c: c.wait(),
                            _sched_copies(w, win_steps, buf0, 0),
                        )
                        _run_pipeline(win_steps, buf0)

                    @pl.when(jnp.logical_not(even))
                    def _compute_buf1():
                        jax.tree.map(
                            lambda c: c.wait(),
                            _sched_copies(w, win_steps, buf1, 1),
                        )
                        _run_pipeline(win_steps, buf1)

                    return None

                jax.lax.fori_loop(0, num_windows_actual, _window, None)

        _run()

    # q_hbm / kv_cache_hbm live after the 2 scalar-prefetch operands and the
    # flattened schedule leaves; compute their input indices so adding schedule
    # fields (e.g. combine_span) can't silently misalign the aliases.
    _n_sched_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    _q_in_idx = 2 + _n_sched_leaves  # q_hbm_ref
    _kv_in_idx = _q_in_idx + 2  # after q_hbm, new_kv_hbm
    _kernel = pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=[q_hbm, kv_cache_hbm],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # visibility_hbm_ref
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(
                    memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            # Paged attention issues many small KV DMAs plus manual double-buffered
            # schedule streaming and kv/q/o BufferedRefs, so per-DMA semaphore
            # validation is significant overhead. Disable it (same rationale as
            # disable_bounds_checks); semaphore usage is covered by the tests.
            disable_semaphore_checks=True,
        ),
        input_output_aliases={
            _q_in_idx: 0,
            _kv_in_idx: 1
        },
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )

    # TODO (alynie): Investigate why this causes regressions in tpu-inference.
    def _constrain_hbm(path, x):
        for p in path:
            key = getattr(p, "name", getattr(p, "key", None))
            if key == "actual_steps":
                return x
        return pltpu.with_memory_space_constraint(x, pltpu.HBM)

    constrained_schedule_hbm_ref = jax.tree_util.tree_map_with_path(
        _constrain_hbm, schedule_hbm)
    return _kernel(
        cu_q_lens,
        kv_lens,
        constrained_schedule_hbm_ref,
        pltpu.with_memory_space_constraint(q_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(new_kv_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(kv_cache_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(visibility_hbm, pltpu.HBM),
    )
