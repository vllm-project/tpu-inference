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
"""General split-K decode for global attention."""

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.stacked_rpa import (configs,
                                                            flash_attention,
                                                            utils)
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    config as decode_config
from tpu_inference.kernels.experimental.stacked_rpa.decode import stitch
from tpu_inference.kernels.experimental.stacked_rpa.decode.non_sliding_window import (
    bref, schedule)


@jax.named_scope("dense_combine_and_store")
def _dense_combine_and_store(
    step_idx: jax.Array,
    schedule_ref: schedule.DecodeSchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    m_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    m_carry_ref: jax.Ref,
    l_carry_ref: jax.Ref,
    acc_carry_ref: jax.Ref,
    carry_valid_ref: jax.Ref,
    *,
    cfgs: decode_config.DecodeConfig,
):
    """Combine dense-packed cells, carrying one request across steps."""
    n = cfgs.batch_size

    # Reduce each multi-cell group in place into its root cell.
    for root in range(n - 1):
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

    # The request straddling from the previous step always continues at cell 0.
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

    # The single non-final group becomes the carry for the next step.
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

    has_output = schedule_ref.is_final[step_idx, 0] == 1
    for root in range(1, n):
        has_output = jnp.logical_or(has_output,
                                    schedule_ref.is_final[step_idx, root] == 1)

    @pl.when(has_output)
    def _emit_all():
        # DenseBatchingORef only DMAs final roots, so idle normalized cells are
        # never written to HBM.
        _normalize_and_store(l_scratch_ref[...],
                             acc_scratch_ref[...],
                             o_vref,
                             cfgs=cfgs)


def _normalize_and_store(l_value, acc_value, o_vref, *, cfgs) -> None:
    """Normalize fresh or combined attention values into the output buffer."""
    denom = jnp.where(l_value == 0.0, 1.0, l_value)
    batch_l = utils.broadcast_minor(denom, acc_value.shape)
    if (cfgs.serve.dtype_out == jnp.float32
            or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
        result = lax.div(acc_value, batch_l)
    else:
        result = acc_value.astype(jnp.float32) * pl.reciprocal(
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


def create_scratch_shapes(cfgs: decode_config.DecodeConfig) -> tuple:
    """Build decode's persistent online-softmax and carry scratch plan."""
    dtype = configs.accum_dtype(cfgs.serve.dtype_out)
    return (
        pltpu.VMEM(cfgs.lm_scratch_shape, dtype=dtype),
        pltpu.VMEM(cfgs.lm_scratch_shape, dtype=dtype),
        pltpu.VMEM(cfgs.acc_scratch_shape, dtype=dtype),
        pltpu.VMEM((1, ) + tuple(cfgs.lm_scratch_shape[1:]), dtype=dtype),
        pltpu.VMEM((1, ) + tuple(cfgs.lm_scratch_shape[1:]), dtype=dtype),
        pltpu.VMEM((1, ) + tuple(cfgs.acc_scratch_shape[1:]), dtype=dtype),
        pltpu.SMEM((1, ), jnp.int32),
    )


def initialize_stacked_scratch(scratches: tuple[jax.Ref, ...]) -> None:
    """Invalidate generic split-K carry state before the first step."""
    *_, carry_valid_ref = scratches
    carry_valid_ref[0] = 0


def _get_kv_block_start(*, k_idx, kv_len, q_len, is_valid, cfgs):
    del kv_len, q_len
    return jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)


def _finish_fresh_step(
    *,
    step,
    schedule_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    o_vref,
    carry_refs,
    cfgs,
) -> None:
    """Restore idle cells to identity and finish a directly initialized block."""
    with jax.named_scope("stacked_idle_restore"):
        for b_idx in range(cfgs.batch_size):
            is_valid_b = schedule_ref.s_idx[step, b_idx] != -1
            m_scratch_ref[b_idx] = jnp.where(is_valid_b, m_scratch_ref[b_idx],
                                             -jnp.inf)
            l_scratch_ref[b_idx] = jnp.where(is_valid_b, l_scratch_ref[b_idx],
                                             0.0)
            acc_scratch_ref[b_idx] = jnp.where(is_valid_b,
                                               acc_scratch_ref[b_idx], 0.0)

    m_carry_ref, l_carry_ref, acc_carry_ref, carry_valid_ref = carry_refs
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


@jax.named_scope("rpa_body")
def rpa_body(
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    o_vref: jax.Ref,
    schedule_ref: schedule.DecodeSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    m_carry_ref: jax.Ref,
    l_carry_ref: jax.Ref,
    acc_carry_ref: jax.Ref,
    carry_valid_ref: jax.Ref,
    new_kv_vref: jax.Ref,
    *,
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    cfgs: decode_config.DecodeConfig,
):
    """Run one decode schedule step."""
    step = pl.program_id(0)

    with jax.named_scope("rpa_metadata"):
        processed_q_len = []
        processed_kv_len = []
        effective_kv_len = []
        bkv_sz_frm_cache_list = []
        new_kv_len_start_list = []
        skip_mask_list = []
        int_ty = cfgs.serve.int_ty
        for b_idx in range(cfgs.batch_size):
            s_idx = schedule_ref.s_idx[step, b_idx]
            is_valid = s_idx != -1
            k_idx = schedule_ref.k_idx[step, b_idx]
            kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
            q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
            q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
            q_len = q_end - q_start
            offset = kv_len - q_len

            k_id = _get_kv_block_start(
                k_idx=k_idx,
                kv_len=kv_len,
                q_len=q_len,
                is_valid=is_valid,
                cfgs=cfgs,
            )

            processed_q_len.append(offset.astype(int_ty))
            processed_kv_len.append(k_id.astype(int_ty))
            effective_kv_len.append(kv_len.astype(int_ty))
            skip_mask_list.append(schedule_ref.skip_mask[step, b_idx])

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

    with jax.named_scope("kv_stitch"):
        if cfgs.update_kv_cache:

            def _stitch_lane(b_idx):
                stitch_result = stitch.stitch_kv_lane(
                    kv_in_vref,
                    b_idx,
                    bkv_sz_frm_cache_list[b_idx],
                    new_kv_len_start_list[b_idx],
                    cfgs=cfgs,
                    new_kv_vref=new_kv_vref,
                )
                stitch.store_kv_lane(
                    kv_in_vref,
                    b_idx,
                    stitch_result,
                    cfgs=cfgs,
                )

            for b_idx in range(cfgs.batch_size):

                @pl.when(schedule_ref.do_writeback[step, b_idx] == 1)
                def _stitch_lane_if_needed(b_idx=b_idx):
                    _stitch_lane(b_idx)

    # Decode stages exactly one query block per sequence, so q1 and multi-token
    # speculative decode share the same fresh-partial path.
    def _compute_decode_tile():
        tile_valid_kv = jnp.stack([
            jnp.clip(
                effective_kv_len[b_idx].astype(jnp.int32) -
                processed_kv_len[b_idx].astype(jnp.int32),
                0,
                cfgs.bkv_sz,
            ) for b_idx in range(cfgs.batch_size)
        ])
        with jax.named_scope("k_load_decode"):
            k = jnp.stack(
                [
                    jnp.stack(
                        [
                            kv_in_vref[
                                b_idx,
                                kv_head * 2,
                                :,
                                :cfgs.bkv_sz,
                            ] for kv_head in range(cfgs.model.num_kv_heads)
                        ],
                        axis=0,
                    ) for b_idx in range(cfgs.batch_size)
                ],
                axis=0,
            )

        p, _, m_next, l_next = flash_attention.flash_attention_qk_softmax(
            q,
            k,
            None,
            None,
            processed_q_len=processed_q_len,
            processed_kv_len=processed_kv_len,
            effective_kv_len=effective_kv_len,
            skip_mask=skip_mask_list,
            cfgs=cfgs,
            bq_start=0,
            initialize=True,
        )
        with jax.named_scope("v_load_decode"):
            v = jnp.stack(
                [
                    jnp.stack(
                        [
                            kv_in_vref[
                                b_idx,
                                kv_head * 2 + 1,
                                :,
                                :cfgs.bkv_sz,
                            ] for kv_head in range(cfgs.model.num_kv_heads)
                        ],
                        axis=0,
                    ) for b_idx in range(cfgs.batch_size)
                ],
                axis=0,
            )
            keep = lax.broadcasted_iota(
                jnp.int32, v.shape,
                v.ndim - 1) < tile_valid_kv.reshape((cfgs.batch_size, ) +
                                                    (1, ) * (v.ndim - 1))
            v = jnp.where(keep, v, 0)

        acc_next = flash_attention.flash_attention_pv(
            p,
            v,
            None,
            None,
            cfgs=cfgs,
            initialize=True,
        )
        return m_next, l_next, acc_next

    m_next, l_next, acc_next = _compute_decode_tile()

    m_scratch_ref[...] = m_next
    l_scratch_ref[...] = l_next
    acc_scratch_ref[...] = acc_next
    _finish_fresh_step(
        step=step,
        schedule_ref=schedule_ref,
        m_scratch_ref=m_scratch_ref,
        l_scratch_ref=l_scratch_ref,
        acc_scratch_ref=acc_scratch_ref,
        o_vref=o_vref,
        carry_refs=(
            m_carry_ref,
            l_carry_ref,
            acc_carry_ref,
            carry_valid_ref,
        ),
        cfgs=cfgs,
    )


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    cfgs: decode_config.DecodeConfig,
):
    """Create decode's buffered pipeline allocations."""
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
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=True),
    )
    kv_cache_alloc = bref.KVBufferedRefSeqAlongLane.input_output(
        spec=kv_cache_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    q_alloc = bref.BatchingQRef.input(
        spec=q_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    o_alloc = bref.DenseBatchingORef.output(
        spec=o_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=2,
        use_lookahead=True,
        cfgs=cfgs,
    )
    return q_alloc, kv_cache_alloc, o_alloc


def get_kernel_name(cfgs: decode_config.DecodeConfig) -> str:
    name = f"RPA{cfgs.mode.symbol}-p{cfgs.serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    return name


def get_kernel_metadata(
    cfgs: decode_config.DecodeConfig, ) -> dict[str, str | int | float]:
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
    schedule_hbm: schedule.DecodeSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: decode_config.DecodeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Launch the decode pipeline and return output plus updated KV cache."""
    cfgs.validate_decode()
    if cfgs.model.sliding_window is not None:
        raise ValueError("Global decode does not accept a sliding window.")
    plan = schedule_hbm.plan
    # pallas_call materializes these scratch specs as trailing kernel arguments.
    q_alloc, kv_cache_alloc, o_alloc = create_allocs(kv_cache_hbm, q_hbm, cfgs)
    final_allocs = (q_alloc, kv_cache_alloc, o_alloc)
    schedule_scope = tuple(
        type(schedule_hbm).create_shape_dtype(
            cfgs, steps=plan.sched_window, plan=plan).scratch_shapes()
        for _ in range(1 if plan.fits_one_window else 2))
    dma_scope = pltpu.SemaphoreType.DMA((2, ) if plan.fits_one_window else (
        3, ))
    attention_scratch = create_scratch_shapes(cfgs) + (pltpu.VMEM(
        cfgs.new_kv_vmem_shape, dtype=cfgs.serve.dtype_kv), )

    def ragged_paged_attention_pipeline(
        cu_q_lens_ref: jax.Ref,
        kv_lens_ref: jax.Ref,
        schedule_hbm_ref: schedule.DecodeSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        final_allocs,
        schedule_ref,
        dma_sem,
        scratches,
    ):
        del o_kv_cache_hbm_ref

        def _run(final_allocs, schedule_ref, dma_sem, scratches):
            actual_steps = schedule_hbm_ref.actual_steps[0]
            w_size = plan.sched_window
            num_windows = plan.num_sched_windows

            initialize_stacked_scratch(scratches[:-1])
            flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)

            def _new_kv_init():
                if not cfgs.new_kv_resident:
                    return None
                sem_idx = 1 if plan.fits_one_window else 2
                copy = pltpu.make_async_copy(
                    new_kv_hbm_ref.at[:, :,
                                      pl.ds(0, cfgs.new_kv_padded_lanes)],
                    scratches[-1].at[:, :,
                                     pl.ds(0, cfgs.new_kv_padded_lanes)],
                    dma_sem.at[sem_idx],
                )
                copy.start()
                return copy

            def _kv_cache_init():
                kv_alloc = final_allocs[1]
                slot0_u32 = kv_alloc.window_ref.at[0].bitcast(jnp.uint32)
                slot0_u32[...] = jnp.zeros_like(slot0_u32)

            def _sched_copies(w, n_steps, buf, sem_idx):
                buf_leaves = jax.tree_util.tree_leaves(buf)
                descs = []
                for h, s in zip(flat_hbm, buf_leaves):
                    if h.shape[0] > 1:
                        s_len = s.shape[0]
                        raw_stride = s_len // w_size
                        copy_len = jnp.minimum(
                            pl.multiple_of(
                                pl.cdiv(n_steps * raw_stride, 1024) * 1024,
                                1024,
                            ),
                            s_len,
                        )
                        descs.append(
                            pltpu.make_async_copy(
                                h.at[pl.ds(
                                    pl.multiple_of(w * s_len, s_len),
                                    copy_len,
                                )],
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
                    in_specs=(q_alloc.spec, kv_cache_alloc.spec),
                    out_specs=(o_alloc.spec, ),
                )
                pipeline_func(
                    (q_hbm_ref, buf),
                    (kv_cache_hbm_ref, new_kv_hbm_ref, buf),
                    (o_hbm_ref, buf),
                    scratches=(buf, ) + scratches,
                    allocations=final_allocs,
                )

            if num_windows == 1:
                buf = schedule_ref[0]
                safe_steps = jnp.minimum(actual_steps, w_size)
                with jax.named_scope("sched_dma_load"):
                    descs = _sched_copies(0, safe_steps, buf, 0)
                    for copy in descs:
                        copy.start()
                    new_kv_copy = _new_kv_init()
                    _kv_cache_init()
                    jax.tree.map(lambda copy: copy.wait(), descs)
                    if new_kv_copy is not None:
                        new_kv_copy.wait()
                _run_pipeline(safe_steps, buf)
            else:
                buf0, buf1 = schedule_ref
                num_windows_actual = jnp.minimum(pl.cdiv(actual_steps, w_size),
                                                 num_windows)

                win0_steps = jnp.minimum(actual_steps, w_size)
                with jax.named_scope("sched_dma_load"):
                    for copy in _sched_copies(0, win0_steps, buf0, 0):
                        copy.start()
                    new_kv_copy = _new_kv_init()
                    _kv_cache_init()
                    if new_kv_copy is not None:
                        new_kv_copy.wait()

                def _window(w, _):
                    win_steps = jnp.clip(actual_steps - w * w_size, 0, w_size)
                    nxt = w + 1
                    nxt_steps = jnp.clip(actual_steps - nxt * w_size, 0,
                                         w_size)
                    even = (w % 2) == 0

                    @pl.when(jnp.logical_and(nxt < num_windows_actual, even))
                    def _prefetch_to_buf1():
                        with jax.named_scope("sched_dma_load"):
                            for copy in _sched_copies(nxt, nxt_steps, buf1, 1):
                                copy.start()

                    @pl.when(
                        jnp.logical_and(
                            nxt < num_windows_actual,
                            jnp.logical_not(even),
                        ))
                    def _prefetch_to_buf0():
                        with jax.named_scope("sched_dma_load"):
                            for copy in _sched_copies(nxt, nxt_steps, buf0, 0):
                                copy.start()

                    @pl.when(even)
                    def _compute_buf0():
                        jax.tree.map(
                            lambda copy: copy.wait(),
                            _sched_copies(w, win_steps, buf0, 0),
                        )
                        _run_pipeline(win_steps, buf0)

                    @pl.when(jnp.logical_not(even))
                    def _compute_buf1():
                        jax.tree.map(
                            lambda copy: copy.wait(),
                            _sched_copies(w, win_steps, buf1, 1),
                        )
                        _run_pipeline(win_steps, buf1)

                    return None

                jax.lax.fori_loop(0, num_windows_actual, _window, None)

        _run(final_allocs, schedule_ref, dma_sem, scratches)

    n_sched_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    q_in_idx = 2 + n_sched_leaves
    kv_in_idx = q_in_idx + 2
    kernel = pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=[q_hbm, kv_cache_hbm],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            scratch_shapes=(
                final_allocs,
                schedule_scope,
                dma_scope,
                attention_scratch,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        input_output_aliases={
            q_in_idx: 0,
            kv_in_idx: 1
        },
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )

    def _constrain_hbm(path, x):
        for p in path:
            key = getattr(p, "name", getattr(p, "key", None))
            if key == "actual_steps":
                return x
        return pltpu.with_memory_space_constraint(x, pltpu.HBM)

    constrained_schedule_hbm_ref = jax.tree_util.tree_map_with_path(
        _constrain_hbm, schedule_hbm)
    return kernel(
        cu_q_lens,
        kv_lens,
        constrained_schedule_hbm_ref,
        pltpu.with_memory_space_constraint(q_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(new_kv_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(kv_cache_hbm, pltpu.HBM),
    )
