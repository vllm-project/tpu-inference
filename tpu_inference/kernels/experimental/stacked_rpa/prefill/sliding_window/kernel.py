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
"""Scratch-free, one-KV-block sliding-window prefill implementation."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import (flash_attention,
                                                            utils)
from tpu_inference.kernels.experimental.stacked_rpa.prefill import (config,
                                                                    stitch)
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import (
    bref, schedule)


def _normalize_and_store_chunk(
    l_value,
    acc_value,
    o_vref,
    bq_start,
    *,
    cfgs: config.PrefillConfig,
) -> None:
    denom = jnp.where(l_value == 0.0, 1.0, l_value)
    chunk_l = utils.broadcast_minor(denom, acc_value.shape)
    if (cfgs.serve.dtype_out == jnp.float32
            or cfgs.serve.dtype_out == chunk_l.dtype == jnp.bfloat16):
        result = lax.div(acc_value, chunk_l)
    else:
        result = acc_value.astype(jnp.float32) * pl.reciprocal(
            chunk_l.astype(jnp.float32), approx=True)
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

    q_packing = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
    out_ref = o_vref.bitcast(jnp.uint32).reshape(
        cfgs.batch_size,
        cfgs.model.num_kv_heads,
        cfgs.bq_sz * q_packing,
        cfgs.aligned_q_head_dim,
    )
    chunk_rows = cfgs.bq_c_sz * q_packing
    out_u32 = pltpu.bitcast(out, jnp.uint32).reshape(
        cfgs.batch_size,
        cfgs.model.num_kv_heads,
        chunk_rows,
        cfgs.aligned_q_head_dim,
    )
    row_start = pl.multiple_of(bq_start * q_packing, chunk_rows)
    out_ref[:, :, pl.ds(row_start, chunk_rows), :] = out_u32


@jax.named_scope("rpa_body_sliding_window_prefill")
def rpa_body(
    q_vref,
    kv_in_vref,
    o_vref,
    schedule_ref,
    *,
    cfgs: config.PrefillConfig,
):
    """Compute one fresh local-attention result for every scheduled Q block."""
    step = pl.program_id(0)
    processed_q_len = []
    processed_kv_len = []
    effective_kv_len = []
    cache_size = []
    new_kv_start = []
    int_ty = cfgs.serve.int_ty

    with jax.named_scope("rpa_metadata"):
        for b_idx in range(cfgs.batch_size):
            (
                _q_src,
                _q_size,
                processed_q,
                processed_kv,
                effective_kv,
                lane_cache_size,
                lane_new_kv_start,
            ) = schedule_ref.get_cell_metadata(step, b_idx)
            processed_q_len.append(processed_q.astype(int_ty))
            processed_kv_len.append(processed_kv.astype(int_ty))
            effective_kv_len.append(effective_kv.astype(int_ty))
            cache_size.append(lane_cache_size.astype(jnp.int32))
            new_kv_start.append(lane_new_kv_start.astype(jnp.int32))

    with jax.named_scope("kv_stitch"):
        for b_idx in range(cfgs.batch_size):
            stitch_result = stitch.stitch_kv_lane(
                kv_in_vref,
                b_idx,
                cache_size[b_idx],
                new_kv_start[b_idx],
                cfgs=cfgs,
            )
            stitch.store_kv_lane(
                kv_in_vref,
                b_idx,
                stitch_result,
                cfgs=cfgs,
            )

    compute_kv_size = schedule_ref.plan.compute_bkv_size
    num_lanes = pltpu.get_tpu_info().num_lanes
    offsets = None
    tile_processed_kv_len = processed_kv_len
    if compute_kv_size < cfgs.bkv_sz:
        offsets = []
        tile_processed_kv_len = []
        for b_idx in range(cfgs.batch_size):
            raw_window_start = jnp.maximum(
                processed_q_len[b_idx].astype(jnp.int32) -
                cfgs.model.sliding_window,
                0,
            )
            offset = jnp.maximum(
                raw_window_start - processed_kv_len[b_idx].astype(jnp.int32),
                0,
            )
            offset = pl.multiple_of(
                (offset // num_lanes) * num_lanes,
                num_lanes,
            )
            offsets.append(offset)
            tile_processed_kv_len.append(processed_kv_len[b_idx] + offset)

    tile_valid_kv = jnp.stack([
        jnp.clip(
            effective_kv_len[b_idx].astype(jnp.int32) -
            tile_processed_kv_len[b_idx].astype(jnp.int32),
            0,
            compute_kv_size,
        ) for b_idx in range(cfgs.batch_size)
    ])

    def kv_slice(b_idx, head_idx):
        if offsets is None:
            return kv_in_vref[b_idx, head_idx, :, :compute_kv_size]
        return kv_in_vref[
            b_idx,
            head_idx,
            :,
            pl.ds(offsets[b_idx], compute_kv_size),
        ]

    def compute_q_chunk(chunk_idx, _):
        bq_start = chunk_idx * cfgs.bq_c_sz
        with jax.named_scope("q_load"):
            q_packing = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
            q_token_start = pl.multiple_of(bq_start, cfgs.bq_c_sz)
            q_chunk = jnp.stack(
                [
                    jnp.stack(
                        [
                            utils.strided_load(
                                q_vref.at[
                                    b_idx,
                                    kv_head,
                                    pl.ds(q_token_start, cfgs.bq_c_sz),
                                    :,
                                    :,
                                    :,
                                ].bitcast(jnp.uint32).reshape(
                                    -1, cfgs.aligned_q_head_dim),
                                0,
                                cfgs.bq_c_sz * q_packing,
                                1,
                                dtype=cfgs.serve.dtype_q,
                            ).reshape(
                                cfgs.bq_c_sz *
                                cfgs.aligned_num_q_heads_per_kv_head,
                                cfgs.aligned_q_head_dim,
                            ) for kv_head in range(cfgs.model.num_kv_heads)
                        ],
                        axis=0,
                    ) for b_idx in range(cfgs.batch_size)
                ],
                axis=0,
            )
            if cfgs.aligned_q_head_dim != cfgs.aligned_kv_head_dim:
                q_chunk = q_chunk[..., :cfgs.aligned_kv_head_dim]

        # Reloading K and V from the buffered VMEM tile for each Q chunk keeps
        # their register values local to QK and PV. The runtime loop prevents
        # every chunk's intermediates from being live in one unrolled program.
        with jax.named_scope("k_load_prefill"):
            k = jnp.stack(
                [
                    jnp.stack(
                        [
                            kv_slice(b_idx, kv_head * 2)
                            for kv_head in range(cfgs.model.num_kv_heads)
                        ],
                        axis=0,
                    ) for b_idx in range(cfgs.batch_size)
                ],
                axis=0,
            )
        p, _, _, l_next = flash_attention.flash_attention_qk_softmax(
            q_chunk,
            k,
            None,
            None,
            processed_q_len=processed_q_len,
            processed_kv_len=tile_processed_kv_len,
            effective_kv_len=effective_kv_len,
            skip_mask=None,
            cfgs=cfgs,
            bq_start=bq_start,
            initialize=True,
        )

        with jax.named_scope("v_load_prefill"):
            v = jnp.stack(
                [
                    jnp.stack(
                        [
                            kv_slice(b_idx, kv_head * 2 + 1)
                            for kv_head in range(cfgs.model.num_kv_heads)
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
        _normalize_and_store_chunk(
            l_next,
            acc_next,
            o_vref,
            bq_start,
            cfgs=cfgs,
        )
        return None

    lax.fori_loop(0, cfgs.bq_sz // cfgs.bq_c_sz, compute_q_chunk, None)


def create_allocs(kv_cache_hbm_ref, q_hbm_ref, cfgs):
    kv_spec = pl.BlockSpec(
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
    kv_alloc = bref.SlidingWindowKVRef.input_output(
        spec=kv_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    q_alloc = bref.SlidingWindowQRef.input(
        spec=q_spec,
        dtype_or_type=q_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    o_alloc = bref.SlidingWindowORef.output(
        spec=o_spec,
        dtype_or_type=q_hbm_ref,
        buffer_count=2,
        use_lookahead=True,
        cfgs=cfgs,
    )
    return q_alloc, kv_alloc, o_alloc


def get_kernel_name(cfgs: config.PrefillConfig) -> str:
    return (f"RPA{cfgs.mode.symbol}-p{cfgs.serve.page_size}"
            f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
            f"-sw{cfgs.model.sliding_window}-one-kv")


def get_kernel_metadata(cfgs: config.PrefillConfig):
    cfgs_dict = dataclasses.asdict(cfgs)
    metadata = {}
    for path, value in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(value, str | int | float):
            value = str(value)
        metadata[key] = value
    return metadata


def rpa_kernel(
    schedule_hbm,
    q_hbm,
    new_kv_hbm,
    kv_cache_hbm,
    *,
    cfgs: config.PrefillConfig,
):
    """Launch the scratch-free sliding-window prefill pipeline."""
    schedule.validate_config(cfgs)
    plan = schedule_hbm.plan
    # pallas_call materializes these scratch specs as trailing kernel arguments.
    q_alloc, kv_alloc, o_alloc = create_allocs(kv_cache_hbm, q_hbm, cfgs)
    final_allocs = (q_alloc, kv_alloc, o_alloc)
    schedule_scope = tuple(
        type(schedule_hbm).create_shape_dtype(
            cfgs, steps=plan.sched_window, plan=plan).scratch_shapes()
        for _ in range(1 if plan.fits_one_window else 2))
    dma_scope = pltpu.SemaphoreType.DMA((1, ) if plan.fits_one_window else (
        2, ))

    def ragged_paged_attention_pipeline(
        schedule_hbm_ref,
        q_hbm_ref,
        new_kv_hbm_ref,
        kv_cache_hbm_ref,
        o_hbm_ref,
        o_kv_cache_hbm_ref,
        final_allocs,
        schedule_ref,
        dma_sem,
    ):
        del o_kv_cache_hbm_ref

        def _run(final_allocs, schedule_ref, dma_sem):
            actual_steps = schedule_hbm_ref.actual_steps[0]
            window_size = plan.sched_window
            num_windows = plan.num_sched_windows
            flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)

            def initialize_kv_cache():
                slot0_u32 = final_allocs[1].window_ref.at[0].bitcast(
                    jnp.uint32)
                slot0_u32[...] = jnp.zeros_like(slot0_u32)

            def schedule_copies(window, num_steps, buffer, sem_idx):
                buffer_leaves = jax.tree_util.tree_leaves(buffer)
                copies = []
                for hbm_ref, smem_ref in zip(flat_hbm, buffer_leaves):
                    if hbm_ref.shape[0] > 1:
                        smem_len = smem_ref.shape[0]
                        raw_stride = smem_len // window_size
                        copy_len = jnp.minimum(
                            pl.multiple_of(
                                pl.cdiv(num_steps * raw_stride, 1024) * 1024,
                                1024,
                            ),
                            smem_len,
                        )
                        copies.append(
                            pltpu.make_async_copy(
                                hbm_ref.at[pl.ds(
                                    pl.multiple_of(
                                        window * smem_len,
                                        smem_len,
                                    ),
                                    copy_len,
                                )],
                                smem_ref.at[pl.ds(0, copy_len)],
                                dma_sem.at[sem_idx],
                            ))
                return copies

            def run_pipeline(num_steps, buffer):
                pipeline = pltpu.emit_pipeline(
                    body=functools.partial(rpa_body, cfgs=cfgs),
                    grid=(num_steps, ),
                    in_specs=(q_alloc.spec, kv_alloc.spec),
                    out_specs=(o_alloc.spec, ),
                )
                pipeline(
                    (q_hbm_ref, buffer),
                    (kv_cache_hbm_ref, new_kv_hbm_ref, buffer),
                    (o_hbm_ref, buffer),
                    scratches=(buffer, ),
                    allocations=final_allocs,
                )

            if num_windows == 1:
                buffer = schedule_ref[0]
                safe_steps = jnp.minimum(actual_steps, window_size)
                copies = schedule_copies(0, safe_steps, buffer, 0)
                for copy in copies:
                    copy.start()
                initialize_kv_cache()
                jax.tree.map(lambda copy: copy.wait(), copies)
                run_pipeline(safe_steps, buffer)
            else:
                buffer0, buffer1 = schedule_ref
                actual_windows = jnp.minimum(
                    pl.cdiv(actual_steps, window_size), num_windows)
                first_steps = jnp.minimum(actual_steps, window_size)
                for copy in schedule_copies(0, first_steps, buffer0, 0):
                    copy.start()
                initialize_kv_cache()

                def run_window(window, _):
                    num_steps = jnp.clip(
                        actual_steps - window * window_size,
                        0,
                        window_size,
                    )
                    next_window = window + 1
                    next_steps = jnp.clip(
                        actual_steps - next_window * window_size,
                        0,
                        window_size,
                    )
                    even = (window % 2) == 0

                    @pl.when(
                        jnp.logical_and(next_window < actual_windows, even))
                    def prefetch_buffer1():
                        for copy in schedule_copies(next_window, next_steps,
                                                    buffer1, 1):
                            copy.start()

                    @pl.when(
                        jnp.logical_and(
                            next_window < actual_windows,
                            jnp.logical_not(even),
                        ))
                    def prefetch_buffer0():
                        for copy in schedule_copies(next_window, next_steps,
                                                    buffer0, 0):
                            copy.start()

                    @pl.when(even)
                    def compute_buffer0():
                        jax.tree.map(
                            lambda copy: copy.wait(),
                            schedule_copies(window, num_steps, buffer0, 0),
                        )
                        run_pipeline(num_steps, buffer0)

                    @pl.when(jnp.logical_not(even))
                    def compute_buffer1():
                        jax.tree.map(
                            lambda copy: copy.wait(),
                            schedule_copies(window, num_steps, buffer1, 1),
                        )
                        run_pipeline(num_steps, buffer1)

                    return None

                jax.lax.fori_loop(0, actual_windows, run_window, None)

        _run(final_allocs, schedule_ref, dma_sem)

    num_schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    q_input_idx = num_schedule_leaves
    kv_input_idx = q_input_idx + 2
    kernel = pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=[q_hbm, kv_cache_hbm],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
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
            scratch_shapes=(final_allocs, schedule_scope, dma_scope),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        input_output_aliases={
            q_input_idx: 0,
            kv_input_idx: 1
        },
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )

    def constrain_hbm(path, value):
        for part in path:
            key = getattr(part, "name", getattr(part, "key", None))
            if key == "actual_steps":
                return value
        return pltpu.with_memory_space_constraint(value, pltpu.HBM)

    constrained_schedule = jax.tree_util.tree_map_with_path(
        constrain_hbm, schedule_hbm)
    return kernel(
        constrained_schedule,
        pltpu.with_memory_space_constraint(q_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(new_kv_hbm, pltpu.HBM),
        pltpu.with_memory_space_constraint(kv_cache_hbm, pltpu.HBM),
    )
