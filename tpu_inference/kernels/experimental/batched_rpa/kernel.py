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
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.lax as lax
import jax.numpy as jnp
from jax._src.pallas.mosaic import pipeline

from tpu_inference.kernels.experimental.batched_rpa import (bref_override,
                                                            flash_attention)
from tpu_inference.kernels.experimental.batched_rpa import \
    schedule as schedule_lib


def make_rpa_kernel(config: schedule_lib.RPAConfig):
    q_packing = schedule_lib.get_dtype_packing(config.q_dtype)
    kv_packing = schedule_lib.get_dtype_packing(config.kv_dtype)

    def ragged_paged_attention_pipeline(
        # metadata inputs
        cu_q_lens_ref,
        kv_lens_ref,
        page_indices_ref,
        # schedule flattened (9 arrays)
        schedule_hbm,
        # hbm inputs
        o_hbm_alias_q_hbm_ref,
        new_kv_hbm_ref,
        kv_cache_hbm_ref,
        # output
        o_hbm_ref,
        kv_cache_out_ref,
        # scratch allocations
        schedule,
        m_scratch,
        l_scratch,
        acc_scratch,
        dma_sem,
    ):

        # q_hbm_ref shape: [q_times_kv, max_tokens, d] or similar
        # We use config values for dimensions where possible, or infer from refs
        kv_heads = config.num_kv_heads
        q_per_kv = config.num_q_heads_per_kv_head
        d_dim = config.head_dim
        _, _, num_kv_x2_packed, _, _ = kv_cache_hbm_ref.shape

        q_vmem_shape = (
            config.batch_size,
            kv_heads,
            config.bq_sz,
            q_per_kv // q_packing,
            q_packing,
            d_dim,
        )
        bkv_stride: int = (kv_heads * 2) // kv_packing

        if schedule_lib.has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        kv_vmem_shape = (
            config.batch_size,
            config.bkv_sz,
            bkv_stride,
            kv_packing,
            d_dim,
        )

        def rpa_body(
            # input
            q_vref,
            kv_in_vref,
            # output
            o_vref,
        ):

            def strided_load(ref, start, sz, step, *, dtype=None):
                assert schedule_lib.get_dtype_packing(ref.dtype) == 1
                assert len(ref.shape) == 2
                r, l = ref.shape  # noqa
                assert l % 128 == 0
                folds = l // 128
                ref = ref.reshape(r * folds, 128)
                start *= folds
                sz *= folds
                step *= folds
                assert sz % step == 0
                vec = jnp.concat(
                    [
                        ref[pl.ds(start + i, sz // step, step)]
                        for i in range(folds)
                    ],
                    axis=1,
                )
                if dtype is not None:
                    vec = pltpu.bitcast(vec, dtype)
                return vec

            def get_dtype_bitwidth(dtype):
                return jax._src.dtypes.itemsize_bits(dtype)

            def _strided_load_bkv(b_idx, start):
                assert start % kv_packing == 0
                start //= kv_packing
                kv_ref = (kv_in_vref.bitcast(jnp.uint32).at[b_idx].reshape(
                    config.bkv_sz * bkv_stride, config.head_dim))

                if kv_packing == 1:
                    k = strided_load(
                        kv_ref,
                        start,
                        config.bkv_sz * bkv_stride,
                        bkv_stride,
                        dtype=config.kv_dtype,
                    )
                    v = strided_load(
                        kv_ref,
                        start + 1,
                        config.bkv_sz * bkv_stride,
                        bkv_stride,
                        dtype=config.kv_dtype,
                    )
                    return [(k, v)]

                kv = strided_load(kv_ref, start, config.bkv_sz * bkv_stride,
                                  bkv_stride)
                bitwidth = 32 // kv_packing

                # If we want to convert 32-bits into 32//N number of N-bits value, naive
                # approach would be to perform 32//N number of 32-bits to N-bits conversion.
                # However, we can reduce number of instructions by utilizing binary tree.
                # 0: [32]
                # 1: [16, 16]
                # ...
                # log2(32//N): [N, N, ... N]

                def _convert_to_target_bitwidth(val, target_bitwidth: int):
                    curr_dtype = val.dtype
                    curr_bitwidth = get_dtype_bitwidth(curr_dtype)
                    assert target_bitwidth != curr_bitwidth, "No conversion is needed."

                    # We split val into two vals (left and right) where each have half of the
                    # original bitwidth.
                    next_bitwidth = curr_bitwidth // 2
                    next_dtype = jnp.dtype(f"uint{next_bitwidth}")

                    left = val.astype(next_dtype)

                    # Bitwise shift is only supported in uint32.
                    val_u32 = pltpu.bitcast(val, jnp.uint32)
                    val_u32_shifted = val_u32 >> next_bitwidth
                    # Convert back to original dtype.
                    val_shifted = pltpu.bitcast(val_u32_shifted, curr_dtype)
                    right = val_shifted.astype(next_dtype)

                    if next_bitwidth == target_bitwidth:
                        k = pltpu.bitcast(left, config.kv_dtype)
                        v = pltpu.bitcast(right, config.kv_dtype)
                        return [(k, v)]
                    else:
                        left_out = _convert_to_target_bitwidth(
                            left,
                            target_bitwidth=target_bitwidth,
                        )
                        right_out = _convert_to_target_bitwidth(
                            right,
                            target_bitwidth=target_bitwidth,
                        )
                        return left_out + right_out

                return _convert_to_target_bitwidth(kv,
                                                   target_bitwidth=bitwidth)

            step = pl.program_id(0)

            processed_q_len = []
            processed_kv_len = []
            effective_kv_len = []
            int_ty = jnp.int16
            for b in range(config.batch_size):
                idx = step * config.batch_size + b
                s_idx = schedule.s_idx[idx]
                is_valid = s_idx != -1
                q_idx = schedule.q_idx[idx]
                k_idx = schedule.k_idx[idx]
                k_id = lax.select(is_valid, k_idx * config.bkv_sz, 0)
                kv_len = lax.select(is_valid, kv_lens_ref[s_idx], 0)
                q_start = lax.select(is_valid, cu_q_lens_ref[s_idx], 0)
                q_end = lax.select(is_valid, cu_q_lens_ref[s_idx + 1], 0)
                q_len = q_end - q_start
                offset = kv_len - q_len

                processed_q_len.append(
                    (q_idx * config.bq_sz + offset).astype(int_ty))
                processed_kv_len.append(k_id.astype(int_ty))
                effective_kv_len.append(kv_len.astype(int_ty))

                start_k_idx = 0
                if config.sliding_window is not None:
                    sw_start_idx = (kv_len - q_len + q_idx * config.bq_sz -
                                    config.sliding_window + 1)
                    start_k_idx = jnp.maximum(0, sw_start_idx) // config.bkv_sz

                is_first_k_block = k_idx == start_k_idx
                reset_cond = is_valid & is_first_k_block
                m_scratch[b] = jnp.where(reset_cond, -jnp.inf, m_scratch[b])
                l_scratch[b] = jnp.where(reset_cond, 0.0, l_scratch[b])
                acc_scratch[b] = jnp.where(reset_cond, 0.0, acc_scratch[b])

            q_b = []
            for b in range(config.batch_size):
                q_p = config.num_q_heads_per_kv_head // q_packing
                q_ref = (q_vref.bitcast(jnp.uint32).at[b].reshape(
                    config.num_kv_heads * config.bq_sz * q_p, config.head_dim))
                q_loaded = strided_load(
                    q_ref,
                    0,
                    config.num_kv_heads * config.bq_sz * q_p,
                    1,
                    dtype=config.q_dtype,
                )
                q_b.append(
                    q_loaded.reshape(
                        config.num_kv_heads,
                        config.bq_sz * config.num_q_heads_per_kv_head,
                        config.head_dim,
                    ))
            q_flat = jnp.stack(q_b, axis=0)

            k_b = []
            v_b = []
            for b in range(config.batch_size):
                heads_per_load = max(1, kv_packing // 2)
                ks = []
                vs = []
                for kv_head_start in range(0, config.num_kv_heads,
                                           heads_per_load):
                    bkv_lst = _strided_load_bkv(
                        b,
                        kv_head_start * 2,
                    )
                    ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                    vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
                k, v = jnp.concatenate(ks, axis=0), jnp.concatenate(vs, axis=0)
                k = k.reshape(-1, config.bkv_sz, config.head_dim)
                v = v.reshape(-1, config.bkv_sz, config.head_dim)

                k = k[:config.num_kv_heads]
                v = v[:config.num_kv_heads]
                k_b.append(k)
                v_b.append(v)

            # Stack to (batch, num_heads, bkv_sz, 128)
            k = jnp.stack(k_b, axis=0)
            v = jnp.stack(v_b, axis=0)

            m_val = m_scratch[...]
            l_val = l_scratch[...]
            acc_val = acc_scratch[...]

            m_next, l_next, o_next = flash_attention.flash_attention(
                q_flat[...],
                k,
                v,
                acc_val,
                m_val,
                l_val,
                processed_q_len=processed_q_len,
                processed_kv_len=processed_kv_len,
                effective_kv_len=effective_kv_len,
                config=config,
            )
            m_scratch[...] = m_next
            l_scratch[...] = l_next
            acc_scratch[...] = o_next

        kv_cache_spec = pl.BlockSpec(
            block_shape=kv_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, ),
            pipeline_mode=pl.Buffered(buffer_count=config.n_buffer,
                                      use_lookahead=True),
        )
        q_spec = pl.BlockSpec(
            block_shape=q_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, ),
            pipeline_mode=pl.Buffered(buffer_count=config.n_buffer,
                                      use_lookahead=True),
        )
        o_spec = pl.BlockSpec(
            block_shape=q_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, ),
            pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
        )

        # hbm_kv_packed_stride = (config.num_kv_heads * 2 + kv_packing - 1) // kv_packing
        kv_cache_alloc = bref_override.KVBufferedRef.create(
            spec=kv_cache_spec,
            source_memory_space=kv_cache_hbm_ref,
            bkv_p_cache=config.bkv_p_cache,
            bkv_p_new=config.bkv_p_new,
            page_size=config.page_size,
            batch_size=config.batch_size,
            hbm_stride=num_kv_x2_packed,
            page_size_log2=config.page_size_log2,
            page_size_mask=config.page_size_mask,
            buffer_count=config.n_buffer,
            use_lookahead=True,
        )
        q_alloc = bref_override.BatchingQRef.create(
            spec=q_spec,
            source_memory_space=o_hbm_alias_q_hbm_ref,
            bq_sz=config.bq_sz,
            batch_size=config.batch_size,
            buffer_count=config.n_buffer,
            use_lookahead=True,
        )
        o_alloc = bref_override.BatchingORef.create(
            spec=o_spec,
            source_memory_space=o_hbm_ref,
            batch_size=config.batch_size,
            config=config,
            buffer_count=2,
            use_lookahead=False,
        )

        def _run(final_allocs):
            actual_steps = schedule_hbm.actual_steps[0]
            safe_steps = jnp.minimum(actual_steps, config.max_steps_ub)
            grid = (safe_steps, )

            sem = dma_sem.at[0]
            flat_hbm = jax.tree_util.tree_leaves(schedule_hbm)
            flat_smem = jax.tree_util.tree_leaves(schedule)
            dma_list = []
            for h, s in zip(flat_hbm, flat_smem):
                if h.shape[0] != 1:
                    fetch_size = (h.shape[0] //
                                  config.max_steps_ub) * safe_steps
                    fetch_size = pl.cdiv(fetch_size, 1024) * 1024

                    copy = pltpu.make_async_copy(
                        h.at[pl.ds(0, fetch_size)],
                        s.at[pl.ds(0, fetch_size)],
                        sem,
                    )
                    copy.start()
                    dma_list.append(copy)

            # Zero-initialize KV cache buffer
            # we do this weird dma thing to overlap with HBM -> SMEM fetch

            kv_alloc = final_allocs[1]
            num_lanes = pltpu.get_tpu_info().num_lanes
            kv_ref_flat = kv_alloc.window_ref.bitcast(jnp.uint32).reshape(
                -1, num_lanes)
            kv_ref_flat[...] = jnp.zeros_like(kv_ref_flat)

            for copy in dma_list:
                copy.wait()

            pipeline_func = pipeline.emit_pipeline(
                body=rpa_body,
                grid=grid,
                in_specs=[
                    q_spec,
                    kv_cache_spec,
                ],
                out_specs=[
                    o_spec,
                ],
            )

            pipeline_func(
                (o_hbm_alias_q_hbm_ref, schedule),
                (kv_cache_hbm_ref, new_kv_hbm_ref, schedule, page_indices_ref),
                (o_hbm_ref, schedule, acc_scratch, l_scratch),
                allocations=final_allocs,
            )

        return pl.run_scoped(_run, (q_alloc, kv_cache_alloc, o_alloc))

    num_pages = config.num_seq * config.pages_per_seq
    if config.total_num_pages is not None:
        num_pages = config.total_num_pages
    q_packing = schedule_lib.get_dtype_packing(config.q_dtype)
    kv_packing = schedule_lib.get_dtype_packing(config.kv_dtype)
    num_kv_heads_x2_packed = (config.num_kv_heads * 2 + kv_packing -
                              1) // kv_packing

    out_shape = [
        jax.ShapeDtypeStruct(
            (
                config.num_kv_heads,
                config.total_q_tokens,
                config.num_q_heads_per_kv_head // q_packing,
                q_packing,
                config.head_dim,
            ),
            config.q_dtype,
        ),
        jax.ShapeDtypeStruct(
            (
                num_pages,
                config.page_size,
                num_kv_heads_x2_packed,
                kv_packing,
                config.head_dim,
            ),
            config.kv_dtype,
        ),
    ]

    schedule_shapes = schedule_lib.RPASchedule.smem_specs(config)
    scratch_shapes = [
        schedule_shapes,
        pltpu.VMEM(
            (
                config.batch_size,
                config.num_kv_heads,
                config.bq_sz * config.num_q_heads_per_kv_head,
                128,
            ),
            dtype=config.out_dtype,
        ),  # m
        pltpu.VMEM(
            (
                config.batch_size,
                config.num_kv_heads,
                config.bq_sz * config.num_q_heads_per_kv_head,
                128,
            ),
            dtype=config.out_dtype,
        ),  # l
        pltpu.VMEM(
            (
                config.batch_size,
                config.num_kv_heads,
                config.bq_sz * config.num_q_heads_per_kv_head,
                config.head_dim,
            ),
            dtype=config.out_dtype,
        ),  # acc
        pltpu.SemaphoreType.DMA((1, )),  # dma_sem
    ]
    in_specs = [
        schedule_lib.RPASchedule.test_specs(config),  # 9 refs
        pl.BlockSpec(memory_space=pltpu.HBM),  # o_hbm_alias_q_hbm_ref
        pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
        pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
    ]
    input_output_aliases = {12: 0, 14: 1}

    scope_name = f"RPA{config.case.symbol}-p{config.page_size}-b{config.batch_size}-q{config.bq_sz}-k{config.bkv_sz}"
    if config.sliding_window:
        scope_name += f"-sw{config.sliding_window}"
    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            grid=(1, ),
            num_scalar_prefetch=3,
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # o_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_out_ref
            ],
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=config.vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        input_output_aliases=input_output_aliases,
        name=scope_name,
    )
