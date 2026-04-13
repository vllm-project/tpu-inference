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

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.batched_rpa import (bref_override,
                                                            configs,
                                                            flash_attention,
                                                            schedule, utils)

# Define inner kernel.


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RPAConfig,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
    kv_ref = kv_u32_ref.reshape(-1, cfgs.model.head_dim)

    if cfgs.serve.packing_kv == 1:
        k = utils.strided_load(
            kv_ref,
            start,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        v = utils.strided_load(
            kv_ref,
            start + 1,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        return [(k, v)]

    kv = utils.strided_load(kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride,
                            cfgs.bkv_stride)
    bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

    return utils.convert_to_target_bitwidth(kv,
                                            target_bitwidth=bitwidth,
                                            kv_dtype=cfgs.serve.dtype_kv)


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RPASchedule,
    acc_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    o_vref: jax.Ref,
    *,
    cfgs: configs.RPAConfig,
):

    def _accum(b_idx: int):
        batch_acc = acc_scratch_ref[b_idx]
        batch_l = l_scratch_ref[b_idx]
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (cfgs.serve.dtype_out == jnp.float32
                or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pl.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.model.head_dim)
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


def rpa_body(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    # Scratches.
    schedule_ref: schedule.RPASchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    # Configs.
    cfgs: configs.RPAConfig,
):
    step = pl.program_id(0)

    # Step 1: Fetch metadata.
    processed_q_len = []
    processed_kv_len = []
    effective_kv_len = []
    int_ty = cfgs.serve.int_ty
    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
        q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
        q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
        q_len = q_end - q_start
        offset = kv_len - q_len

        processed_q_len.append((q_idx * cfgs.bq_sz + offset).astype(int_ty))
        processed_kv_len.append(k_id.astype(int_ty))
        effective_kv_len.append(kv_len.astype(int_ty))

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = kv_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

        is_first_k_block = k_idx == start_k_idx
        reset_cond = jnp.logical_and(is_valid, is_first_k_block)
        m_scratch_ref[b_idx] = jnp.where(reset_cond, -jnp.inf,
                                         m_scratch_ref[b_idx])
        l_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0, l_scratch_ref[b_idx])
        acc_scratch_ref[b_idx] = jnp.where(reset_cond, 0.0,
                                           acc_scratch_ref[b_idx])

    # Step 2: Fetch inputs.
    q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
    q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.model.head_dim)
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
        cfgs.model.head_dim,
    )

    # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
    # where bkv_stride ~= num_kv_heads * 2 // kv_packing
    # to 2x (batch, num_kv_heads, bkv_sz, d)
    # We use strided_load to avoid the expensive transpose.
    k_b = []
    v_b = []
    for b_idx in range(cfgs.batch_size):
        heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
        ks = []
        vs = []
        for kv_head_start in range(0, cfgs.model.num_kv_heads, heads_per_load):
            bkv_lst = strided_load_bkv(
                kv_in_vref,
                b_idx,
                kv_head_start * 2,
                cfgs=cfgs,
            )
            ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
            vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
        k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
        k = k.reshape(-1, cfgs.bkv_sz, cfgs.model.head_dim)
        v = v.reshape(-1, cfgs.bkv_sz, cfgs.model.head_dim)

        k = k[:cfgs.model.num_kv_heads]
        v = v[:cfgs.model.num_kv_heads]
        k_b.append(k)
        v_b.append(v)
    # Stack to (batch, num_heads, bkv_sz, num_lanes)
    k = jnp.stack(k_b, axis=0)
    v = jnp.stack(v_b, axis=0)

    # Step 3: Perform compute.
    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]

    m_next, l_next, o_next = flash_attention.flash_attention(
        q,
        k,
        v,
        acc_val,
        m_val,
        l_val,
        processed_q_len=processed_q_len,
        processed_kv_len=processed_kv_len,
        effective_kv_len=effective_kv_len,
        cfgs=cfgs,
    )
    m_scratch_ref[...] = m_next
    l_scratch_ref[...] = l_next
    acc_scratch_ref[...] = o_next

    # Step 4: Write back outputs.
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
    kv_cache_hbm_ref: jax.Ref, o_hbm_ref: jax.Ref, cfgs: configs.RPAConfig
) -> tuple[
        bref_override.BatchingQRef,
        bref_override.KVBufferedRef,
        bref_override.BatchingORef,
]:
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

    kv_cache_alloc = bref_override.KVBufferedRef.input_output(
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

    return q_alloc, kv_cache_alloc, o_alloc


def get_kernel_name(cfgs: configs.RPAConfig) -> str:
    name = f"RPA{cfgs.mode.symbol}-p{cfgs.serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    if cfgs.model.sliding_window:
        name += f"-sw{cfgs.model.sliding_window}"
    return name


def get_kernel_metadata(
    cfgs: configs.RPAConfig, ) -> dict[str, str | int | float]:
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
    page_indices: jax.Array,
    schedule_hbm: schedule.RPASchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: configs.RPAConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention with scheduler data.

    Args:
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
        page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        schedule: Output of scheduler kernel. It informs which: 1. seqs 2. q block
            3. kv block that should be processed at a given step.
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
        page_indices_ref: jax.Ref,
        # Inputs.
        schedule_hbm_ref: schedule.RPASchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
    ):

        del o_kv_cache_hbm_ref

        q_alloc, kv_cache_alloc, o_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, cfgs)

        actual_steps = schedule_hbm_ref.actual_steps[0]
        safe_steps = jnp.minimum(actual_steps, cfgs.max_steps_ub)
        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                rpa_body,
                cfgs=cfgs,
                cu_q_lens_ref=cu_q_lens_ref,
                kv_lens_ref=kv_lens_ref,
            ),
            grid=(safe_steps, ),
            in_specs=(q_alloc.spec, kv_cache_alloc.spec),
            out_specs=(o_alloc.spec, ),
        )

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc),
            schedule_ref=schedule_hbm_ref.scratch_shapes(),
            dma_sem=pltpu.SemaphoreType.DMA((1, )),
            scratches=(
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # m
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # l
                pltpu.VMEM(
                    cfgs.acc_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # acc
            ),
        )
        def _run(final_allocs, schedule_ref, dma_sem, scratches):

            # Transfer schedule from HBM to SMEM --- we only copy what we need. Since
            # we almost always over-allocate schedule size, we only want to copy a
            # small portion of it from HBM to SMEM.
            flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
            flat_smem = jax.tree_util.tree_leaves(schedule_ref)
            dma_list = []
            for h, s in zip(flat_hbm, flat_smem):
                if h.memory_space == pltpu.HBM:
                    read_size = (h.shape[0] // cfgs.max_steps_ub) * safe_steps
                    read_size = utils.align_to(read_size, 1024)

                    copy = pltpu.make_async_copy(
                        h.at[pl.ds(0, read_size)],
                        s.at[pl.ds(0, read_size)],
                        dma_sem.at[0],
                    )
                    copy.start()
                    dma_list.append(copy)

            # Initialize KV cache to zeros.
            # When perfomring p * v, we perform causal masking on lhs (p) by zeroing
            # out columns that should not be processed for a given row. Even if we
            # don't perform masking on rows of rhs (v), the output is still correct
            # since reuslt of multiplication will be zero thanks zero on lhs. However,
            # this assumption does not hold if a row of rhs has NaNs. To avoid this,
            # we initiallize scratch memory with non-zero values. Even if the scratch
            # memory is storing kv cache from previous step, as long as the data is
            # not NaNs, there will be no numeric concerns.
            num_lanes = pltpu.get_tpu_info().num_lanes
            kv_alloc = final_allocs[1]
            kv_ref_flat = kv_alloc.window_ref.bitcast(jnp.uint32).reshape(
                -1, num_lanes)
            kv_ref_flat[...] = jnp.zeros_like(kv_ref_flat)

            jax.tree.map(lambda x: x.wait(), dma_list)

            pipeline_func(
                (q_hbm_ref, schedule_ref),
                (kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref,
                 page_indices_ref),
                (o_hbm_ref, schedule_ref),
                scratches=(schedule_ref, ) + scratches,
                allocations=final_allocs,
            )

        _run()

    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=[q_hbm, kv_cache_hbm],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
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
        ),
        input_output_aliases={
            12: 0,
            14: 1
        },
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        cu_q_lens,
        kv_lens,
        page_indices,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
    )
