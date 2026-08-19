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

# yapf: disable
from tpu_inference.kernels.experimental.batched_rpa import (bref_override,
                                                            configs,
                                                            flash_attention,
                                                            ring,
                                                            schedule,
                                                            stitch_utils,
                                                            utils)

# yapf: enable

# Define inner kernel.


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
    kv_ref = kv_u32_ref.reshape(-1, cfgs.aligned_kv_head_dim)

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
    schedule_ref: schedule.RpaSchedule,
    acc_val: jax.Array,
    l_val: jax.Array,
    m_val: jax.Array,
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
):

    def _accum(b_idx: jax.Array, batch_acc: jax.Array, batch_l: jax.Array):
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (cfgs.serve.dtype_out == jnp.float32
                or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pl.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        pad_width = [[0, 0] for _ in range(out.ndim)]
        pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.aligned_kv_head_dim
        out = jnp.pad(out, pad_width, constant_values=0)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    def _stage_lse(b_idx: int, batch_m: jax.Array, batch_l: jax.Array):
        lse_val = batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
        lse_val = lse_val.astype(cfgs.serve.dtype_out)
        aligned_q = cfgs.aligned_num_q_heads_per_kv_head
        if cfgs.lse_row_stride != aligned_q:
            # Pad each token's rows to the sublane-aligned stride the HBM
            # buffer is laid out with.
            kv, tq, lanes = lse_val.shape
            lse_val = jnp.pad(
                lse_val.reshape(kv, tq // aligned_q, aligned_q, lanes),
                ((0, 0), (0, 0), (0, cfgs.lse_row_stride - aligned_q),
                 (0, 0)),
            ).reshape(kv, -1, lanes)
        lse_o_vref[b_idx] = lse_val

    if cfgs.fuse_accum:
        for b in range(cfgs.batch_size):
            _accum(b, acc_val[b], l_val[b])
    else:
        # Adding a conditional causes a scheduling barrier. In prefill, we often
        # use small block sizes, so it's not worth executing the accumulation
        # on every block. In decode, because of the large block sizes / and or
        # batch sizes, we almost always use accumulation on every block. Please
        # tune `fuse_accum` for your use case.
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            b_acc = acc_val[b]
            b_l = l_val[b]
            accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
            jax.lax.cond(is_last_k, accum_named_call, lambda *_: None, b,
                         b_acc, b_l)

    if cfgs.serve.return_lse:
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            b_m = m_val[b]
            b_l = l_val[b]
            jax.lax.cond(is_last_k, _stage_lse, lambda *_: None, b, b_m, b_l)


def rpa_body(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    cp_rank_ref: jax.Array | None,
    ring_refs: ring.RingRefs | None = None,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    # Configs.
    cfgs: configs.RpaConfigs,
    chunk_start: jax.Array | int = 0,
):
    step = pl.program_id(0)

    pcp_ring = (ring.PcpRing(cfgs, ring_refs, step=step,
                             chunk_start=chunk_start)
                if cfgs.ring_enabled else None)

    # Step 1: Fetch metadata.
    processed_q_len = []
    processed_kv_len = []
    # Lists to hold the 2 variables needed for stitching
    bkv_sz_frm_cache_list = []
    new_kv_len_start_list = []
    # List to support AttentionScope.CACHE_ONLY and AttentionScope.NEW_TOKENS_ONLY.
    kv_new_start = []  # for new tokens.
    cache_len = []  # for cache only.

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        if pcp_ring is not None:
            # k_idx is ring-encoded (see schedule.py); decode before any
            # block arithmetic.
            k_idx, ring_round_b = pcp_ring.lane(k_idx, is_valid)
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
        q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
        q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
        q_len = q_end - q_start
        offset = kv_len - q_len

        if cfgs.serve.attention_scope == configs.AttentionScope.CACHE_ONLY:
            # Mirror q_loop in schedule
            local_cache_len = offset
            if cfgs.serve.cp_group_size is not None:
                cp_group_size = cfgs.serve.cp_group_size
                rank = cp_rank_ref[0]
                if pcp_ring is not None:
                    # Mask by the source rank's shard length, not our own.
                    rank = pcp_ring.src_rank(ring_round_b)
                local_cache_len = utils.cp_local_cache_len(
                    offset, cp_group_size, rank, cfgs.serve.page_size)
            # We add this to make sure flash_attention.py can mask non-cache tokens out.
            cache_len.append(local_cache_len)
        elif cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
            kv_new_start.append(kv_len - q_len)

        processed_q_len.append((q_idx * cfgs.bq_sz + offset))
        processed_kv_len.append(k_id)

        # Stitching metadata
        if (cfgs.serve.attention_scope == configs.AttentionScope.CACHE_ONLY
                and cfgs.serve.cp_group_size is not None):
            kv_left = jnp.maximum(local_cache_len - k_id, 0)
            bkv_sz_frm_cache = jnp.minimum(kv_left, cfgs.bkv_sz)
            new_kv_len_start = q_end  # no new tokens in CACHE_ONLY
        else:
            kv_left = jnp.maximum(kv_len - k_id, 0)
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
            kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
            bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
            new_kv_len_start = q_end - kv_left_frm_new

        bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
        new_kv_len_start_list.append(new_kv_len_start)

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = offset + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz
        if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
            start_k_idx = jnp.maximum(start_k_idx, offset // cfgs.bkv_sz)

    if pcp_ring is not None:
        # Rotate the KV blocks one micro-step per lane and stage received
        # blocks into kv_in_vref, so Step 2 below reads every lane the same
        # way whether or not the ring is on. See ring.py for the protocol.
        pcp_ring.stage(kv_in_vref)

    # Step 2: Fetch inputs.
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
    k_b = []
    v_b = []

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
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
        for b_idx in range(cfgs.batch_size):
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                k_slice = kv_in_vref.at[b_idx, kv_head * 2].bitcast(jnp.uint32)
                v_slice = kv_in_vref.at[b_idx,
                                        kv_head * 2 + 1].bitcast(jnp.uint32)

                target_shape = (-1, cfgs.bkv_sz + 2 * cfgs.serve.page_size)
                k_head_ref = k_slice.reshape(target_shape)
                v_head_ref = v_slice.reshape(target_shape)
                pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv

                k_head_loaded = utils.strided_load(k_head_ref,
                                                   0,
                                                   pack_dim,
                                                   1,
                                                   dtype=cfgs.serve.dtype_kv)
                v_head_loaded = utils.strided_load(v_head_ref,
                                                   0,
                                                   pack_dim,
                                                   1,
                                                   dtype=cfgs.serve.dtype_kv)

                k_head = k_head_loaded[:, :cfgs.bkv_sz]
                v_head = v_head_loaded[:, :cfgs.bkv_sz]

                ks.append(k_head.reshape(cfgs.aligned_kv_head_dim,
                                         cfgs.bkv_sz))
                vs.append(v_head.reshape(cfgs.aligned_kv_head_dim,
                                         cfgs.bkv_sz))
            k_b.append(jnp.stack(ks, axis=0))
            v_b.append(jnp.stack(vs, axis=0))
    else:
        for b_idx in range(cfgs.batch_size):
            heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
            ks = []
            vs = []
            for kv_head_start in range(0, cfgs.model.num_kv_heads,
                                       heads_per_load):
                bkv_lst = strided_load_bkv(
                    kv_in_vref,
                    b_idx,
                    kv_head_start * 2,
                    cfgs=cfgs,
                )
                ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
            k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
            k = k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
            v = v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)

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

    l_new_list = []
    m_new_list = []
    acc_new_list = []

    prev_p = prev_alpha_list = prev_q_slice = None
    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
        bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
        q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
        q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
        q_slice = slice(q_start, q_end)

        p, alpha_list, m_next_step, m_next, l_next = flash_attention.flash_attention_qk_softmax(
            step,
            q[:, :, q_slice],
            k,
            m_val[:, q_slice],
            l_val[:, q_slice],
            schedule_ref.is_last_k,
            processed_q_len=processed_q_len,
            processed_kv_len=processed_kv_len,
            kv_new_start=kv_new_start,
            cache_len=cache_len,
            cfgs=cfgs,
            bq_start=bq_start,
        )
        m_scratch_ref[:, q_slice] = m_next_step
        l_scratch_ref[:, q_slice] = l_next[-1]
        m_new_list.append(m_next)
        l_new_list.append(l_next)

        if prev_p is not None:
            o_next = flash_attention.flash_attention_pv(
                step,
                prev_p,
                v,
                prev_alpha_list,
                acc_val[:, prev_q_slice],
                schedule_ref.is_last_k,
                cfgs=cfgs,
            )
            acc_scratch_ref[:, prev_q_slice] = o_next[-1]
            acc_new_list.append(o_next)

        prev_p = p
        prev_alpha_list = alpha_list
        prev_q_slice = q_slice

    assert prev_p is not None
    o_next = flash_attention.flash_attention_pv(
        step,
        prev_p,
        v,
        prev_alpha_list,
        acc_val[:, prev_q_slice],
        schedule_ref.is_last_k,
        cfgs=cfgs,
    )
    acc_scratch_ref[:, prev_q_slice] = o_next[-1]
    acc_new_list.append(o_next)

    m_next = jnp.concatenate(m_new_list, axis=2)
    l_next = jnp.concatenate(l_new_list, axis=2)
    acc_next = jnp.concatenate(acc_new_list, axis=2)

    # Step 4: Write back outputs.
    calculate_and_store_out(
        step,
        schedule_ref,
        acc_next,
        l_next,
        m_next,
        o_vref,
        lse_o_vref,
        cfgs=cfgs,
    )


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    lse_hbm_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[
        bref_override.BatchingQRef,
        bref_override.KVBufferedRefSeqAlongLane
        | bref_override.KVBufferedRefHeadAlongSublane,
        bref_override.BatchingORef,
        bref_override.BatchingLSERef | None,
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

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_alloc_cls = bref_override.KVBufferedRefSeqAlongLane
    else:
        kv_cache_alloc_cls = bref_override.KVBufferedRefHeadAlongSublane

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

    lse_alloc = None
    if cfgs.serve.return_lse:
        lse_spec = pl.BlockSpec(
            block_shape=cfgs.lse_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, ),
            pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
        )
        lse_alloc = bref_override.BatchingLSERef.output(
            spec=lse_spec,
            dtype_or_type=lse_hbm_ref,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )

    return q_alloc, kv_cache_alloc, o_alloc, lse_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
    serve = cfgs.serve
    name = f"RPA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
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
    page_indices: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    cp_rank: jax.Array | None = None,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
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
    lse_hbm: pre-allocated buffer for LSE output. None when return_lse=False.
    cp_rank: scalar rank of this device within the CP group. None when
      cp_group_size is None.
    cfgs: Configuration of the kernel.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
      head_dim]. Result of new kv cache.
    lse_out: [max_num_tokens, num_q_heads] LSE values, or None.
  """
    return_lse = cfgs.serve.return_lse
    cp_group_size = cfgs.serve.cp_group_size

    def ragged_paged_attention_pipeline(
        # Scalar prefetch.
        cu_q_lens_ref: jax.Ref,
        kv_lens_ref: jax.Ref,
        page_indices_ref: jax.Ref,
        cp_rank_ref: jax.Array | None,
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        lse_hbm_ref: jax.Ref | None,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        o_lse_hbm_ref: jax.Ref | None = None,
    ):

        del o_kv_cache_hbm_ref
        if o_lse_hbm_ref is not None:
            del o_lse_hbm_ref

        q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs)

        actual_steps = schedule_hbm_ref.actual_steps[0]
        num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
            schedule_ref=schedule.RpaSchedule.create_shape_dtype(
                cfgs).scratch_shapes(),
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
            ) + ring.scratch_refs(cfgs),
        )
        def _run(final_allocs, schedule_ref, dma_sem, scratches):
            # Initialize KV cache to zeros.
            # When perfomring p * v, we perform causal masking on lhs (p) by zeroing
            # out columns that should not be processed for a given row. Even if we
            # don't perform masking on rows of rhs (v), the output is still correct
            # since reuslt of multiplication will be zero thanks zero on lhs. However,
            # this assumption does not hold if a row of rhs has NaNs. To avoid this,
            # we initiallize scratch memory with non-zero values. Even if the scratch
            # memory is storing kv cache from previous step, as long as the data is
            # not NaNs, there will be no numeric concerns.

            scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
            scratches[1][...] = jnp.zeros_like(scratches[1])
            scratches[2][...] = jnp.zeros_like(scratches[2])

            num_lanes = pltpu.get_tpu_info().num_lanes
            kv_alloc = final_allocs[1]
            kv_ref_flat = kv_alloc.window_ref.bitcast(jnp.uint32).reshape(
                -1, num_lanes)
            kv_ref_flat[...] = jnp.zeros_like(kv_ref_flat)

            if cfgs.ring_enabled:
                ring.zero_init(scratches[3])

            def execute_schedule_chunk(start_step, num_steps):
                # All reads are aligned to 128 and some extra steps are copied in the
                # process.
                aligned_start_step = (start_step // 128) * 128
                prefix_steps = start_step - aligned_start_step

                flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
                flat_smem = jax.tree_util.tree_leaves(schedule_ref)
                dma_list = []
                for h, s in zip(flat_hbm, flat_smem):
                    if h.memory_space == pltpu.HBM:
                        element_size = s.shape[0] // cfgs.max_steps_ub
                        read_size = element_size * (num_steps + prefix_steps)
                        read_size = utils.align_to(read_size, 1024)
                        read_size = jnp.minimum(read_size, s.shape[0])

                        src_off = element_size * aligned_start_step
                        src_off = pl.multiple_of(src_off, 128)

                        copy = pltpu.make_async_copy(
                            h.at[pl.ds(src_off, read_size)],
                            s.at[pl.ds(0, read_size)],
                            dma_sem.at[0],
                        )
                        copy.start()
                        dma_list.append(copy)
                jax.tree.map(lambda x: x.wait(), dma_list)
                pipeline_func = pltpu.emit_pipeline(
                    body=functools.partial(
                        rpa_body,
                        cfgs=cfgs,
                        cu_q_lens_ref=cu_q_lens_ref,
                        kv_lens_ref=kv_lens_ref,
                        chunk_start=aligned_start_step,
                    ),
                    grid=(num_steps + prefix_steps, ),
                    in_specs=(q_alloc.spec, kv_cache_alloc.spec),
                    out_specs=(o_alloc.spec,
                               lse_alloc.spec if return_lse else None),
                )
                pipeline_func(
                    (q_hbm_ref, schedule_ref),
                    (kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref,
                     page_indices_ref),
                    (o_hbm_ref, schedule_ref),
                    (lse_hbm_ref, schedule_ref) if return_lse else None,
                    scratches=(schedule_ref, ) + scratches[:3] +
                    (cp_rank_ref, ) + scratches[3:],
                    allocations=final_allocs,
                )

            @pl.loop(0, num_safe_step_iterations)
            def loop_body(step_idx):
                start = step_idx * cfgs.max_steps_ub
                rem = actual_steps % cfgs.max_steps_ub
                last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
                is_last_step = step_idx == num_safe_step_iterations - 1
                size = jnp.where(is_last_step, last_step_size,
                                 cfgs.max_steps_ub)

                execute_schedule_chunk(start, size)

        _run()

    scalar_prefetches = (
        cu_q_lens,
        kv_lens,
        page_indices,
        cp_rank if cp_group_size is not None else None,
    )
    num_scalar_prefetch = len(scalar_prefetches)
    num_active_scalers = sum(1 for s in scalar_prefetches if s is not None)

    out_shape = [q_hbm, kv_cache_hbm, lse_hbm if return_lse else None]

    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    input_output_aliases = {
        num_active_scalers + schedule_leaves: 0,
        num_active_scalers + schedule_leaves + 2: 1,
    }
    if return_lse:
        input_output_aliases[num_active_scalers + schedule_leaves + 3] = (
            2  # lse_hbm -> out[2]
        )

    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(
                    memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            **ring.compiler_params(cfgs),
        ),
        input_output_aliases=input_output_aliases,
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        cu_q_lens,
        kv_lens,
        page_indices,
        cp_rank if cp_group_size is not None else None,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
    )
