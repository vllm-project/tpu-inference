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
"""Wrapper for RPA kernel to match expected interface.

NOTE: all of the code in this directory is experimental and not fully tested!
To enable usage of this kernel in full run, you can pass the USE_BATCHED_RPA_KERNEL=1
environment variable.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops. 

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via 
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model. 

Note: batched_rpa is build on top / derived from RPA3. 
"""

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference import envs
from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            ring, schedule,
                                                            schedule_cp, utils)
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import \
    get_tuned_params


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    total_kv_tokens, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                             num_sublanes * kv_packing)
    else:
        aligned_kv_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2,
                                             kv_packing)

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_kv_tokens, align_tokens)
        new_kv_hbm = (jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_kv_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_head_dim),
            (
                (0, padded_total_tokens - total_kv_tokens),
                (0, 0),
                (0, aligned_kv_head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            padded_total_tokens,
            actual_num_kv_heads_x2,
            aligned_kv_head_dim // kv_packing,
            kv_packing,
        ).transpose(1, 2, 3, 0))
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_kv_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_kv_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
):
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(actual_head_dim, num_sublanes * kv_packing) //
            kv_packing,
            kv_packing,
            page_size,
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    # Calculate aligned model dimensions.
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q)
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads

    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_num_kv_heads_x2 = model_cfgs.num_kv_heads * 2
    else:
        bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2,
                             serve_cfgs.packing_kv)
        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

    def calculate_vmem_usage(batch_size: int, n_buffer: int, bq_sz: int,
                             bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            bkv_array_size = ((bkv_sz + 2 * serve_cfgs.page_size) *
                              aligned_num_kv_heads_x2 * aligned_head_dim)
        else:
            bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. For output, we always use double buffer.
        bq_bytes *= n_buffer
        bkv_bytes *= n_buffer
        bo_bytes *= 2

        # Sum up all buffer memory usage.
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

        # Step 2: Compute-time live values, per lane. Empirically (compile-time
        # scoped-VMEM reports on v7x, bf16/fp8, nq/nkv 8/2..32/8, bq 256-2048,
        # bkv 256-1024, batch 1-4) Mosaic keeps several f32 [rows, 128] softmax
        # statistics (m, l, alpha and the acc; ~3.7 fitted, 4.0 used) and ~0.9
        # bf16-sized [rows, bkv] qk/p intermediate per lane for the WHOLE q
        # block (the bq_c loop is unrolled), with rows = bq_sz * aligned q
        # heads. Predicts >= 0.97x of every measured config (cap is 0.85x);
        # over-predicts small ones (safe side).
        rows = bq_sz * aligned_num_q_heads
        stats_bytes = rows * num_lanes * 4
        inter_bytes = rows * bkv_sz * 2
        per_lane_bytes = buffer_bytes + 4.0 * stats_bytes + 0.9 * inter_bytes

        # Step 3: Account for batch size.
        return int(batch_size * per_lane_bytes)

    def calculate_compute_buffer_time(batch_size: int, bq_c_sz: int,
                                      bkv_sz: int) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def find_best_block_sizes(
        max_batch_size: int,
        max_n_buffer: int,
        fixed_bq_sz: int | None = None,
        cap_fraction: float = 0.8,
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * cap_fraction

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size
        n_buffer = max_n_buffer

        # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
        # size can trigger OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (batch_size > 1
               and calculate_vmem_usage(batch_size, n_buffer, bq_sz,
                                        bkv_sz) > capped_vmem_limit_bytes):
            batch_size -= 1

        # The compute-time terms of the model (softmax statistics and the qk
        # intermediate) scale with bq_sz * q heads and not with n_buffer, so
        # with many q heads per device (e.g. 96 under pcp8 x tp1) even the
        # smallest bq can exceed the cap on its own. Halve bq first (when it
        # is not fixed) so the buffer-count fallback below has something to
        # trade; it must never drive n_buffer below 1.
        min_bq_sz = 16
        while (fixed_bq_sz is None and bq_sz // 2 >= min_bq_sz
               and calculate_vmem_usage(batch_size, n_buffer, bq_sz,
                                        bkv_sz) > capped_vmem_limit_bytes):
            bq_sz //= 2
            bq_stride = bq_sz

        # As a last resort, attempt to decrease number of buffers to avoid OOM.
        while (n_buffer > 1
               and calculate_vmem_usage(batch_size, n_buffer, bq_sz,
                                        bkv_sz) > capped_vmem_limit_bytes):
            n_buffer -= 1

        # Indicates OOM was triggered even when batch_size=1 and n_buffer=1.
        if (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
                > capped_vmem_limit_bytes):
            raise ValueError(
                "Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        max_seq_len = serve_cfgs.pages_per_seq * serve_cfgs.page_size
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               < capped_vmem_limit_bytes and bkv_sz <= max_seq_len
               # and bkv_sz <= 8192
               ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        bkv_sz -= bkv_stride
        bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.

        # Fixed threshold value based on hardware spec.
        # TODO(kyuyeunk): Use different threshold based on hardware and precision.
        threshold = 1500

        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz)
               > threshold or bq_c_rem != 0) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=last_valid_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    def ring_vmem_usage(batch_size: int, n_buffer: int, bq_sz: int,
                        bkv_sz: int) -> int:
        """calculate_vmem_usage plus two lane-sized slots of headroom.

        The ring rotates blocks directly between the pipeline's window
        buffers and no longer allocates private slots; the term is kept as
        headroom until the model is refit for the direct protocol."""
        kv_buf = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim * kv_bytes
        ring_slots = 2 * kv_buf
        return calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz) + int(
            1.3 * ring_slots)

    def find_ring_block_sizes(max_n_buffer: int) -> configs.BlockSizes:
        """Block sizes for the PCP ring cache phase (CACHE_ONLY).

        Every q block drives one full rotation of the KV shards around the
        ring, so the ICI traffic is (num q blocks) x (cache bytes): bq_sz
        comes first and is the whole local query block when VMEM allows,
        then the largest bkv_sz (fewer, larger ring steps), then two lanes
        per step if that still leaves a useful bkv_sz. Falls back to the
        generic search when nothing fits.
        """
        cap = vmem_limit_bytes * 0.85
        page = serve_cfgs.page_size
        max_seq_len = serve_cfgs.pages_per_seq * page
        bq_max = min(utils.align_to(serve_cfgs.total_q_tokens, num_lanes),
                     4096)
        bkv_cap = max(page, min(2048, max_seq_len // page * page))
        bkv_good = min(512, bkv_cap)

        def largest_bkv(batch_size, bq_sz):
            # Powers of two only: shard lengths are page multiples and
            # (with a power-of-two page) usually power-of-two aligned, so an
            # odd block size like 768 spends up to a third of every ring
            # step on masked columns.
            bkv_sz = 1 << (bkv_cap.bit_length() - 1)
            while bkv_sz >= page:
                if ring_vmem_usage(batch_size, max_n_buffer, bq_sz,
                                   bkv_sz) <= cap:
                    return bkv_sz
                bkv_sz //= 2
            return None

        chosen = None
        for bq_sz in range(bq_max, 0, -num_lanes):
            fits = {b: largest_bkv(b, bq_sz) for b in (2, 1)}
            if fits[2] is not None and fits[2] >= bkv_good:
                chosen = (bq_sz, fits[2], 2)
            elif fits[1] is not None and fits[1] >= bkv_good:
                chosen = (bq_sz, fits[1], 1)
            elif fits[1] is not None and bq_sz == num_lanes:
                chosen = (bq_sz, fits[1], 1)
            if chosen is not None:
                break
        if chosen is None:
            return find_best_block_sizes(1, max_n_buffer)
        bq_sz, bkv_sz, batch_size = chosen

        # The ring cost is set by the number of q blocks, not their size: the
        # smallest bq_sz with the same block count leaves VMEM for a larger
        # bkv_sz or a second lane.
        num_q_blocks = pl.cdiv(bq_max, bq_sz)
        bq_sz = utils.align_to(pl.cdiv(bq_max, num_q_blocks), num_lanes)
        fits = {b: largest_bkv(b, bq_sz) for b in (2, 1)}
        if fits[2] is not None and fits[2] >= bkv_good:
            bkv_sz, batch_size = fits[2], 2
        else:
            bkv_sz, batch_size = fits[1], 1

        # Compute sub-tile: ~256 rows per kv head is where the compiled
        # kernel's VMEM bottoms out (each unrolled tile carries its own
        # temporaries; a single huge tile keeps them all live at once).
        target = max(8, 256 // aligned_num_q_heads_per_kv_head)
        bq_c_sz = next(
            (c for c in range(min(target, bq_sz), 0, -1) if bq_sz % c == 0),
            bq_sz)
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=max_n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    # Fixed value based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    if serve_cfgs.cp is not None:
        # Decode-shaped blocks (bq = 1, many lanes, large bkv) carry up to
        # ~1.5x their buffer bytes in compute-time state the model does not
        # capture (compile probes on v7x: fp8/hd256 b8 k2048 65.8 MiB for a
        # 48 MiB window; bf16/hd128 b8 k1024 36 for 24; fp8/hd128/nkv8 b8
        # k1024 81 for 60). Under CP the DECODE launch only ever sees a
        # single request's decode step (a few tokens per rank), so its blocks
        # are not performance-relevant; search them against a budget that
        # absorbs that factor instead of tuning the generic path.
        decode_block_sizes = find_best_block_sizes(decode_batch_size,
                                                   n_buffer,
                                                   1,
                                                   cap_fraction=0.55)
    else:
        decode_block_sizes = find_best_block_sizes(decode_batch_size, n_buffer,
                                                   1)
    if serve_cfgs.cp is not None and serve_cfgs.cp.ring_axis_name is not None:
        prefill_block_sizes = find_ring_block_sizes(n_buffer)
    else:
        prefill_block_sizes = find_best_block_sizes(prefill_batch_size,
                                                    n_buffer)

    return decode_block_sizes, prefill_block_sizes


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
        "kv_layout",
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "write_last_seq_only",
        "pcp_ring_axis_name",
        "pcp_ring_mesh_axis_names",
    ),
    # Donation of transient inputs can fail for some runtime buffer layouts in
    # the experimental tuning path. Keep donation only for kv_cache, which is
    # the intended long-lived mutable state.
    donate_argnames=("kv_cache", ),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    pcp_ring_axis_name: str | None = None,
    pcp_ring_mesh_axis_names: tuple[str, ...] | None = None,
    global_kv_cache_lens: jax.Array | None = None,
    global_new_kv_lens: jax.Array | None = None,
    q_pos_offsets: jax.Array | None = None,
    write_last_seq_only: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    Args:
      queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
      keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
      values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
      kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
        kv_packing, head_dim]. Stores existing kv cache data where k & vs are
        concatenated along num kv heads dim.
      kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
      page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
        sequence.
      cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
        length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
        b=cu_q_lens[i+1] represents q/k/v of sequence i.
      distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
        sequences. distribution[2] represents total number of sequences.
      sm_scale: Softmax scale value.
      sliding_window: Size of sliding window (also known as local attention). kvs
        outside of the window is not fetched from hbm and masked out during
        computation.
      soft_cap: Cap values of softmax inputs.
      mask_value: Value to use for causal masking. Defaults to smallest
        representable value of the activation dtype.
      q_scale: Quantization scale value of queries.
      k_scale: Quantization scale value of keys.
      v_scale: Quantization scale value of values.
      chunk_prefill_size: Not used.
      decode_block_sizes: Kernel block size to use during decode.
      prefill_block_sizes: Kernel block size to use during prefill.
      vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
        size of the hardware.
      debug_mode: Not used.
      out_dtype: Dtype of output. Defaults to dtype of queries.
      use_causal_mask: Not used.
      cp_group_size: Size of the context parallelism (CP) group. KV cache is
        sharded across devices in this group. Defaults to None.
      cp_rank: Rank of the current device within the CP group, which determine
        the token ownership. Defaults to None.
      attention_scope: Which KV positions to attend to. FULL attends all
        positions, CACHE_ONLY skips new tokens, NEW_TOKENS_ONLY skips cached
        tokens. Defaults to FULL.
      return_lse: If True, return log-sum-exp (lse) values along with the
        output. Defaults to False.
      pcp_ring_axis_name: PCP cache phase only. When set, CACHE_ONLY streams
        each rank's KV cache shard around this mesh axis in-kernel so every
        rank attends the full cache with its local Q; one online softmax
        accumulates all rounds. Requires CACHE_ONLY, an even cp_group_size,
        cp_rank, and the HEAD_ALONG_SUBLANE layout.
      pcp_ring_mesh_axis_names: All axis names of the mesh the ring runs on,
        in order. Defaults to a one-axis mesh.
      global_kv_cache_lens: [max_num_seqs]. Cached length per sequence. When
        given it replaces kv_lens - q_len, so the new kv can be longer than
        this device's queries (PCP: the chunk's kv is all-gathered).
      global_new_kv_lens: [max_num_seqs]. Length of each sequence's new kv, which
        then starts at keys/values[0] for every sequence (PCP current phase:
        the head and tail chunks share the whole chunk's kv). Defaults to
        the sequence's own query length at cu_q_lens[i].
      q_pos_offsets: [max_num_seqs]. Position of each sequence's first query
        relative to the cache end (PCP head/tail chunks). Defaults to 0.
      write_last_seq_only: Only the last sequence writes new kv back, and
        visits every new kv block it owns (PCP: one write of the chunk per
        rank).

    Returns:
      out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
      new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
        kv_packing, head_dim]. Result of new kv cache where k & vs are
        concatenated along num kv heads dim.
      lse (only when return_lse=True): [max_num_tokens, num_q_heads].
        Log-sum-exp values (m + log(l)) for each query token and head,
        needed for merging partial attention results in CP.
    """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    out_dtype = jnp.dtype(queries.dtype if out_dtype is None else out_dtype)

    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    kv_packing = utils.get_dtype_packing(kv_cache.dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    cp_cfg = None
    if cp_group_size is not None:
        cp_cfg = configs.CPConfig(
            group_size=cp_group_size,
            ring_axis_name=pcp_ring_axis_name,
            ring_mesh_axis_names=pcp_ring_mesh_axis_names,
            write_last_seq_only=write_last_seq_only,
        )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=kv_layout,
        cp=cp_cfg,
        attention_scope=attention_scope,
        return_lse=return_lse,
    )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        kv_layout=kv_layout,
        page_size=page_size,
    )

    default_decode = default_prefill = None
    if decode_block_sizes is None or prefill_block_sizes is None:
        default_decode, default_prefill = calculate_block_sizes(
            model_cfgs, serve_cfgs, vmem_limit_bytes)
    # Pre-allocate LSE buffer.
    lse_hbm_init: jax.Array | None = None
    if return_lse:
        num_lanes = pltpu.get_tpu_info().num_lanes
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        q_packing = utils.get_dtype_packing(queries.dtype)
        num_q_heads_per_kv_head = num_q_heads // num_kv_heads
        aligned_num_q_heads_per_kv_head = utils.align_to(
            num_q_heads_per_kv_head, q_packing)
        lse_row_stride = utils.align_to(aligned_num_q_heads_per_kv_head,
                                        num_sublanes)
        max_tokens = queries.shape[0]
        lse_hbm_init = jnp.full(
            [
                num_kv_heads,
                max_tokens * lse_row_stride,
                num_lanes,
            ],
            -jnp.inf,
            dtype=out_dtype,
        )

    # Compute per-sequence length parameters for the kernel.
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    if global_kv_cache_lens is None:
        global_kv_cache_lens = kv_lens - q_lens
    if global_new_kv_lens is None:
        new_lens = q_lens
        new_kv_starts = cu_q_lens[:-1]
    else:
        new_lens = global_new_kv_lens
        new_kv_starts = jnp.zeros_like(q_lens)

    if attention_scope == configs.AttentionScope.CACHE_ONLY:
        if cp_group_size is not None:
            rank = cp_rank[0] if cp_rank is not None else 0
            local_kv_cache_lens = utils.cp_local_cache_len(
                global_kv_cache_lens, cp_group_size, rank, page_size)
        else:
            local_kv_cache_lens = global_kv_cache_lens
        kv_new_lens = jnp.zeros_like(q_lens)
        q_offsets = local_kv_cache_lens
        if pcp_ring_axis_name is not None:
            # The ring attends every rank's shard, some longer than ours, and
            # all queries follow the whole cache: the causal offset is the
            # global cache length.
            q_offsets = global_kv_cache_lens
    elif attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        local_kv_cache_lens = global_kv_cache_lens
        kv_new_lens = new_lens
        q_offsets = global_kv_cache_lens
    else:  # FULL
        local_kv_cache_lens = global_kv_cache_lens
        kv_new_lens = new_lens
        q_offsets = global_kv_cache_lens
    if q_pos_offsets is not None:
        q_offsets = q_offsets + q_pos_offsets
    # The CP schedule always receives a rank scalar (extra_refs[0]).
    cp_rank_scalar = (cp_rank if cp_rank is not None else jnp.zeros(
        (1, ), jnp.int32))

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm_in: jax.Array | None,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
        else:
            effective_blocks = prefill_block_sizes or default_prefill

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )
        kernel_kv_cache_lens = local_kv_cache_lens
        step_metadata_cls = kernel.StepMetadataComputer
        extra_scratch_shapes = ()
        collective_id = None
        post_rpa_hook = None
        if cfgs.ring_enabled:
            computer_cls = ring.RingMetadataComputer
            step_metadata_cls = ring.RingStepMetadataComputer
            extra_scratch_shapes = (ring.RingSems(
                dma_sems=pltpu.SemaphoreType.DMA((2, )),
                sync_sem=pltpu.SemaphoreType.REGULAR,
            ), )
            # The ring's startup barrier needs a barrier semaphore.
            collective_id = 0
            post_rpa_hook = ring.wait_ring_sends
            extra_scalars = (cp_rank_scalar, new_kv_starts,
                             global_kv_cache_lens)
            kernel_kv_cache_lens = global_kv_cache_lens
        elif cp_group_size is not None:
            computer_cls = schedule_cp.CPMetadataComputer
            extra_scalars = (cp_rank_scalar, new_kv_starts)
        else:
            computer_cls = schedule.BaseMetadataComputer
            extra_scalars = ()

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            local_kv_cache_lens,
            kv_new_lens,
            distribution,
            cfgs=cfgs,
            computer_cls=computer_cls,
            extra_scalars=extra_scalars,
        )
        result = kernel.rpa_kernel(
            cu_q_lens,
            q_offsets,
            kernel_kv_cache_lens,
            kv_new_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            lse_hbm_in,
            cfgs=cfgs,
            computer_cls=computer_cls,
            step_metadata_cls=step_metadata_cls,
            extra_scratch_shapes=extra_scratch_shapes,
            collective_id=collective_id,
            post_rpa_hook=post_rpa_hook,
        )
        if return_lse:
            o_out, kv_out, lse_out = result
        else:
            o_out, kv_out, _ = result
            lse_out = None
        if not serve_cfgs.writes_kv_cache:
            kv_out = kv_cache
        return o_out, kv_out, lse_out

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init)
    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm)

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    if not return_lse:
        return o_hbm, kv_cache

    # Reshape LSE from [num_kv_heads, max_tokens * lse_row_stride, num_lanes]
    # to [max_tokens, num_q_heads].
    max_tokens = queries.shape[0]
    # Extract first lane (scalar LSE value per token-head pair).
    lse = lse_hbm.reshape(num_kv_heads, max_tokens, lse_row_stride,
                          num_lanes)[:, :max_tokens, :num_q_heads_per_kv_head,
                                     0]
    lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

    return o_hbm, kv_cache, lse
