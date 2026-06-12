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
"""Pallas kernel for GDN recurrent scan."""

# pylint: disable=invalid-name

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.v2 import \
    compute_schedule_v2 as compute_schedule_table_v2
from tpu_inference.kernels.gdn.v2 import recurrent_scan_impl


def invert_triangular_matrix(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


def inner_kernel(
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Prefill chunk
    prefill_qkv_ref,
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Decode batch
    decode_qkv_ref,
    # VMEM: (C, 128). Raw a values for Prefill chunk
    prefill_a_raw_ref,
    # VMEM: (BT, 128). Raw a values for Decode batch
    decode_a_raw_ref,
    # VMEM: (C, 128). Raw b values for Prefill chunk
    prefill_b_raw_ref,
    # VMEM: (BT, 128). Raw b values for Decode batch
    decode_b_raw_ref,
    # VMEM: (n_v,). A_log for gate computation
    a_log_ref,
    # VMEM: (n_v,). dt_bias for gate computation
    dt_bias_ref,
    # VMEM: (C, n_v * d_v). Scanned outputs for prefill
    prefill_output_ref,
    # VMEM: (BT, n_v * d_v). Scanned outputs for decode
    decode_output_ref,
    # SMEM: (max_blocks, 8). Schedule table
    schedule_table,
    # SMEM: (max_reqs,). State indices
    state_indices,
    # SMEM: (max_reqs,). Whether each request has prior recurrent state
    has_initial_state,
    *,
    # HBM: (B, n_v, d_k, d_v). All recurrent states
    recurrent_state_in,
    recurrent_state_out,
    # Chunk size for prefill
    C: int,
    # Batch size for decode
    BT: int,
    #  Number of key/query heads
    n_kq: int,
    #  Number of value heads
    n_v: int,
    #  Key dimension
    d_k: int,
    #  Value dimension
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    # VMEM scratchpad: (2, n_v, d_k, d_v). To carry state across chunks
    # (double buffered)
    prefill_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). Per-iter safe-copy of the loaded
    # state. Required as a separate buffer from decode_load_scratch because
    # the prefetch DMA for iter b+2 writes to the same slot concurrently with
    # this iter's compute; this buffer isolates the reads. Stored as bf16;
    # per-head fp32 cast happens in VREG inside the compute loop.
    decode_state_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). Aliased to slot 0 of
    # decode_store_scratch in _run_with_scratch (decode drains its stores
    # before prefill runs, so the slot is free for prefill bf16 staging).
    state_commit_scratch,
    # VMEM scratchpad: (2, n_v, d_k, d_v). Double-buffered staging for
    # fully-async decode loads. iter b lands in slot (b % 2).
    decode_load_scratch,
    # VMEM scratchpad: (2, n_v, d_k, d_v). Double-buffered staging for
    # fully-async decode stores. iter b uses slot (b % 2).
    decode_store_scratch,
    # VMEM scratchpad: (BT, n_v * d_v). To hold decode outputs before DMA
    decode_output_scratch,
    # Array of C semaphores for decode state loads
    decode_read_semaphores,
    # 2 semaphores (one per decode_store_scratch slot) for async decode stores
    decode_write_semaphore,
    # 1 semaphore for prefill DMA (stores only)
    prefill_semaphore,
    # Number of decode tokens (requests) in the batch
    decode_tokens,
):
    """Inner kernel for recurrent scan processing both prefill and decode.

  This function is called for each step in the schedule table and dispatches
  work to either `process_decode` or
  `process_regular_prefill`/`process_transition_prefill`.
  """
    step = pl.program_id(0)

    # Instantiate helper config structures
    model_dims = recurrent_scan_impl.ModelDims(n_kq=n_kq,
                                               n_v=n_v,
                                               d_k=d_k,
                                               d_v=d_v)
    tiling_cfg = recurrent_scan_impl.TilingConfig(C=C,
                                                  BT=BT,
                                                  sublanesize=sublanesize)
    config = recurrent_scan_impl.ScanConfig(
        model=model_dims,
        tiling=tiling_cfg,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        decode_tokens=decode_tokens,
    )

    schedule = recurrent_scan_impl.ScheduleStep(schedule_table, step)

    shared_refs = recurrent_scan_impl.SharedRefs(
        a_log=a_log_ref,
        dt_bias=dt_bias_ref,
        recurrent_state_in=recurrent_state_in,
        recurrent_state_out=recurrent_state_out,
    )

    prefill_refs = recurrent_scan_impl.BranchRefs(
        qkv=prefill_qkv_ref,
        a_raw=prefill_a_raw_ref,
        b_raw=prefill_b_raw_ref,
        output=prefill_output_ref,
    )
    prefill_scratch_refs = recurrent_scan_impl.PrefillScratchRefs(
        scratch=prefill_scratch,
        semaphore=prefill_semaphore,
    )
    prefill_dma = recurrent_scan_impl.DMAHelper(
        state_in=recurrent_state_in,
        state_out=recurrent_state_out,
        commit_scratch=state_commit_scratch,
        semaphore=prefill_semaphore,
    )

    prefill_processor = recurrent_scan_impl.PrefillProcessor(
        config=config,
        schedule=schedule,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        refs=prefill_refs,
        shared=shared_refs,
        scratch=prefill_scratch_refs,
        dma=prefill_dma,
    )

    decode_refs = recurrent_scan_impl.BranchRefs(
        qkv=decode_qkv_ref,
        a_raw=decode_a_raw_ref,
        b_raw=decode_b_raw_ref,
        output=decode_output_ref,
    )
    decode_scratch = recurrent_scan_impl.DecodeScratchRefs(
        state=decode_state_scratch,
        load=decode_load_scratch,
        store=decode_store_scratch,
        output=decode_output_scratch,
        read_semaphores=decode_read_semaphores,
        write_semaphore=decode_write_semaphore,
    )
    decode_dma_in = recurrent_scan_impl.DMAHelper(
        state_in=recurrent_state_in,
        state_out=recurrent_state_out,
        commit_scratch=decode_load_scratch,
        semaphore=decode_read_semaphores,
    )
    decode_processor = recurrent_scan_impl.DecodeProcessor(
        config=config,
        schedule=schedule,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        refs=decode_refs,
        shared=shared_refs,
        scratch=decode_scratch,
        dma=decode_dma_in,
    )

    # READ table

    prefill_valid = schedule_table[step,
                                   recurrent_scan_impl.COL_PREFILL_VALID][...]
    decode_valid = schedule_table[step,
                                  recurrent_scan_impl.COL_DECODE_VALID][...]
    decode_offset = schedule_table[step,
                                   recurrent_scan_impl.COL_DECODE_OFFSET][...]
    prefill_offset = schedule_table[
        step, recurrent_scan_impl.COL_PREFILL_OFFSET][...]
    is_transition = schedule_table[step,
                                   recurrent_scan_impl.COL_IS_TRANSITION][...]

    # 2. Decode Branch
    @pl.when(decode_valid > 0)
    def decode_wrapper():
        decode_processor.process()
        return None

    # Prefill Branch
    # Process prefill if there is valid prefill work in this step
    @pl.when(prefill_valid > 0)
    def process_prefill():
        prefill_processor.process()
        return None

    # For transition block at boundary of decode and prefill we will have
    # overlap:
    # - Decode block BT contains prefill tokens
    # - Sublane size transition prefill block contains some decode tokens in the
    #   sublane
    # So we need to stitch the outputs so they don't overwrite each other in the
    # global index. We exchange decode and prefill outputs so:
    # - Prefill output ref has decode token outputs at decode token indexes in its
    #   out ref
    # - Decode output ref has prefill token outputs at prefill token indexes in
    #   its out ref
    def do_stitch():
        local_start = prefill_offset - decode_offset
        local_split = decode_tokens - prefill_offset

        # Need to hint compiler, or it complains in DMA added by emit pipeline
        safe_local_start = pl.multiple_of(local_start, sublanesize)

        decode_overlap = decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :]
        prefill_arr = prefill_output_ref[pl.ds(0, sublanesize), :]

        # 3. Build sublane size mask
        iota = jax.lax.broadcasted_iota(jnp.int32, (sublanesize, ), 0)
        is_decode_mask = (iota < local_split).astype(jnp.int32)[:, None]

        # 4. Merge
        merged_overlap = jnp.where(is_decode_mask, decode_overlap, prefill_arr)

        decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :] = merged_overlap
        prefill_output_ref[pl.ds(0, sublanesize), :] = merged_overlap

        return None

    is_first_block = pl.program_id(0) == 0
    needs_stitching = (is_transition > 0) & is_first_block & (decode_valid > 0)
    jax.lax.cond(needs_stitching, do_stitch, lambda: None)


def get_qkv_index_map_v2(
    step,
    schedule_table,
    valid_col,
    offset_col,
    alignment=16,
    block_size=64,
    sink_offset=0,
):
    valid = schedule_table[step, valid_col][...]
    offset = schedule_table[step, offset_col][...]
    offset = pl.multiple_of(offset, alignment)

    safe_offset = jnp.where(valid > 0, offset, sink_offset)
    safe_offset = pl.multiple_of(safe_offset, alignment)

    return (pl.ds(safe_offset, block_size), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    alignment=16,
    sink_offset=0,
):
    """Creates block specs for recurrent scan kernel."""

    prefill_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=recurrent_scan_impl.COL_PREFILL_VALID,
        offset_col=recurrent_scan_impl.COL_PREFILL_OFFSET,
        alignment=alignment,
        block_size=chunk_size,
        sink_offset=sink_offset,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=recurrent_scan_impl.COL_DECODE_VALID,
        offset_col=recurrent_scan_impl.COL_DECODE_OFFSET,
        alignment=alignment,
        block_size=BT,
        sink_offset=sink_offset,
    )

    prefill_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), d),
        index_map=prefill_qkv_index_map,
    )
    decode_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), d),
        index_map=decode_qkv_index_map,
    )

    prefill_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
        index_map=prefill_qkv_index_map,
    )
    decode_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), n_v * d_v),
        index_map=decode_qkv_index_map,
    )

    a_log_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    dt_bias_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    prefill_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )
    prefill_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )

    return [
        prefill_qkv_spec,
        decode_qkv_spec,
        prefill_a_raw_spec,
        decode_a_raw_spec,
        prefill_b_raw_spec,
        decode_b_raw_spec,
        a_log_spec,
        dt_bias_spec,
    ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    has_initial_state_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    schedule_table_ref,
    decode_tokens_ref,
    total_blocks_ref,
    recurrent_state_ref,
    output_ref,
    *,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
):
    """Fused kernel for recurrent scan."""
    decode_tokens = decode_tokens_ref[0]
    total_blocks = total_blocks_ref[0]

    d = mixed_qkv_ref.shape[-1]
    pad_size = max(C, BT)
    sink_offset = mixed_qkv_ref.shape[0] - pad_size

    in_specs, out_specs = create_block_specs(
        schedule_table_ref,
        C,
        BT,
        d,
        n_v,
        d_v,
        alignment=sublanesize,
        sink_offset=sink_offset,
    )

    def _run_with_scratch(
        scratch_ref,
        decode_state_scratch_ref,
        decode_load_scratch_ref,
        decode_store_scratch_ref,
        decode_output_scratch_ref,
        decode_read_sems,
        decode_write_sem,
        prefill_sem,
    ):
        # Alias state_commit_scratch to slot 0 of decode_store_scratch.
        # Decode drains its store DMAs before prefill runs in any step, so
        # the slot is free to use as prefill's bf16 HBM staging.
        state_commit_scratch_ref = decode_store_scratch_ref.at[pl.ds(0, 1)]

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                inner_kernel,
                C=C,
                BT=BT,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                sublanesize=sublanesize,
                prefill_scratch=scratch_ref,
                decode_state_scratch=decode_state_scratch_ref,
                decode_output_scratch=decode_output_scratch_ref,
                state_commit_scratch=state_commit_scratch_ref,
                decode_load_scratch=decode_load_scratch_ref,
                decode_store_scratch=decode_store_scratch_ref,
                decode_read_semaphores=decode_read_sems,
                decode_write_semaphore=decode_write_sem,
                prefill_semaphore=prefill_sem,
                decode_tokens=decode_tokens,
                recurrent_state_in=aliased_recurrent_state_ref,
                recurrent_state_out=recurrent_state_ref,
            ),
            grid=(total_blocks, ),
            in_specs=in_specs,
            out_specs=out_specs,
        )

        pipeline_func(
            mixed_qkv_ref,
            mixed_qkv_ref,
            a_raw_ref,
            a_raw_ref,
            b_raw_ref,
            b_raw_ref,
            a_log_ref,
            dt_bias_ref,
            output_ref,
            output_ref,
            scratches=[
                schedule_table_ref,
                state_indices_ref,
                has_initial_state_ref,
            ],
        )

    pl.run_scoped(
        # TODO: Move this to outer pallas call and get rid of run_scoped
        _run_with_scratch,
        pltpu.VMEM((2, n_v, d_k, d_v),
                   jnp.float32),  # prefill_scratch (double buffered)
        pltpu.VMEM((1, n_v, d_k, d_v),
                   recurrent_state_ref.dtype),  # decode_state_scratch
        # state_commit_scratch aliased to slot 0 of decode_store_scratch in
        # _run_with_scratch; no separate allocation.
        pltpu.VMEM((2, n_v, d_k, d_v), recurrent_state_ref.dtype
                   ),  # decode_load_scratch (double-buffered)
        pltpu.VMEM(
            (2, n_v, d_k, d_v), recurrent_state_ref.dtype
        ),  # decode_store_scratch (double-buffered; slot 0 also used as prefill's state_commit staging)
        pltpu.VMEM((BT, n_v * d_v),
                   mixed_qkv_ref.dtype),  # decode_output_scratch
        pltpu.SemaphoreType.DMA(
            (2, )),  # decode_read_semaphores (one per slot)
        pltpu.SemaphoreType.DMA(
            (2, )),  # decode_write_semaphore (one per slot)
        pltpu.SemaphoreType.DMA((2, )),  # prefill_semaphore
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "BT",
        "use_qk_norm_in_gdn",
        "vmem_limit_bytes",
        "race_detect_enable",
    ],
)
def recurrent_scan(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 128,
    BT: int = 128,
    use_qk_norm_in_gdn: bool = True,
    has_initial_state: jax.Array | None = None,
    vmem_limit_bytes: int | None = None,
    race_detect_enable: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused recurrent scan kernel for GDN on TPU v7.

  Args:
    mixed_qkv: jax.Array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
      Packed Query, Key, and Value tokens.
    b: jax.Array of shape [num_tokens, n_v]. Input for beta gate.
    a: jax.Array of shape [num_tokens, n_v]. Input for g gate.
    recurrent_state: jax.Array of shape [max_reqs, n_v, d_k, d_v]. Current
      recurrent states.
    A_log: jax.Array of shape [n_v]. Log of parameter A.
    dt_bias: jax.Array of shape [n_v]. Bias for dt.
    query_start_loc: jax.Array of shape [num_requests + 1]. Start indices of
      each request in mixed_qkv.
    state_indices: jax.Array of shape [num_requests] or larger. Mapping from
      request ID to state index.
    distribution: jax.Array of shape [3]. Contains [decode_tokens,
      total_tokens].
    n_kq: Number of query/key heads.
    n_v: Number of value heads.
    d_k: Dimension of query/key features.
    d_v: Dimension of value features.
    chunk_size: Block size for processing (default 128).
    BT: Block size for decode requests (default 128).
    use_qk_norm_in_gdn: Whether to use QK normalization.
    vmem_limit_bytes: Per-kernel scoped VMEM ceiling passed to Mosaic.
    race_detect_enable: If True, run the kernel under Pallas interpret mode with
      DMA/buffer race detection enabled.

  Returns:
    A tuple containing:
      - Updated recurrent state of shape [max_reqs, n_v, d_k, d_v].
      - The mixed_qkv array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
  """
    if has_initial_state is None:
        has_initial_state = jnp.zeros(state_indices.shape[0], dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    num_tokens = mixed_qkv.shape[0]
    tpu_info = pltpu.get_tpu_info()
    sublanesize = 4 // mixed_qkv.itemsize * tpu_info.num_sublanes

    # Default the scoped VMEM ceiling. This value could be tuned for different state cache numerics and chunk sizes.
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.8)

    # Pad token dimension so invalid pipeline steps DMA into a safe sink area.
    # Sink offset must be aligned to sublanesize for Mosaic tile compatibility.
    block_size = max(chunk_size, BT)
    sink_offset = ((num_tokens + sublanesize - 1) // sublanesize) * sublanesize
    pad_rows = sink_offset + block_size - num_tokens
    mixed_qkv = jnp.pad(mixed_qkv, ((0, pad_rows), (0, 0)))

    # Pad raw a and b to (num_tokens + pad_rows, 128) for sublanes
    a_padded = jnp.pad(a, ((0, pad_rows), (0, 128 - n_v)))
    b_padded = jnp.pad(b, ((0, pad_rows), (0, 128 - n_v)))

    # decode_tokens: scalar, number of decode tokens.
    # Assuming length 1 per decode request, this is also the number of decode
    # requests.
    decode_tokens = distribution[0]
    schedule_table, total_blocks = (
        compute_schedule_table_v2.compute_schedule_table_v2(
            query_start_loc,
            decode_tokens,
            distribution[2],
            num_tokens,
            chunk_size,
            BT,
            alignment=sublanesize,
        ))

    # sublane,128
    decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
    total_blocks_arr = jnp.expand_dims(total_blocks, 0)

    grid_spec = pl.GridSpec(
        grid=(1, ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
    )

    updated_recurrent_state, output_padded = pl.pallas_call(
        functools.partial(
            fused_kernel,
            C=chunk_size,
            BT=BT,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            sublanesize=sublanesize,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
            jax.ShapeDtypeStruct((sink_offset + block_size, n_v * d_v),
                                 mixed_qkv.dtype),
        ),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},
        interpret=(pltpu.InterpretParams(
            detect_races=True) if race_detect_enable else False),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(
        mixed_qkv,
        recurrent_state,
        state_indices,
        has_initial_state,
        a_padded,
        b_padded,
        A_log,
        dt_bias,
        schedule_table,
        decode_tokens_arr,
        total_blocks_arr,
    )
    return updated_recurrent_state, output_padded[:num_tokens]
