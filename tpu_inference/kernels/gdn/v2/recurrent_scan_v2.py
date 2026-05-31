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

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tpu_inference.kernels.gdn.v2 import \
    compute_schedule_v2 as compute_schedule_table_v2


# def invert_triangular_matrix(A, block_size=None):
#   """Inverts a unit lower triangular matrix A using Neumann doubling.

#   Algorithm: Neumann doubling. For L strictly lower triangular of size N x N,
#   since L^N = 0 we have:
#     (I + L)^{-1} = (I - L)(I + L^2)(I + L^4) ... (I + L^(N/2))

#   Args:
#     A: Unit lower triangular matrix of shape (B, N, N).
#     block_size: Size of the blocks for Gaussian elimination (unused).

#   Returns:
#     Inverse of A, of shape (B, N, N).

#   For N=128 this is exactly 6 iterations = 12 matmuls, all (B, 128, 128).
#   """
#   B, N, _ = A.shape
#   num_iters = max(1, (N - 1).bit_length() - 1)
#   in_dtype = A.dtype
#   A_f32 = A.astype(jnp.float32)
#   L = jnp.tril(A_f32, k=-1)
#   eye = jnp.broadcast_to(jnp.eye(N, dtype=jnp.float32), (B, N, N))
#   Y = eye - L
#   for _ in range(num_iters):
#     L = jnp.matmul(L, L, precision=jax.lax.Precision.HIGHEST)
#     Y = Y + jnp.matmul(Y, L, precision=jax.lax.Precision.HIGHEST)
#   return Y.astype(in_dtype)


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

    # READ table

    prefill_valid = schedule_table[step, 0][...]
    prefill_req_id = schedule_table[step, 2][...]

    decode_valid = schedule_table[step, 4][...]
    decode_offset = schedule_table[step, 5][...]
    decode_req_id = schedule_table[step, 6][...]
    decode_count = schedule_table[step, 7][...]

    prefill_offset = schedule_table[step, 1][...]
    is_transition = schedule_table[step, 10][...]

    is_last_chunk = schedule_table[step, 8][...]
    is_first_chunk = schedule_table[step, 9][...]

    def l2_normalize(x, eps=1e-6):
        norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
        return x / norm

    # 2. Decode Branch
    @pl.when(decode_valid > 0)
    def decode_wrapper():

        def get_target_idx(b):
            safe_req_id = jnp.minimum(decode_req_id + b,
                                      state_indices.shape[0] - 1)
            return state_indices[safe_req_id][...]

        # Pre-loop: kick off async loads for iters 0 and 1.
        # iter b consumes the load that lands in decode_load_scratch[b % 2].
        # Subsequent loads (iter b+2 for each iter b) are issued from inside
        # the loop as prefetches.
        @pl.when(decode_count >= 1)
        def _preload_slot_0():
            tgt = get_target_idx(0)
            op = pltpu.make_async_copy(
                src_ref=recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=decode_load_scratch.at[pl.ds(0, 1)],
                sem=decode_read_semaphores.at[0],
            )
            op.start()

        @pl.when(decode_count >= 2)
        def _preload_slot_1():
            tgt = get_target_idx(1)
            op = pltpu.make_async_copy(
                src_ref=recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=decode_load_scratch.at[pl.ds(1, 1)],
                sem=decode_read_semaphores.at[1],
            )
            op.start()

        def process_decode(b, store_inflight):
            # store_inflight: tuple (s0_inflight, s1_inflight) of int32 scalars.
            # s{n}_inflight == 1 iff slot n has an in-flight async store DMA.
            s0_inflight, s1_inflight = store_inflight
            is_valid = b < decode_count
            slot = b % 2
            using_slot_0 = slot == 0
            cur_slot_inflight = jax.lax.select(using_slot_0, s0_inflight,
                                               s1_inflight)

            @pl.when(is_valid)
            def do_work():
                # Wait for THIS iter's load (issued by preload or by iter b-2 prefetch).
                wait_load = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(0, 1)],
                    dst_ref=decode_load_scratch.at[pl.ds(slot, 1)],
                    sem=decode_read_semaphores.at[slot],
                )
                wait_load.wait()

                # Safe-copy of loaded state. Isolates compute from the
                # prefetch DMA below that writes the same slot of
                # decode_load_scratch concurrently. bf16 -> bf16, no cast.
                decode_state_scratch[pl.ds(0, 1)] = decode_load_scratch[pl.ds(
                    slot, 1)][...]

                # Prefetch load for iter b+2 (same slot, since (b+2) % 2 == b % 2).
                # This DMA overlaps with the compute below.
                next_b = b + 2

                @pl.when(next_b < decode_count)
                def _prefetch_next_load():
                    next_tgt = get_target_idx(next_b)
                    op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(next_tgt, 1)],
                        dst_ref=decode_load_scratch.at[pl.ds(slot, 1)],
                        sem=decode_read_semaphores.at[slot],
                    )
                    op.start()

                target_idx = get_target_idx(b)

                key_dim = n_kq * d_k
                b_aligned = (b // sublanesize) * sublanesize
                # Workaround: Upcast to fp32 to avoid NaNs
                qkv_block_data = decode_qkv_ref[
                    pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
                    qkv_block_data.dtype)[:, None]
                qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)
                # Fused SiLU
                qkv_row = jax.nn.silu(qkv_row)
                q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
                k = qkv_row[:, key_dim:2 * key_dim].reshape(n_kq, d_k)
                v = qkv_row[:, 2 * key_dim:].reshape(n_v, d_v)

                if use_qk_norm_in_gdn:
                    q = l2_normalize(q)
                    k = l2_normalize(k)

                # Head repetition
                repeat_factor = n_v // n_kq
                if repeat_factor > 1:
                    q = jnp.repeat(q, repeat_factor, axis=0)
                    k = jnp.repeat(k, repeat_factor, axis=0)

                scale = d_k**-0.5
                q = q * scale

                b_aligned = (b // sublanesize) * sublanesize

                g_block_new = decode_a_raw_ref[
                    pl.ds(b_aligned, sublanesize), :]
                beta_block_new = decode_b_raw_ref[
                    pl.ds(b_aligned, sublanesize), :]

                mask_new = (jnp.arange(sublanesize) == (
                    b % sublanesize)).astype(g_block_new.dtype)[:, None]

                curr_g_slice_new = jnp.sum(g_block_new * mask_new,
                                           axis=0,
                                           keepdims=True)
                curr_beta_slice_new = jnp.sum(beta_block_new * mask_new,
                                              axis=0,
                                              keepdims=True)

                a_raw_new = curr_g_slice_new[:, :n_v].reshape(n_v).astype(
                    jnp.float32)
                b_raw_new = (curr_beta_slice_new[:, :n_v].reshape(n_v).astype(
                    jnp.float32))

                # Compute gate
                curr_beta = jax.nn.sigmoid(b_raw_new)
                curr_g = -jnp.exp(a_log_ref[...].astype(
                    jnp.float32)) * jax.nn.softplus(
                        a_raw_new + dt_bias_ref[...].astype(jnp.float32))
                curr_g = jnp.maximum(curr_g, -100.0)
                decay = jnp.exp(curr_g)

                current_state = decode_state_scratch[0]

                # TODO: compare MXU vs VPU, MXU doesn't support FP32, VPU does
                # (n_v, d_k, 1) * (n_v, 1, d_v) -> (n_v, d_k, d_v)
                out_list = []
                new_state_list = []
                for h in range(n_v):
                    q_h = q[h:h + 1, :]  # (1, d_k)
                    k_h = k[h:h + 1, :]  # (1, d_k)
                    v_h = v[h:h + 1, :]  # (1, d_v)

                    state_h = current_state[h].astype(
                        jnp.float32)  # (d_k, d_v)

                    k_state_h = pl.dot(
                        k_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    # v_diff_h = v_h - decay[h].astype(jnp.float32) * k_state_h
                    decay_k_state = jnp.where(
                        jnp.isinf(k_state_h),
                        0.0,
                        decay[h].astype(jnp.float32) * k_state_h,
                    )
                    v_diff_h = v_h - decay_k_state
                    v_new_h = curr_beta[h].astype(jnp.float32) * v_diff_h

                    q_state_h = pl.dot(
                        q_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    q_k_h = jnp.sum(q_h * k_h, axis=-1,
                                    keepdims=True)  # (1, 1)

                    # Defensive code to handle NaNs and infs in state,
                    # Saw similar issue while trying newton schulz
                    # which can happen due to large decay or long sequences.
                    # TODO: analyze perf impact and risk of removing this.
                    decay_q_state = jnp.where(jnp.isinf(q_state_h), 0.0,
                                              decay[h] * q_state_h)
                    out_h = decay_q_state + q_k_h * v_new_h
                    out_list.append(out_h)

                    k_v_new_h = pl.dot(k_h,
                                       v_new_h,
                                       trans_a=True,
                                       precision=jax.lax.Precision.HIGHEST
                                       )  # (d_k, 1) @ (1, d_v) -> (d_k, d_v)
                    # Defensive code to handle NaNs and infs in state,
                    # which can happen due to large decay or long sequences.
                    # In such cases, we reset the state contribution to zero and rely solely on the new value
                    # TODO: analyze perf impact and risk of removing this.
                    decay_state = jnp.where(jnp.isinf(state_h), 0.0,
                                            state_h * decay[h])
                    new_state_h = decay_state + k_v_new_h
                    new_state_list.append(new_state_h)

                out = jnp.concatenate(out_list, axis=0)  # (n_v, d_v)
                new_state = jnp.stack(new_state_list,
                                      axis=0)  # (n_v, d_k, d_v)

                # Accumulate output in scratchpad
                current_output = decode_output_scratch[...]
                mask = (jnp.arange(BT) == b).astype(current_output.dtype)[:,
                                                                          None]
                new_output = jnp.where(
                    mask,
                    out.reshape(1, n_v * d_v),
                    current_output,
                )
                decode_output_scratch[...] = new_output.astype(
                    current_output.dtype)

                # Async store. Before writing to decode_store_scratch[slot],
                # wait for the previous same-slot store DMA (from iter b-2)
                # so we don't clobber a buffer that's still being read.
                @pl.when(cur_slot_inflight > 0)
                def _wait_same_slot_store():
                    temp_desc = pltpu.make_async_copy(
                        src_ref=decode_store_scratch.at[pl.ds(slot, 1)],
                        dst_ref=recurrent_state_out.at[pl.ds(0, 1)],
                        sem=decode_write_semaphore.at[slot],
                    )
                    temp_desc.wait()

                decode_store_scratch[slot] = new_state.astype(
                    decode_store_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=decode_store_scratch.at[pl.ds(slot, 1)],
                    dst_ref=recurrent_state_out.at[pl.ds(target_idx, 1)],
                    sem=decode_write_semaphore.at[slot],
                )
                copy_op.start()
                # No wait — drained after fori_loop.

            # Update carry: this iter marked its slot as having an in-flight
            # store iff is_valid.
            next_s0_inflight = jax.lax.select(
                is_valid & using_slot_0,
                jnp.int32(1),
                s0_inflight,
            )
            next_s1_inflight = jax.lax.select(
                is_valid & (~using_slot_0),
                jnp.int32(1),
                s1_inflight,
            )
            return (next_s0_inflight, next_s1_inflight)

        # loop over bt, could be for loop, BT is static anyway, unroll
        final_s0_inflight, final_s1_inflight = jax.lax.fori_loop(
            0,
            BT,
            process_decode,
            (jnp.int32(0), jnp.int32(0)),
        )

        # Drain any remaining async store DMAs (at most one per slot).
        @pl.when(final_s0_inflight > 0)
        def _drain_slot_0():
            temp_desc = pltpu.make_async_copy(
                src_ref=decode_store_scratch.at[pl.ds(0, 1)],
                dst_ref=recurrent_state_out.at[pl.ds(0, 1)],
                sem=decode_write_semaphore.at[0],
            )
            temp_desc.wait()

        @pl.when(final_s1_inflight > 0)
        def _drain_slot_1():
            temp_desc = pltpu.make_async_copy(
                src_ref=decode_store_scratch.at[pl.ds(1, 1)],
                dst_ref=recurrent_state_out.at[pl.ds(0, 1)],
                sem=decode_write_semaphore.at[1],
            )
            temp_desc.wait()

        # Mask and write accumulated outputs to HBM
        mask = (jnp.arange(BT)
                < decode_count).astype(decode_output_scratch.dtype)[:, None]
        decode_output_scratch_masked = decode_output_scratch[...] * mask
        decode_output_ref[...] = decode_output_scratch_masked

        return None

    # Prefill Branch
    # Process prefill if there is valid prefill work in this step
    @pl.when(prefill_valid > 0)
    def process_prefill():
        # TODO: eliminate k.transpose in matmuls by directly slicing in the right shape above

        # not used meaningfully, because dma is sync.
        # intention is to index into scratch for storing state and not overwrite each other
        prefill_slot = prefill_req_id % 2

        def process_regular_prefill():
            init_has_init = has_initial_state[prefill_req_id][...]
            init_state_idx = state_indices[prefill_req_id][...]
            should_load_init = (is_first_chunk > 0) & (init_has_init > 0)
            should_zero_init = (is_first_chunk > 0) & (init_has_init == 0)

            @pl.when(should_load_init)
            def _start_init_load():
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(init_state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[prefill_slot],
                )
                copy_op.start()

            @pl.when(should_zero_init)
            def _zero_init_state():
                prefill_scratch[prefill_slot] = jnp.zeros(
                    (n_v, d_k, d_v), dtype=prefill_scratch.dtype)

            ### Preparataion for chunk wise math,
            ### this kernel design could be optimized lot by not doing this every chunk
            # 1. Extract Q, K, V, g, beta for the chunk
            key_dim = n_kq * d_k

            # Workaround: Upcast to fp32 to avoid NaNs in long sequences
            qkv_chunk = prefill_qkv_ref[...].astype(jnp.float32)  # (C, d)
            # Fused SiLU
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim:2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim:]

            # Load a, b
            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            # Slice and transpose to match expected shape (n_v, C),
            # TODO: this transpose can be eliminated
            a_raw_processed = a_raw_chunk[:, :n_v].T
            b_raw_processed = b_raw_chunk[:, :n_v].T


            a_raw_processed = a_raw_processed.astype(jnp.float32)
            b_raw_processed = b_raw_processed.astype(jnp.float32)
            beta = jax.nn.sigmoid(b_raw_processed)
            g = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))
            # Workaround: Clamp g to avoid underflow to negative inf
            # g is always negative, from above line
            # for long prefill sequence this negative value will get more negative
            # pow(e,-100) is close to 0.
            g = jnp.maximum(g, -100.0)
            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
            q = jnp.where(mask_float[:, None] > 0, q, 0.0)
            k = jnp.where(mask_float[:, None] > 0, k, 0.0)
            g = jnp.where(mask_float[None, :] > 0, g, 0.0)
            v = jnp.where(mask_float[:, None] > 0, v, 0.0)
            beta = jnp.where(mask_float[None, :] > 0, beta, 0.0)

            q = q.reshape(C, n_kq, d_k)
            k = k.reshape(C, n_kq, d_k)
            v = v.reshape(C, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            # Transpose first, then repeat: the transpose operates on the
            # smaller (C, n_kq, d_k) tensor; the subsequent repeat on the
            # (now-leading) axis produces the same final layout.
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=0)
                k = jnp.repeat(k, repeat_factor, axis=0)

            scale = d_k**-0.5
            q = q * scale

            g_cumsum_list = []
            current_sum = jnp.zeros((n_v, ), dtype=jnp.float32)
            # cumsum not implemented in pallas
            for i in range(C):
                current_sum = current_sum + g[:, i].astype(jnp.float32)
                g_cumsum_list.append(current_sum)
            g_cumsum = jnp.stack(g_cumsum_list, axis=-1)
            k_beta = k * beta[..., None]

            # Fuse S and S_q into a single matmul.
            k_T = k.transpose(0, 2, 1)
            kbeta_q = jnp.concatenate([k_beta, q], axis=1)  # (n_v, 2C, d_k)
            S_both = jnp.matmul(
                kbeta_q.astype(jnp.float32),
                k_T.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )  # (n_v, 2C, C)
            S = S_both[:, :C, :]
            S_q = S_both[:, C:, :]

            g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
            i = jnp.arange(C)[:, None]
            j = jnp.arange(C)[None, :]
            mask_float = (i > j).astype(jnp.float32)

            # Defensive code to handle large positive g_diff which can cause
            # overflow in exp,
            # TODO: analyze if this is a common case and if we can remove this or do
            # by other means (like clipping g values before cumsum or using a
            # different data type for g/g_cumsum)
            g_diff_safe = jnp.minimum(g_diff, 0.0)
            S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe),
                          0.0)

            mask_float_q = (i >= j).astype(jnp.float32)
            g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (
                1.0 - mask_float_q[None, ...]) * (-1e30)
            S_q = S_q * jnp.exp(g_diff_Sq)
            S_q = S_q * mask_float_q[None, ...]

            I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
            # TODO: call the function in kernels file
            A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

            # Fuse u and w into a single matmul. Both compute
            # A_inv @ <something>; stack v_beta and k_beta_g along the last
            # axis (d_v -> d_v + d_k), do one matmul, then split.
            v_beta = v * beta[..., None]
            k_beta_g = k_beta * jnp.exp(g_cumsum)[..., None]
            vk_in = jnp.concatenate(
                [
                    v_beta.astype(jnp.float32),
                    k_beta_g.astype(jnp.float32),
                ],
                axis=2,
            )  # (n_v, C, d_v + d_k)
            uw = jnp.matmul(A_inv, vk_in, precision=jax.lax.Precision.HIGHEST)
            u = uw[..., :d_v]
            w = uw[..., d_v:]

            q_g = q * jnp.exp(g_cumsum)[..., None]

            # Fuse attn_inter and v_prime into a single matmul. With attentionDP, the async copy leads to very small perf gain.
            @pl.when(should_load_init)
            def _finish_init_load():
                temp_desc = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(init_state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[prefill_slot],
                )
                temp_desc.wait()
                prefill_scratch[prefill_slot] = state_commit_scratch[0].astype(
                    prefill_scratch.dtype)

            current_state = prefill_scratch[prefill_slot]
            qw = jnp.concatenate([q_g.astype(jnp.float32), w], axis=1)
            comb = jnp.matmul(
                qw,
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )  # (n_v, 2C, d_v)
            attn_inter = comb[:, :C, :]
            v_prime = comb[:, C:, :]

            v_new = u - v_prime
            term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
            o_c = attn_inter + term2

            g_i_last_exp = jnp.exp(g_cumsum[..., -1, None, None])
            g_diff_exp_state = jnp.exp(g_cumsum[..., -1, None] -
                                       g_cumsum)[..., None]
            k_i_g_diff = k * g_diff_exp_state

            update_term = jnp.matmul(
                k_i_g_diff.transpose(0, 2, 1).astype(jnp.float32),
                v_new,
                precision=jax.lax.Precision.HIGHEST,
            )
            h_new = current_state * g_i_last_exp + update_term

            prefill_scratch[prefill_slot] = h_new.astype(prefill_scratch.dtype)

            # Store state only if it's the last chunk of the request.
            store_state_idx = state_indices[prefill_req_id][...]

            @pl.when(is_last_chunk > 0)
            def store_state():
                # TODO: if dtype of state in HBM is always f32,
                # then we can eliminate this copy and directly write from scratch to HBM
                state_commit_scratch[0] = prefill_scratch[prefill_slot].astype(
                    state_commit_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(store_state_idx, 1)],
                    sem=prefill_semaphore.at[prefill_slot],
                )
                copy_op.start()
                copy_op.wait()

            # TODO: eliminate this transpose and reshape by directly writing in the right shape above
            o_c_tr = o_c.transpose(1, 0, 2)
            o_c_flat = o_c_tr.reshape(C, n_v * d_v)

            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
            o_c_flat_masked = o_c_flat * mask_float[:, None]
            prefill_output_ref[...] = o_c_flat_masked.astype(
                prefill_output_ref.dtype)

            return None

        def process_transition_prefill():
            # this is processing prefill sequences in a sublane that has multiple sequences
            C_trans = sublanesize
            key_dim = n_kq * d_k

            first_req_id = schedule_table[step, 11][...]
            first_is_first = schedule_table[step, 11 + C_trans][...]
            first_slot = first_req_id % 2
            first_has_init = has_initial_state[first_req_id][...]
            should_load_first = (first_is_first > 0) & (first_has_init > 0)
            first_state_idx = state_indices[first_req_id][...]

            # Async initial load
            @pl.when(should_load_first)
            def _start_first_load():
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(first_state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[first_slot],
                )
                copy_op.start()

            # Workaround: Upcast to fp32 to avoid NaNs
            qkv_chunk = prefill_qkv_ref[:C_trans, :].astype(jnp.float32)
            # Fused SiLU TODO: maybe 'SiLU' needs to be parametrized,
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim:2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim:]

            # Load untransposed a and b
            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            # Slice and transpose to match expected shape (n_v, C_trans)
            a_raw_processed = a_raw_chunk[:C_trans, :n_v].T
            b_raw_processed = b_raw_chunk[:C_trans, :n_v].T

            a_raw_processed = a_raw_processed.astype(jnp.float32)
            b_raw_processed = b_raw_processed.astype(jnp.float32)
            beta_chunk = jax.nn.sigmoid(b_raw_processed)
            g_chunk = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))
            g_chunk = jnp.maximum(g_chunk, -100.0)
            q = q.reshape(C_trans, n_kq, d_k)
            k = k.reshape(C_trans, n_kq, d_k)
            v = v.reshape(C_trans, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            # Transpose first, then repeat
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=0)
                k = jnp.repeat(k, repeat_factor, axis=0)

            scale = d_k**-0.5
            q = q * scale

            # Cold-start: zero the slot in place when the first sequence has
            # no carried-over state, then read.
            @pl.when((first_is_first > 0) & (first_has_init == 0))
            def _zero_first_slot():
                prefill_scratch[first_slot] = jnp.zeros(
                    (n_v, d_k, d_v), dtype=prefill_scratch.dtype)

            # Finish the async initial load started above
            @pl.when(should_load_first)
            def _finish_first_load():
                temp_desc = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(first_state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[first_slot],
                )
                temp_desc.wait()
                prefill_scratch[first_slot] = state_commit_scratch[0].astype(
                    prefill_scratch.dtype)

            h = prefill_scratch[first_slot]

            current_r = first_req_id
            sequence_valid = True

            # Loop over tokens in the sublane.
            for i in range(sublanesize):
                t_req = schedule_table[step, 11 + i][...]
                t_is_first = schedule_table[step, 11 + C_trans + i][...]
                t_is_last = schedule_table[step, 11 + 2 * C_trans + i][...]

                is_new_seq = t_req != current_r
                sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

                # Ignore tokens that belong to decode requests,
                # (assumes decode tokens are at packed at head)
                is_decode_token = t_req < decode_tokens
                sequence_valid = jnp.where(is_decode_token, False,
                                           sequence_valid)

                c_slot = current_r % 2

                # Commit the previous iter's h to the current request's slot.
                # The other slot is untouched.
                prefill_scratch[c_slot] = h

                def do_write():
                    state_idx = state_indices[current_r][...]
                    # Stage h only when we actually DMA out.
                    state_commit_scratch[0] = h.astype(
                        state_commit_scratch.dtype)
                    copy_op = pltpu.make_async_copy(
                        src_ref=state_commit_scratch,
                        dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                        sem=prefill_semaphore.at[c_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    return None

                is_current_r_prefill = current_r >= decode_tokens
                should_write = is_current_r_prefill & is_new_seq
                jax.lax.cond(should_write, do_write, lambda: None)

                t_slot = t_req % 2
                t_has_init = has_initial_state[t_req][...]

                def load_t_state():
                    state_idx = state_indices[t_req][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=prefill_semaphore.at[t_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    prefill_scratch[t_slot] = state_commit_scratch[0].astype(
                        prefill_scratch.dtype)

                should_load_t = (t_is_first > 0) & (t_has_init > 0)
                jax.lax.cond(should_load_t, load_t_state, lambda: None)

                # Cold-start: zero the slot in place for a new sequence with
                # no carried-over state. Mutually exclusive with load_t_state.
                @pl.when((t_is_first > 0) & (t_has_init == 0))
                def _zero_t_slot():
                    prefill_scratch[t_slot] = jnp.zeros(
                        (n_v, d_k, d_v), dtype=prefill_scratch.dtype)

                h = prefill_scratch[t_slot]

                current_r = t_req

                k_i = k[:, i, :]
                v_i = v[:, i, :]
                g_i = g_chunk[:, i]
                beta_i = beta_chunk[:, i]
                q_i = q[:, i, :]

                decay = jnp.exp(g_i)[..., None]

                k_state = jnp.sum(k_i[..., None] * h, axis=1)
                v_diff = v_i - decay * k_state
                v_new = beta_i[:, None] * v_diff

                q_state = jnp.sum(q_i[..., None] * h, axis=1)
                q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

                out_i = decay * q_state + q_k * v_new

                k_v_new = k_i[..., None] * v_new[:, None, :]
                h_new = h * decay[..., None] + k_v_new

                h = jnp.where(sequence_valid, h_new, h)

                # Mask output before invalidating the sequence for the next token.
                out_i = jnp.where(sequence_valid, out_i, 0.0)

                sequence_valid = jnp.where(t_is_last > 0, False,
                                           sequence_valid)

                prefill_output_ref[i, :] = out_i.reshape(n_v * d_v).astype(
                    prefill_output_ref.dtype)

            final_slot = current_r % 2
            prefill_scratch[final_slot] = h

            is_current_r_prefill = current_r >= decode_tokens

            # Store state if the current request is a prefill.
            # At the end of the transition prefill. No need to make this async.
            @pl.when(is_current_r_prefill)
            def do_final_write():
                state_idx = state_indices[current_r][...]
                # Stage h into state_commit only when we actually DMA out.
                state_commit_scratch[0] = h.astype(state_commit_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                    sem=prefill_semaphore.at[final_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            return None

        is_transition = schedule_table[step, 10][...]

        def process_prefill_dispatch():
            return jax.lax.cond(
                is_transition > 0,
                lambda _: process_transition_prefill(),
                lambda _: process_regular_prefill(),
                operand=None,
            )

        process_prefill_dispatch()
        return None

    # For transition block at boundary of decode and prefill we will have overlap
    # decode block BT contains prefill tokens
    # sublane size transition prefill block contains some decode tokens in the sublane
    # so we need to stitch the outputs so they don't overwrite each other in global index
    # we exchange decode and prefill outputs so
    # prefill output ref has decode token outputs at decode token indexes in its out ref
    # decode output ref has prefill token outputs have prefill token indexes in its out ref
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

        # 4. Merge the tensors
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
    count_col,
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
        valid_col=0,
        offset_col=1,
        count_col=3,
        alignment=alignment,
        block_size=chunk_size,
        sink_offset=sink_offset,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=4,
        offset_col=5,
        count_col=7,
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
    distribution: jax.Array of shape [2]. Contains [decode_tokens,
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
    # if vmem_limit_bytes is None:
    #     vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.8)

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
            # vmem_limit_bytes=vmem_limit_bytes,
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