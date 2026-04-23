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
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.compute_schedule_v2 import \
    compute_schedule_table_v2


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
    # VMEM: (C, C). Identity matrix for Newton-Schulz (currently unused)
    identity_ref,
    # VMEM: (C, n_v * d_v). Scanned outputs for prefill
    prefill_output_ref,
    # VMEM: (BT, n_v * d_v). Scanned outputs for decode
    decode_output_ref,
    # SMEM: (max_blocks, 8). Schedule table
    schedule_table,
    # SMEM: (max_reqs,). State indices
    state_indices,
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
    # VMEM scratchpad: (2, 2, n_v, d_k, d_v). To hold decode states (double
    # buffered, loop double buffered)
    decode_state_scratch,
    state_commit_scratch,
    # VMEM scratchpad: (C, n_v * d_v). To hold prefill outputs before DMA
    o_c_flat_scratch,
    # VMEM scratchpad: (BT, n_v * d_v). To hold decode outputs before DMA
    decode_output_scratch,
    # Array of C semaphores for decode state loads
    decode_read_semaphores,
    # 1 semaphore for decode state stores
    decode_write_semaphore,
    # 1 semaphore for prefill DMA (stores only)
    prefill_semaphore,
    # Number of decode tokens (requests) in the batch
    decode_tokens,
):
    step = pl.program_id(0)
    print(
        f"A log shape in kernel: {a_log_ref.shape}, dt_bias shape in kernel: {dt_bias_ref.shape}"
    )

    # READ table

    prefill_valid = schedule_table[step, 0][...]
    prefill_req_id = schedule_table[step, 2][...]

    decode_valid = schedule_table[step, 4][...]
    decode_offset = schedule_table[step, 5][...]
    decode_req_id = schedule_table[step, 6][...]
    decode_count = schedule_table[step, 7][...]

    prefill_offset = schedule_table[step, 1][...]
    is_transition = schedule_table[step, 11][...]

    is_last_chunk = schedule_table[step, 8][...]
    is_first_chunk = schedule_table[step, 9][...]

    def l2_normalize(x, eps=1e-6):
        norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
        return x / norm

    # 2. Decode Branch
    def decode_wrapper():

        def get_target_idx(b):
            safe_req_id = jnp.minimum(decode_req_id + b,
                                      state_indices.shape[0] - 1)
            return state_indices[safe_req_id][...]

        def process_decode(b, _):
            # token by token check if decode token or not
            is_valid = b < decode_count

            def do_work():
                target_idx = get_target_idx(b)

                # Load state TODO: make async
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(target_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=decode_read_semaphores.at[0],
                )
                copy_op.start()
                copy_op.wait()
                decode_state_scratch[pl.ds(
                    0, 1)] = state_commit_scratch[...].astype(jnp.float32)

                key_dim = n_kq * d_k
                b_aligned = (b // sublanesize) * sublanesize
                # Workaround: Upcast to fp32 to avoid NaNs
                qkv_block_data = decode_qkv_ref[
                    pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
                    qkv_block_data.dtype)[:, None]
                qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)
                # Fused SiLU
                qkv_row = qkv_row * jax.nn.sigmoid(qkv_row)

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

                    state_h = current_state[h]  # (d_k, d_v)

                    k_state_h = pl.dot(
                        k_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    v_diff_h = v_h - decay[h].astype(jnp.float32) * k_state_h
                    v_new_h = curr_beta[h].astype(jnp.float32) * v_diff_h

                    q_state_h = pl.dot(
                        q_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    q_k_h = jnp.sum(q_h * k_h, axis=-1,
                                    keepdims=True)  # (1, 1)

                    out_h = decay[h] * q_state_h + q_k_h * v_new_h  # (1, d_v)
                    out_list.append(out_h)

                    k_v_new_h = pl.dot(k_h,
                                       v_new_h,
                                       trans_a=True,
                                       precision=jax.lax.Precision.HIGHEST
                                       )  # (d_k, 1) @ (1, d_v) -> (d_k, d_v)
                    new_state_h = state_h * decay[h] + k_v_new_h
                    new_state_list.append(new_state_h)

                out = jnp.concatenate(out_list, axis=0)  # (n_v, d_v)
                new_state = jnp.stack(new_state_list,
                                      axis=0)  # (n_v, d_k, d_v)

                # decay_exp = decay[..., None]  # (n_v, 1)

                # k_state = jnp.sum(k[..., None] * current_state, axis=1)  # (n_v, d_v)
                # v_diff = v - decay_exp * k_state
                # v_new = curr_beta[..., None] * v_diff  # (n_v, d_v)

                # q_state = jnp.sum(q[..., None] * current_state, axis=1)  # (n_v, d_v)
                # q_k = jnp.sum(q * k, axis=-1, keepdims=True)  # (n_v, 1)

                # out = decay_exp * q_state + q_k * v_new  # (n_v, d_v)
                # k_v_new = k[..., None] * v_new[:, None, :]
                # new_state = current_state * decay_exp[..., None] + k_v_new

                decode_state_scratch[pl.ds(
                    0, 1)] = new_state[None, ...].astype(current_state.dtype)

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

                # Store state (Synchronous)
                state_commit_scratch[0] = decode_state_scratch[0].astype(
                    state_commit_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(target_idx, 1)],
                    sem=decode_write_semaphore.at[0],
                )
                copy_op.start()
                copy_op.wait()

                return None

            jax.lax.cond(is_valid, do_work, lambda: None)
            return None

        # loop over bt, could be for loop, BT is static anyway, unroll
        jax.lax.fori_loop(0, BT, process_decode, None)

        # Mask and write accumulated outputs to HBM
        mask = (jnp.arange(BT)
                < decode_count).astype(decode_output_scratch.dtype)[:, None]
        decode_output_scratch_masked = decode_output_scratch[...] * mask
        decode_output_ref[...] = decode_output_scratch_masked

        return None

    # check current iteration had decode work
    jax.lax.cond(
        decode_valid > 0,
        decode_wrapper,
        lambda: None,
    )

    # Prefill Branch
    def process_prefill():
        # not used meaningfully, because dma is sync.
        # intention is to index into scratch for storing state and not overwrite each other
        prefill_slot = prefill_req_id % 2

        def process_regular_prefill():
            # 1. Initialize state to zero if first chunk
            def init_state():
                prefill_scratch[prefill_slot] = jnp.zeros(
                    (n_v, d_k, d_v), dtype=prefill_scratch.dtype)
                return None

            # init only if first chunk
            jax.lax.cond(is_first_chunk > 0, init_state, lambda: None)

            ### Preparataion for chunk wise math,
            ### this kernel design could be optimized lot by not doing this every chunk
            # 1. Extract Q, K, V, g, beta for the chunk
            key_dim = n_kq * d_k
            # Workaround: Upcast to fp32 to avoid NaNs in long sequences
            qkv_chunk = prefill_qkv_ref[...].astype(jnp.float32)  # (C, d)
            # Fused SiLU
            qkv_chunk = qkv_chunk * jax.nn.sigmoid(qkv_chunk)
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

            # Compute gates in VMEM in full fp32, not sure if needed.
            beta = jax.nn.sigmoid(b_raw_processed)
            g = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))

            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)

            q = q * mask_float[:, None]
            k = k * mask_float[:, None]
            g = g * mask_float[None, :]
            v = v * mask_float[:, None]
            beta = beta * mask_float[None, :]

            q = q.reshape(C, n_kq, d_k)
            k = k.reshape(C, n_kq, d_k)
            v = v.reshape(C, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

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

            S = jnp.matmul(
                k_beta.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
            i = jnp.arange(C)[:, None]
            j = jnp.arange(C)[None, :]
            mask_float = (i > j).astype(jnp.float32)

            g_diff_safe = jnp.minimum(g_diff, 0.0)
            g_diff_S = g_diff_safe * mask_float + (1.0 - mask_float) * (-1e30)
            S = S * jnp.exp(g_diff_S)
            S = S * mask_float

            S_q = jnp.matmul(
                q.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            mask_float_q = (i >= j).astype(jnp.float32)
            g_diff_Sq = g_diff_safe * mask_float_q + (1.0 -
                                                      mask_float_q) * (-1e30)
            S_q = S_q * jnp.exp(g_diff_Sq)
            S_q = S_q * mask_float_q

            # I_plus_S = identity_ref[...] + S
            # maybe better to initialize in kernel instead of wasting bandwidth
            I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
            # TODO: cleanup its inligned, duplicated code
            A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

            # UW
            v_beta = v * beta[..., None]
            u = jnp.matmul(A_inv,
                           v_beta.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST)

            k_beta_g = k_beta * jnp.exp(g_cumsum)[..., None]
            w = jnp.matmul(
                A_inv,
                k_beta_g.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            q_g = q * jnp.exp(g_cumsum)[..., None]
            current_state = prefill_scratch[prefill_slot]
            attn_inter = jnp.matmul(
                q_g.astype(jnp.float32),
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            v_prime = jnp.matmul(
                w,
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
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

            def store_state():
                state_commit_scratch[0] = prefill_scratch[prefill_slot].astype(
                    state_commit_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(prefill_req_id, 1)],
                    sem=prefill_semaphore.at[prefill_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            jax.lax.cond(is_last_chunk > 0, store_state, lambda: None)

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

            # Workaround: Upcast to fp32 to avoid NaNs
            qkv_chunk = prefill_qkv_ref[...]
            qkv_chunk = qkv_chunk[:C_trans, :].astype(jnp.float32)
            # Fused SiLU
            qkv_chunk = qkv_chunk * jax.nn.sigmoid(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim:2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim:]

            # Load untransposed a and b
            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            # Slice and transpose to match expected shape (n_v, C_trans)
            a_raw_processed = a_raw_chunk[:C_trans, :n_v].T
            b_raw_processed = b_raw_chunk[:C_trans, :n_v].T

            # Compute gates in VMEM in full fp32
            beta_chunk = jax.nn.sigmoid(b_raw_processed)
            g_chunk = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))

            q = q.reshape(C_trans, n_kq, d_k)
            k = k.reshape(C_trans, n_kq, d_k)
            v = v.reshape(C_trans, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            scale = d_k**-0.5
            q = q * scale

            first_req_id = schedule_table[step, 12][...]
            first_is_first = schedule_table[step, 12 + C_trans][...]
            first_slot = first_req_id % 2
            h = prefill_scratch[first_slot]
            h = jnp.where(first_is_first > 0, jnp.zeros_like(h), h)

            current_r = first_req_id
            sequence_valid = True

            # loop over token by token
            for i in range(sublanesize):
                # read transition token metadata
                t_req = schedule_table[step, 12 + i][...]
                t_is_first = schedule_table[step, 12 + C_trans + i][...]
                t_is_last = schedule_table[step, 12 + 2 * C_trans + i][...]

                is_new_seq = t_req != current_r
                sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

                # Ignore tokens that belong to decode requests
                is_decode_token = t_req < decode_tokens
                sequence_valid = jnp.where(is_decode_token, False,
                                           sequence_valid)

                c_slot = current_r % 2

                h0 = prefill_scratch[0]
                h1 = prefill_scratch[1]
                prefill_scratch[0] = jnp.where(c_slot == 0, h, h0)
                prefill_scratch[1] = jnp.where(c_slot == 1, h, h1)

                state_commit_scratch[0] = prefill_scratch[c_slot].astype(
                    state_commit_scratch.dtype)

                def do_write():
                    # TODO: Make async
                    copy_op = pltpu.make_async_copy(
                        src_ref=state_commit_scratch,
                        dst_ref=recurrent_state_out.at[pl.ds(current_r, 1)],
                        sem=prefill_semaphore.at[c_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    return None

                is_current_r_prefill = current_r >= decode_tokens
                should_write = is_current_r_prefill & is_new_seq
                jax.lax.cond(should_write, do_write, lambda: None)

                t_slot = t_req % 2
                h0_new = prefill_scratch[0]
                h1_new = prefill_scratch[1]
                new_h = jnp.where(t_slot == 0, h0_new, h1_new)

                new_h = jnp.where(t_is_first > 0, jnp.zeros_like(new_h), new_h)
                h = new_h

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

                # Mask output BEFORE invalidating the sequence for the next token
                out_i = jnp.where(sequence_valid, out_i, 0.0)

                sequence_valid = jnp.where(t_is_last > 0, False,
                                           sequence_valid)

                prefill_output_ref[i, :] = out_i.reshape(n_v * d_v).astype(
                    prefill_output_ref.dtype)

            final_slot = current_r % 2
            prefill_scratch[final_slot] = h
            state_commit_scratch[0] = h.astype(state_commit_scratch.dtype)

            def do_final_write():
                # TODO: make async
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(current_r, 1)],
                    sem=prefill_semaphore.at[final_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            is_current_r_prefill = current_r >= decode_tokens
            jax.lax.cond(is_current_r_prefill, do_final_write, lambda: None)

            return None

        is_transition = schedule_table[step, 11][...]

        def process_prefill_dispatch():
            return jax.lax.cond(
                is_transition > 0,
                lambda _: process_transition_prefill(),
                lambda _: process_regular_prefill(),
                operand=None,
            )

        process_prefill_dispatch()
        return None

    jax.lax.cond(prefill_valid > 0, process_prefill, lambda: None)

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
    num_tokens,
    valid_col,
    offset_col,
    count_col,
    alignment=16,
):
    valid = schedule_table[step, valid_col][...]
    offset = schedule_table[step, offset_col][...]
    offset = pl.multiple_of(offset, alignment)
    count = schedule_table[step, count_col][...]

    safe_offset = jnp.where(valid > 0, offset, num_tokens - 1)
    safe_offset = pl.multiple_of(safe_offset, alignment)

    safe_count = pl.cdiv(count, alignment) * alignment
    return (pl.ds(safe_offset, safe_count), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    recurrent_state_shape,
    num_tokens,
    alignment=16,
):
    """Creates block specs for recurrent scan kernel."""

    prefill_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        num_tokens=num_tokens,
        valid_col=0,
        offset_col=1,
        count_col=3,
        alignment=alignment,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        num_tokens=num_tokens,
        valid_col=4,
        offset_col=5,
        count_col=7,
        alignment=alignment,
    )

    prefill_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), d),
        index_map=prefill_qkv_index_map,
    )
    decode_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), d),
        index_map=decode_qkv_index_map,
    )

    identity_spec = pl.BlockSpec(
        block_shape=(chunk_size, chunk_size),
        index_map=lambda step: (0, 0),
    )

    prefill_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
        index_map=prefill_qkv_index_map,
    )
    decode_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), n_v * d_v),
        index_map=decode_qkv_index_map,
    )

    a_log_spec = pl.BlockSpec(block_shape=(n_v, ),
                              index_map=lambda step: (0, ))
    dt_bias_spec = pl.BlockSpec(block_shape=(n_v, ),
                                index_map=lambda step: (0, ))
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
        identity_spec,
    ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    query_start_loc_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    identity_ref,
    schedule_table_ref,
    decode_tokens_ref,
    prefill_tokens_ref,
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
    print(f"Launching fused_kernel with chunk_size={C}")
    print(f"fused_kernel: mixed_qkv_ref.shape={mixed_qkv_ref.shape}")
    print(
        f"fused_kernel: recurrent_state_ref.shape={recurrent_state_ref.shape}")
    print(f"fused_kernel: state_indices_ref.shape={state_indices_ref.shape}")

    decode_tokens = decode_tokens_ref[0]
    # prefill_tokens = prefill_tokens_ref[0]
    total_blocks = total_blocks_ref[0]

    # num_decode_batches = (decode_tokens + BT - 1) // BT
    print(f"fused_kernel: total_blocks={total_blocks}")

    d = mixed_qkv_ref.shape[-1]

    in_specs, out_specs = create_block_specs(
        schedule_table_ref,
        C,
        BT,
        d,
        n_v,
        d_v,
        recurrent_state_ref.shape,
        mixed_qkv_ref.shape[0],
    )
    print(f"fused_kernel: len(in_specs)={len(in_specs)},"
          f" len(out_specs)={len(out_specs)}")

    def _run_with_scratch(
        scratch_ref,
        decode_state_scratch_ref,
        state_commit_scratch_ref,
        o_c_flat_scratch_ref,
        decode_output_scratch_ref,
        decode_read_sems,
        decode_write_sem,
        prefill_sem,
    ):
        print(f"Running fused kernel body A_log shape {a_log_ref.shape}")
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
                o_c_flat_scratch=o_c_flat_scratch_ref,
                decode_output_scratch=decode_output_scratch_ref,
                state_commit_scratch=state_commit_scratch_ref,
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
            identity_ref,
            output_ref,
            output_ref,
            scratches=[schedule_table_ref, state_indices_ref],
        )

    pl.run_scoped(
        _run_with_scratch,
        pltpu.VMEM((2, n_v, d_k, d_v),
                   jnp.float32),  # prefill_scratch (double buffered)
        pltpu.VMEM((1, n_v, d_k, d_v), jnp.float32),  # decode_state_scratch
        pltpu.VMEM((1, n_v, d_k, d_v),
                   recurrent_state_ref.dtype),  # state_commit_scratch
        # TODO: Move this to outer pallas call and get rid of run_scoped
        pltpu.VMEM((C, n_v * d_v), mixed_qkv_ref.dtype),  # o_c_flat_scratch
        pltpu.VMEM((BT, n_v * d_v),
                   mixed_qkv_ref.dtype),  # decode_output_scratch
        pltpu.SemaphoreType.DMA((1, )),  # decode_read_semaphores
        pltpu.SemaphoreType.DMA((1, )),  # decode_write_semaphore
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
        "max_reqs",
        "use_qk_norm_in_gdn",
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
    max_reqs: int = 3,
    use_qk_norm_in_gdn: bool = True,
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
    max_reqs: Maximum number of requests supported by state buffer.
    use_qk_norm_in_gdn: Whether to use QK normalization.

  Returns:
    A tuple containing:
      - Updated recurrent state of shape [max_reqs, n_v, d_k, d_v].
      - The mixed_qkv array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
  """
    jax.debug.print("recurrent_scan: query_start_loc={}", query_start_loc)
    # TODO(kunjanp): Compute beta and g inside the kernel to save HBM bandwidth.
    # We could pass raw a and b and compute sigmoid/softplus on the fly.
    # beta = jax.nn.sigmoid(b)
    # g = -jnp.exp(A_log[:, jnp.newaxis]) * jax.nn.softplus(
    #     a + dt_bias[:, jnp.newaxis]
    # )
    # # Workaround: Clamp g to avoid underflow to -inf in g_cumsum
    # g = jnp.maximum(g, -100.0)

    # Pad raw a and b to (num_tokens, 128) for sublanes
    a_padded = jnp.pad(a, ((0, 0), (0, 128 - n_v)))
    b_padded = jnp.pad(b, ((0, 0), (0, 128 - n_v)))

    print(
        f"From recurrent scan n_kq={n_kq}, n_v={n_v}, d_k={d_k}, d_v={d_v}, chunk_size={chunk_size}, BT={BT}, max_reqs={max_reqs}, use_qk_norm_in_gdn={use_qk_norm_in_gdn}"
    )
    print(
        f"recurrent_scan: mixed_qkv.shape={mixed_qkv.shape}, recurrent_state.shape={recurrent_state.shape}, state_indices.shape={state_indices.shape}, A_log.shape={A_log.shape}, dt_bias.shape={dt_bias.shape}, query_start_loc.shape={query_start_loc.shape}, distribution.shape={distribution.shape} "
    )
    # TODO(kunjanp): Materialize identity matrix directly inside the kernel
    identity = jnp.eye(chunk_size, dtype=jnp.float32)

    # decode_tokens: scalar, number of decode tokens.
    # Assuming length 1 per decode request, this is also the number of decode
    # requests.
    decode_tokens = distribution[0]
    prefill_tokens = distribution[1] - distribution[0]

    # change to by dtype size
    sublanesize = 8 if mixed_qkv.dtype == jnp.float32 else 16
    schedule_table, total_blocks = (compute_schedule_table_v2(
        query_start_loc,
        decode_tokens,
        mixed_qkv.shape[0],
        chunk_size,
        BT,
        alignment=sublanesize,
    ))

    # sublane,128
    decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
    prefill_tokens_arr = jnp.expand_dims(prefill_tokens, 0)
    total_blocks_arr = jnp.expand_dims(total_blocks, 0)

    grid_spec = pl.GridSpec(
        grid=(1, ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda step: (0, )),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda step: (0, )),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda step: (0, )),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
    )

    updated_recurrent_state, output = pl.pallas_call(
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
            jax.ShapeDtypeStruct((mixed_qkv.shape[0], n_v * d_v),
                                 mixed_qkv.dtype),
        ),
        grid_spec=grid_spec,
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(
        mixed_qkv,
        query_start_loc,
        recurrent_state,
        state_indices,
        a_padded,
        b_padded,
        A_log,
        dt_bias,
        identity,
        schedule_table,
        decode_tokens_arr,
        prefill_tokens_arr,
        total_blocks_arr,
    )
    return updated_recurrent_state, output
