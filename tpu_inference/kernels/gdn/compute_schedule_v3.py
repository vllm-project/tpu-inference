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
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel(
    query_metadata_ref: jax.Array,
    num_seqs_ref: jax.Array,
    decode_tokens_ref: jax.Array,
    chunk_size_ref: jax.Array,
    BT_ref: jax.Array,
    total_num_seqs_ref: jax.Array,
    table_ref: jax.Array,
    *,
    alignment: int,
):

    # Read scalars from the first element of the padded vectors
    num_seqs = num_seqs_ref[0]
    decode_tokens = decode_tokens_ref[0]
    chunk_size = chunk_size_ref[0]
    BT = BT_ref[0]
    total_num_seqs = total_num_seqs_ref[0]

    max_metadata_idx = query_metadata_ref.shape[0] - 1
    total_tokens = query_metadata_ref[jnp.minimum(num_seqs, max_metadata_idx)]
    last_seq_id = total_num_seqs - 1

    # 1. Prefill loop over requests
    def process_request(r, running_sum):
        seq_start = query_metadata_ref[jnp.minimum(r, max_metadata_idx)]
        seq_end = query_metadata_ref[jnp.minimum(r + 1, max_metadata_idx)]

        effective_start = jnp.where(
            seq_start % alignment != 0,
            (seq_start // alignment) * alignment + alignment,
            seq_start,
        )

        is_decode_boundary = seq_start == decode_tokens
        is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

        next_aligned_start = (seq_end // alignment) * alignment
        is_last_seq = r == last_seq_id
        needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                            (~is_swallowed))

        needs_start_transition = ((seq_start % alignment != 0) &
                                  (~is_swallowed) & is_decode_boundary)

        effective_end = jnp.where(needs_transition, next_aligned_start,
                                  seq_end)
        effective_end = jnp.maximum(effective_start, effective_end)

        num_regular_blocks = (effective_end - effective_start + chunk_size -
                              1) // chunk_size

        total_blocks_per_seq = (num_regular_blocks +
                                needs_transition.astype(jnp.int32) +
                                needs_start_transition.astype(jnp.int32))
        total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)
        is_pure_decode = seq_end <= decode_tokens
        total_blocks_per_seq = jnp.where(is_pure_decode, 0,
                                         total_blocks_per_seq)

        # Inner loop over blocks for current request
        def block_loop(b, _):
            # index of block within blocks for a sequence
            local_b = b
            start_trans_offset = (seq_start // alignment) * alignment

            is_start_trans = needs_start_transition & (local_b == 0)
            # Adjust local_b for regular blocks if there was a start transition
            adj_local_b = jnp.where(needs_start_transition, local_b - 1,
                                    local_b)
            is_end_trans = needs_transition & (adj_local_b
                                               == num_regular_blocks)

            reg_offset = effective_start + adj_local_b * chunk_size
            reg_count = jnp.minimum(chunk_size, effective_end - reg_offset)

            trans_offset = next_aligned_start

            # Apply predication
            block_offset = jnp.where(
                is_start_trans,
                start_trans_offset,
                jnp.where(is_end_trans, trans_offset, reg_offset),
            )

            block_count = jnp.where(
                is_start_trans,
                effective_start - seq_start,
                jnp.where(is_end_trans, alignment, reg_count),
            )

            is_trans_block = is_start_trans | is_end_trans

            is_first_block = block_offset <= seq_start
            is_last_block = (block_offset + block_count) >= seq_end

            # Metadata for shared sublane tiles
            glob_idxs = block_offset + lax.broadcasted_iota(
                jnp.int32, (alignment, ), 0)
            valid_mask = glob_idxs < total_tokens

            def body_func(i, current_sum):
                return current_sum + (glob_idxs >= query_metadata_ref[
                    jnp.minimum(i, max_metadata_idx)]).astype(jnp.int32)

            sum_ge = lax.fori_loop(0, num_seqs, body_func,
                                   jnp.zeros((alignment, ), dtype=jnp.int32))
            t_reqs = sum_ge - 1

            t_reqs = jnp.where(valid_mask, t_reqs, num_seqs - 1)
            t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

            start_locs = jnp.zeros((alignment, ), dtype=jnp.int32)
            end_locs = jnp.zeros((alignment, ), dtype=jnp.int32)
            iota_alignment = lax.broadcasted_iota(jnp.int32, (alignment, ), 0)

            carry = (start_locs, end_locs)
            for k in range(alignment):
                s_locs, e_locs = carry
                req_idx = t_reqs[k]
                val_start = query_metadata_ref[jnp.minimum(
                    req_idx, max_metadata_idx)]
                val_end = query_metadata_ref[jnp.minimum(
                    req_idx + 1, max_metadata_idx)]

                mask = iota_alignment == k
                s_locs = jnp.where(mask, val_start, s_locs)
                e_locs = jnp.where(mask, val_end, e_locs)
                carry = (s_locs, e_locs)
            start_locs, end_locs = carry

            is_first_tok = (glob_idxs == start_locs).astype(jnp.int32)
            is_last_tok = (glob_idxs == end_locs - 1).astype(jnp.int32)

            row_idx = running_sum + b

            # Construct full row
            prefill_parts = [
                # 0: prefill_valid
                jnp.ones((1, ), dtype=jnp.int32),
                # 1: block_offset
                jnp.expand_dims(block_offset, 0).astype(jnp.int32),
                # 2: r_for_block
                jnp.expand_dims(r, 0).astype(jnp.int32),
                # 3: block_count
                jnp.expand_dims(block_count, 0).astype(jnp.int32),
                # 4: decode_valid
                jnp.zeros((1, ), dtype=jnp.int32),
                # 5: decode_offsets
                jnp.zeros((1, ), dtype=jnp.int32),
                # 6: decode_req_ids
                jnp.zeros((1, ), dtype=jnp.int32),
                # 7: decode_counts
                jnp.zeros((1, ), dtype=jnp.int32),
                ## Conditions
                # 8: block_is_last
                jnp.expand_dims(is_last_block, 0).astype(jnp.int32),
                # 9: block_is_first
                jnp.expand_dims(is_first_block, 0).astype(jnp.int32),
                # 10: is_trans_block
                jnp.expand_dims(is_trans_block, 0).astype(jnp.int32),
                ## Sublane info
                # 11 to 11 + alignment - 1: t_reqs
                t_reqs.astype(jnp.int32),
                # 11 + alignment to 11 + 2 * alignment - 1: is_first_tok
                is_first_tok.astype(jnp.int32),
                # 11 + 2 * alignment to 11 + 3 * alignment - 1: is_last_tok
                is_last_tok.astype(jnp.int32),
            ]

            prefill_metadata = jnp.concatenate(prefill_parts)
            table_ref[row_idx, :] = prefill_metadata

            return None

        lax.fori_loop(0, total_blocks_per_seq, block_loop, None)

        return running_sum + total_blocks_per_seq

    running_sum = lax.fori_loop(0, num_seqs, process_request, 0)

    # 2. Decode loop
    num_decode_batches = (decode_tokens + BT - 1) // BT

    def decode_loop(b, _):
        # Reverse the batch index to match v2 row ordering
        decode_batch_idx = (num_decode_batches - 1) - b
        decode_offsets = decode_batch_idx * BT
        decode_req_ids = decode_batch_idx * BT
        decode_counts = jnp.minimum(BT, decode_tokens - decode_offsets)

        # For decode, each token belongs to a separate request (length 1)

        existing_row = table_ref[b, :]

        is_real_prefill = b < running_sum

        decode_parts = [
            # 4: decode_valid
            jnp.ones((1, ), dtype=jnp.int32),
            # 5: decode_offsets
            jnp.expand_dims(decode_offsets, 0).astype(jnp.int32),
            # 6: decode_req_ids
            jnp.expand_dims(decode_req_ids, 0).astype(jnp.int32),
            # 7: decode_counts
            jnp.expand_dims(decode_counts, 0).astype(jnp.int32),
            ## Conditions
            # 8: block_is_last
            jnp.where(
                is_real_prefill,
                existing_row[8:9],
                jnp.zeros((1, ), dtype=jnp.int32),
            ),
            # 9: block_is_first
            jnp.where(
                is_real_prefill,
                existing_row[9:10],
                jnp.zeros((1, ), dtype=jnp.int32),
            ),
            # 10: is_trans_block
            jnp.where(
                is_real_prefill,
                existing_row[10:11],
                jnp.zeros((1, ), dtype=jnp.int32),
            ),
            ## Sublane info
            # 11 to 11 + alignment - 1: t_reqs
            jnp.where(
                is_real_prefill,
                existing_row[11:11 + alignment],
                jnp.zeros((alignment, ), dtype=jnp.int32),
            ),
            # 11 + alignment to 11 + 2 * alignment - 1: is_first_tok
            jnp.where(
                is_real_prefill,
                existing_row[11 + alignment:11 + 2 * alignment],
                jnp.zeros((alignment, ), dtype=jnp.int32),
            ),
            # 11 + 2 * alignment to 11 + 3 * alignment - 1: is_last_tok
            jnp.where(
                is_real_prefill,
                existing_row[11 + 2 * alignment:11 + 3 * alignment],
                jnp.zeros((alignment, ), dtype=jnp.int32),
            ),
        ]
        decode_metadata = jnp.concatenate(decode_parts)

        new_row = jnp.concatenate([
            jnp.where(is_real_prefill, existing_row[:4],
                      jnp.zeros((4, ), dtype=jnp.int32)),
            decode_metadata,
        ])

        table_ref[b, :] = new_row
        return None

    lax.fori_loop(0, num_decode_batches, decode_loop, None)

    return None


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def compute_schedule_table_v3(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jnp.ndarray]:
    """Compute number of iterations in grid and work each iteration will do.

  This is a Pallas kernel implementation of schedule table generation.
  At high level
    - each iteration of grid is either prefill and or decode
    - grid moves in size of bt decode tokens (sequence) backwards starting from
    boundary
    - and prefill moves in chunk sized tokens forward from boundary to end
  Input characteristics
    - each sequence start and end may not be sublane aligned,
    boundary between decode and prefill maybe in shared sublane
    - sequence may not divide chunk size

  hardware req
    - offset for each block has to be sublane aligned

  So for this we have transition blocks at boundaries between prefill sequences,
  including first one with decode, token by token math is done here instead of
  chunk wise.
  """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (max_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)
    padded_safe_max_blocks = max(((safe_max_blocks + 7) // 8) * 8, 128)

    table_width = 11 + 3 * alignment

    query_metadata_spec = pl.BlockSpec(block_shape=query_start_loc.shape,
                                       memory_space=pltpu.SMEM)

    # BlockSpecs for the padded scalar inputs (vectors of size 8)
    scalar_spec = pl.BlockSpec(block_shape=(8, ), memory_space=pltpu.SMEM)

    table_spec = pl.BlockSpec(block_shape=(padded_safe_max_blocks,
                                           table_width))

    # Create padded arrays for scalars to avoid
    num_seqs_arr = jnp.zeros((8, ), dtype=jnp.int32).at[0].set(num_valid_seqs)
    decode_tokens_arr = jnp.zeros((8, ),
                                  dtype=jnp.int32).at[0].set(decode_tokens)
    chunk_size_arr = jnp.zeros((8, ), dtype=jnp.int32).at[0].set(chunk_size)
    bt_arr = jnp.zeros((8, ), dtype=jnp.int32).at[0].set(BT)
    total_num_seqs_arr = jnp.zeros((8, ), dtype=jnp.int32).at[0].set(num_seqs)

    table = pl.pallas_call(
        kernel=functools.partial(kernel, alignment=alignment),
        out_shape=[
            jax.ShapeDtypeStruct((padded_safe_max_blocks, table_width),
                                 jnp.int32)
        ],
        in_specs=[
            query_metadata_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
        ],
        grid=(1, ),
        out_specs=[table_spec],
    )(
        query_start_loc,
        num_seqs_arr,
        decode_tokens_arr,
        chunk_size_arr,
        bt_arr,
        total_num_seqs_arr,
    )[0]

    seq_end = query_start_loc[1:]
    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )
    is_decode_boundary = prev_seq_end == decode_tokens
    is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)
    next_aligned_start = (seq_end // alignment) * alignment
    is_last_seq = jnp.arange(num_seqs) == num_seqs - 1
    needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                        (~is_swallowed))
    needs_start_transition = ((prev_seq_end % alignment != 0) & (~is_swallowed)
                              & is_decode_boundary)
    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)
    num_regular_blocks = (effective_end - effective_start + chunk_size -
                          1) // chunk_size
    total_blocks_per_seq = (num_regular_blocks +
                            needs_transition.astype(jnp.int32) +
                            needs_start_transition.astype(jnp.int32))
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)
    is_pure_decode = seq_end <= decode_tokens
    total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)
    total_prefill_blocks = jnp.sum(total_blocks_per_seq)

    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return table[:safe_max_blocks, :], total_blocks
