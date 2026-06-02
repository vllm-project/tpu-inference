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
"""Utility functions for the KV cache fusion in MLA."""

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu


def unsigned_mod(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bitmask instead of modulo for efficiency.
        return a & (b - 1)
    return a % b


def unsigned_cdiv(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bit shift instead of division for efficiency.
        return (a + b - 1) >> exponent
    return (a + b - 1) // b


def unsigned_floor_div(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bit shift instead of division for efficiency.
        return a >> exponent
    return a // b


def unsigned_align_to(a, b):
    exponent = int(math.log2(b))
    if b == int(math.pow(2, exponent)):
        # Use bitmask instead of division and multiply for efficiency.
        return (a + b - 1) & (-int(b))

    return unsigned_cdiv(a, b) * b


# Need to explicitly multiply with 128 to avoid TPU compile error. (async_copy)
def align_to(a, b):
    return ((a + b - 1) // b) * b


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def pack_new_kv(bkvc_vmem_ref, bkvpe_vmem_ref, offset, update_sz, q_end,
                kv_len, bkv_sz):
    _, kv_packing, lkv_dim = bkvc_vmem_ref.shape
    _, _, r_dim = bkvpe_vmem_ref.shape

    num_sublanes_for_kv_packing = kv_packing // get_dtype_packing(
        bkvc_vmem_ref.dtype)

    update_kv_packing_iters = unsigned_cdiv(
        unsigned_mod(offset, kv_packing) + update_sz, kv_packing)
    kv_packing_offset = unsigned_mod(offset, kv_packing)
    new_kv_len_start = q_end - kv_len + offset
    new_kv_packing_offset = unsigned_mod(new_kv_len_start, kv_packing)

    token_offset_in_bkv = unsigned_mod(offset, bkv_sz)
    kv_packing_idx = unsigned_floor_div(token_offset_in_bkv, kv_packing)

    # Compute the shift amount for each word in bits
    offset_diff = kv_packing_offset - new_kv_packing_offset
    roll_amount = unsigned_mod(offset_diff, kv_packing)

    bits_per_element = get_dtype_bitwidth(bkvc_vmem_ref.dtype)
    shift_bits = bits_per_element * roll_amount

    roll_amount = roll_amount.astype(jnp.int32)
    roll_shift_max = 32 // bits_per_element
    roll_shift_bits = unsigned_mod(roll_amount,
                                   roll_shift_max) * bits_per_element
    roll_rolling_amount = unsigned_floor_div(roll_amount, roll_shift_max)

    shift_bits = shift_bits.astype(jnp.uint32)
    roll_shift_bits = roll_shift_bits.astype(jnp.uint32)

    # Calculate the starting index in the KV buffer corresponding to the new KV
    # to fetch the data from. This index accounts for the potential offset
    # caused by the shift_amount.
    # (-offset_diff) // kv_packing will be:
    #   0 if new_kv_packing_offset <= kv_packing_offset
    #  -1 if new_kv_packing_offset > kv_packing_offset.
    kv_packing_idx_new = unsigned_cdiv(token_offset_in_bkv, kv_packing) + (
        (-offset_diff) // kv_packing)
    curr_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new, :, :]
    curr_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new, :, :]
    next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
    next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]

    def merge_loop_body(i, vals):
        (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        ) = vals
        if num_sublanes_for_kv_packing == 1:
            curr_kvc_reg_u32 = pltpu.bitcast(curr_kvc_reg, jnp.uint32)
            curr_kpe_reg_u32 = pltpu.bitcast(curr_kpe_reg, jnp.uint32)
            next_kvc_reg_u32 = pltpu.bitcast(next_kvc_reg, jnp.uint32)
            next_kpe_reg_u32 = pltpu.bitcast(next_kpe_reg, jnp.uint32)

            shifted_kvc_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kvc_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kvc_reg_u32, shift_bits),
            )
            shifted_kpe_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kpe_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kpe_reg_u32, shift_bits),
            )

            # If shift_bits is 0, we should use the current word. Otherwise,
            # shifting by 32 bits would result in shifted_*_u32 becoming
            # next_*_reg_u32, which is incorrect.
            rotated_kvc_u32 = lax.select(shift_bits == 0, curr_kvc_reg_u32,
                                         shifted_kvc_u32)
            rotated_kpe_u32 = lax.select(shift_bits == 0, curr_kpe_reg_u32,
                                         shifted_kpe_u32)

            rolled_kvc = pltpu.bitcast(rotated_kvc_u32, next_kvc_reg.dtype)
            rolled_kpe = pltpu.bitcast(rotated_kpe_u32, next_kpe_reg.dtype)
        else:
            kvc_cur_cond = lax.broadcasted_iota(dtype=jnp.int32,
                                                shape=[kv_packing, lkv_dim],
                                                dimension=0) < roll_amount
            dtype = curr_kvc_reg.dtype

            def shift_roll(reg, roll_rolling_amount, roll_shift_bits):
                reg_u32 = pltpu.bitcast(reg, jnp.uint32)
                reg_u32 = pltpu.roll(reg_u32,
                                     shift=roll_rolling_amount,
                                     axis=0)
                reg_u32_next = pltpu.roll(reg_u32, shift=1, axis=0)
                result = lax.bitwise_or(
                    lax.shift_left(reg_u32, roll_shift_bits),
                    lax.shift_right_logical(reg_u32_next,
                                            32 - roll_shift_bits),
                )
                return pltpu.bitcast(result, dtype)

            rolled_kvc = lax.select(
                kvc_cur_cond,
                shift_roll(curr_kvc_reg, roll_rolling_amount, roll_shift_bits),
                shift_roll(next_kvc_reg, roll_rolling_amount, roll_shift_bits),
            )
            kpe_cur_cond = lax.broadcasted_iota(dtype=jnp.int32,
                                                shape=[kv_packing, r_dim],
                                                dimension=0) < roll_amount
            rolled_kpe = lax.select(
                kpe_cur_cond,
                shift_roll(curr_kpe_reg, roll_rolling_amount, roll_shift_bits),
                shift_roll(next_kpe_reg, roll_rolling_amount, roll_shift_bits),
            )
            rolled_kvc = lax.select(roll_amount == 0, curr_kvc_reg, rolled_kvc)
            rolled_kpe = lax.select(roll_amount == 0, curr_kpe_reg, rolled_kpe)

        offset_in_word = i * kv_packing + lax.broadcasted_iota(
            dtype=jnp.int32, shape=[kv_packing, lkv_dim], dimension=0)
        kvc_mask = jnp.logical_and(
            offset_in_word >= kv_packing_offset,
            offset_in_word < kv_packing_offset + update_sz,
        )
        updated_kvc_reg = lax.select(
            kvc_mask,
            rolled_kvc,
            bkvc_vmem_ref[kv_packing_idx, :, :],
        )
        offset_in_word_pe = i * kv_packing + lax.broadcasted_iota(
            dtype=jnp.int32, shape=[kv_packing, r_dim], dimension=0)
        kpe_mask = jnp.logical_and(
            offset_in_word_pe >= kv_packing_offset,
            offset_in_word_pe < kv_packing_offset + update_sz,
        )
        updated_kpe_reg = lax.select(
            kpe_mask,
            rolled_kpe,
            bkvpe_vmem_ref[kv_packing_idx, :, :],
        )

        # Store back the merged word
        bkvc_vmem_ref[kv_packing_idx, :, :] = updated_kvc_reg
        bkvpe_vmem_ref[kv_packing_idx, :, :] = updated_kpe_reg

        # Move to the next word.
        kv_packing_idx += 1
        kv_packing_idx_new += 1
        curr_kvc_reg = next_kvc_reg
        curr_kpe_reg = next_kpe_reg
        next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
        next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]
        return (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        )

    lax.fori_loop(
        0,
        update_kv_packing_iters,
        merge_loop_body,
        (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        ),
    )


def pack_new_kv_reference(bkvc_vmem_ref, bkvpe_vmem_ref, offset, update_sz,
                          q_end, kv_len, bkv_sz):
    _, kv_packing, lkv_dim = bkvc_vmem_ref.shape
    _, _, r_dim = bkvpe_vmem_ref.shape

    update_kv_packing_iters = unsigned_cdiv(
        unsigned_mod(offset, kv_packing) + update_sz, kv_packing)
    kv_packing_offset = unsigned_mod(offset, kv_packing)
    new_kv_len_start = q_end - kv_len + offset
    new_kv_packing_offset = unsigned_mod(new_kv_len_start, kv_packing)

    token_offset_in_bkv = unsigned_mod(offset, bkv_sz)
    kv_packing_idx = unsigned_floor_div(token_offset_in_bkv, kv_packing)

    # Compute the shift amount for each word in bits
    offset_diff = kv_packing_offset - new_kv_packing_offset
    roll_amount = unsigned_mod(offset_diff, kv_packing)
    roll_amount = roll_amount.astype(jnp.int32)

    # Calculate the starting index in the KV buffer corresponding to the new KV
    # to fetch the data from. This index accounts for the potential offset
    # caused by the shift_amount.
    # (-offset_diff) // kv_packing will be:
    #   0 if new_kv_packing_offset <= kv_packing_offset
    #  -1 if new_kv_packing_offset > kv_packing_offset.
    kv_packing_idx_new = unsigned_cdiv(token_offset_in_bkv, kv_packing) + (
        (-offset_diff) // kv_packing)
    curr_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new, :, :]
    curr_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new, :, :]
    next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
    next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]

    def merge_loop_body(i, vals):
        (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        ) = vals
        kvc_cur_cond = lax.broadcasted_iota(dtype=jnp.int32,
                                            shape=[kv_packing, lkv_dim],
                                            dimension=0) < roll_amount
        dtype = curr_kvc_reg.dtype
        rolled_kvc = lax.select(
            kvc_cur_cond,
            pltpu.roll(curr_kvc_reg.astype(jnp.float32),
                       shift=roll_amount,
                       axis=0).astype(dtype),
            pltpu.roll(next_kvc_reg.astype(jnp.float32),
                       shift=roll_amount,
                       axis=0).astype(dtype),
        )
        kpe_cur_cond = lax.broadcasted_iota(dtype=jnp.int32,
                                            shape=[kv_packing, r_dim],
                                            dimension=0) < roll_amount
        rolled_kpe = lax.select(
            kpe_cur_cond,
            pltpu.roll(curr_kpe_reg.astype(jnp.float32),
                       shift=roll_amount,
                       axis=0).astype(dtype),
            pltpu.roll(next_kpe_reg.astype(jnp.float32),
                       shift=roll_amount,
                       axis=0).astype(dtype),
        )
        rolled_kvc = lax.select(roll_amount == 0, curr_kvc_reg, rolled_kvc)
        rolled_kpe = lax.select(roll_amount == 0, curr_kpe_reg, rolled_kpe)

        offset_in_word = i * kv_packing + lax.broadcasted_iota(
            dtype=jnp.int32, shape=[kv_packing, lkv_dim], dimension=0)
        kvc_mask = jnp.logical_and(
            offset_in_word >= kv_packing_offset,
            offset_in_word < kv_packing_offset + update_sz,
        )
        updated_kvc_reg = lax.select(
            kvc_mask,
            rolled_kvc,
            bkvc_vmem_ref[kv_packing_idx, :, :],
        )
        offset_in_word_pe = i * kv_packing + lax.broadcasted_iota(
            dtype=jnp.int32, shape=[kv_packing, r_dim], dimension=0)
        kpe_mask = jnp.logical_and(
            offset_in_word_pe >= kv_packing_offset,
            offset_in_word_pe < kv_packing_offset + update_sz,
        )
        updated_kpe_reg = lax.select(
            kpe_mask,
            rolled_kpe,
            bkvpe_vmem_ref[kv_packing_idx, :, :],
        )

        # Store back the merged word
        bkvc_vmem_ref[kv_packing_idx, :, :] = updated_kvc_reg
        bkvpe_vmem_ref[kv_packing_idx, :, :] = updated_kpe_reg

        # Move to the next word.
        kv_packing_idx += 1
        kv_packing_idx_new += 1
        curr_kvc_reg = next_kvc_reg
        curr_kpe_reg = next_kpe_reg
        next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
        next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]
        return (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        )

    lax.fori_loop(
        0,
        update_kv_packing_iters,
        merge_loop_body,
        (
            kv_packing_idx,
            kv_packing_idx_new,
            curr_kvc_reg,
            curr_kpe_reg,
            next_kvc_reg,
            next_kpe_reg,
        ),
    )
