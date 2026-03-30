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
import jax.numpy as jnp

from tpu_inference.kernels.experimental.batched_rpa import \
    schedule as schedule_lib


def align_to(a, b):
    """Returns 'a' aligned to 'b'."""
    return pl.cdiv(a, b) * b


def broadcast_minor(src, shape):
    """Broadcasts 'src' to 'shape' in the minor dimension."""
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])],
                           axis=-1)[..., :shape[-1]]


def get_dtype_bitwidth(dtype):
    """Returns the bitwidth of a JAX dtype."""
    return jax._src.dtypes.itemsize_bits(dtype)


def strided_load(ref, start_row, num_rows, step, *, dtype=None):
    """Loads data from HBM with strided access, handling 128-lane alignment."""
    _, row_width = ref.shape
    num_sub_lanes = row_width // 128
    ref_flat = ref.reshape(-1, 128)

    # scale indices to match flattened arraw.
    v_start = start_row * num_sub_lanes
    v_num = num_rows * num_sub_lanes
    v_step = step * num_sub_lanes

    # Gather the chunks into the original head dimension.
    chunks = [
        ref_flat[pl.ds(v_start + i, v_num // v_step, v_step)]
        for i in range(num_sub_lanes)
    ]
    vec = jnp.concat(chunks, axis=1)

    return pltpu.bitcast(vec, dtype) if dtype is not None else vec


def strided_store(ref, start, sz, step, val):
    """Stores data to HBM with strided access, handling 128-lane alignment."""
    assert schedule_lib.get_dtype_packing(ref.dtype) == 1
    assert ref.dtype == val.dtype
    assert ref.shape == val.shape
    assert len(ref.shape) == 2
    r, l = ref.shape  # noqa
    assert l % 128 == 0
    folds = l // 128
    ref = ref.reshape(r * folds, 128)
    start *= folds
    sz *= folds
    step *= folds
    assert sz % step == 0
    for i in range(folds):
        ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128:(i + 1) * 128]


# If we want to convert 32-bits into 32//N number of N-bits value, naive
# approach would be to perform 32//N number of 32-bits to N-bits conversion.
# However, we can reduce number of instructions by utilizing binary tree.
# 0: [32]
# 1: [16, 16]
# ...
# log2(32//N): [N, N, ... N]
def convert_to_target_bitwidth(val, target_bitwidth: int, kv_dtype: jnp.dtype):
    """Converts a value to a target bitwidth."""
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
        k = pltpu.bitcast(left, kv_dtype)
        v = pltpu.bitcast(right, kv_dtype)
        return [(k, v)]
    else:
        left_out = convert_to_target_bitwidth(left,
                                              target_bitwidth=target_bitwidth,
                                              kv_dtype=kv_dtype)
        right_out = convert_to_target_bitwidth(right,
                                               target_bitwidth=target_bitwidth,
                                               kv_dtype=kv_dtype)
        return left_out + right_out
