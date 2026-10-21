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
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# NOTE: When performing strided ldst, using pltpu.bitcast on vreg data should
# be avoided as it can trigger unintended relayout.


def load_large_to_compact(vmem_ref,
                          dst_dtype: jnp.dtype | None = None) -> jax.Array:
    assert vmem_ref.ndim == 2

    row_size = vmem_ref.shape[0]
    src_dtype = vmem_ref.dtype
    should_unpack = dst_dtype is not None and dst_dtype != src_dtype
    packing = 4 // src_dtype.itemsize

    unpacked_list = []
    for row_start in range(0, row_size, packing):
        row_end = row_start + packing
        if should_unpack:
            packed_row = row_start // packing
            u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
            packed = u32_vmem_ref[packed_row:packed_row + 1]

            for p in range(packing):
                unpacked = pltpu.unpack_elementwise(
                    packed,
                    index=p,
                    packed_dtype=src_dtype,
                    unpacked_dtype=dst_dtype,
                )
                unpacked_list.append(unpacked)
        else:
            unpacked_list.append(vmem_ref[row_start:row_end])

    return jnp.stack(unpacked_list, axis=0)


def store_compact_to_large(vmem_ref, vreg: jax.Array):
    row_size = vmem_ref.shape[0]
    src_dtype = vreg.dtype
    dst_dtype = vmem_ref.dtype
    should_pack = src_dtype != dst_dtype
    src_packing = 4 // src_dtype.itemsize
    dst_packing = 4 // dst_dtype.itemsize

    assert vreg.ndim == 3
    assert vreg.shape[-2] == src_packing
    assert vmem_ref.ndim == 2
    assert vmem_ref.shape[-1] == vreg.shape[-1]

    for row_start in range(0, row_size, dst_packing):
        row_end = row_start + dst_packing
        packed_row = row_start // dst_packing
        if should_pack:
            assert src_dtype.itemsize == 4
            assert dst_dtype.itemsize == 2

            unpacked_list = [vreg[i] for i in range(row_start, row_end)]
            packed = pltpu.pack_elementwise(unpacked_list,
                                            packed_dtype=dst_dtype)
            u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
            u32_vmem_ref[packed_row:packed_row + 1] = packed
        else:
            vmem_ref[row_start:row_end] = vreg[packed_row]
