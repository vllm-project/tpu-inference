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

import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ref_ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.per_token_scale_kernel import \
    ref_ragged_paged_attention_per_token
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common.quantization import quantize_kv

seq_lens = [(192, 328), (128, 180), (64, 255)]
num_heads = (32, 8)
head_dim = 128
page_size = 16
num_pages = 1000
max_num_batched_tokens = 4096
max_num_seq = 512
q_dtype = jnp.bfloat16
kv_dtype = jnp.float4_e2m1fn

rng = np.random.default_rng(1234)


def gen_random(shape, dtype):
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)


cu_q_lens = [0]
kv_lens = []
for q_len, kv_len in seq_lens:
    assert q_len <= kv_len
    cu_q_lens.append(cu_q_lens[-1] + q_len)
    kv_lens.append(kv_len)

max_num_batched_tokens = max(align_to(cu_q_lens[-1], 128),
                             max_num_batched_tokens)
max_num_seq = max(align_to(len(seq_lens), 8), max_num_seq)
max_kv_len = max(kv_lens)
pages_per_seq = cdiv(max_kv_len, page_size)
num_q_heads, num_kv_heads = num_heads

q = gen_random((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), jnp.float16)
v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), jnp.bfloat16)

page_cnt = 0
page_indices_list = []
kv_pages_list = []
kv_packing = get_dtype_packing(kv_dtype)
padded_head_dim = align_to(head_dim, 128)
num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
for kv_len in kv_lens:
    kv = gen_random((
        kv_len,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        padded_head_dim,
    ), kv_dtype)
    kv = jnp.pad(
        kv,
        (
            (
                0,
                cdiv(kv_len, page_size) * page_size - kv_len,
            ),
            (0, 0),
            (0, 0),
            (0, 0),
        ),
        constant_values=0,
    ).reshape(
        -1,
        page_size,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        padded_head_dim,
    )
    indices = page_cnt + jnp.arange(kv.shape[0], dtype=jnp.int32)
    indices = jnp.pad(
        indices,
        ((0, pages_per_seq - indices.shape[0]), ),
        constant_values=0,
    )
    page_indices_list.append(indices)
    page_cnt += kv.shape[0]
    kv_pages_list.append(kv)

kv_cache = jnp.concatenate(kv_pages_list, axis=0)
kv_cache = jnp.pad(
    kv_cache,
    ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0), (0, 0), (0, 0)),
    constant_values=0,
)
page_indices = jnp.stack(page_indices_list, axis=0)
page_indices = jnp.pad(
    page_indices,
    ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
    constant_values=0,
)
page_indices = page_indices.reshape(-1)

cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

k_scale = gen_random((1000, 16, 8, 1), jnp.float32)
v_scale = gen_random((1000, 16, 8, 1), jnp.float32)

args = (
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    k_scale,
    v_scale,
)

kwargs = {
    # "sliding_window": sliding_window,
    # "soft_cap": soft_cap,
    # "q_scale": q_scale,
    # "k_scale": k_scale,
    # "v_scale": v_scale,
}

expected, expected_kv_cache, _, _ = ref_ragged_paged_attention_per_token(
    *args,
    **kwargs,
)

print(expected)
print("FIRST IS OVER \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

args = (
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
)

print("NEXT")

k, v = quantize_kv(jnp.float4_e2m1fn, k, v, 1.0, 1.0)

args = (
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
)

expected, expected_kv_cache = ref_ragged_paged_attention(
    *args,
    **kwargs,
)

print(expected)

print("LAST")
k = k.astype(jnp.bfloat16)
v = v.astype(jnp.bfloat16)
expected, expected_kv_cache = ref_ragged_paged_attention(
    *args,
    **kwargs,
)

print(expected)
# output, updated_kv_cache = ragged_paged_attention(
#     *args,
#     **kwargs,
# )
# output = output[:cu_q_lens[distribution[-1]]]

# dtype_bits = dtypes.bit_width(jnp.dtype(kv_dtype))
# tols = {
#     32: 0.15,
#     16: 0.2,
#     8: 0.2,
#     4: 0.2,
# }
# tol = tols[dtype_bits]
# np.testing.assert_allclose(output, expected, atol=tol, rtol=tol)
