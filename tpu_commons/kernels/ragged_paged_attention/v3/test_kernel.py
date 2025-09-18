
from tpu_commons.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention)
import jax 
import jax.numpy as jnp
import numpy as np


kv_dtype = jnp.bfloat16
q_dtype = jnp.bfloat16
head_dim = 128
page_size = 64
max_num_batched_tokens = 32
num_q_heads = 32
num_kv_heads = 8
num_kv_pages_per_block = 8 
num_queries_per_block = 64
vmem_limit_bytes=100 * 1024 * 1024

def gen_random(shape, dtype):
    x = jax.numpy.arange(np.prod(shape), dtype=dtype).reshape(shape) / 1000.0
    return x.astype(dtype)

q = gen_random((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)

page_indices = jnp.zeros((512,), dtype=jnp.int32)
page_indices = page_indices.at[0].set(1)
cu_q_lens = jnp.ones((33,), dtype=jnp.int32)
cu_q_lens = cu_q_lens.at[0].set(0)
cu_q_lens = cu_q_lens.at[1].set(6)
kv_lens = jnp.zeros((32,), dtype=jnp.int32)
kv_lens = kv_lens.at[0].set(6)
distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
kv_cache = jnp.zeros((16, 64, 8, 2, 128), dtype=kv_dtype)

print("kv_cache", kv_cache.shape)
print("query vector", q.shape, jnp.sum(q), q)
print("key vector",k.shape, jnp.sum(k), k)
print("value vector",v.shape, jnp.sum(v), v)
print("page_indices", page_indices.shape, page_indices)
print("cu_q_lens", cu_q_lens.shape, cu_q_lens)
print("kv_lens", kv_lens.shape, kv_lens)
print("distribution", distribution)

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

kwargs = {
    "sliding_window": None,
    "soft_cap": None,
    "q_scale": None,
    "k_scale": None,
    "v_scale": None,
}

expected, expected_kv_cache = ref_ragged_paged_attention(
            *args,
            **kwargs,
        )

print("expected", expected.ravel()[:10])
print("expected_kv_cache", jnp.sum(expected_kv_cache))

output, updated_kv_cache = ragged_paged_attention(
    *args,
    **kwargs,
    num_kv_pages_per_block=num_kv_pages_per_block,
    num_queries_per_block=num_queries_per_block,
    vmem_limit_bytes=vmem_limit_bytes,
)

output = output[: cu_q_lens[distribution[-1]]]
print("output", output.ravel()[:10])
print("updated_kv_cache",jnp.sum(updated_kv_cache))



############# INPUT #################

# kv_cache (16, 64, 8, 2, 128)
# query vector (32, 32, 128) 8.58522e+06 [[[0 0.000999451 0.0019989 ... 0.125 0.125977 0.126953]
#   [0.12793 0.128906 0.129883 ... 0.253906 0.253906 0.255859]
#   [0.255859 0.255859 0.257812 ... 0.380859 0.382812 0.384766]
#   ...
#   [131 131 131 ... 131 131 131]
#   [131 131 131 ... 131 131 131]
#   [131 131 131 ... 131 131 131]]]
# key vector (32, 8, 128) 536576 [[[0 0.000999451 0.0019989 ... 0.125 0.125977 0.126953]
#   [0.12793 0.128906 0.129883 ... 0.253906 0.253906 0.255859]
#   [0.255859 0.255859 0.257812 ... 0.380859 0.382812 0.384766]
#   ...
#   [0.640625 0.640625 0.640625 ... 0.765625 0.769531 0.769531]
#   [0.769531 0.769531 0.769531 ... 0.890625 0.894531 0.894531]
#   [0.894531 0.894531 0.894531 ... 1.02344 1.02344 1.02344]]
#   ...
#   [32.5 32.5 32.5 ... 32.5 32.5 32.5]
#   [32.5 32.5 32.5 ... 32.75 32.75 32.75]
#   [32.75 32.75 32.75 ... 32.75 32.75 32.75]]]
# value vector (32, 8, 128) 536576 [[[0 0.000999451 0.0019989 ... 0.125 0.125977 0.126953]
#   [0.12793 0.128906 0.129883 ... 0.253906 0.253906 0.255859]
#   [0.255859 0.255859 0.257812 ... 0.380859 0.382812 0.384766]
#   ...
#   [0.640625 0.640625 0.640625 ... 0.765625 0.769531 0.769531]
#   [0.769531 0.769531 0.769531 ... 0.890625 0.894531 0.894531]
#   [0.894531 0.894531 0.894531 ... 1.02344 1.02344 1.02344]]
#   ...
#   [32.5 32.5 32.5 ... 32.5 32.5 32.5]
#   [32.5 32.5 32.5 ... 32.75 32.75 32.75]
#   [32.75 32.75 32.75 ... 32.75 32.75 32.75]]]
# page_indices (512,) [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# cu_q_lens (33,) [0 6 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
# kv_lens (32,) [6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# distribution [0 0 1]

############### OUTPUT #################

# G3 
# output [0 0.000999451 0.0019989 0.00300598 0.0039978 0.00500488 0.00601196
#  0.00698853 0.00799561 0.00897217]
# updated_kv_cache 37632

# Cloud v6e

# expected [0 0.000999451 0.0019989 0.00300598 0.0039978 0.00500488 0.00601196
#  0.00698853 0.00799561 0.00897217]
# expected_kv_cache 37632

# output [0 0.000999451 0.0019989 0.00299072 0.0039978 0.00500488 0.00598145
#  0.00698853 0.00799561 0.00897217]
# updated_kv_cache 0
