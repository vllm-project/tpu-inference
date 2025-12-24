import jax
import jax.numpy as jnp

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_per_token import \
    get_kv_cache_shape

seed = 42
key = jax.random.PRNGKey(seed)

kv_cache_dtype = jnp.float4_e2m1fn

actual_head_dim = 128
actual_num_q_heads = 32
actual_num_kv_heads = 8
page_size = 128
total_num_pages = 1000

# Distribution: 0 decode, 0 chunked, 2 mixed (total 2 sequences)
distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
kv_lens = jnp.array([32, 48], dtype=jnp.int32)  # length of tokens in KV
cu_q_lens = jnp.array([0, 16, 32], dtype=jnp.int32)  # 16 queries per sequence
max_num_tokens = cu_q_lens[-1]

# Page indices: seq0 uses pages 0-1, seq1 uses pages 2-4
page_indices = jnp.array([0, 1, -1, -1, 2, 3, 4, -1], dtype=jnp.int32)

# 2. Generate BF16 Inputs
k1, k2, k3 = jax.random.split(key, 3)
queries = jax.random.normal(
    k1, (max_num_tokens, actual_num_q_heads, actual_head_dim),
    dtype=jnp.bfloat16)
keys = jax.random.normal(
    k2, (max_num_tokens, actual_num_kv_heads, actual_head_dim),
    dtype=jnp.bfloat16)
keys = keys.at[0].set(keys[0] * 10.0)
values = jax.random.normal(
    k3, (max_num_tokens, actual_num_kv_heads, actual_head_dim),
    dtype=jnp.bfloat16)
# values = values.at[1].set(values[1] * 100.0)
# 3. Initialize Caches
# Calculate shapes using the utility function from your code
kv_cache_shape_quantized = get_kv_cache_shape(total_num_pages, page_size,
                                              actual_num_kv_heads,
                                              actual_head_dim, kv_cache_dtype)
kv_cache_shape_unquantized = get_kv_cache_shape(total_num_pages, page_size,
                                                actual_num_kv_heads,
                                                actual_head_dim, jnp.bfloat16)
kv_cache_quantized = jnp.zeros(kv_cache_shape_quantized, dtype=kv_cache_dtype)

# Scale caches: [Total Pages, Page Size, Num KV Heads, 1]
k_scale_cache_expected = jnp.ones(
    (total_num_pages, page_size, actual_num_kv_heads, 1), dtype=jnp.float32)
v_scale_cache_expected = jnp.ones(
    (total_num_pages, page_size, actual_num_kv_heads, 1), dtype=jnp.float32)

keys_q, values_q = utils.quantize_kv(keys, values, kv_cache_dtype, 1.0, 1.0)

per_tensor_output_quantized_ref, per_tensor_kv_cache_quantized_ref = ref_ragged_paged_attention(
    queries=queries,
    keys=keys_q,
    values=values_q,
    kv_cache=kv_cache_quantized,
    kv_lens=kv_lens,
    page_indices=page_indices,
    cu_q_lens=cu_q_lens,
    distribution=distribution,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=1.0 / (actual_head_dim**0.5))

kv_cache_quantized = jnp.zeros(kv_cache_shape_quantized, dtype=kv_cache_dtype)
print(kv_cache_quantized.dtype)
per_tensor_output_quantized, per_tensor_kv_cache_quantized = ragged_paged_attention(
    queries=queries,
    keys=keys_q,
    values=values_q,
    kv_cache=kv_cache_quantized,
    kv_lens=kv_lens,
    page_indices=page_indices,
    cu_q_lens=cu_q_lens,
    distribution=distribution,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=1.0 / (actual_head_dim**0.5))

print(per_tensor_output_quantized_ref, per_tensor_output_quantized_ref.dtype)
print(per_tensor_output_quantized, per_tensor_output_quantized.dtype)
