import jax
import jax.numpy as jnp

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ref_ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_per_token import (
    get_kv_cache_shape, ref_ragged_paged_attention_per_token)

seed = 42
key = jax.random.PRNGKey(seed)

kv_cache_dtype = jnp.float8_e4m3fn

actual_head_dim = 128
actual_num_q_heads = 32
actual_num_kv_heads = 8
page_size = 256
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
values = values.at[1].set(values[1] * 100.0)
# 3. Initialize Caches
# Calculate shapes using the utility function from your code
kv_cache_shape_quantized = get_kv_cache_shape(total_num_pages, page_size,
                                              actual_num_kv_heads,
                                              actual_head_dim, kv_cache_dtype)
kv_cache_shape_unquantized = get_kv_cache_shape(total_num_pages, page_size,
                                                actual_num_kv_heads,
                                                actual_head_dim, jnp.bfloat16)
kv_cache_quantized = jnp.zeros(kv_cache_shape_quantized, dtype=kv_cache_dtype)
kv_cache_unquantized = jnp.zeros(kv_cache_shape_unquantized,
                                 dtype=jnp.bfloat16)

# Scale caches: [Total Pages, Page Size, Num KV Heads, 1]
k_scale_cache_expected = jnp.ones(
    (total_num_pages, page_size, actual_num_kv_heads, 1), dtype=jnp.float32)
v_scale_cache_expected = jnp.ones(
    (total_num_pages, page_size, actual_num_kv_heads, 1), dtype=jnp.float32)

# per_token_output, per_token_kv_cache, expected_k_scales, expected_v_scales = ref_ragged_paged_attention_per_token(
#     queries=queries,
#     keys=keys,
#     values=values,
#     kv_cache=kv_cache_quantized,
#     kv_lens=kv_lens,
#     page_indices=page_indices,
#     cu_q_lens=cu_q_lens,
#     distribution=distribution,
#     k_scale_cache=k_scale_cache_expected,
#     v_scale_cache=v_scale_cache_expected,
#     sm_scale=1.0 / (actual_head_dim**0.5))

# k_scale_cache_expected = jnp.ones(
#     (total_num_pages, page_size, actual_num_kv_heads, 1),
#     dtype=jnp.float32)
# v_scale_cache_expected = jnp.ones(
#     (total_num_pages, page_size, actual_num_kv_heads, 1),
#     dtype=jnp.float32)

keys_q, values_q = utils.quantize_kv(keys, values, jnp.float8_e4m3fn, 1.0, 1.0)

per_tensor_output_quantized, per_tensor_kv_cache_quantized = ref_ragged_paged_attention(
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

per_token_output_quantized, per_token_kv_cache_quantized, _, _ = ref_ragged_paged_attention_per_token(
    queries=queries,
    keys=keys,
    values=values,
    kv_cache=kv_cache_quantized,
    kv_lens=kv_lens,
    page_indices=page_indices,
    cu_q_lens=cu_q_lens,
    distribution=distribution,
    k_scale_cache=k_scale_cache_expected,
    v_scale_cache=v_scale_cache_expected,
    sm_scale=1.0 / (actual_head_dim**0.5))

per_tensor_output_unquantized, per_tensor_kv_cache_unquantized = ref_ragged_paged_attention(
    queries=queries,
    keys=keys,
    values=values,
    kv_cache=kv_cache_unquantized,
    kv_lens=kv_lens,
    page_indices=page_indices,
    cu_q_lens=cu_q_lens,
    distribution=distribution,
    sm_scale=1.0 / (actual_head_dim**0.5))


def compare_arrays(ref: jax.Array, actual: jax.Array, name: str = "Tensor"):
    """
    Compares two JAX arrays and prints detailed error statistics.
    Useful for debugging quantization drift.
    """
    # Ensure they are on CPU for printing/formatting if they are small,
    # or keep on GPU for calculation if large.

    diff = jnp.abs(ref - actual)
    max_abs_err = jnp.max(diff)
    mean_abs_err = jnp.mean(diff)

    # Safe Relative Error: Avoid div-by-zero by clamping denominator
    # For BF16/FP16, 1e-5 is a standard epsilon
    denominator = jnp.maximum(jnp.abs(ref), 1e-5)
    rel_diff = diff / denominator
    max_rel_err = jnp.max(rel_diff)
    mean_rel_err = jnp.mean(rel_diff)

    # Find index of max error
    flat_idx = jnp.argmax(diff)
    # Unravel index to get multidimensional coordinate
    idx_coords = jnp.unravel_index(flat_idx, ref.shape)

    print(f"--- Comparison: {name} ---")
    print(f"Shape        : {ref.shape}")
    print(f"Max Abs Err  : {max_abs_err}")
    print(f"Mean Abs Err : {mean_abs_err}")
    print(f"Max Rel Err  : {max_rel_err}")
    print(f"Mean Rel Err : {mean_rel_err}")
    print("-" * 30)

    return max_abs_err, max_rel_err


print(per_tensor_output_quantized)
print(per_tensor_output_unquantized)

# check the diff between the two
compare_arrays(per_tensor_output_quantized,
               per_tensor_output_unquantized,
               name="Output")
compare_arrays(per_token_output_quantized,
               per_tensor_output_unquantized,
               name="Output")
