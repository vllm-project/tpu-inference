import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention, ref_ragged_paged_attention)


def test_per_token_quantization_and_kernel():
    print("Starting Per-Token Quantization Test (BF16 -> FP8)...")

    # 1. Configuration
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
    cu_q_lens = jnp.array([0, 16, 32],
                          dtype=jnp.int32)  # 16 queries per sequence
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
    values = jax.random.normal(
        k3, (max_num_tokens, actual_num_kv_heads, actual_head_dim),
        dtype=jnp.bfloat16)

    # 3. Initialize Caches
    # Calculate shapes using the utility function from your code
    kv_cache_shape = get_kv_cache_shape(total_num_pages, page_size,
                                        actual_num_kv_heads, actual_head_dim,
                                        kv_cache_dtype)
    kv_cache = jnp.zeros(kv_cache_shape, dtype=kv_cache_dtype)

    keys_q, values_q = utils.quantize_kv(keys, values, jnp.float8_e4m3fn, 1.0,
                                         1.0)

    # 4. Run Reference Per-Token Attention
    # This simulates the logic where BF16 K/V come in, get quantized to FP4,
    # stored, and then used for attention with scales.
    out_q, updated_kv = ref_ragged_paged_attention(queries=queries,
                                                   keys=keys_q,
                                                   values=values_q,
                                                   kv_cache=kv_cache,
                                                   kv_lens=kv_lens,
                                                   page_indices=page_indices,
                                                   cu_q_lens=cu_q_lens,
                                                   distribution=distribution,
                                                   sm_scale=1.0 /
                                                   (actual_head_dim**0.5))

    # 5. Verification
    print(f"Output shape: {out_q.shape}")
    print(f"KV Cache dtype: {updated_kv.dtype}")

    # Check for NaNs
    assert not jnp.isnan(out_q).any(), "NaNs detected in output!"

    # 6. Compare with "Unquantized" Ref to check magnitude
    # Note: We expect some divergence due to FP4 precision
    expected, expected_kv_cache = ref_ragged_paged_attention(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=jnp.zeros(get_kv_cache_shape(total_num_pages, page_size,
                                              actual_num_kv_heads,
                                              actual_head_dim, jnp.bfloat16),
                           dtype=jnp.bfloat16),
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=1.0 / (actual_head_dim**0.5))

    output, updated_kv_cache = ragged_paged_attention(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=jnp.zeros(get_kv_cache_shape(total_num_pages, page_size,
                                              actual_num_kv_heads,
                                              actual_head_dim, jnp.bfloat16),
                           dtype=jnp.bfloat16),
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=1.0 / (actual_head_dim**0.5))
    diff = jnp.abs(output - expected)
    print(
        f"Mean absolute difference (BF16 vs Quantized FP8): {jnp.mean(diff)}")
    print(f"Max absolute difference: {jnp.max(diff)}")

    dtype_bits = 8
    tols = {
        32: 0.15,
        16: 0.2,
        8: 0.2,
        4: 0.2,
    }
    tol = tols[dtype_bits]
    np.testing.assert_allclose(output, expected, atol=tol, rtol=tol)
    mask = ~jnp.isnan(expected_kv_cache)
    np.testing.assert_array_equal(updated_kv_cache[mask],
                                  expected_kv_cache[mask])
    np.testing.assert_equal(output.shape[-1], actual_head_dim)


if __name__ == "__main__":
    # Ensure we are visible to TPU if running in TPU environment
    test_per_token_quantization_and_kernel()
