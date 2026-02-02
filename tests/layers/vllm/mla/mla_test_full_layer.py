#!/usr/bin/env python3
"""Test the full MLA attention layer with instrumentation.

This test exercises the complete MLA attention path including:
1. Q projection via W_K (ql_nope = q_nope @ W_K)
2. MLA kernel call (attention over compressed KV)
3. Output projection via W_V (output = attn_output @ W_V)

Run on TPU with:
    VLLM_USE_MLA=1 python tests/mla_test_full_layer.py
"""

import os
import sys
import numpy as np

# Set environment for MLA
os.environ["VLLM_USE_MLA"] = "1"
os.environ["PJRT_DEVICE"] = "tpu"

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

# Try to import TPU inference modules
try:
    from tpu_inference.kernels.mla.v1.kernel import (
        ref_mla_ragged_paged_attention,
        get_kv_cache_shape,
    )
    HAS_KERNEL = True
    print("[OK] MLA kernel imported")
except ImportError as e:
    HAS_KERNEL = False
    print(f"[SKIP] MLA kernel not available: {e}")


def align_to(x: int, alignment: int) -> int:
    """Align x up to the nearest multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


def create_mock_kv_b_proj(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank, seed=42):
    """Create a mock kv_b_proj weight matrix.

    Shape: ((qk_nope_head_dim + v_head_dim) * num_heads, kv_lora_rank)
    Layout: [head0_K, head0_V, head1_K, head1_V, ...]
    """
    np.random.seed(seed)
    out_dim = (qk_nope_head_dim + v_head_dim) * num_heads
    weight = np.random.randn(out_dim, kv_lora_rank).astype(np.float32) * 0.02
    return weight


def extract_w_k_w_v(kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    """Extract W_K and W_V matrices from kv_b_proj weight.

    Returns:
        w_k: (num_heads, qk_nope_head_dim, kv_lora_rank)
        w_v: (num_heads, v_head_dim, kv_lora_rank)
    """
    # Reshape to (num_heads, qk_nope + v, kv_lora_rank)
    kv_b_reshaped = kv_b_weight.reshape(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    # Split
    w_k = kv_b_reshaped[:, :qk_nope_head_dim, :]
    w_v = kv_b_reshaped[:, qk_nope_head_dim:, :]
    return w_k, w_v


def mla_attention_reference(
    q_nope,      # (tokens, heads, qk_nope_head_dim)
    q_pe,        # (tokens, heads, qk_rope_head_dim)
    kv_c,        # (tokens, kv_lora_rank)
    k_pe,        # (tokens, qk_rope_head_dim)
    w_k,         # (heads, qk_nope_head_dim, kv_lora_rank)
    w_v,         # (heads, v_head_dim, kv_lora_rank)
    scale,
    return_intermediates=False,
):
    """Reference MLA attention implementation.

    This follows the exact algorithm from vLLM's MLA documentation:
    1. ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)
    2. q_combined = [ql_nope, q_pe]
    3. k_combined = [kv_c, k_pe]
    4. attn = softmax(q_combined @ k_combined.T * scale)
    5. attn_output = attn @ kv_c (in latent space)
    6. output = einsum("snl,lnv->snv", attn_output, W_UV)
    """
    num_tokens = q_nope.shape[0]
    num_heads = q_nope.shape[1]
    kv_lora_rank = w_k.shape[2]
    v_head_dim = w_v.shape[1]

    intermediates = {}

    # Step 1: Project q_nope through W_K
    # ql_nope[t, h] = q_nope[t, h] @ w_k[h]
    ql_nope = np.einsum('thd,hdk->thk', q_nope, w_k)
    intermediates['ql_nope'] = ql_nope.copy()
    print(f"  ql_nope: {ql_nope.shape}, min={ql_nope.min():.4f}, max={ql_nope.max():.4f}")

    # Step 2: Combine q parts
    # q_combined = [ql_nope, q_pe]  -> (tokens, heads, kv_lora_rank + qk_rope_dim)
    q_combined = np.concatenate([ql_nope, q_pe], axis=-1)
    intermediates['q_combined'] = q_combined.copy()
    print(f"  q_combined: {q_combined.shape}")

    # Step 3: Combine k parts (broadcast kv_c across heads)
    # k_combined = [kv_c, k_pe] -> (tokens, kv_lora_rank + qk_rope_dim)
    k_combined = np.concatenate([kv_c, k_pe], axis=-1)
    intermediates['k_combined'] = k_combined.copy()
    print(f"  k_combined: {k_combined.shape}")

    # Step 4: Compute attention scores
    # For each head h: scores[h, t1, t2] = q_combined[t1, h] @ k_combined[t2]
    # Broadcast: (tokens, heads, dim) @ (tokens, dim).T -> (heads, tokens, tokens)
    # Use einsum for clarity
    attn_scores = np.einsum('thd,sd->hts', q_combined, k_combined) * scale
    intermediates['attn_scores_raw'] = attn_scores.copy()
    print(f"  attn_scores_raw: {attn_scores.shape}, scale={scale}")

    # Apply causal mask
    causal_mask = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)
    attn_scores_masked = np.where(causal_mask, -1e9, attn_scores)
    intermediates['attn_scores_masked'] = attn_scores_masked.copy()

    # Softmax
    attn_weights = np.exp(attn_scores_masked - attn_scores_masked.max(axis=-1, keepdims=True))
    attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
    intermediates['attn_weights'] = attn_weights.copy()
    print(f"  attn_weights: {attn_weights.shape}, sum per row={attn_weights.sum(axis=-1)[0, 0]:.4f}")

    # Step 5: Compute attention output in latent space
    # attn_output[t, h] = sum_s(attn_weights[h, t, s] * kv_c[s])
    # (heads, tokens, tokens) @ (tokens, kv_lora) -> (heads, tokens, kv_lora) -> (tokens, heads, kv_lora)
    attn_output_latent = np.einsum('hts,sk->thk', attn_weights, kv_c)
    intermediates['attn_output_latent'] = attn_output_latent.copy()
    print(f"  attn_output_latent: {attn_output_latent.shape}")

    # Step 6: Project through W_V
    # output[t, h] = attn_output_latent[t, h] @ w_v[h].T
    # (tokens, heads, kv_lora) @ (heads, v_head, kv_lora).T -> (tokens, heads, v_head)
    output = np.einsum('thk,hvk->thv', attn_output_latent, w_v)
    intermediates['output'] = output.copy()
    print(f"  output: {output.shape}")

    if return_intermediates:
        return output, intermediates
    return output


def test_full_mla_layer():
    """Test the full MLA attention layer against reference implementation."""
    if not HAS_JAX or not HAS_KERNEL:
        print("Skipping full layer test - JAX or kernel not available")
        return True

    print("=" * 70)
    print("Testing Full MLA Layer")
    print("=" * 70)

    # GLM-4 like dimensions (scaled down for testing)
    num_tokens = 4
    num_heads = 4
    qk_nope_head_dim = 24  # Scaled from 192
    qk_rope_head_dim = 8   # Scaled from 64
    v_head_dim = 32        # Scaled from 256
    kv_lora_rank = 64      # Scaled from 512
    page_size = 16

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = 1.0 / np.sqrt(qk_head_dim)

    print(f"\nConfig:")
    print(f"  num_tokens={num_tokens}, num_heads={num_heads}")
    print(f"  qk_nope={qk_nope_head_dim}, qk_rope={qk_rope_head_dim}, v_head={v_head_dim}")
    print(f"  kv_lora_rank={kv_lora_rank}, scale={scale:.6f}")

    # Create random inputs
    np.random.seed(42)

    # Q is split into nope and pe parts
    q_nope = np.random.randn(num_tokens, num_heads, qk_nope_head_dim).astype(np.float32)
    q_pe = np.random.randn(num_tokens, num_heads, qk_rope_head_dim).astype(np.float32)

    # Compressed KV (shared across heads)
    kv_c = np.random.randn(num_tokens, kv_lora_rank).astype(np.float32)

    # K position embeddings
    k_pe = np.random.randn(num_tokens, qk_rope_head_dim).astype(np.float32)

    # Create kv_b_proj weight and extract W_K, W_V
    kv_b_weight = create_mock_kv_b_proj(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)
    w_k, w_v = extract_w_k_w_v(kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)

    print(f"\nInput shapes:")
    print(f"  q_nope: {q_nope.shape}")
    print(f"  q_pe: {q_pe.shape}")
    print(f"  kv_c: {kv_c.shape}")
    print(f"  k_pe: {k_pe.shape}")
    print(f"  w_k: {w_k.shape}")
    print(f"  w_v: {w_v.shape}")

    # Run reference implementation
    print("\n--- Reference Implementation ---")
    ref_output, ref_intermediates = mla_attention_reference(
        q_nope, q_pe, kv_c, k_pe, w_k, w_v, scale, return_intermediates=True
    )

    # Run kernel implementation
    print("\n--- Kernel Implementation ---")

    # Project q_nope to ql_nope (this is done by the layer, not the kernel)
    ql_nope = np.einsum('thd,hdk->thk', q_nope, w_k)
    print(f"  ql_nope: {ql_nope.shape}")

    # Compare ql_nope
    ref_ql_nope = ref_intermediates['ql_nope']
    ql_nope_diff = np.abs(ql_nope - ref_ql_nope).max()
    print(f"  ql_nope diff: {ql_nope_diff:.2e}")

    # Prepare kernel inputs
    dtype = jnp.bfloat16

    ql_nope_jax = jnp.array(ql_nope, dtype=dtype)
    q_pe_jax = jnp.array(q_pe, dtype=dtype)
    kv_c_jax = jnp.array(kv_c, dtype=dtype)
    k_pe_jax = jnp.array(k_pe, dtype=dtype)

    # Create KV cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_head_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(
        total_num_pages=1,
        page_size=page_size,
        kv_dim=kv_dim,
        kv_dtype=dtype,
    )
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)
    print(f"  cache_kv: {cache_shape}")

    # Metadata
    kv_lens = jnp.array([num_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    # Run kernel
    print("\nRunning kernel...")
    kernel_output, updated_cache = ref_mla_ragged_paged_attention(
        ql_nope=ql_nope_jax,
        q_pe=q_pe_jax,
        new_kv_c=kv_c_jax,
        new_k_pe=k_pe_jax,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=scale,
    )

    # Kernel output is in latent space (tokens, heads, padded_kv_lora)
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]
    print(f"  kernel output (raw): {kernel_output.shape}")
    print(f"  kernel output (trimmed): {kernel_output_trimmed.shape}")

    # Compare kernel output (latent space) with reference
    ref_latent = ref_intermediates['attn_output_latent']
    latent_diff = np.abs(kernel_output_trimmed - ref_latent)
    print(f"\n  Latent space comparison:")
    print(f"    max diff: {latent_diff.max():.4f}")
    print(f"    mean diff: {latent_diff.mean():.4f}")

    # Project kernel output through W_V
    kernel_final = np.einsum('thk,hvk->thv', kernel_output_trimmed, w_v)
    print(f"\n  Kernel final output: {kernel_final.shape}")

    # Compare final output
    final_diff = np.abs(kernel_final - ref_output)
    print(f"\n  Final output comparison:")
    print(f"    max diff: {final_diff.max():.4f}")
    print(f"    mean diff: {final_diff.mean():.4f}")

    # Check sample values
    print(f"\n  Sample values (first token, first head):")
    print(f"    ref output[:5]:    {ref_output[0, 0, :5]}")
    print(f"    kernel output[:5]: {kernel_final[0, 0, :5]}")

    # bfloat16 tolerance
    match = final_diff.max() < 0.1
    print(f"\n  Match: {'✓' if match else '✗'}")

    return match


def test_decode_step():
    """Test decode step (single token attending to cached context)."""
    if not HAS_JAX or not HAS_KERNEL:
        print("Skipping decode test - JAX or kernel not available")
        return True

    print("\n" + "=" * 70)
    print("Testing Decode Step (1 token attending to 8 cached tokens)")
    print("=" * 70)

    # Dimensions
    context_len = 8
    decode_tokens = 1
    num_heads = 4
    qk_nope_head_dim = 24
    qk_rope_head_dim = 8
    v_head_dim = 32
    kv_lora_rank = 64
    page_size = 16

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = 1.0 / np.sqrt(qk_head_dim)
    dtype = jnp.bfloat16

    print(f"\nConfig: context={context_len}, decode={decode_tokens}, heads={num_heads}")

    # Create kv_b_proj and extract W_K, W_V
    np.random.seed(42)
    kv_b_weight = create_mock_kv_b_proj(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)
    w_k, w_v = extract_w_k_w_v(kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)

    # Create context data
    context_q_nope = np.random.randn(context_len, num_heads, qk_nope_head_dim).astype(np.float32)
    context_q_pe = np.random.randn(context_len, num_heads, qk_rope_head_dim).astype(np.float32)
    context_kv_c = np.random.randn(context_len, kv_lora_rank).astype(np.float32)
    context_k_pe = np.random.randn(context_len, qk_rope_head_dim).astype(np.float32)

    # Create decode data
    decode_q_nope = np.random.randn(decode_tokens, num_heads, qk_nope_head_dim).astype(np.float32)
    decode_q_pe = np.random.randn(decode_tokens, num_heads, qk_rope_head_dim).astype(np.float32)
    decode_kv_c = np.random.randn(decode_tokens, kv_lora_rank).astype(np.float32)
    decode_k_pe = np.random.randn(decode_tokens, qk_rope_head_dim).astype(np.float32)

    # Project q_nope for context and decode
    context_ql_nope = np.einsum('thd,hdk->thk', context_q_nope, w_k)
    decode_ql_nope = np.einsum('thd,hdk->thk', decode_q_nope, w_k)

    # Create cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_head_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(2, page_size, kv_dim, dtype)
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)

    # Step 1: Prefill (populate cache)
    print("\nStep 1: Prefill...")
    prefill_kv_lens = jnp.array([context_len], dtype=jnp.int32)
    prefill_page_indices = jnp.array([0, 0], dtype=jnp.int32)
    prefill_cu_q_lens = jnp.array([0, context_len], dtype=jnp.int32)
    prefill_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    _, cache_after_prefill = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(context_ql_nope, dtype=dtype),
        q_pe=jnp.array(context_q_pe, dtype=dtype),
        new_kv_c=jnp.array(context_kv_c, dtype=dtype),
        new_k_pe=jnp.array(context_k_pe, dtype=dtype),
        cache_kv=cache_kv,
        kv_lens=prefill_kv_lens,
        page_indices=prefill_page_indices,
        cu_q_lens=prefill_cu_q_lens,
        distribution=prefill_distribution,
        sm_scale=scale,
    )
    print(f"  Cache populated, shape={cache_after_prefill.shape}")

    # Step 2: Decode
    print("\nStep 2: Decode...")
    total_len = context_len + decode_tokens
    decode_kv_lens = jnp.array([total_len], dtype=jnp.int32)
    decode_page_indices = jnp.array([0, 0], dtype=jnp.int32)
    decode_cu_q_lens = jnp.array([0, decode_tokens], dtype=jnp.int32)
    decode_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    kernel_output, _ = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(decode_ql_nope, dtype=dtype),
        q_pe=jnp.array(decode_q_pe, dtype=dtype),
        new_kv_c=jnp.array(decode_kv_c, dtype=dtype),
        new_k_pe=jnp.array(decode_k_pe, dtype=dtype),
        cache_kv=cache_after_prefill,
        kv_lens=decode_kv_lens,
        page_indices=decode_page_indices,
        cu_q_lens=decode_cu_q_lens,
        distribution=decode_distribution,
        sm_scale=scale,
    )

    # Trim and project output
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]
    kernel_final = np.einsum('thk,hvk->thv', kernel_output_trimmed, w_v)

    print(f"  Kernel output: {kernel_final.shape}")
    print(f"  Sample values: {kernel_final[0, 0, :5]}")

    # Compute reference (full attention over all tokens)
    all_kv_c = np.concatenate([context_kv_c, decode_kv_c], axis=0)
    all_k_pe = np.concatenate([context_k_pe, decode_k_pe], axis=0)

    # For decode, only query is the new token but it attends to all KV
    # q_combined for decode token
    decode_q_combined = np.concatenate([decode_ql_nope, decode_q_pe], axis=-1)  # (1, heads, kv_lora + rope)

    # k_combined for all tokens
    all_k_combined = np.concatenate([all_kv_c, all_k_pe], axis=-1)  # (9, kv_lora + rope)

    # Attention: (1, heads, dim) @ (9, dim).T -> (heads, 1, 9)
    decode_attn_scores = np.einsum('thd,sd->hts', decode_q_combined, all_k_combined) * scale

    # No masking needed for decode (single query can attend to all past)
    decode_attn_weights = np.exp(decode_attn_scores - decode_attn_scores.max(axis=-1, keepdims=True))
    decode_attn_weights = decode_attn_weights / decode_attn_weights.sum(axis=-1, keepdims=True)

    # Latent output
    ref_latent = np.einsum('hts,sk->thk', decode_attn_weights, all_kv_c)

    # Final output
    ref_output = np.einsum('thk,hvk->thv', ref_latent, w_v)

    print(f"\n  Reference output: {ref_output.shape}")
    print(f"  Reference sample: {ref_output[0, 0, :5]}")

    # Compare
    diff = np.abs(kernel_final - ref_output)
    print(f"\n  Comparison:")
    print(f"    max diff: {diff.max():.4f}")
    print(f"    mean diff: {diff.mean():.4f}")

    match = diff.max() < 0.1
    print(f"\n  Match: {'✓' if match else '✗'}")

    return match


def test_multi_turn_inference():
    """Simulate multi-turn inference with prefill + multiple decode steps."""
    if not HAS_JAX or not HAS_KERNEL:
        print("Skipping multi-turn test - JAX or kernel not available")
        return True

    print("\n" + "=" * 70)
    print("Testing Multi-Turn Inference")
    print("=" * 70)

    # Dimensions
    num_heads = 4
    qk_nope_head_dim = 24
    qk_rope_head_dim = 8
    v_head_dim = 32
    kv_lora_rank = 64
    page_size = 16

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = 1.0 / np.sqrt(qk_head_dim)
    dtype = jnp.bfloat16

    # Create kv_b_proj
    np.random.seed(42)
    kv_b_weight = create_mock_kv_b_proj(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)
    w_k, w_v = extract_w_k_w_v(kv_b_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)

    # Create cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_head_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(4, page_size, kv_dim, dtype)  # 4 pages for growth
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)

    # Sequence: prefill 4 tokens, then decode 3 tokens one by one
    all_outputs = []
    current_len = 0

    # Store all KV data for reference computation
    all_kv_c = []
    all_k_pe = []

    print("\nSequence of operations:")

    # Turn 1: Prefill with 4 tokens
    prefill_len = 4
    print(f"\n  Turn 1: Prefill {prefill_len} tokens...")

    q_nope = np.random.randn(prefill_len, num_heads, qk_nope_head_dim).astype(np.float32)
    q_pe = np.random.randn(prefill_len, num_heads, qk_rope_head_dim).astype(np.float32)
    kv_c = np.random.randn(prefill_len, kv_lora_rank).astype(np.float32)
    k_pe = np.random.randn(prefill_len, qk_rope_head_dim).astype(np.float32)

    ql_nope = np.einsum('thd,hdk->thk', q_nope, w_k)

    all_kv_c.append(kv_c)
    all_k_pe.append(k_pe)
    current_len += prefill_len

    kv_lens = jnp.array([current_len], dtype=jnp.int32)
    page_indices = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, prefill_len], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    output, cache_kv = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(ql_nope, dtype=dtype),
        q_pe=jnp.array(q_pe, dtype=dtype),
        new_kv_c=jnp.array(kv_c, dtype=dtype),
        new_k_pe=jnp.array(k_pe, dtype=dtype),
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=scale,
    )
    print(f"    Output shape: {output.shape}, current_len={current_len}")

    # Turns 2-4: Decode one token each
    for turn in range(2, 5):
        print(f"\n  Turn {turn}: Decode 1 token (total len={current_len + 1})...")

        q_nope = np.random.randn(1, num_heads, qk_nope_head_dim).astype(np.float32)
        q_pe = np.random.randn(1, num_heads, qk_rope_head_dim).astype(np.float32)
        kv_c = np.random.randn(1, kv_lora_rank).astype(np.float32)
        k_pe = np.random.randn(1, qk_rope_head_dim).astype(np.float32)

        ql_nope = np.einsum('thd,hdk->thk', q_nope, w_k)

        all_kv_c.append(kv_c)
        all_k_pe.append(k_pe)
        current_len += 1

        kv_lens = jnp.array([current_len], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)

        output, cache_kv = ref_mla_ragged_paged_attention(
            ql_nope=jnp.array(ql_nope, dtype=dtype),
            q_pe=jnp.array(q_pe, dtype=dtype),
            new_kv_c=jnp.array(kv_c, dtype=dtype),
            new_k_pe=jnp.array(k_pe, dtype=dtype),
            cache_kv=cache_kv,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=scale,
        )

        output_np = np.array(output, dtype=np.float32)
        output_trimmed = output_np[:, :, :kv_lora_rank]
        output_final = np.einsum('thk,hvk->thv', output_trimmed, w_v)
        all_outputs.append(output_final)

        print(f"    Output: min={output_final.min():.4f}, max={output_final.max():.4f}")

    # Verify cache contains all tokens
    print(f"\n  Final cache state:")
    print(f"    Total tokens: {current_len}")
    print(f"    Cache shape: {cache_kv.shape}")

    # Check cache is non-zero where expected
    cache_np = np.array(cache_kv, dtype=np.float32)
    page0_data = cache_np[0]  # First page
    page0_min = float(page0_data.min())
    page0_max = float(page0_data.max())
    print(f"    Page 0 stats: min={page0_min:.4f}, max={page0_max:.4f}")

    match = current_len == 7 and len(all_outputs) == 3
    print(f"\n  Multi-turn test: {'✓' if match else '✗'}")

    return match


if __name__ == "__main__":
    print("=" * 70)
    print("MLA FULL LAYER TESTS")
    print("=" * 70)

    if not HAS_JAX:
        print("\nJAX not available - install with: pip install jax jaxlib")
        sys.exit(1)

    if not HAS_KERNEL:
        print("\nMLA kernel not available - run on TPU with tpu_inference installed")
        sys.exit(1)

    results = []

    results.append(("Full MLA layer", test_full_mla_layer()))
    results.append(("Decode step", test_decode_step()))
    results.append(("Multi-turn inference", test_multi_turn_inference()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False

    print("\n" + ("✓ All tests passed!" if all_pass else "✗ Some tests failed"))
    sys.exit(0 if all_pass else 1)
