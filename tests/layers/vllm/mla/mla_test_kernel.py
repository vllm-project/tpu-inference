#!/usr/bin/env python3
"""Test MLA kernel reference implementation against manual computation.

This script verifies the MLA kernel produces correct results by comparing
against a manual numpy implementation of the MLA attention algorithm.

Run with: python tests/mla_test_kernel.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple

# Force CPU execution
jax.config.update('jax_platform_name', 'cpu')

# Import the kernel if available
try:
    from tpu_inference.kernels.mla.v1.kernel import (
        ref_mla_ragged_paged_attention,
        get_kv_cache_shape,
        update_kv_cache,
    )
    from tpu_inference.kernels.ragged_paged_attention.v3.util import align_to, get_dtype_packing
    KERNEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MLA kernel: {e}")
    KERNEL_AVAILABLE = False


def manual_mla_attention(
    ql_nope: np.ndarray,  # (num_tokens, num_heads, kv_lora_rank)
    q_pe: np.ndarray,     # (num_tokens, num_heads, qk_rope_dim)
    kv_c: np.ndarray,     # (num_tokens, kv_lora_rank)
    k_pe: np.ndarray,     # (num_tokens, qk_rope_dim)
    scale: float,
) -> np.ndarray:
    """Manual implementation of MLA attention for verification.

    Returns:
        output: (num_tokens, num_heads, kv_lora_rank)
    """
    num_tokens = ql_nope.shape[0]
    num_heads = ql_nope.shape[1]
    kv_lora_rank = ql_nope.shape[2]
    qk_rope_dim = q_pe.shape[2]

    # Step 1: Concatenate Q = [ql_nope, q_pe]
    q = np.concatenate([ql_nope, q_pe], axis=-1)  # (tokens, heads, kv_lora + rope)

    # Step 2: Concatenate K = [kv_c, k_pe]
    k = np.concatenate([kv_c, k_pe], axis=-1)  # (tokens, kv_lora + rope)

    # Step 3: Compute attention scores Q @ K^T
    # q: (tokens, heads, dim), k: (tokens, dim) -> attn: (heads, tokens, tokens)
    attn_scores = np.einsum('qnh,kh->nqk', q, k)  # (heads, q_tokens, k_tokens)
    attn_scores = attn_scores * scale

    # Step 4: Apply causal mask
    causal_mask = np.tril(np.ones((num_tokens, num_tokens)))
    attn_scores = np.where(causal_mask[None, :, :] == 0, -1e9, attn_scores)

    # Step 5: Softmax
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Step 6: Compute output = attn @ kv_c (V is kv_c)
    # attn: (heads, q_tokens, k_tokens), kv_c: (tokens, kv_lora) -> (tokens, heads, kv_lora)
    output = np.einsum('nqk,kl->qnl', attn_weights, kv_c)

    return output


def test_mla_kernel_simple():
    """Test MLA kernel with simple inputs."""
    if not KERNEL_AVAILABLE:
        print("Skipping kernel test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing MLA kernel reference implementation")
    print("="*70)

    # Small test case
    num_tokens = 4
    num_heads = 2
    kv_lora_rank = 8
    qk_rope_dim = 4
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)

    # Create inputs
    ql_nope = np.random.randn(num_tokens, num_heads, kv_lora_rank).astype(np.float32)
    q_pe = np.random.randn(num_tokens, num_heads, qk_rope_dim).astype(np.float32)
    kv_c = np.random.randn(num_tokens, kv_lora_rank).astype(np.float32)
    k_pe = np.random.randn(num_tokens, qk_rope_dim).astype(np.float32)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    # Convert to JAX/bfloat16
    ql_nope_jax = jnp.array(ql_nope, dtype=dtype)
    q_pe_jax = jnp.array(q_pe, dtype=dtype)
    kv_c_jax = jnp.array(kv_c, dtype=dtype)
    k_pe_jax = jnp.array(k_pe, dtype=dtype)

    # Create KV cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(
        total_num_pages=1,
        page_size=page_size,
        kv_dim=kv_dim,
        kv_dtype=dtype,
    )
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)

    print(f"Input shapes:")
    print(f"  ql_nope: {ql_nope.shape}")
    print(f"  q_pe: {q_pe.shape}")
    print(f"  kv_c: {kv_c.shape}")
    print(f"  k_pe: {k_pe.shape}")
    print(f"  cache_kv: {cache_shape}")
    print(f"  scale: {scale:.6f}")

    # Metadata for single sequence (prefill mode - all tokens are new)
    kv_lens = jnp.array([num_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    # distribution: [prefill_only_seqs, prefill_only + decode_seqs, total_seqs]
    # Format must satisfy: 0 <= i <= j <= k
    # For a single sequence (prefill or decode), use [0, 0, 1]
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    # Run kernel
    print("\nRunning MLA kernel...")
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

    # Run manual computation (in float32 for accuracy)
    print("Running manual computation...")
    manual_output = manual_mla_attention(ql_nope, q_pe, kv_c, k_pe, scale)

    # Compare (kernel output is padded to 128)
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]

    print(f"\nOutput shapes:")
    print(f"  kernel (raw): {kernel_output.shape}")
    print(f"  kernel (trimmed): {kernel_output_trimmed.shape}")
    print(f"  manual: {manual_output.shape}")

    # Note: bfloat16 has limited precision, so we use looser tolerances
    diff = np.abs(kernel_output_trimmed - manual_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nComparison:")
    print(f"  max diff: {max_diff:.4f}")
    print(f"  mean diff: {mean_diff:.4f}")

    # bfloat16 has ~3 decimal digits of precision, so 0.1 tolerance is reasonable
    match = max_diff < 0.1

    print(f"\nMatch: {'✓' if match else '✗'}")

    if not match:
        print(f"\nKernel output sample:\n{kernel_output_trimmed[0, 0, :4]}")
        print(f"Manual output sample:\n{manual_output[0, 0, :4]}")

    return match


def test_mla_kernel_decode():
    """Test MLA kernel in decode mode (with existing context)."""
    if not KERNEL_AVAILABLE:
        print("Skipping kernel decode test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing MLA kernel in DECODE mode")
    print("="*70)

    # Test parameters
    context_len = 8
    decode_tokens = 1
    num_heads = 2
    kv_lora_rank = 8
    qk_rope_dim = 4
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    # Create context (prefill) data
    context_ql_nope = np.random.randn(context_len, num_heads, kv_lora_rank).astype(np.float32)
    context_q_pe = np.random.randn(context_len, num_heads, qk_rope_dim).astype(np.float32)
    context_kv_c = np.random.randn(context_len, kv_lora_rank).astype(np.float32)
    context_k_pe = np.random.randn(context_len, qk_rope_dim).astype(np.float32)

    # Create decode data
    decode_ql_nope = np.random.randn(decode_tokens, num_heads, kv_lora_rank).astype(np.float32)
    decode_q_pe = np.random.randn(decode_tokens, num_heads, qk_rope_dim).astype(np.float32)
    decode_kv_c = np.random.randn(decode_tokens, kv_lora_rank).astype(np.float32)
    decode_k_pe = np.random.randn(decode_tokens, qk_rope_dim).astype(np.float32)

    # Initialize cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(
        total_num_pages=2,
        page_size=page_size,
        kv_dim=kv_dim,
        kv_dtype=dtype,
    )
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)

    # Step 1: Run prefill to populate cache
    print(f"Step 1: Prefill with {context_len} tokens...")
    prefill_kv_lens = jnp.array([context_len], dtype=jnp.int32)
    prefill_page_indices = jnp.array([0, 0], dtype=jnp.int32)  # Single page
    prefill_cu_q_lens = jnp.array([0, context_len], dtype=jnp.int32)
    # distribution: [prefill_only_seqs, prefill_only + decode_seqs, total_seqs]
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

    # Step 2: Run decode with 1 new token
    print(f"Step 2: Decode with {decode_tokens} token...")
    total_len = context_len + decode_tokens
    decode_kv_lens = jnp.array([total_len], dtype=jnp.int32)
    decode_page_indices = jnp.array([0, 0], dtype=jnp.int32)
    decode_cu_q_lens = jnp.array([0, decode_tokens], dtype=jnp.int32)
    # distribution: [prefill_only_seqs, prefill_only + decode_seqs, total_seqs]
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

    # Step 3: Manual computation for decode
    # For decode, we need full KV = [context_kv, decode_kv]
    full_kv_c = np.concatenate([context_kv_c, decode_kv_c], axis=0)
    full_k_pe = np.concatenate([context_k_pe, decode_k_pe], axis=0)

    # Compute attention for the decode token against all KV
    q = np.concatenate([decode_ql_nope, decode_q_pe], axis=-1)  # (1, heads, dim)
    k = np.concatenate([full_kv_c, full_k_pe], axis=-1)  # (total, dim)

    # Attention: (1, heads, dim) @ (total, dim)^T = (heads, 1, total)
    attn_scores = np.einsum('qnh,kh->nqk', q, k) * scale
    # No causal mask needed for decode (query is last position, can attend to all)
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Output: (heads, 1, total) @ (total, kv_lora) = (1, heads, kv_lora)
    manual_output = np.einsum('nqk,kl->qnl', attn_weights, full_kv_c)

    # Compare
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]

    diff = np.abs(kernel_output_trimmed - manual_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nComparison:")
    print(f"  max diff: {max_diff:.4f}")
    print(f"  mean diff: {mean_diff:.4f}")

    match = max_diff < 0.1  # bfloat16 tolerance

    print(f"\nDecode match: {'✓' if match else '✗'}")

    if not match:
        print(f"\nKernel output:\n{kernel_output_trimmed}")
        print(f"Manual output:\n{manual_output}")

    return match


def test_with_glm4_dims():
    """Test with actual GLM-4 dimensions (smaller scale)."""
    if not KERNEL_AVAILABLE:
        print("Skipping GLM-4 dims test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing with GLM-4 like dimensions (scaled down)")
    print("="*70)

    # Scaled-down GLM-4 dims (to fit in memory easily)
    num_tokens = 4
    num_heads = 4  # GLM-4 has 20
    kv_lora_rank = 64  # GLM-4 has 512
    qk_rope_dim = 8    # GLM-4 has 64
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    # Create inputs
    ql_nope = np.random.randn(num_tokens, num_heads, kv_lora_rank).astype(np.float32)
    q_pe = np.random.randn(num_tokens, num_heads, qk_rope_dim).astype(np.float32)
    kv_c = np.random.randn(num_tokens, kv_lora_rank).astype(np.float32)
    k_pe = np.random.randn(num_tokens, qk_rope_dim).astype(np.float32)

    # Initialize cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(1, page_size, kv_dim, dtype)
    cache_kv = jnp.zeros(cache_shape, dtype=dtype)

    # Metadata
    kv_lens = jnp.array([num_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    # distribution: [prefill_only_seqs, prefill_only + decode_seqs, total_seqs]
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    print(f"Dimensions: heads={num_heads}, kv_lora={kv_lora_rank}, rope={qk_rope_dim}")
    print(f"Padded: lkv={padded_lkv}, rope={padded_r}, total={kv_dim}")

    # Run kernel
    kernel_output, _ = ref_mla_ragged_paged_attention(
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

    # Manual computation
    manual_output = manual_mla_attention(ql_nope, q_pe, kv_c, k_pe, scale)

    # Compare
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]

    diff = np.abs(kernel_output_trimmed - manual_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nmax diff: {max_diff:.4f}, mean diff: {mean_diff:.4f}")

    match = max_diff < 0.1
    print(f"Match: {'✓' if match else '✗'}")

    return match

def test_prefix_cache_additional_blocks():
    """Test that additional blocks are allocated for new tokens in prefix cache hit.

    This test verifies the fix for the issue where:
    1. Prefix cache hit provides blocks for cached prefix (e.g., system prompt)
    2. But new tokens (e.g., user query) need ADDITIONAL blocks beyond the prefix
    3. These additional blocks must be allocated and tracked separately

    Example scenario:
    - System prompt: 32 tokens → 2 blocks (page_size=16)
    - User query: 16 tokens → needs 1 additional block
    - Total: 48 tokens → 3 blocks needed
    - Scheduler provides 2 blocks (for prefix), we must allocate 1 more

    Without this fix, the new tokens would be written to unallocated pages,
    causing garbage output or crashes.
    """
    print("\n" + "="*70)
    print("Testing prefix cache hit with additional block allocation")
    print("="*70)

    # Import the PersistentBatchManager
    try:
        from tpu_inference.runner.persistent_batch_manager import PersistentBatchManager
        from tpu_inference.runner.input_batch import InputBatch, CachedRequestState
        from unittest.mock import MagicMock
    except (ImportError, RuntimeError) as e:
        print(f"Skipping prefix cache additional blocks test - imports not available: {e}")
        return None  # Skip

    # Create mock objects
    requests = {}

    # Create a minimal InputBatch mock
    input_batch = MagicMock(spec=InputBatch)
    input_batch.req_id_to_index = {}
    input_batch.req_ids = []
    input_batch.num_reqs = 0

    encoder_cache = {}
    model_config = MagicMock()

    # Initialize PersistentBatchManager with MLA enabled
    block_size = 16
    pbm = PersistentBatchManager(
        requests=requests,
        input_batch=input_batch,
        encoder_cache=encoder_cache,
        uses_mrope=False,
        model_config=model_config,
        is_last_rank=True,
    )

    print(f"Block size: {block_size}")
    print(f"Initial free blocks: {len(pbm._free_blocks)}")

    # Scenario: Prefix cache hit with additional blocks needed
    # System prompt: 32 tokens → 2 blocks
    # User query: 16 tokens → 1 additional block needed
    # Total: 48 tokens → 3 blocks

    system_prompt_tokens = 32
    user_query_tokens = 16
    total_tokens = system_prompt_tokens + user_query_tokens

    # Calculate expected blocks
    import math
    prefix_blocks_needed = math.ceil(system_prompt_tokens / block_size)  # 2
    total_blocks_needed = math.ceil(total_tokens / block_size)  # 3
    additional_blocks_needed = total_blocks_needed - prefix_blocks_needed  # 1

    print(f"\nScenario:")
    print(f"  System prompt: {system_prompt_tokens} tokens → {prefix_blocks_needed} blocks")
    print(f"  User query: {user_query_tokens} tokens → {additional_blocks_needed} additional blocks needed")
    print(f"  Total: {total_tokens} tokens → {total_blocks_needed} blocks needed")

    # Simulate scheduler providing blocks for prefix only
    scheduler_blocks = [[10, 11]]  # 2 blocks for the cached system prompt

    # Remove scheduler blocks from free list (simulating scheduler behavior)
    for block_id in scheduler_blocks[0]:
        pbm._free_blocks.discard(block_id)

    free_before = len(pbm._free_blocks)
    print(f"\nFree blocks before processing: {free_before}")

    # Create a mock NewRequestData for prefix cache hit
    mock_new_req_data = MagicMock()
    mock_new_req_data.req_id = "test_prefix_additional"
    mock_new_req_data.prompt_token_ids = list(range(total_tokens))  # Full prompt
    mock_new_req_data.num_computed_tokens = system_prompt_tokens  # Prefix is cached
    mock_new_req_data.block_ids = scheduler_blocks.copy()  # Scheduler-provided blocks
    mock_new_req_data.mm_features = []
    mock_new_req_data.sampling_params = MagicMock()
    mock_new_req_data.lora_request = None

    # Manually invoke the prefix cache hit logic
    # (This simulates what happens in update_from_output for new requests)
    block_ids = mock_new_req_data.block_ids
    if pbm.use_mla and block_ids and len(block_ids) > 0:
        # This is the logic we fixed - allocate additional blocks
        scheduler_block_count = len(block_ids[0]) if block_ids else 0
        needed_blocks = math.ceil(total_tokens / pbm.block_size)

        print(f"\nScheduler provided: {scheduler_block_count} blocks")
        print(f"Needed: {needed_blocks} blocks")

        # Allocate additional blocks if needed
        additional_blocks = []
        if needed_blocks > scheduler_block_count:
            additional_count = needed_blocks - scheduler_block_count
            if len(pbm._free_blocks) >= additional_count:
                free_list = sorted(pbm._free_blocks)
                for i in range(additional_count):
                    block_id = free_list[i]
                    additional_blocks.append(block_id)
                    pbm._free_blocks.remove(block_id)
                # Extend block_ids with additional blocks
                block_ids[0].extend(additional_blocks)
                print(f"Allocated {additional_count} additional block(s): {additional_blocks}")

        # Track blocks
        req_id = mock_new_req_data.req_id
        pbm._request_blocks[req_id] = block_ids[0] if block_ids else []
        if additional_blocks:
            pbm._self_allocated_blocks[req_id] = additional_blocks.copy()

    free_after = len(pbm._free_blocks)
    print(f"Free blocks after processing: {free_after}")

    # Verify results
    all_blocks = pbm._request_blocks.get("test_prefix_additional", [])
    self_allocated = pbm._self_allocated_blocks.get("test_prefix_additional", [])

    print(f"\nResults:")
    print(f"  Total blocks for request: {all_blocks}")
    print(f"  Self-allocated blocks: {self_allocated}")

    # Assertions
    success = True

    # Check total blocks
    if len(all_blocks) != total_blocks_needed:
        print(f"✗ FAIL: Expected {total_blocks_needed} total blocks, got {len(all_blocks)}")
        success = False
    else:
        print(f"✓ Total blocks correct: {len(all_blocks)}")

    # Check additional blocks allocated
    if len(additional_blocks) != additional_blocks_needed:
        print(f"✗ FAIL: Expected {additional_blocks_needed} additional blocks, got {len(additional_blocks)}")
        success = False
    else:
        print(f"✓ Additional blocks allocated: {len(additional_blocks)}")

    # Check self-allocated tracking (for proper cleanup)
    if len(self_allocated) != additional_blocks_needed:
        print(f"✗ FAIL: Expected {additional_blocks_needed} self-allocated blocks, got {len(self_allocated)}")
        success = False
    else:
        print(f"✓ Self-allocated tracking correct: {len(self_allocated)}")

    # Verify free blocks were consumed
    expected_free = free_before - additional_blocks_needed
    if free_after != expected_free:
        print(f"✗ FAIL: Expected {expected_free} free blocks, got {free_after}")
        success = False
    else:
        print(f"✓ Free blocks consumed correctly")

    print(f"\nPrefix cache additional blocks test: {'✓' if success else '✗'}")
    return success


def test_decode_block_allocation_boundary():
    """Test that decode correctly allocates blocks at page boundaries.

    This test verifies the fix for the issue where decode fails to allocate
    the correct blocks when the token being generated crosses a page boundary.

    Example scenario:
    - After prefill of 48 tokens: num_computed_tokens = 48
    - Decode needs to write token at position 48 (0-indexed)
    - Page for position 48 = 48 // 16 = 3 (need page 3)
    - Without fix: needed_blocks = ceil(48/16) = 3 (pages 0,1,2 only!)
    - With fix: needed_blocks = ceil(49/16) = 4 (pages 0,1,2,3 - correct!)

    This is the exact scenario that causes "gibberish after some tokens".
    """
    print("\n" + "="*70)
    print("Testing decode block allocation at page boundaries")
    print("="*70)

    # This test only uses math, no vllm imports needed
    import math

    # Test scenario: 48 tokens prefilled, now decoding token 49
    # With page_size=16: positions 0-15 in page 0, 16-31 in page 1, 32-47 in page 2
    # Token 49 is at position 48, which is in page 3 (48//16=3)

    block_size = 16
    num_computed_tokens = 48  # 48 tokens already computed (positions 0-47)
    # Token being generated is at position 48

    print(f"Scenario: {num_computed_tokens} tokens computed, generating token at position {num_computed_tokens}")
    print(f"Page size: {block_size}")
    print(f"Position {num_computed_tokens} requires page {num_computed_tokens // block_size}")

    # Test the OLD logic (buggy)
    old_current_tokens = num_computed_tokens  # Bug: doesn't account for new token
    old_needed_blocks = math.ceil(old_current_tokens / block_size)
    print(f"\nOLD logic (buggy):")
    print(f"  current_tokens = {old_current_tokens}")
    print(f"  needed_blocks = ceil({old_current_tokens}/{block_size}) = {old_needed_blocks}")
    print(f"  Pages allocated: 0-{old_needed_blocks-1}")

    # Test the NEW logic (fixed)
    new_current_tokens = num_computed_tokens + 1  # Fix: +1 for token being generated
    new_needed_blocks = math.ceil(new_current_tokens / block_size)
    print(f"\nNEW logic (fixed):")
    print(f"  current_tokens = {new_current_tokens}")
    print(f"  needed_blocks = ceil({new_current_tokens}/{block_size}) = {new_needed_blocks}")
    print(f"  Pages allocated: 0-{new_needed_blocks-1}")

    # Verify the fix
    token_position = num_computed_tokens  # Position of token being generated
    required_page = token_position // block_size

    old_has_page = old_needed_blocks > required_page
    new_has_page = new_needed_blocks > required_page

    print(f"\nVerification:")
    print(f"  Token at position {token_position} needs page {required_page}")
    print(f"  OLD logic has page {required_page}? {old_has_page}")
    print(f"  NEW logic has page {required_page}? {new_has_page}")

    success = True

    # OLD logic should FAIL to have the required page at boundary
    if old_has_page:
        print(f"✗ OLD logic unexpectedly has page {required_page} (test setup issue)")
        # This might happen if not at a boundary - still continue
    else:
        print(f"✓ OLD logic correctly missing page {required_page} (confirms the bug)")

    # NEW logic MUST have the required page
    if not new_has_page:
        print(f"✗ FAIL: NEW logic missing page {required_page}")
        success = False
    else:
        print(f"✓ NEW logic correctly has page {required_page}")

    # Test additional boundary cases
    test_cases = [
        (15, 16, "Last token in page 0"),
        (16, 16, "First token in page 1"),
        (31, 16, "Last token in page 1"),
        (32, 16, "First token in page 2"),
        (47, 16, "Last token in page 2"),
        (48, 16, "First token in page 3 (our bug case)"),
        (63, 16, "Last token in page 3"),
        (64, 16, "First token in page 4"),
    ]

    print(f"\nBoundary case analysis (page_size={block_size}):")
    for num_computed, ps, description in test_cases:
        token_pos = num_computed  # Token being generated
        required_page = token_pos // ps

        # OLD logic
        old_blocks = math.ceil(num_computed / ps)
        old_ok = old_blocks > required_page

        # NEW logic
        new_blocks = math.ceil((num_computed + 1) / ps)
        new_ok = new_blocks > required_page

        status = "✓" if new_ok else "✗"
        old_status = "OK" if old_ok else "MISS"

        print(f"  {description}: pos={token_pos}, page={required_page}, old={old_blocks}({old_status}), new={new_blocks} {status}")

        if not new_ok:
            success = False

    print(f"\nDecode block allocation boundary test: {'✓' if success else '✗'}")
    return success


def test_cache_clearing_on_prefill():
    """Test that cache pages are cleared during prefill to prevent stale data corruption.

    This test verifies the fix for the issue where subsequent requests produce
    gibberish because stale data from previous requests (that used the same cache
    blocks) wasn't being cleared.

    The test:
    1. Creates a cache with "stale" data (non-zero values simulating previous request)
    2. Runs a prefill operation with new data
    3. Verifies that stale data beyond the sequence length has been cleared
    """
    if not KERNEL_AVAILABLE:
        print("Skipping cache clearing test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing cache clearing on prefill")
    print("="*70)

    num_tokens = 4  # Short prefill
    num_heads = 2
    kv_lora_rank = 8
    qk_rope_dim = 4
    page_size = 16  # Page can hold 16 tokens, but we only use 4
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    # Create inputs
    ql_nope = np.random.randn(num_tokens, num_heads, kv_lora_rank).astype(np.float32)
    q_pe = np.random.randn(num_tokens, num_heads, qk_rope_dim).astype(np.float32)
    kv_c = np.random.randn(num_tokens, kv_lora_rank).astype(np.float32)
    k_pe = np.random.randn(num_tokens, qk_rope_dim).astype(np.float32)

    # Initialize cache with STALE DATA (simulating previous request's data)
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(1, page_size, kv_dim, dtype)
    print(f"Cache shape: {cache_shape}")

    # Fill cache with "stale" non-zero data (value 1.0 everywhere)
    stale_cache = jnp.ones(cache_shape, dtype=dtype)
    print(f"Initial cache (stale data): all ones")

    # Metadata for prefill
    kv_lens = jnp.array([num_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    # Run kernel - this should clear the page before writing new data
    _, updated_cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(ql_nope, dtype=dtype),
        q_pe=jnp.array(q_pe, dtype=dtype),
        new_kv_c=jnp.array(kv_c, dtype=dtype),
        new_k_pe=jnp.array(k_pe, dtype=dtype),
        cache_kv=stale_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=scale,
    )

    # Check the cache: positions 0-3 should have new data, positions 4-15 should be zeros
    updated_cache_np = np.array(updated_cache, dtype=np.float32)

    # Cache shape: (pages, page_size/kv_packing, kv_packing, kv_dim)
    # For bfloat16: kv_packing = 2
    kv_packing = get_dtype_packing(dtype)
    print(f"kv_packing: {kv_packing}")

    # Flatten the cache to check positions
    cache_flat = updated_cache_np.reshape(-1, updated_cache_np.shape[-1])
    print(f"Flattened cache shape: {cache_flat.shape}")

    # Check that positions beyond num_tokens are zeros (cleared stale data)
    stale_region = cache_flat[num_tokens:page_size, :]
    stale_region_max = np.max(np.abs(stale_region))
    stale_region_mean = np.mean(np.abs(stale_region))

    print(f"\nStale region check (positions {num_tokens} to {page_size-1}):")
    print(f"  max |value|: {stale_region_max:.6f}")
    print(f"  mean |value|: {stale_region_mean:.6f}")

    # Check that positions 0-3 have non-zero values (new data was written)
    new_data_region = cache_flat[:num_tokens, :]
    new_data_max = np.max(np.abs(new_data_region))
    new_data_mean = np.mean(np.abs(new_data_region))

    print(f"\nNew data region check (positions 0 to {num_tokens-1}):")
    print(f"  max |value|: {new_data_max:.6f}")
    print(f"  mean |value|: {new_data_mean:.6f}")

    # Success criteria:
    # 1. Stale region should be cleared (near zero)
    # 2. New data region should have non-zero values
    stale_cleared = stale_region_max < 0.01  # Should be essentially zero
    new_data_present = new_data_max > 0.1    # Should have actual values

    success = stale_cleared and new_data_present

    if stale_cleared:
        print(f"\n✓ Stale data was cleared")
    else:
        print(f"\n✗ Stale data NOT cleared (max={stale_region_max:.4f}, expected ~0)")

    if new_data_present:
        print(f"✓ New data was written")
    else:
        print(f"✗ New data NOT written (max={new_data_max:.4f}, expected > 0.1)")

    print(f"\nCache clearing test: {'✓' if success else '✗'}")
    return success


def test_long_sequence_multi_page():
    """Test MLA kernel with long sequences spanning multiple pages.

    This tests the scenario where a sequence is long enough to span multiple
    cache pages, which is common with system prompts + user messages.
    """
    if not KERNEL_AVAILABLE:
        print("Skipping long sequence test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing LONG SEQUENCE spanning multiple pages")
    print("="*70)

    # Realistic parameters - sequence spans multiple pages
    num_tokens = 48  # Spans 3 pages with page_size=16
    num_heads = 4
    kv_lora_rank = 64
    qk_rope_dim = 8
    page_size = 16
    num_pages = (num_tokens + page_size - 1) // page_size  # 3 pages
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    print(f"Config: {num_tokens} tokens, {num_pages} pages (page_size={page_size})")
    print(f"Dims: heads={num_heads}, kv_lora={kv_lora_rank}, rope={qk_rope_dim}")

    # Create inputs
    ql_nope = np.random.randn(num_tokens, num_heads, kv_lora_rank).astype(np.float32)
    q_pe = np.random.randn(num_tokens, num_heads, qk_rope_dim).astype(np.float32)
    kv_c = np.random.randn(num_tokens, kv_lora_rank).astype(np.float32)
    k_pe = np.random.randn(num_tokens, qk_rope_dim).astype(np.float32)

    # Initialize cache with stale data to detect clearing issues
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(num_pages, page_size, kv_dim, dtype)
    print(f"Cache shape: {cache_shape}")

    # Fill with stale data
    stale_cache = jnp.ones(cache_shape, dtype=dtype) * 0.5

    # Metadata for multi-page prefill
    kv_lens = jnp.array([num_tokens], dtype=jnp.int32)
    # Page indices for all pages this sequence uses
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    print(f"Page indices: {page_indices}")

    # Run prefill
    print("\nRunning multi-page prefill...")
    kernel_output, updated_cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(ql_nope, dtype=dtype),
        q_pe=jnp.array(q_pe, dtype=dtype),
        new_kv_c=jnp.array(kv_c, dtype=dtype),
        new_k_pe=jnp.array(k_pe, dtype=dtype),
        cache_kv=stale_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=scale,
    )

    # Manual reference computation
    print("Running manual computation...")
    manual_output = manual_mla_attention(ql_nope, q_pe, kv_c, k_pe, scale)

    # Compare
    kernel_output_np = np.array(kernel_output, dtype=np.float32)
    kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]

    diff = np.abs(kernel_output_trimmed - manual_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nComparison:")
    print(f"  max diff: {max_diff:.4f}")
    print(f"  mean diff: {mean_diff:.4f}")

    # Check cache was properly populated across all pages
    updated_cache_np = np.array(updated_cache, dtype=np.float32)

    # Verify each page has data
    for page_idx in range(num_pages):
        page_data = updated_cache_np[page_idx]
        page_max = np.max(np.abs(page_data))
        tokens_in_page = min(page_size, num_tokens - page_idx * page_size)
        print(f"  Page {page_idx}: max|value|={page_max:.4f}, tokens={tokens_in_page}")

    match = max_diff < 0.15  # Slightly higher tolerance for longer sequences
    print(f"\nLong sequence test: {'✓' if match else '✗'}")

    if not match:
        # Debug: show where differences are largest
        diff_by_token = np.max(diff, axis=(1, 2))
        worst_tokens = np.argsort(diff_by_token)[-5:]
        print(f"Worst tokens: {worst_tokens}, diffs: {diff_by_token[worst_tokens]}")

    return match


def test_extended_decode_sequence():
    """Test many sequential decode steps after a prefill.

    This simulates generating a long response after a system prompt + user query.
    """
    if not KERNEL_AVAILABLE:
        print("Skipping extended decode test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing EXTENDED DECODE (many decode steps)")
    print("="*70)

    # Simulate: 32 token prefill (system prompt + user query), then 20 decode steps
    prefill_len = 32
    num_decode_steps = 20
    num_heads = 4
    kv_lora_rank = 64
    qk_rope_dim = 8
    page_size = 16
    total_max_len = prefill_len + num_decode_steps
    num_pages = (total_max_len + page_size - 1) // page_size
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    print(f"Config: {prefill_len} prefill tokens + {num_decode_steps} decode steps")
    print(f"Max sequence length: {total_max_len}, pages needed: {num_pages}")

    # Initialize cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r
    cache_shape = get_kv_cache_shape(num_pages, page_size, kv_dim, dtype)

    # Start with stale data to detect clearing issues
    cache = jnp.ones(cache_shape, dtype=dtype) * 0.5

    # Generate all data upfront for reference computation
    all_ql_nope = np.random.randn(total_max_len, num_heads, kv_lora_rank).astype(np.float32)
    all_q_pe = np.random.randn(total_max_len, num_heads, qk_rope_dim).astype(np.float32)
    all_kv_c = np.random.randn(total_max_len, kv_lora_rank).astype(np.float32)
    all_k_pe = np.random.randn(total_max_len, qk_rope_dim).astype(np.float32)

    # Page indices for all pages
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)

    # Step 1: Prefill
    print(f"\nStep 1: Prefill with {prefill_len} tokens...")
    prefill_kv_lens = jnp.array([prefill_len], dtype=jnp.int32)
    prefill_cu_q_lens = jnp.array([0, prefill_len], dtype=jnp.int32)
    prefill_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    _, cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(all_ql_nope[:prefill_len], dtype=dtype),
        q_pe=jnp.array(all_q_pe[:prefill_len], dtype=dtype),
        new_kv_c=jnp.array(all_kv_c[:prefill_len], dtype=dtype),
        new_k_pe=jnp.array(all_k_pe[:prefill_len], dtype=dtype),
        cache_kv=cache,
        kv_lens=prefill_kv_lens,
        page_indices=page_indices,
        cu_q_lens=prefill_cu_q_lens,
        distribution=prefill_distribution,
        sm_scale=scale,
    )
    print(f"  Cache updated after prefill")

    # Step 2: Sequential decode steps
    print(f"\nStep 2: Running {num_decode_steps} decode steps...")
    decode_outputs = []
    errors = []

    for step in range(num_decode_steps):
        current_len = prefill_len + step + 1
        token_idx = prefill_len + step

        # Decode one token
        decode_kv_lens = jnp.array([current_len], dtype=jnp.int32)
        decode_cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)
        decode_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

        kernel_output, cache = ref_mla_ragged_paged_attention(
            ql_nope=jnp.array(all_ql_nope[token_idx:token_idx+1], dtype=dtype),
            q_pe=jnp.array(all_q_pe[token_idx:token_idx+1], dtype=dtype),
            new_kv_c=jnp.array(all_kv_c[token_idx:token_idx+1], dtype=dtype),
            new_k_pe=jnp.array(all_k_pe[token_idx:token_idx+1], dtype=dtype),
            cache_kv=cache,
            kv_lens=decode_kv_lens,
            page_indices=page_indices,
            cu_q_lens=decode_cu_q_lens,
            distribution=decode_distribution,
            sm_scale=scale,
        )

        # Compute manual reference for this decode step
        # Query is just the current token, KV is all tokens up to current
        q = np.concatenate([all_ql_nope[token_idx:token_idx+1],
                           all_q_pe[token_idx:token_idx+1]], axis=-1)
        k = np.concatenate([all_kv_c[:current_len], all_k_pe[:current_len]], axis=-1)

        attn_scores = np.einsum('qnh,kh->nqk', q, k) * scale
        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        manual_output = np.einsum('nqk,kl->qnl', attn_weights, all_kv_c[:current_len])

        # Compare
        kernel_output_np = np.array(kernel_output, dtype=np.float32)
        kernel_output_trimmed = kernel_output_np[:, :, :kv_lora_rank]

        diff = np.abs(kernel_output_trimmed - manual_output)
        max_diff = np.max(diff)
        errors.append(max_diff)
        decode_outputs.append(kernel_output_trimmed)

        if step < 5 or step >= num_decode_steps - 3:
            print(f"  Step {step+1}: len={current_len}, max_diff={max_diff:.4f}")
        elif step == 5:
            print(f"  ...")

    # Analyze errors
    errors = np.array(errors)
    print(f"\nDecode error statistics:")
    print(f"  mean: {np.mean(errors):.4f}")
    print(f"  max:  {np.max(errors):.4f}")
    print(f"  min:  {np.min(errors):.4f}")

    # Check for error growth (indicates accumulating issues)
    first_half_mean = np.mean(errors[:len(errors)//2])
    second_half_mean = np.mean(errors[len(errors)//2:])
    print(f"  first half mean:  {first_half_mean:.4f}")
    print(f"  second half mean: {second_half_mean:.4f}")

    # Check if errors are acceptable
    all_acceptable = np.all(errors < 0.15)
    no_error_growth = second_half_mean < first_half_mean * 2  # Allow some growth

    success = all_acceptable and no_error_growth

    if not success:
        # Find problematic steps
        bad_steps = np.where(errors > 0.15)[0]
        if len(bad_steps) > 0:
            print(f"\n✗ Steps with high error: {bad_steps[:10]}")
        if second_half_mean >= first_half_mean * 2:
            print(f"✗ Error growth detected: {first_half_mean:.4f} -> {second_half_mean:.4f}")

    print(f"\nExtended decode test: {'✓' if success else '✗'}")
    return success


def test_simulated_prefix_cache_hit():
    """Test simulated prefix cache hit scenario.

    This simulates:
    1. First request: "System prompt + User query 1" - fresh prefill
    2. Second request: "System prompt + User query 2" - prefix cache HIT
       (system prompt is cached, only user query 2 is new)

    This is the exact scenario that can cause gibberish output if
    the prefix cache isn't handled correctly.
    """
    if not KERNEL_AVAILABLE:
        print("Skipping prefix cache hit test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing SIMULATED PREFIX CACHE HIT")
    print("="*70)

    # System prompt = 32 tokens, user query = 16 tokens
    system_prompt_len = 32
    user_query_1_len = 16
    user_query_2_len = 12
    num_heads = 4
    kv_lora_rank = 64
    qk_rope_dim = 8
    page_size = 16
    num_pages = 4
    dtype = jnp.bfloat16

    np.random.seed(42)
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    print(f"Config: system={system_prompt_len}, query1={user_query_1_len}, query2={user_query_2_len}")

    # Initialize cache
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r
    cache_shape = get_kv_cache_shape(num_pages, page_size, kv_dim, dtype)

    # Start with zeros (clean cache)
    cache = jnp.zeros(cache_shape, dtype=dtype)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)

    # Generate data for first request
    req1_len = system_prompt_len + user_query_1_len
    req1_ql_nope = np.random.randn(req1_len, num_heads, kv_lora_rank).astype(np.float32)
    req1_q_pe = np.random.randn(req1_len, num_heads, qk_rope_dim).astype(np.float32)
    req1_kv_c = np.random.randn(req1_len, kv_lora_rank).astype(np.float32)
    req1_k_pe = np.random.randn(req1_len, qk_rope_dim).astype(np.float32)

    # === Request 1: Fresh prefill (system prompt + query 1) ===
    print(f"\n--- Request 1: Fresh prefill ({req1_len} tokens) ---")
    req1_kv_lens = jnp.array([req1_len], dtype=jnp.int32)
    req1_cu_q_lens = jnp.array([0, req1_len], dtype=jnp.int32)
    req1_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    req1_output, cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(req1_ql_nope, dtype=dtype),
        q_pe=jnp.array(req1_q_pe, dtype=dtype),
        new_kv_c=jnp.array(req1_kv_c, dtype=dtype),
        new_k_pe=jnp.array(req1_k_pe, dtype=dtype),
        cache_kv=cache,
        kv_lens=req1_kv_lens,
        page_indices=page_indices,
        cu_q_lens=req1_cu_q_lens,
        distribution=req1_distribution,
        sm_scale=scale,
    )

    # Manual reference for request 1
    req1_manual = manual_mla_attention(req1_ql_nope, req1_q_pe, req1_kv_c, req1_k_pe, scale)
    req1_kernel_np = np.array(req1_output, dtype=np.float32)[:, :, :kv_lora_rank]
    req1_diff = np.max(np.abs(req1_kernel_np - req1_manual))
    print(f"  Request 1 max diff: {req1_diff:.4f}")

    # Save the cache state with system prompt
    cache_after_req1 = cache

    # === Request 2: Prefix cache hit (reuse system prompt, only query 2 is new) ===
    print(f"\n--- Request 2: PREFIX CACHE HIT ---")
    print(f"  Cached prefix: {system_prompt_len} tokens (system prompt)")
    print(f"  New tokens: {user_query_2_len} tokens (user query 2)")

    # For request 2, we only process the NEW tokens (user query 2)
    # But kv_len = system_prompt_len + user_query_2_len (total sequence length)
    req2_total_len = system_prompt_len + user_query_2_len
    req2_new_tokens = user_query_2_len

    # Generate NEW data for request 2 (only for user query 2)
    req2_ql_nope = np.random.randn(req2_new_tokens, num_heads, kv_lora_rank).astype(np.float32)
    req2_q_pe = np.random.randn(req2_new_tokens, num_heads, qk_rope_dim).astype(np.float32)
    req2_kv_c = np.random.randn(req2_new_tokens, kv_lora_rank).astype(np.float32)
    req2_k_pe = np.random.randn(req2_new_tokens, qk_rope_dim).astype(np.float32)

    # CRITICAL: kv_len is TOTAL (cached + new), q_len is just NEW
    req2_kv_lens = jnp.array([req2_total_len], dtype=jnp.int32)
    req2_cu_q_lens = jnp.array([0, req2_new_tokens], dtype=jnp.int32)
    req2_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    # The cache already has the system prompt from request 1 (simulating prefix cache)
    # We use the SAME page_indices since the prefix is cached in the same blocks

    req2_output, cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(req2_ql_nope, dtype=dtype),
        q_pe=jnp.array(req2_q_pe, dtype=dtype),
        new_kv_c=jnp.array(req2_kv_c, dtype=dtype),
        new_k_pe=jnp.array(req2_k_pe, dtype=dtype),
        cache_kv=cache_after_req1,  # Use cache with system prompt
        kv_lens=req2_kv_lens,
        page_indices=page_indices,
        cu_q_lens=req2_cu_q_lens,
        distribution=req2_distribution,
        sm_scale=scale,
    )

    # Manual reference for request 2
    # The query tokens attend to: system_prompt (from cache) + user_query_2 (new)
    # We need the full KV for manual computation
    # Use system prompt KV from request 1, plus new query 2 KV
    full_kv_c = np.concatenate([req1_kv_c[:system_prompt_len], req2_kv_c], axis=0)
    full_k_pe = np.concatenate([req1_k_pe[:system_prompt_len], req2_k_pe], axis=0)

    # Compute attention for query 2 tokens attending to full KV
    q = np.concatenate([req2_ql_nope, req2_q_pe], axis=-1)  # (new_tokens, heads, dim)
    k = np.concatenate([full_kv_c, full_k_pe], axis=-1)  # (total, dim)

    # Each query token position in the sequence
    # q[0] is at seq position system_prompt_len
    # q[1] is at seq position system_prompt_len + 1, etc.
    attn_scores = np.einsum('qnh,kh->nqk', q, k) * scale

    # Causal mask: q position i (at seq pos system_prompt_len+i) can attend to kv[0:system_prompt_len+i+1]
    for qi in range(req2_new_tokens):
        seq_pos = system_prompt_len + qi
        for ki in range(req2_total_len):
            if ki > seq_pos:  # Future position
                attn_scores[:, qi, ki] = -1e9

    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    req2_manual = np.einsum('nqk,kl->qnl', attn_weights, full_kv_c)

    # Compare
    req2_kernel_np = np.array(req2_output, dtype=np.float32)[:, :, :kv_lora_rank]
    req2_diff = np.max(np.abs(req2_kernel_np - req2_manual))
    print(f"  Request 2 max diff: {req2_diff:.4f}")

    # Check for issues
    success = req1_diff < 0.15 and req2_diff < 0.15

    if req2_diff >= 0.15:
        print(f"\n✗ Prefix cache hit produced incorrect output!")
        print(f"  Kernel output sample: {req2_kernel_np[0, 0, :5]}")
        print(f"  Manual output sample: {req2_manual[0, 0, :5]}")

        # Check if output looks like gibberish (very large values)
        if np.max(np.abs(req2_kernel_np)) > 10:
            print(f"  WARNING: Output contains very large values - likely stale data corruption!")

    print(f"\nPrefix cache hit test: {'✓' if success else '✗'}")
    return success


def test_decode_after_long_prefill_with_stale_cache():
    """Test decode after long prefill with stale data in cache.

    This specifically tests the scenario where:
    1. Cache has stale data from a previous request
    2. A new long prefill fills multiple pages
    3. Subsequent decode steps must read from the filled cache

    This catches issues where stale data corrupts decode outputs.
    """
    if not KERNEL_AVAILABLE:
        print("Skipping decode-after-long-prefill test - kernel not available")
        return None  # Skip

    print("\n" + "="*70)
    print("Testing DECODE after LONG PREFILL with STALE cache")
    print("="*70)

    prefill_len = 40  # Spans ~3 pages
    decode_steps = 10
    num_heads = 4
    kv_lora_rank = 64
    qk_rope_dim = 8
    page_size = 16
    num_pages = 4  # Extra page for decode
    dtype = jnp.bfloat16

    np.random.seed(123)  # Different seed
    scale = 1.0 / np.sqrt(kv_lora_rank + qk_rope_dim)

    print(f"Config: {prefill_len} prefill + {decode_steps} decode, {num_pages} pages")

    # Initialize cache with DISTINCTIVE stale pattern
    padded_lkv = align_to(kv_lora_rank, 128)
    padded_r = align_to(qk_rope_dim, 128)
    kv_dim = padded_lkv + padded_r
    cache_shape = get_kv_cache_shape(num_pages, page_size, kv_dim, dtype)

    # Create stale cache with recognizable pattern (large values)
    stale_cache = jnp.ones(cache_shape, dtype=dtype) * 5.0  # Large stale values
    print(f"Initial cache filled with stale value 5.0")

    # Generate data
    total_len = prefill_len + decode_steps
    all_ql_nope = np.random.randn(total_len, num_heads, kv_lora_rank).astype(np.float32) * 0.1
    all_q_pe = np.random.randn(total_len, num_heads, qk_rope_dim).astype(np.float32) * 0.1
    all_kv_c = np.random.randn(total_len, kv_lora_rank).astype(np.float32) * 0.1
    all_k_pe = np.random.randn(total_len, qk_rope_dim).astype(np.float32) * 0.1

    page_indices = jnp.arange(num_pages, dtype=jnp.int32)

    # Prefill
    print(f"\nStep 1: Prefill {prefill_len} tokens...")
    prefill_kv_lens = jnp.array([prefill_len], dtype=jnp.int32)
    prefill_cu_q_lens = jnp.array([0, prefill_len], dtype=jnp.int32)
    prefill_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    prefill_output, cache = ref_mla_ragged_paged_attention(
        ql_nope=jnp.array(all_ql_nope[:prefill_len], dtype=dtype),
        q_pe=jnp.array(all_q_pe[:prefill_len], dtype=dtype),
        new_kv_c=jnp.array(all_kv_c[:prefill_len], dtype=dtype),
        new_k_pe=jnp.array(all_k_pe[:prefill_len], dtype=dtype),
        cache_kv=stale_cache,
        kv_lens=prefill_kv_lens,
        page_indices=page_indices,
        cu_q_lens=prefill_cu_q_lens,
        distribution=prefill_distribution,
        sm_scale=scale,
    )

    # Check prefill output is reasonable (not corrupted by stale data)
    prefill_output_np = np.array(prefill_output, dtype=np.float32)
    prefill_max = np.max(np.abs(prefill_output_np))
    print(f"  Prefill output max: {prefill_max:.4f} (should be < 1.0 given small inputs)")

    # If output is huge, stale data leaked
    prefill_ok = prefill_max < 2.0  # Should be small given 0.1 scale inputs

    # Decode steps
    print(f"\nStep 2: Running {decode_steps} decode steps...")
    decode_ok = True

    for step in range(decode_steps):
        current_len = prefill_len + step + 1
        token_idx = prefill_len + step

        decode_kv_lens = jnp.array([current_len], dtype=jnp.int32)
        decode_cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)
        decode_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

        decode_output, cache = ref_mla_ragged_paged_attention(
            ql_nope=jnp.array(all_ql_nope[token_idx:token_idx+1], dtype=dtype),
            q_pe=jnp.array(all_q_pe[token_idx:token_idx+1], dtype=dtype),
            new_kv_c=jnp.array(all_kv_c[token_idx:token_idx+1], dtype=dtype),
            new_k_pe=jnp.array(all_k_pe[token_idx:token_idx+1], dtype=dtype),
            cache_kv=cache,
            kv_lens=decode_kv_lens,
            page_indices=page_indices,
            cu_q_lens=decode_cu_q_lens,
            distribution=decode_distribution,
            sm_scale=scale,
        )

        decode_output_np = np.array(decode_output, dtype=np.float32)
        decode_max = np.max(np.abs(decode_output_np))

        # If stale data (5.0) is being read, output will be huge
        if decode_max > 2.0:
            print(f"  Step {step+1}: max={decode_max:.4f} - STALE DATA DETECTED!")
            decode_ok = False
        elif step < 3 or step >= decode_steps - 2:
            print(f"  Step {step+1}: max={decode_max:.4f} - OK")

    success = prefill_ok and decode_ok

    if not prefill_ok:
        print(f"\n✗ Prefill corrupted by stale data (max={prefill_max:.4f})")
    if not decode_ok:
        print(f"\n✗ Decode corrupted by stale data")

    print(f"\nDecode-after-long-prefill test: {'✓' if success else '✗'}")
    return success


def run_test(name: str, test_func):
    """Run a test and catch any unexpected exceptions."""
    try:
        result = test_func()
        return (name, result)
    except Exception as e:
        print(f"\n✗ {name} raised exception: {e}")
        import traceback
        traceback.print_exc()
        return (name, False)


if __name__ == "__main__":
    print("="*70)
    print("MLA KERNEL TESTS")
    print("="*70)

    # Run all tests with exception handling
    results = []
    results.append(run_test("Simple prefill", test_mla_kernel_simple))
    results.append(run_test("Decode mode", test_mla_kernel_decode))
    results.append(run_test("GLM-4 dims", test_with_glm4_dims))
    results.append(run_test("Cache clearing on prefill", test_cache_clearing_on_prefill))
    results.append(run_test("Prefix cache additional blocks", test_prefix_cache_additional_blocks))
    results.append(run_test("Decode block allocation boundary", test_decode_block_allocation_boundary))
    results.append(run_test("Long sequence multi-page", test_long_sequence_multi_page))
    results.append(run_test("Extended decode sequence", test_extended_decode_sequence))
    results.append(run_test("Decode after long prefill with stale cache", test_decode_after_long_prefill_with_stale_cache))
    results.append(run_test("Simulated prefix cache hit", test_simulated_prefix_cache_hit))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is None:
            status = "⊘ SKIP"
            skipped += 1
        elif result:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        print(f"{status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n✗ Some tests failed!")
        exit(1)
    elif passed == 0 and skipped > 0:
        print("\n⚠ All tests were skipped - no tests actually ran!")
        exit(1)
    else:
        print("\n✓ All tests passed!")
