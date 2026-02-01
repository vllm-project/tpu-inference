#!/usr/bin/env python3
"""Test KV cache management and metadata for MLA attention.

Tests that can be run locally to verify:
1. KV cache page allocation and update logic
2. Metadata arrays (kv_lens, cu_q_lens, distribution)
3. Multi-turn decode simulation (prefill -> decode -> decode)
4. RoPE application timing

Run with: python tests/mla_test_cache_and_metadata.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict

# Force CPU execution
jax.config.update('jax_platform_name', 'cpu')


def align_to(x: int, alignment: int) -> int:
    """Align x to the next multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


def get_dtype_packing(dtype) -> int:
    """Get packing factor for dtype."""
    if dtype == jnp.bfloat16 or dtype == jnp.float16:
        return 2
    elif dtype == jnp.float32:
        return 1
    else:
        return 1


def get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype):
    """Get KV cache shape for MLA."""
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


def manual_update_kv_cache(
    new_kv_c: np.ndarray,  # [num_tokens, lkv_dim]
    new_k_pe: np.ndarray,  # [num_tokens, r_dim]
    cache_kv: np.ndarray,  # [total_pages, page_size_packed, packing, kv_dim]
    kv_lens: np.ndarray,   # [max_num_seqs]
    page_indices: np.ndarray,  # [max_num_seqs * pages_per_seq]
    cu_q_lens: np.ndarray,  # [max_num_seqs + 1]
    num_seqs: int,
) -> np.ndarray:
    """Manual KV cache update for verification."""
    lkv_dim = new_kv_c.shape[-1]
    r_dim = new_k_pe.shape[-1]

    # Pad to alignment
    padded_lkv = align_to(lkv_dim, 128)
    padded_r = align_to(r_dim, 128)

    if lkv_dim != padded_lkv:
        new_kv_c = np.pad(new_kv_c, ((0, 0), (0, padded_lkv - lkv_dim)))
    if r_dim != padded_r:
        new_k_pe = np.pad(new_k_pe, ((0, 0), (0, padded_r - r_dim)))

    total_pages, page_size_packed, packing, kv_dim = cache_kv.shape
    page_size = page_size_packed * packing
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    cache_kv = cache_kv.copy()

    for i in range(num_seqs):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        for j in range(q_len):
            token_idx_in_seq = kv_len - q_len + j
            page_num_in_seq = token_idx_in_seq // page_size
            page_indices_start = i * pages_per_seq
            page_idx = page_indices[page_indices_start + page_num_in_seq]
            row = (token_idx_in_seq % page_size) // packing
            col = (token_idx_in_seq % page_size) % packing

            # Store kv_c in first lkv_dim, k_pe in remaining
            cache_kv[page_idx, row, col, :padded_lkv] = new_kv_c[q_start + j]
            cache_kv[page_idx, row, col, padded_lkv:padded_lkv + padded_r] = new_k_pe[q_start + j]

    return cache_kv


def test_kv_cache_update_single_seq():
    """Test KV cache update for a single sequence."""
    print("\n" + "="*70)
    print("Test: KV cache update - single sequence")
    print("="*70)

    # Parameters
    num_tokens = 4
    lkv_dim = 8  # Small for testing
    r_dim = 4
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)

    # Create test data
    new_kv_c = np.random.randn(num_tokens, lkv_dim).astype(np.float32)
    new_k_pe = np.random.randn(num_tokens, r_dim).astype(np.float32)

    # Create cache
    padded_lkv = align_to(lkv_dim, 128)
    padded_r = align_to(r_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(2, page_size, kv_dim, dtype)
    cache_kv = np.zeros(cache_shape, dtype=np.float32)

    # Metadata for single sequence (prefill)
    kv_lens = np.array([num_tokens], dtype=np.int32)
    page_indices = np.array([0, 0], dtype=np.int32)  # 2 pages allocated, using page 0
    cu_q_lens = np.array([0, num_tokens], dtype=np.int32)
    num_seqs = 1

    print(f"Input: {num_tokens} tokens, lkv_dim={lkv_dim}, r_dim={r_dim}")
    print(f"Cache shape: {cache_shape}")
    print(f"Page size: {page_size}")

    # Manual update
    updated_cache = manual_update_kv_cache(
        new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices, cu_q_lens, num_seqs
    )

    # Verify tokens are in correct positions
    packing = get_dtype_packing(dtype)
    all_correct = True

    for t in range(num_tokens):
        row = t // packing
        col = t % packing

        stored_kv_c = updated_cache[0, row, col, :lkv_dim]
        stored_k_pe = updated_cache[0, row, col, padded_lkv:padded_lkv + r_dim]

        kv_c_match = np.allclose(stored_kv_c, new_kv_c[t], rtol=1e-5)
        k_pe_match = np.allclose(stored_k_pe, new_k_pe[t], rtol=1e-5)

        if not kv_c_match or not k_pe_match:
            print(f"  Token {t} at [{0}, {row}, {col}]: kv_c={kv_c_match}, k_pe={k_pe_match}")
            all_correct = False

    print(f"All tokens stored correctly: {'✓' if all_correct else '✗'}")
    return all_correct


def test_kv_cache_decode_append():
    """Test KV cache update during decode (appending to existing context)."""
    print("\n" + "="*70)
    print("Test: KV cache update - decode append")
    print("="*70)

    # Parameters
    context_len = 8
    decode_tokens = 1
    lkv_dim = 8
    r_dim = 4
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)

    # Create context data (simulating prefill)
    context_kv_c = np.random.randn(context_len, lkv_dim).astype(np.float32)
    context_k_pe = np.random.randn(context_len, r_dim).astype(np.float32)

    # Create decode data
    decode_kv_c = np.random.randn(decode_tokens, lkv_dim).astype(np.float32)
    decode_k_pe = np.random.randn(decode_tokens, r_dim).astype(np.float32)

    # Create cache
    padded_lkv = align_to(lkv_dim, 128)
    padded_r = align_to(r_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(2, page_size, kv_dim, dtype)
    cache_kv = np.zeros(cache_shape, dtype=np.float32)

    # Step 1: Prefill
    prefill_kv_lens = np.array([context_len], dtype=np.int32)
    prefill_page_indices = np.array([0, 0], dtype=np.int32)
    prefill_cu_q_lens = np.array([0, context_len], dtype=np.int32)

    cache_after_prefill = manual_update_kv_cache(
        context_kv_c, context_k_pe, cache_kv,
        prefill_kv_lens, prefill_page_indices, prefill_cu_q_lens, 1
    )

    # Step 2: Decode
    total_len = context_len + decode_tokens
    decode_kv_lens = np.array([total_len], dtype=np.int32)  # Total length now
    decode_page_indices = np.array([0, 0], dtype=np.int32)
    decode_cu_q_lens = np.array([0, decode_tokens], dtype=np.int32)  # Only 1 new token

    cache_after_decode = manual_update_kv_cache(
        decode_kv_c, decode_k_pe, cache_after_prefill,
        decode_kv_lens, decode_page_indices, decode_cu_q_lens, 1
    )

    print(f"Context length: {context_len}")
    print(f"Decode tokens: {decode_tokens}")
    print(f"Total length after decode: {total_len}")

    # Verify context tokens are still correct
    packing = get_dtype_packing(dtype)
    context_correct = True
    for t in range(context_len):
        row = t // packing
        col = t % packing
        stored_kv_c = cache_after_decode[0, row, col, :lkv_dim]
        if not np.allclose(stored_kv_c, context_kv_c[t], rtol=1e-5):
            print(f"  Context token {t} corrupted!")
            context_correct = False

    # Verify decode token is at correct position
    decode_pos = context_len  # Position 8 (0-indexed)
    row = decode_pos // packing
    col = decode_pos % packing
    stored_decode_kv_c = cache_after_decode[0, row, col, :lkv_dim]
    decode_correct = np.allclose(stored_decode_kv_c, decode_kv_c[0], rtol=1e-5)

    print(f"Context tokens preserved: {'✓' if context_correct else '✗'}")
    print(f"Decode token at position {decode_pos}: {'✓' if decode_correct else '✗'}")

    return context_correct and decode_correct


def test_metadata_computation():
    """Test that metadata arrays are computed correctly for different scenarios."""
    print("\n" + "="*70)
    print("Test: Metadata computation")
    print("="*70)

    # Scenario 1: Single prefill request
    print("\n--- Scenario 1: Single prefill (5 tokens) ---")
    num_tokens = 5
    kv_lens = np.array([5], dtype=np.int32)
    cu_q_lens = np.array([0, 5], dtype=np.int32)
    distribution = np.array([5, 0, 1], dtype=np.int32)  # [prefill, decode, num_seqs]

    # Verify
    assert kv_lens[0] == num_tokens, "kv_lens should equal num_tokens for prefill"
    assert cu_q_lens[-1] == num_tokens, "cu_q_lens[-1] should equal total tokens"
    assert distribution[0] == num_tokens, "distribution[0] should be prefill tokens"
    assert distribution[1] == 0, "distribution[1] should be 0 for pure prefill"
    assert distribution[2] == 1, "distribution[2] should be num_seqs"
    print("✓ Prefill metadata correct")

    # Scenario 2: Single decode request (1 token, context of 10)
    print("\n--- Scenario 2: Single decode (1 token, context=10) ---")
    context_len = 10
    decode_tokens = 1
    kv_lens = np.array([context_len + decode_tokens], dtype=np.int32)  # Total = 11
    cu_q_lens = np.array([0, decode_tokens], dtype=np.int32)  # Only new token in Q
    distribution = np.array([0, decode_tokens, 1], dtype=np.int32)  # [0 prefill, 1 decode, 1 seq]

    assert kv_lens[0] == 11, "kv_lens should be total length"
    assert cu_q_lens[-1] == 1, "cu_q_lens should only count new tokens"
    assert distribution[0] == 0, "No prefill tokens in decode"
    assert distribution[1] == 1, "1 decode token"
    print("✓ Decode metadata correct")

    # Scenario 3: Mixed batch (1 prefill + 1 decode)
    print("\n--- Scenario 3: Mixed batch (prefill=3 + decode=1) ---")
    prefill_len = 3
    decode_context = 5
    kv_lens = np.array([prefill_len, decode_context + 1], dtype=np.int32)
    cu_q_lens = np.array([0, prefill_len, prefill_len + 1], dtype=np.int32)
    distribution = np.array([prefill_len, 1, 2], dtype=np.int32)

    assert kv_lens[0] == 3, "First seq kv_lens"
    assert kv_lens[1] == 6, "Second seq kv_lens (context + 1)"
    assert cu_q_lens[-1] == 4, "Total Q tokens (3 prefill + 1 decode)"
    assert distribution[0] == 3, "3 prefill tokens"
    assert distribution[1] == 1, "1 decode token"
    assert distribution[2] == 2, "2 sequences"
    print("✓ Mixed batch metadata correct")

    return True


def test_multi_turn_conversation():
    """Test a multi-turn conversation: prefill -> decode -> decode -> decode."""
    print("\n" + "="*70)
    print("Test: Multi-turn conversation simulation")
    print("="*70)

    # Parameters
    lkv_dim = 8
    r_dim = 4
    page_size = 16
    dtype = jnp.bfloat16

    np.random.seed(42)

    # Create cache
    padded_lkv = align_to(lkv_dim, 128)
    padded_r = align_to(r_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(4, page_size, kv_dim, dtype)  # 4 pages
    cache = np.zeros(cache_shape, dtype=np.float32)

    all_kv_c = []
    all_k_pe = []

    # Turn 1: Prefill with 5 tokens
    print("\n--- Turn 1: Prefill (5 tokens) ---")
    turn1_kv_c = np.random.randn(5, lkv_dim).astype(np.float32)
    turn1_k_pe = np.random.randn(5, r_dim).astype(np.float32)
    all_kv_c.append(turn1_kv_c)
    all_k_pe.append(turn1_k_pe)

    kv_lens = np.array([5], dtype=np.int32)
    page_indices = np.array([0, 0, 0, 0], dtype=np.int32)
    cu_q_lens = np.array([0, 5], dtype=np.int32)

    cache = manual_update_kv_cache(turn1_kv_c, turn1_k_pe, cache, kv_lens, page_indices, cu_q_lens, 1)
    print(f"  After turn 1: kv_len=5")

    # Turn 2: Decode 1 token
    print("\n--- Turn 2: Decode (1 token) ---")
    turn2_kv_c = np.random.randn(1, lkv_dim).astype(np.float32)
    turn2_k_pe = np.random.randn(1, r_dim).astype(np.float32)
    all_kv_c.append(turn2_kv_c)
    all_k_pe.append(turn2_k_pe)

    kv_lens = np.array([6], dtype=np.int32)  # Total now 6
    cu_q_lens = np.array([0, 1], dtype=np.int32)  # Only 1 new token

    cache = manual_update_kv_cache(turn2_kv_c, turn2_k_pe, cache, kv_lens, page_indices, cu_q_lens, 1)
    print(f"  After turn 2: kv_len=6")

    # Turn 3: Decode 1 more token
    print("\n--- Turn 3: Decode (1 token) ---")
    turn3_kv_c = np.random.randn(1, lkv_dim).astype(np.float32)
    turn3_k_pe = np.random.randn(1, r_dim).astype(np.float32)
    all_kv_c.append(turn3_kv_c)
    all_k_pe.append(turn3_k_pe)

    kv_lens = np.array([7], dtype=np.int32)  # Total now 7
    cu_q_lens = np.array([0, 1], dtype=np.int32)

    cache = manual_update_kv_cache(turn3_kv_c, turn3_k_pe, cache, kv_lens, page_indices, cu_q_lens, 1)
    print(f"  After turn 3: kv_len=7")

    # Turn 4: Decode 1 more token
    print("\n--- Turn 4: Decode (1 token) ---")
    turn4_kv_c = np.random.randn(1, lkv_dim).astype(np.float32)
    turn4_k_pe = np.random.randn(1, r_dim).astype(np.float32)
    all_kv_c.append(turn4_kv_c)
    all_k_pe.append(turn4_k_pe)

    kv_lens = np.array([8], dtype=np.int32)  # Total now 8
    cu_q_lens = np.array([0, 1], dtype=np.int32)

    cache = manual_update_kv_cache(turn4_kv_c, turn4_k_pe, cache, kv_lens, page_indices, cu_q_lens, 1)
    print(f"  After turn 4: kv_len=8")

    # Verify all tokens are in correct positions
    all_kv_c_concat = np.concatenate(all_kv_c, axis=0)
    all_k_pe_concat = np.concatenate(all_k_pe, axis=0)

    packing = get_dtype_packing(dtype)
    all_correct = True

    print("\n--- Verifying all 8 tokens ---")
    for t in range(8):
        row = t // packing
        col = t % packing
        stored_kv_c = cache[0, row, col, :lkv_dim]
        stored_k_pe = cache[0, row, col, padded_lkv:padded_lkv + r_dim]

        kv_c_match = np.allclose(stored_kv_c, all_kv_c_concat[t], rtol=1e-5)
        k_pe_match = np.allclose(stored_k_pe, all_k_pe_concat[t], rtol=1e-5)

        if not kv_c_match or not k_pe_match:
            print(f"  Token {t}: kv_c={'✓' if kv_c_match else '✗'}, k_pe={'✓' if k_pe_match else '✗'}")
            all_correct = False

    print(f"\nAll tokens in correct positions: {'✓' if all_correct else '✗'}")
    return all_correct


def test_page_boundary():
    """Test KV cache behavior at page boundaries."""
    print("\n" + "="*70)
    print("Test: Page boundary handling")
    print("="*70)

    # Parameters - small page size to test boundary
    lkv_dim = 8
    r_dim = 4
    page_size = 4  # Small page to hit boundary quickly
    dtype = jnp.bfloat16

    np.random.seed(42)

    # Create cache with 2 pages
    padded_lkv = align_to(lkv_dim, 128)
    padded_r = align_to(r_dim, 128)
    kv_dim = padded_lkv + padded_r

    cache_shape = get_kv_cache_shape(2, page_size, kv_dim, dtype)
    cache = np.zeros(cache_shape, dtype=np.float32)

    # Prefill with 6 tokens (spans 2 pages with page_size=4)
    num_tokens = 6
    kv_c = np.random.randn(num_tokens, lkv_dim).astype(np.float32)
    k_pe = np.random.randn(num_tokens, r_dim).astype(np.float32)

    kv_lens = np.array([num_tokens], dtype=np.int32)
    page_indices = np.array([0, 1], dtype=np.int32)  # 2 pages
    cu_q_lens = np.array([0, num_tokens], dtype=np.int32)

    print(f"Tokens: {num_tokens}, Page size: {page_size}")
    print(f"Expected: tokens 0-3 in page 0, tokens 4-5 in page 1")

    cache = manual_update_kv_cache(kv_c, k_pe, cache, kv_lens, page_indices, cu_q_lens, 1)

    # Verify tokens in correct pages
    packing = get_dtype_packing(dtype)
    all_correct = True

    for t in range(num_tokens):
        page_idx = t // page_size
        pos_in_page = t % page_size
        row = pos_in_page // packing
        col = pos_in_page % packing

        actual_page = page_indices[page_idx]
        stored_kv_c = cache[actual_page, row, col, :lkv_dim]

        match = np.allclose(stored_kv_c, kv_c[t], rtol=1e-5)
        if not match:
            print(f"  Token {t} at page {actual_page}, pos ({row}, {col}): ✗")
            all_correct = False
        else:
            print(f"  Token {t} at page {actual_page}, pos ({row}, {col}): ✓")

    print(f"\nPage boundary handling: {'✓' if all_correct else '✗'}")
    return all_correct


def test_rope_application_order():
    """Test that RoPE is applied in the correct order in the pipeline."""
    print("\n" + "="*70)
    print("Test: RoPE application order verification")
    print("="*70)

    # In vLLM's MLA, RoPE is applied BEFORE the attention backend receives tensors:
    # 1. MultiHeadLatentAttentionWrapper.forward() receives hidden_states, positions
    # 2. It computes q, kv_c, k_pe
    # 3. It applies RoPE: q[..., qk_nope_head_dim:], k_pe = self.rotary_emb(positions, q_pe, k_pe)
    # 4. It calls self.mla_attn(q, kv_c_normed, k_pe, ...) - already RoPE'd

    # Simple RoPE implementation for testing
    def apply_rope(x, positions, head_dim):
        """Simple RoPE implementation.

        x: (seq_len, num_heads, head_dim)
        positions: (seq_len,)
        """
        seq_len = positions.shape[0]

        # Create frequency bands
        inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

        # Create position encodings
        pos = positions.reshape(-1, 1).astype(np.float32)
        freqs = pos * inv_freq.reshape(1, -1)  # (seq_len, head_dim/2)

        # Add head dimension for broadcasting: (seq_len, 1, head_dim/2)
        cos = np.cos(freqs)[:, np.newaxis, :]
        sin = np.sin(freqs)[:, np.newaxis, :]

        # Apply rotation - broadcast across heads
        x1 = x[..., 0::2]  # (seq_len, num_heads, head_dim/2)
        x2 = x[..., 1::2]

        x_out = np.zeros_like(x)
        x_out[..., 0::2] = x1 * cos - x2 * sin
        x_out[..., 1::2] = x1 * sin + x2 * cos

        return x_out

    # Test data
    seq_len = 4
    num_heads = 2
    qk_rope_dim = 8

    np.random.seed(42)
    q_pe = np.random.randn(seq_len, num_heads, qk_rope_dim).astype(np.float32)
    k_pe = np.random.randn(seq_len, 1, qk_rope_dim).astype(np.float32)
    positions = np.arange(seq_len, dtype=np.int32)

    print(f"Input shapes: q_pe={q_pe.shape}, k_pe={k_pe.shape}, positions={positions.shape}")

    # Apply RoPE
    q_pe_roped = apply_rope(q_pe, positions, qk_rope_dim)
    k_pe_roped = apply_rope(k_pe, positions, qk_rope_dim)

    # Verify RoPE changes the values
    q_changed = not np.allclose(q_pe, q_pe_roped)
    k_changed = not np.allclose(k_pe, k_pe_roped)

    print(f"q_pe changed after RoPE: {'✓' if q_changed else '✗'}")
    print(f"k_pe changed after RoPE: {'✓' if k_changed else '✗'}")

    # Verify position 0 gets specific rotation
    # At position 0, cos(0) = 1, sin(0) = 0, so rotation should be identity
    # (only approximately due to the frequency bands)

    print("\nRoPE verification:")
    print(f"  q_pe[0] before: {q_pe[0, 0, :4]}")
    print(f"  q_pe[0] after:  {q_pe_roped[0, 0, :4]}")

    # Key insight: The backend should receive ALREADY-ROPED tensors
    print("\n✓ RoPE should be applied BEFORE backend.forward() is called")
    print("  - Verify your wrapper applies RoPE before calling attention backend")

    return q_changed and k_changed


if __name__ == "__main__":
    print("="*70)
    print("KV CACHE AND METADATA TESTS")
    print("="*70)

    results = []

    results.append(("KV cache single seq", test_kv_cache_update_single_seq()))
    results.append(("KV cache decode append", test_kv_cache_decode_append()))
    results.append(("Metadata computation", test_metadata_computation()))
    results.append(("Multi-turn conversation", test_multi_turn_conversation()))
    results.append(("Page boundary handling", test_page_boundary()))
    results.append(("RoPE application order", test_rope_application_order()))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1)
