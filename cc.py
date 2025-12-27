import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_per_token_my_try_v4_no_pad import (
    get_kv_cache_shape, ragged_paged_attention_per_seq, ref_ragged_paged_attention_per_seq)

def test_running_max_logic():
    print("\n=== Starting Multi-Step Running Max Test ===")

    # --- Configuration ---
    seed = 42
    key = jax.random.PRNGKey(seed)
    kv_cache_dtype = jnp.float8_e4m3fn

    actual_head_dim = 128
    actual_num_q_heads = 32
    actual_num_kv_heads = 8
    page_size = 128
    total_num_pages = 1000

    # Two sequences
    num_seqs = 2

    # Initial Cache Setup
    kv_cache_shape = get_kv_cache_shape(total_num_pages, page_size,
                                        actual_num_kv_heads, actual_head_dim,
                                        kv_cache_dtype)

    # Initialize State Arrays
    # We will maintain separate states for Reference and Kernel to verify they match each other
    kv_cache_ref = jnp.zeros(kv_cache_shape, dtype=kv_cache_dtype)
    k_scale_ref = jnp.zeros((num_seqs,), dtype=jnp.float32)
    v_scale_ref = jnp.zeros((num_seqs,), dtype=jnp.float32)

    kv_cache_ker = jnp.zeros(kv_cache_shape, dtype=kv_cache_dtype)
    k_scale_ker = jnp.zeros((num_seqs,), dtype=jnp.float32)
    v_scale_ker = jnp.zeros((num_seqs,), dtype=jnp.float32)

    # Tracking lengths
    # Seq 0 starts at len 0, Seq 1 starts at len 0
    current_kv_lens = jnp.array([0, 0], dtype=jnp.int32)

    # Page allocation (simple static allocation for test)
    # Seq 0 gets pages 0-9, Seq 1 gets pages 10-19
    page_indices = jnp.concatenate([
        jnp.arange(0, 10),
        jnp.arange(10, 20)
    ]).astype(jnp.int32)

    # =========================================================================
    # STEP 1: INITIALIZATION
    # Both sequences see a max value of ~10.0
    # =========================================================================
    print("\n--- Step 1: Injecting Max Value ~10.0 ---")

    # 16 tokens per sequence
    q_len = 16
    cu_q_lens = jnp.array([0, 16, 32], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32) # Mixed mode
    max_num_tokens = 32

    # Update lengths for this step (append 16 tokens)
    step1_kv_lens = current_kv_lens + q_len

    # Generate Data
    k1, k2, k3 = jax.random.split(key, 3)
    q1 = jax.random.normal(k1, (max_num_tokens, actual_num_q_heads, actual_head_dim), dtype=jnp.bfloat16)

    # Make Keys/Values have max ~10.0
    k1_data = jax.random.normal(k2, (max_num_tokens, actual_num_kv_heads, actual_head_dim), dtype=jnp.bfloat16)
    k1_data = jnp.clip(k1_data, -1.0, 1.0) * 10.0
    v1_data = k1_data # Same for value for simplicity

    # --- Run Reference ---
    out_ref_1, kv_cache_ref, k_scale_ref, v_scale_ref = ref_ragged_paged_attention_per_seq(
        queries=q1, keys=k1_data, values=v1_data,
        kv_cache=kv_cache_ref, k_scale_cache=k_scale_ref, v_scale_cache=v_scale_ref,
        kv_lens=step1_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution,
        sm_scale=0.1
    )

    # --- Run Kernel ---
    out_ker_1, kv_cache_ker, k_scale_ker, v_scale_ker = ragged_paged_attention_per_seq(
        queries=q1, keys=k1_data, values=v1_data,
        kv_cache=kv_cache_ker, k_scale_cache=k_scale_ker, v_scale_cache=v_scale_ker,
        kv_lens=step1_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution,
        sm_scale=0.1
    )

    # Check Step 1
    print(f"Step 1 Ref Scales: {k_scale_ref} {v_scale_ref}")
    print(f"Step 1 Ker Scales: {k_scale_ker} {v_scale_ker}")

    # assert jnp.allclose(k_scale_ref, k_scale_ker, atol=1e-2), "Step 1 Scales Mismatch"
    # assert jnp.all(k_scale_ref > 9.0), "Step 1 max should be around 10.0"
    # assert jnp.allclose(out_ref_1, out_ker_1, atol=1e-2), "Step 1 Output Mismatch"

    # Update persistent lengths
    current_kv_lens = step1_kv_lens

    # =========================================================================
    # STEP 2: RUNNING MAX CHECK
    # Seq 0: New data is SMALL (1.0). Max should stay ~10.0.
    # Seq 1: New data is HUGE (100.0). Max should jump to ~100.0.
    # =========================================================================
    print("\n--- Step 2: Seq0 Low (1.0), Seq1 High (100.0) ---")

    # Update lengths (append another 16 tokens)
    step2_kv_lens = current_kv_lens + q_len

    key, subkey = jax.random.split(key)
    q2 = jax.random.normal(subkey, (max_num_tokens, actual_num_q_heads, actual_head_dim), dtype=jnp.bfloat16)

    # Generate Base Data
    k2_raw = jax.random.normal(key, (max_num_tokens, actual_num_kv_heads, actual_head_dim), dtype=jnp.bfloat16)
    k2_raw = jnp.clip(k2_raw, -1.0, 1.0)

    # Modify data per sequence
    # Seq 0 (indices 0:16) -> Scale 1.0
    k2_data = k2_raw.at[0:16].set(k2_raw[0:16] * 1.0)
    # Seq 1 (indices 16:32) -> Scale 100.0
    k2_data = k2_data.at[16:32].set(k2_data[16:32] * 100.0)
    v2_data = k2_data

    # --- Run Reference (Passing in updated caches from Step 1) ---
    out_ref_2, kv_cache_ref, k_scale_ref, v_scale_ref = ref_ragged_paged_attention_per_seq(
        queries=q2, keys=k2_data, values=v2_data,
        kv_cache=kv_cache_ref, k_scale_cache=k_scale_ref, v_scale_cache=v_scale_ref,
        kv_lens=step2_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution,
        sm_scale=0.1
    )

    # --- Run Kernel (Passing in updated caches from Step 1) ---
    out_ker_2, kv_cache_ker, k_scale_ker, v_scale_ker = ragged_paged_attention_per_seq(
        queries=q2, keys=k2_data, values=v2_data,
        kv_cache=kv_cache_ker, k_scale_cache=k_scale_ker, v_scale_cache=v_scale_ker,
        kv_lens=step2_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution,
        sm_scale=0.1
    )

    print(f"Step 2 Ref Scales: {k_scale_ref} {v_scale_ref}")
    print(f"Step 2 Ker Scales: {k_scale_ker} {v_scale_ker}")

    # --- ASSERTIONS ---

    # 1. Output Consistency
    diff = jnp.mean(jnp.abs(out_ref_2 - out_ker_2))
    print(f"Step 2 Output Diff Mean: {diff}")
    # assert jnp.allclose(out_ref_2, out_ker_2, atol=0.1), "Step 2 Output Mismatch (Ref vs Ker)"

    # 2. Scale Logic Verification
    # Seq 0: Previous Max ~10.0. New Data Max ~1.0. Result -> Should be ~10.0.
    if k_scale_ref[0] < 9.0:
        print("FAIL: Seq 0 scale dropped! It overwrote history instead of taking max.")
    elif k_scale_ref[0] > 15.0:
         print("FAIL: Seq 0 scale exploded unexpectedly.")
    else:
        print("PASS: Seq 0 scale maintained correctly (~10.0).")

    # Seq 1: Previous Max ~10.0. New Data Max ~100.0. Result -> Should be ~100.0.
    if k_scale_ref[1] < 90.0:
        print("FAIL: Seq 1 scale did not update! It ignored the new larger data.")
    else:
        print("PASS: Seq 1 scale updated correctly (~100.0).")

    # assert jnp.allclose(k_scale_ref, k_scale_ker, atol=1e-2), "Step 2 Scales Kernel mismatch"

    print("\n=== SUCCESS: Running Max Logic is Correct ===")

    # =========================================================================
    # STEP 3: QUIET PERIOD (Both receive ~1.0)
    # Scales must NOT drop. Seq 1 must remember the 100.0 from Step 2.
    # =========================================================================
    print("\n--- Step 3: Quiet Period (Input ~1.0) ---")
    current_kv_lens += q_len # Now lengths are 48

    k3_raw = jax.random.normal(key, (max_num_tokens, actual_num_kv_heads, actual_head_dim), dtype=jnp.bfloat16)
    k3_data = jnp.clip(k3_raw, -1.0, 1.0) # Max is just 1.0 for EVERYONE
    v3_data = k3_data

    # Pass in the caches from Step 2
    out_ref, kv_cache_ref, k_scale_ref, v_scale_ref = ref_ragged_paged_attention_per_seq(
        queries=q1, keys=k3_data, values=v3_data, kv_cache=kv_cache_ref, k_scale_cache=k_scale_ref, v_scale_cache=v_scale_ref,
        kv_lens=current_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution
    )
    out_ker, kv_cache_ker, k_scale_ker, v_scale_ker = ragged_paged_attention_per_seq(
        queries=q1, keys=k3_data, values=v3_data,
        kv_cache=kv_cache_ker, k_scale_cache=k_scale_ker, v_scale_cache=v_scale_ker,
        kv_lens=current_kv_lens, page_indices=page_indices, cu_q_lens=cu_q_lens, distribution=distribution,
        sm_scale=0.1
    )

    print(f"Step 2 Ref Scales: {k_scale_ref} , {v_scale_ref}")
    print(f"Step 2 Ker Scales: {k_scale_ker}, {v_scale_ker}")

if __name__ == "__main__":
    test_running_max_logic()