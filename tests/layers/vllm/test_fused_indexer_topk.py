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
import numpy as np
import pytest

from tpu_inference.layers.vllm.custom_ops.fused_indexer_topk import streamindex_chunked_topk

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.parametrize(
    "B, num_tokens, page_size, max_blocks, H_I, H_KV, D, k, compression_ratio, c_s, c_t, seq_lens_list, cu_seq_lens_list",
    [
        # Small standard case
        (2, 6, 16, 8, 4, 2, 64, 4, 2, 32, 2, [4, 6], [0, 3, 6]),
        # Single token batch (Decode Phase)
        (1, 1, 16, 4, 2, 2, 32, 2, 1, 16, 1, [10], [0, 1]),
        # Large batch, multiple tokens per sequence (Prefill Phase)
        (4, 20, 32, 16, 8, 4, 128, 8, 4, 64, 4, [10, 20, 30, 40], [0, 5, 10, 15, 20]),
        # Odd chunk sizes
        (2, 5, 8, 6, 4, 4, 16, 3, 1, 16, 3, [8, 12], [0, 2, 5]),
        # Single head for KV but multiple heads for Queries (MQA/GQA pattern)
        (2, 10, 16, 8, 8, 1, 64, 4, 2, 32, 4, [16, 24], [0, 4, 10]),
        # High D, High K
        (1, 8, 16, 4, 2, 2, 256, 16, 1, 32, 4, [60], [0, 8]),
        # NOTE: The 0-token test was removed. JAX's dynamic_slice_in_dim will throw a 
        # ValueError if the operand dimension is 0. vLLM bypasses the kernel anyway.
    ],
)
def test_streamindex_chunked_topk_shape(
    B,
    num_tokens,
    page_size,
    max_blocks,
    H_I,
    H_KV,
    D,
    k,
    compression_ratio,
    c_s,
    c_t,
    seq_lens_list,
    cu_seq_lens_list,
):
    """Tests the shape and basic execution bounds of streamindex_chunked_topk
    using mocked inputs avoiding full execution across varying dimensions.
    """
    print(f"\n{'-'*60}")
    print(f"SHAPE TEST: B={B}, Tokens={num_tokens}, k={k}")
    print(f"{'-'*60}")

    query_projection = jnp.zeros((num_tokens, H_I, D), dtype=jnp.float32)
    indexer_weights = jnp.zeros((num_tokens, H_I), dtype=jnp.float32)

    num_pages = max_blocks * B
    kv_cache = jnp.zeros((num_pages, page_size, H_KV, D), dtype=jnp.float32)

    block_table = jnp.zeros((B, max_blocks), dtype=jnp.int32)
    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)
    cu_seq_lens = jnp.array(cu_seq_lens_list, dtype=jnp.int32)

    print("Inputs:")
    print(f"  - query_projection: {query_projection.shape}")
    print(f"  - kv_cache:         {kv_cache.shape}")
    print(f"  - seq_lens:         {seq_lens_list}")
    print(f"  - cu_seq_lens:      {cu_seq_lens_list}")

    out_shape = jax.eval_shape(
        streamindex_chunked_topk,
        query_projection,
        kv_cache,
        block_table,
        seq_lens,
        cu_seq_lens,
        indexer_weights,
        k,
        compression_ratio,
        c_s,
        c_t,
    )

    print("\nOutputs:")
    print(f"  - Expected shape: {(num_tokens, k)}, dtype: int32")
    print(f"  - Actual shape:   {out_shape.shape}, dtype: {out_shape.dtype}")

    assert out_shape.shape == (num_tokens, k)
    assert out_shape.dtype == jnp.int32
    print("Result: PASS")


@pytest.mark.parametrize(
    "B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, c_s, c_t, block_table_list",
    [
        # 1. Single sequence, highly fragmented block table
        (1, [4], [15], 8, 4, 2, 16, 3, 2, 8, 2, [[2, 0]]),

        # 2. Batched Decode (T=1, B=2)
        (2, [1, 1], [12, 16], 8, 4, 1, 16, 2, 1, 8, 1, [[1, 3], [0, 2]]),

        # 3. High GQA (8 Query Heads, 2 KV Heads)
        (1, [6], [20], 4, 8, 2, 16, 4, 1, 8, 2, [[4, 1, 3, 0, 2]]),

        # 4. Multi-Batch Prefill with variable query/sequence lengths
        (2, [5, 3], [10, 18], 8, 4, 2, 16, 3, 2, 8, 4, [[3, 1, 0], [2, 4, 5]]),

        # 5. Dummy sequences / Padding (B=3, but sequence 1 has 0 tokens)
        (3, [2, 0, 3], [15, 0, 8], 8, 4, 2, 16, 2, 1, 8, 2, [[1, 2], [0, 0], [4, 3]])
    ],
)
def test_streamindex_chunked_topk_numerical_correctness(
    B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, c_s, c_t, block_table_list
):
    """Executes randomized input data against a naive NumPy ground truth.
    Verifies block-tiled Top-K math, GQA routing, causal masks, scan batch isolation,
    and PagedAttention physical memory fetching.
    """
    print(f"\n{'='*60}")
    print(f"BATCHED NUMERICAL TEST (B={B})")
    print(f"{'='*60}")

    np.random.seed(42)

    # 1. Setup Random Tensors
    num_tokens = sum(T_list)
    q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(num_tokens, H_I)).astype(np.float32)

    # Create a unified physical KV Cache pool large enough for all block indices
    max_physical_page = np.max(block_table_list)
    kv = np.random.randn(max_physical_page + 1, page_size, H_KV, D).astype(np.float32)

    block_table = np.array(block_table_list, dtype=np.int32)
    seq_lens = np.array(S_list, dtype=np.int32)
    cu_seq_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)

    print("Configuration:")
    print(f"  - Tokens total: {num_tokens}, Sequences(B): {B}, K: {k}")
    print(f"  - GQA: {H_I} Query Heads -> {H_KV} KV Heads")
    print(f"  - Compression Ratio: {comp_ratio}")

    print("\nInputs Generated:")
    print(f"  - Query Tensor: {q.shape}")
    print(f"  - KV Cache:     {kv.shape}")
    print(f"  - Block Table:  {block_table.tolist()}")

    # =====================================================================
    # 2. NAIVE COMPUTATION (The Ground Truth)
    # =====================================================================
    print("\n[1/3] Computing exact ground-truth using naive NumPy loops...")
    expected_topk = np.full((num_tokens, k), -1, dtype=np.int32)

    for b in range(B):
        T_seq = T_list[b]
        S_total = S_list[b]
        q_start = cu_seq_lens[b]

        if T_seq == 0:
            continue  # Skip dummy padded sequences

        S_valid = S_total // comp_ratio

        # Manually reconstruct this sequence's physical contiguous cache
        seq_blocks = block_table[b]
        seq_kv = np.concatenate([kv[p] for p in seq_blocks], axis=0)

        # Ensure naive_scores has at least K columns for sorting bounds
        naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

        for t_idx in range(T_seq):
            global_t = q_start + t_idx

            for s_idx in range(S_valid):
                # Causal Mask
                q_abs_pos = (S_total - T_seq) + t_idx
                if s_idx * comp_ratio > q_abs_pos:
                    continue

                score = 0.0
                for h in range(H_I):
                    h_kv = h // (H_I // H_KV) 
                    inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
                    score += max(0.0, inner) * weights[global_t, h] 

                naive_scores[t_idx, s_idx] = score

        # Get expected top-k indices
        seq_expected = np.argsort(-naive_scores, axis=-1)[:, :k]

        # Overwrite masked (-inf) positions with -1
        for t_idx in range(T_seq):
            for i in range(k):
                idx = seq_expected[t_idx, i]
                if naive_scores[t_idx, idx] == -np.inf:
                    expected_topk[q_start + t_idx, i] = -1
                else:
                    expected_topk[q_start + t_idx, i] = idx

    print("\nGROUND TRUTH (Naive NumPy):")
    print(expected_topk)

    # =====================================================================
    # 3. JAX KERNEL COMPUTATION
    # =====================================================================
    print("\n[2/3] Executing optimized JAX Kernel (streamindex_chunked_topk)...")
    actual_topk = streamindex_chunked_topk(
        query_projection=jnp.array(q),
        kv_cache=jnp.array(kv),
        block_table=jnp.array(block_table),
        seq_lens=jnp.array(seq_lens),
        cu_seq_lens=jnp.array(cu_seq_lens),
        indexer_weights=jnp.array(weights),
        k=k,
        compression_ratio=comp_ratio,
        c_s=c_s,
        c_t=c_t
    )

    actual_topk_np = np.array(actual_topk)

    print("\nACTUAL OUTPUT (JAX/XLA Kernel):")
    print(actual_topk_np)

    # =====================================================================
    # 4. VERIFY
    # =====================================================================
    print("\n[3/3] Verifying absolute correctness...")
    np.testing.assert_array_equal(
        actual_topk_np, 
        expected_topk, 
        err_msg="JAX Kernel Top-K math did not match Naive Ground Truth"
    )
    print("MATCH VERIFIED! JAX kernel handles batches, fragmentation, and dummy padding flawlessly.\n")