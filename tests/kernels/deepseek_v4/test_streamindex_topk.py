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
"""Tests for Deepseek V4 StreamIndex Top-K kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from google3.experimental.users.hwanginho.deepseek_v4.streamindex_topk.streamindex_topk import \
        streamindex_topk
except ImportError:
    from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import \
        streamindex_topk


# =====================================================================
# Helper functions from the compressor implementation
# =====================================================================
def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


# =====================================================================

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def streamindex_topk_ref(
    q,
    weights,
    kv,
    block_table,
    T_list,
    S_list,
    cu_q_lens,
    k,
    comp_ratio,
    H_I,
    H_KV,
):
    """Naive NumPy reference implementation for StreamIndex Top-K."""
    num_tokens = q.shape[0]
    expected_topk = np.full((num_tokens, k), -1, dtype=np.int32)
    B = len(T_list)

    for b in range(B):
        T_seq = T_list[b]
        S_total = S_list[b]
        q_start = cu_q_lens[b]

        if T_seq == 0:
            continue

        S_valid = S_total // comp_ratio
        seq_blocks = block_table[b]
        seq_kv = np.concatenate([kv[p] for p in seq_blocks], axis=0)

        naive_scores = np.full((T_seq, max(S_valid, k)),
                               -np.inf,
                               dtype=np.float32)

        for t_idx in range(T_seq):
            global_t = q_start + t_idx
            q_abs_pos = (S_total - T_seq) + t_idx

            for s_idx in range(S_valid):
                if s_idx * comp_ratio > q_abs_pos:
                    continue

                score = 0.0
                for h in range(H_I):
                    h_kv = h // (H_I // H_KV)
                    inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
                    score += max(0.0, inner) * weights[global_t, h]

                naive_scores[t_idx, s_idx] = score

        seq_expected = np.argsort(-naive_scores, axis=-1)[:, :k]

        for t_idx in range(T_seq):
            for i in range(k):
                idx = seq_expected[t_idx, i]
                if naive_scores[t_idx, idx] == -np.inf:
                    expected_topk[q_start + t_idx, i] = -1
                else:
                    expected_topk[q_start + t_idx, i] = idx

    return expected_topk


@pytest.mark.parametrize(
    "B, num_tokens, page_size, max_blocks, H_I, H_KV, D, k, compression_ratio,"
    " bq_sz, bkv_p, seq_lens_list, cu_q_lens_list",
    [
        # Small standard case
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 2, [4, 6], [0, 3, 6]),
        # # Single token batch (Decode Phase)
        (1, 1, 16, 4, 2, 1, 32, 512, 1, 16, 1, [10], [0, 1]),
        # Large batch, multiple tokens per sequence (Prefill Phase)
        (
            4,
            20,
            32,
            16,
            8,
            1,
            128,
            512,
            4,
            64,
            4,
            [10, 20, 30, 40],
            [0, 5, 10, 15, 20],
        ),
        # Odd chunk sizes
        (2, 5, 8, 6, 4, 1, 16, 512, 1, 16, 3, [8, 12], [0, 2, 5]),
        # Single head for KV but multiple heads for Queries (MQA/GQA pattern)
        (2, 10, 16, 8, 8, 1, 64, 512, 2, 32, 4, [16, 24], [0, 4, 10]),
        # # High D, High K
        (1, 8, 16, 4, 2, 1, 256, 512, 1, 32, 4, [60], [0, 8]),
        # Mixed batch: sequence 0 has 1 token (decode), sequence 1 has 5 tokens
        # (prefill)
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 2, [4, 6], [0, 1, 6]),
    ],
)
def test_streamindex_topk_shape(
    B,
    num_tokens,
    page_size,
    max_blocks,
    H_I,
    H_KV,
    D,
    k,
    compression_ratio,
    bq_sz,
    bkv_p,
    seq_lens_list,
    cu_q_lens_list,
):
    """Tests the shape and basic execution bounds of streamindex_topk."""
    _ = H_KV
    print(f"\n{'-'*60}")
    print(f"SHAPE TEST: B={B}, Tokens={num_tokens}, k={k}")
    print(f"{'-'*60}")

    query_projection = jnp.zeros((num_tokens, H_I, D), dtype=jnp.float32)
    indexer_weights = jnp.zeros((num_tokens, H_I), dtype=jnp.float32)

    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128
    kv_cache = jnp.zeros((num_pages, page_size // 4, 4, width),
                         dtype=jnp.uint8)

    block_table = jnp.zeros((B, max_blocks), dtype=jnp.int32)
    page_indices = block_table.flatten()
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    print("Inputs:")
    print(f"  - query_projection: {query_projection.shape}")
    print(f"  - kv_cache:         {kv_cache.shape}")
    print(f"  - seq_lens:         {seq_lens_list}")
    print(f"  - cu_q_lens:        {cu_q_lens_list}")

    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while (num_decodes < B
           and (cu_q_lens_list[num_decodes + 1] - cu_q_lens_list[num_decodes])
           == 1):
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)
    expected_shape = (num_tokens, k)

    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)

    out_shape_idxs = jax.eval_shape(
        streamindex_topk,
        query_projection,
        indexer_weights,
        kv_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    print("\nOutputs:")
    print(f"  - Expected shape: {expected_shape}, dtype: int32")
    print(f"  - Actual shape:   {out_shape_idxs.shape}, dtype:"
          f" {out_shape_idxs.dtype}")

    assert out_shape_idxs.shape == expected_shape
    assert out_shape_idxs.dtype == jnp.int32
    print("Result: PASS")


@pytest.mark.parametrize(
    "B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, bq_sz, bkv_p,"
    " block_table_list",
    [
        # 1. Single sequence, highly fragmented block table
        (1, [4], [16], 8, 4, 1, 16, 1024, 2, 8, 2, [[2, 0]]),
        # 2. Batched Decode (T=1, B=2)
        (2, [1, 1], [12, 16], 8, 4, 1, 16, 1024, 1, 8, 1, [[1, 3], [0, 2]]),
        # 3. High GQA (8 Query Heads, 2 KV Heads)
        (1, [6], [20], 4, 8, 1, 16, 1024, 1, 8, 2, [[4, 1, 3, 0, 2]]),
        # 4. Multi-Batch Prefill with variable query/sequence lengths
        (2, [5, 3], [16, 16], 8, 4, 1, 16, 1024, 1, 8, 4, [[3, 1], [2, 4]]),
        # 5. Dummy sequences / Padding (B=3, but sequence 1 has 0 tokens)
        (
            3,
            [2, 0, 3],
            [16, 0, 8],
            8,
            4,
            1,
            16,
            1024,
            1,
            8,
            2,
            [[1, 2], [0, 0], [4, 3]],
        ),
        # 6. Mixed batch: sequence 0 has 1 token, sequence 1 has 5 tokens
        (2, [1, 5], [12, 16], 8, 4, 1, 16, 1024, 2, 8, 2, [[1, 3], [2, 0]]),
    ],
)
def test_streamindex_topk_numerical_correctness(
    B,
    T_list,
    S_list,
    page_size,
    H_I,
    H_KV,
    D,
    k,
    comp_ratio,
    bq_sz,
    bkv_p,
    block_table_list,
):
    """Executes randomized input data against a naive NumPy ground truth."""
    print(f"\n{'='*60}")
    print(f"BATCHED NUMERICAL TEST (B={B})")
    print(f"{'='*60}")

    np.random.seed(42)

    # 1. Setup Random Tensors
    num_tokens = sum(T_list)
    q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
    weights = np.random.uniform(-1.5, 1.5,
                                size=(num_tokens, H_I)).astype(np.float32)

    # Create a unified physical KV Cache pool large enough for all block indices
    max_physical_page = np.max(block_table_list)
    float32_kv = np.random.randn(max_physical_page + 1, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    cache_kv = np.zeros((max_physical_page + 1, page_size // 4, 4, width),
                        dtype=np.uint8)
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(max_physical_page + 1):
        for s in range(page_size):
            for h in range(H_KV):
                row_kv = float32_kv[p, s, h]
                # Quantize using D as block size (each head query key has dimension D)
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), D)
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h] = dq * float(scale_jax[0])
                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    block_table = np.array(block_table_list, dtype=np.int32)
    page_indices = block_table.flatten()
    seq_lens = np.array(S_list, dtype=np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)

    print("Configuration:")
    print(f"  - Tokens total: {num_tokens}, Sequences(B): {B}, K: {k}")
    print(f"  - GQA: {H_I} Query Heads -> {H_KV} KV Heads")
    print(f"  - Compression Ratio: {comp_ratio}")

    print("\nInputs Generated:")
    print(f"  - Query Tensor: {q.shape}")
    print(f"  - KV Cache:     {cache_kv.shape}")
    print(f"  - Block Table:  {block_table.tolist()}")

    # =====================================================================
    # 2. NAIVE COMPUTATION (The Ground Truth)
    # =====================================================================
    print("\n[1/3] Computing exact ground-truth using naive NumPy loops...")
    expected_topk = streamindex_topk_ref(
        q=q,
        weights=weights,
        kv=dequantized_kv,
        block_table=block_table,
        T_list=T_list,
        S_list=S_list,
        cu_q_lens=cu_q_lens,
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=H_KV,
    )

    print("\nGROUND TRUTH (Naive NumPy):")
    print(expected_topk)

    # =====================================================================
    # 3. Pallas KERNEL COMPUTATION
    # =====================================================================
    print(
        "\n[2/3] Executing optimized JAX Kernel (streamindex_pallas_topk)...")
    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while num_decodes < B and T_list[num_decodes] == 1:
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)

    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array(seq_lens),
        page_indices=jnp.array(page_indices),
        cu_q_lens=jnp.array(cu_q_lens),
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    print("\nACTUAL OUTPUT (JAX/XLA Kernel):")
    print(actual_topk_np)

    # =====================================================================
    # 4. VERIFY
    # =====================================================================
    print("\n[3/3] Verifying absolute correctness...")
    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
        err_msg="JAX Pallas Kernel Top-K math did not match Naive Ground Truth",
    )
    print(
        "MATCH VERIFIED! JAX kernel handles batches, fragmentation, and dummy"
        " padding.\n")


def test_streamindex_topk_quantized():
    """Verifies correctness of streamindex_topk on FP8 packed cache."""
    np.random.seed(42)

    T_seq = 4
    S_seq = 512
    page_size = 16
    H_I = 2
    H_KV = 1
    D = 128
    k = 512
    comp_ratio = 4
    bkv_p = 1
    bq_sz = 1

    q = np.random.randn(T_seq, H_I, D).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(T_seq, H_I)).astype(np.float32)

    S_valid = S_seq // comp_ratio
    # We need enough pages to hold S_valid compressed tokens.
    num_pages = (S_valid + page_size - 1) // page_size
    float32_kv = np.random.randn(num_pages, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    width = 256
    cache_kv = np.zeros((num_pages, page_size // 4, 4, width), dtype=np.uint8)

    # Dequantized KV for naive ground truth
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(num_pages):
        for s in range(page_size):
            for h_kv in range(H_KV):
                row_kv = float32_kv[p, s, h_kv]

                # Quantize using compressor helper
                # block_size = 128 since we want 1 scale factor for D=128
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), 128)

                # Convert to bytes
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))

                # Store dequantized version exactly as hardware sees it for accurate
                # validation
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h_kv] = dq * float(scale_jax[0])

                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    # Compute expected top-k using exact dequantized keys
    seq_kv = np.concatenate([dequantized_kv[p] for p in range(num_pages)],
                            axis=0)

    naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

    for t_idx in range(T_seq):
        for s_idx in range(S_valid):
            score = 0.0
            for h in range(H_I):
                h_kv = h // (H_I // H_KV)
                inner = np.dot(q[t_idx, h], seq_kv[s_idx, h_kv])
                score += max(0.0, inner) * weights[t_idx, h]
            naive_scores[t_idx, s_idx] = score

    expected_topk = np.argsort(-naive_scores, axis=-1)[:, :k]
    for t_idx in range(T_seq):
        for i in range(k):
            idx = expected_topk[t_idx, i]
            if naive_scores[t_idx, idx] == -np.inf:
                expected_topk[t_idx, i] = -1

    # Pallas parameters
    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array([S_seq], dtype=jnp.int32),
        page_indices=jnp.arange(num_pages, dtype=jnp.int32),
        cu_q_lens=jnp.array([0, T_seq], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
    )


def main(argv):
    del argv

    raise SystemExit(pytest.main([__file__, "-p", "no:cacheprovider"]))


if __name__ == "__main__":
    from absl import app

    app.run(main)
