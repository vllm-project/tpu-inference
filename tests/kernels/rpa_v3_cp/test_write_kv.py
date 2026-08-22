"""Unit tests for write_kv.py Pallas kernel.

Tests:
  1. Basic correctness: owned slots are written, others untouched.
  2. DCP rank 1 writes to odd positions.
  3. Padding sequences (kv_lens=0) are ignored.
  4. Multiple sequences with different kv_lens.
  5. Match against reference JAX implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    get_kv_cache_shape, merge_kv)
from tpu_inference.kernels.experimental.rpa_v3_cp.write_kv import write_decode_kv

# ── Config ────────────────────────────────────────────────────────────────────
PAGE_SIZE = 16   # small for easy hand-verification
DCP       = 2
KV_HEADS  = 1    # keep kv cache head dim minimal
HEAD_DIM  = 128


def make_cache(total_pages, page_size=PAGE_SIZE):
    shape = get_kv_cache_shape(total_pages, page_size, KV_HEADS, HEAD_DIM, jnp.bfloat16)
    return jnp.zeros(shape, dtype=jnp.bfloat16)


def ref_write(merged_kv, kv_cache, kv_lens, page_indices, cu_q_lens,
              distribution, cp_rank_val, cp_group_size):
    """Pure-Python reference: brute-force loop over real decode seqs."""
    cache = np.array(kv_cache)
    merged = np.array(merged_kv)
    kv_lens_np = np.array(kv_lens)
    pi_np = np.array(page_indices)
    cu_np = np.array(cu_q_lens)
    num_seqs = kv_lens_np.shape[0]
    pages_per_seq = pi_np.shape[0] // num_seqs
    local_page_size = kv_cache.shape[1]
    num_decode = int(distribution[0])

    for seq_i in range(num_decode):
        kv_len = int(kv_lens_np[seq_i])
        if kv_len == 0:
            continue
        write_pos = kv_len - 1
        if write_pos % cp_group_size != cp_rank_val:
            continue
        local_pos = (write_pos + cp_group_size - 1 - cp_rank_val) // cp_group_size
        kv_p  = local_pos // local_page_size
        kv_off = local_pos % local_page_size
        phys = int(pi_np[seq_i * pages_per_seq + kv_p])
        tok  = int(cu_np[seq_i])
        cache[phys, kv_off] = merged[tok]

    return jnp.array(cache)


def run_test(num_seqs, kv_lens_list, cp_rank_val, num_seqs_pad=None,
             pages_per_seq_override=None):
    """Helper: build inputs, run kernel and reference, compare."""
    if num_seqs_pad is None:
        num_seqs_pad = num_seqs
    max_kv = max(kv_lens_list) if kv_lens_list else 1
    pages_per_seq = pages_per_seq_override or ((max_kv + PAGE_SIZE - 1) // PAGE_SIZE)
    total_pages = num_seqs_pad * pages_per_seq

    # Local (DCP-sharded) page size
    local_ps = PAGE_SIZE // DCP

    # Build cache with per-slot sentinel values so we can detect spurious writes.
    cache_shape = get_kv_cache_shape(total_pages, local_ps, KV_HEADS, HEAD_DIM, jnp.bfloat16)
    # Fill cache with -1 so zeroed-out writes are distinguishable.
    kv_cache = jnp.full(cache_shape, -1, dtype=jnp.bfloat16)

    # One token per decode seq, each with a distinct value so we can track it.
    k = jnp.stack([jnp.full((KV_HEADS, HEAD_DIM), float(i+1), dtype=jnp.bfloat16)
                   for i in range(num_seqs_pad)], axis=0)
    v = k * 2
    merged = merge_kv(k, v)

    kv_lens = jnp.array(kv_lens_list + [0] * (num_seqs_pad - num_seqs),
                        dtype=jnp.int32)

    # Sequential page allocation; padding seqs get zeros.
    pil = []
    for i in range(num_seqs):
        pil.extend(range(i * pages_per_seq, (i + 1) * pages_per_seq))
    for _ in range(num_seqs_pad - num_seqs):
        pil.extend([0] * pages_per_seq)
    page_indices = jnp.array(pil, dtype=jnp.int32)

    cu_q_lens = jnp.arange(num_seqs_pad + 1, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    cp_rank = jnp.array([cp_rank_val], dtype=jnp.int32)

    # Compute reference before kernel call: donate_argnames invalidates kv_cache after.
    out_ref = ref_write(merged, kv_cache, kv_lens, page_indices, cu_q_lens,
                        distribution, cp_rank_val, DCP)

    out_kernel = write_decode_kv(
        merged, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
        cp_rank, cp_group_size=DCP)
    jax.block_until_ready(out_kernel)

    np.testing.assert_array_equal(
        np.array(out_kernel), np.array(out_ref),
        err_msg=f"Mismatch for num_seqs={num_seqs} cp_rank={cp_rank_val} "
                f"kv_lens={kv_lens_list}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_single_seq_rank0_even_position():
    """kv_len=4 → write_pos=3, 3%2=1 → rank 0 does NOT own; cache unchanged."""
    run_test(num_seqs=1, kv_lens_list=[4], cp_rank_val=0)


def test_single_seq_rank1_even_position():
    """kv_len=4 → write_pos=3, 3%2=1 → rank 1 DOES own."""
    run_test(num_seqs=1, kv_lens_list=[4], cp_rank_val=1)


def test_single_seq_rank0_odd_kv_len():
    """kv_len=5 → write_pos=4, 4%2=0 → rank 0 owns."""
    run_test(num_seqs=1, kv_lens_list=[5], cp_rank_val=0)


def test_multiple_seqs_rank0():
    """8 seqs with varying kv_lens, rank 0."""
    kv_lens = [4, 5, 6, 7, 8, 9, 10, 11]
    run_test(num_seqs=8, kv_lens_list=kv_lens, cp_rank_val=0)


def test_multiple_seqs_rank1():
    """8 seqs with varying kv_lens, rank 1."""
    kv_lens = [4, 5, 6, 7, 8, 9, 10, 11]
    run_test(num_seqs=8, kv_lens_list=kv_lens, cp_rank_val=1)


def test_padding_seqs_ignored():
    """Padded sequences (kv_lens=0) must not corrupt any cache slot."""
    run_test(num_seqs=4, kv_lens_list=[6, 8, 10, 12], cp_rank_val=0,
             num_seqs_pad=16)


def test_kv_len_at_page_boundary():
    """Write position exactly at a page boundary (local_pos % local_ps == 0)."""
    local_ps = PAGE_SIZE // DCP   # 8
    # write_pos = local_ps → local slot 0 of page 1 for rank 0
    kv_len = local_ps * DCP + 1  # write_pos = local_ps*2, even → rank 0
    run_test(num_seqs=1, kv_lens_list=[kv_len], cp_rank_val=0,
             pages_per_seq_override=4)


def test_kv_len_1():
    """Very first decode token: kv_len=1, write_pos=0, rank 0 owns."""
    run_test(num_seqs=1, kv_lens_list=[1], cp_rank_val=0)


def test_large_batch():
    """64 decode sequences padded to 256, rank 0."""
    rng = np.random.default_rng(42)
    kv_lens = (rng.integers(1, 4 * PAGE_SIZE, size=64)).tolist()
    run_test(num_seqs=64, kv_lens_list=kv_lens, cp_rank_val=0, num_seqs_pad=256,
             pages_per_seq_override=8)


def test_large_batch_rank1():
    """64 decode sequences padded to 256, rank 1."""
    rng = np.random.default_rng(7)
    kv_lens = (rng.integers(1, 4 * PAGE_SIZE, size=64)).tolist()
    run_test(num_seqs=64, kv_lens_list=kv_lens, cp_rank_val=1, num_seqs_pad=256,
             pages_per_seq_override=8)


if __name__ == "__main__":
    tests = [
        test_single_seq_rank0_even_position,
        test_single_seq_rank1_even_position,
        test_single_seq_rank0_odd_kv_len,
        test_multiple_seqs_rank0,
        test_multiple_seqs_rank1,
        test_padding_seqs_ignored,
        test_kv_len_at_page_boundary,
        test_kv_len_1,
        test_large_batch,
        test_large_batch_rank1,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
