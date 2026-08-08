#!/usr/bin/env python3
"""Benchmark for _jax_write_decode_kv implementations.

Compares:
  (A) fori_loop over distribution[0] real seqs  (current impl)
  (B) vectorized scatter + mode='drop'           (proposed)
  (C) Pallas DMA kernel write_decode_kv          (rpa_v3_cp)

Usage:
  NEW_MODEL_DESIGN=1 python3 benchmarks/write_decode_kv_benchmark.py
"""

import os, time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, Mesh
import numpy as np

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    get_kv_cache_shape, merge_kv)
from tpu_inference.kernels.experimental.rpa_v3_cp.write_kv import write_decode_kv

# ── Config ────────────────────────────────────────────────────────────────────
DCP_SIZE     = int(os.environ.get("DCP_SIZE",    2))
TP_SIZE      = int(os.environ.get("TP_SIZE",     8))
NUM_KV_HEADS = int(os.environ.get("NUM_KV_HEADS", 4))
HEAD_DIM     = int(os.environ.get("HEAD_DIM",   128))
PAGE_SIZE    = int(os.environ.get("PAGE_SIZE",  128))
NUM_SEQS_PAD = int(os.environ.get("NUM_SEQS_PAD", 256))
NUM_DECODE   = int(os.environ.get("NUM_DECODE",   64))
KV_LEN       = int(os.environ.get("KV_LEN",    4096))  # incl new decode token
WARMUP       = int(os.environ.get("WARMUP",       5))
BENCH        = int(os.environ.get("BENCH",      200))

KV_DTYPE      = jnp.bfloat16
PAGES_PER_SEQ = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE

# ── Mesh: only dcp axis matters for the write kernel ─────────────────────────
devices = jax.devices()
assert len(devices) >= DCP_SIZE, f"Need {DCP_SIZE} devices, got {len(devices)}"
mesh = Mesh(np.array(devices[:DCP_SIZE]), axis_names=('dcp',))

# kv_cache sharded along page_size dim by dcp; everything else replicated.
CACHE_SPEC = P(None, 'dcp', None, None, None)
REP        = P()   # fully replicated
DCP_SPEC   = P('dcp')

# ── Data ──────────────────────────────────────────────────────────────────────
def make_inputs():
    rng = np.random.default_rng(42)

    total_pages = NUM_SEQS_PAD * PAGES_PER_SEQ
    cache_shape = get_kv_cache_shape(
        total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, KV_DTYPE)
    kv_cache = jnp.zeros(cache_shape, dtype=KV_DTYPE)

    # one token per decode seq
    k = jnp.array(rng.standard_normal((NUM_DECODE, NUM_KV_HEADS, HEAD_DIM)),
                  dtype=KV_DTYPE)
    v = jnp.array(rng.standard_normal((NUM_DECODE, NUM_KV_HEADS, HEAD_DIM)),
                  dtype=KV_DTYPE)

    kv_lens = jnp.array(
        [KV_LEN] * NUM_DECODE + [0] * (NUM_SEQS_PAD - NUM_DECODE),
        dtype=jnp.int32)

    # Real seqs: sequential pages.  Padding seqs: all point to page 0.
    pil = []
    for i in range(NUM_DECODE):
        pil.extend(range(i * PAGES_PER_SEQ, (i + 1) * PAGES_PER_SEQ))
    for _ in range(NUM_SEQS_PAD - NUM_DECODE):
        pil.extend([0] * PAGES_PER_SEQ)
    page_indices = jnp.array(pil, dtype=jnp.int32)

    # cu_q_lens[i] = i (one token per decode seq)
    cu_q_lens    = jnp.arange(NUM_SEQS_PAD + 1, dtype=jnp.int32)
    distribution = jnp.array([NUM_DECODE, NUM_DECODE, NUM_DECODE], dtype=jnp.int32)

    return kv_cache, k, v, kv_lens, page_indices, cu_q_lens, distribution


# ── Shard functions (run on each DCP device) ──────────────────────────────────

def _shard_fori(k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
                cp_rank_arr):
    merged = merge_kv(k, v)
    lps    = cache.shape[1]           # local page size = PAGE_SIZE // DCP_SIZE
    cp     = cp_rank_arr[0]
    ns     = kv_lens.shape[0]
    pps    = page_indices.shape[0] // ns
    nt     = merged.shape[0]
    dcp    = DCP_SIZE

    def write_one(seq_i, state):
        kv_len    = kv_lens[seq_i]
        write_pos = kv_len - 1
        local_pos = (write_pos + dcp - 1 - cp) // dcp
        kv_p      = local_pos // lps
        kv_off    = local_pos % lps
        page      = page_indices[seq_i * pps + kv_p]
        tok       = jnp.minimum(cu_q_lens[seq_i], jnp.int32(nt - 1))
        own       = (write_pos % dcp == cp)
        val       = jnp.where(own, merged[tok], state[page, kv_off])
        return state.at[page, kv_off].set(val)

    return jax.lax.fori_loop(0, distribution[0], write_one, cache)


def _shard_drop(k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
                cp_rank_arr):
    merged = merge_kv(k, v)
    lps    = cache.shape[1]
    cp     = cp_rank_arr[0]
    ns     = kv_lens.shape[0]
    pps    = page_indices.shape[0] // ns
    nt     = merged.shape[0]
    dcp    = DCP_SIZE

    seq_idx   = jnp.arange(ns, dtype=jnp.int32)
    write_pos = jnp.maximum(kv_lens, jnp.int32(1)) - 1
    local_pos = (write_pos + dcp - 1 - cp) // dcp
    kv_p      = jnp.minimum(local_pos // lps, jnp.int32(pps - 1))
    kv_off    = local_pos % lps
    pages     = page_indices[seq_idx * pps + kv_p]
    tok_idx   = jnp.minimum(cu_q_lens[seq_idx], jnp.int32(nt - 1))

    should_write = (seq_idx < distribution[0]) & (write_pos % dcp == cp)
    # OOB page → mode='drop' discards non-owner writes; no old_val read needed
    safe_pages   = jnp.where(should_write, pages, jnp.int32(cache.shape[0]))
    return cache.at[safe_pages, kv_off].set(merged[tok_idx], mode='drop')


def _shard_pallas(k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
                  cp_rank_arr):
    merged = merge_kv(k, v)
    return write_decode_kv(
        merged, cache, kv_lens, page_indices, cu_q_lens, distribution,
        cp_rank_arr,
        cp_group_size=DCP_SIZE,
    )


# ── Build jitted shard_map functions ─────────────────────────────────────────

def _make_fn(shard_fn, mesh, k, v, kv_lens, page_indices, cu_q_lens,
             distribution):
    cp_rank_global = jnp.arange(DCP_SIZE, dtype=jnp.int32)

    @jax.jit
    def fn(cache):
        return jax.shard_map(
            shard_fn, mesh=mesh,
            in_specs=(REP, REP, CACHE_SPEC, REP, REP, REP, REP, DCP_SPEC),
            out_specs=CACHE_SPEC,
            check_vma=False,
        )(k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
          cp_rank_global)

    return fn


# ── Timing ────────────────────────────────────────────────────────────────────

def time_fn(label, fn, cache_init):
    out = fn(cache_init)
    for _ in range(WARMUP - 1):
        out = fn(cache_init)
    jax.block_until_ready(out)

    @jax.jit
    def run_bench(c):
        return jax.lax.fori_loop(0, BENCH, lambda _, c: fn(c), c)

    out = run_bench(cache_init)
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    out = run_bench(cache_init)
    jax.block_until_ready(out)
    us = (time.perf_counter() - t0) * 1e6 / BENCH
    print(f"  {label:<40s}  {us:8.1f} us  ({us/1e3:.4f} ms)", flush=True)
    return us


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72, flush=True)
    print("  write_decode_kv benchmark  (fori_loop vs vectorized+drop)")
    print("=" * 72, flush=True)
    print(f"  dcp={DCP_SIZE}  kv_heads={NUM_KV_HEADS}  head_dim={HEAD_DIM}  "
          f"page_size={PAGE_SIZE}", flush=True)
    print(f"  num_seqs_pad={NUM_SEQS_PAD}  num_decode={NUM_DECODE}  "
          f"kv_len={KV_LEN}  pages_per_seq={PAGES_PER_SEQ}", flush=True)
    print(f"  warmup={WARMUP}  bench={BENCH}", flush=True)
    print(flush=True)

    kv_cache, k, v, kv_lens, page_indices, cu_q_lens, distribution = make_inputs()

    fn_fori   = _make_fn(_shard_fori,   mesh, k, v, kv_lens, page_indices,
                         cu_q_lens, distribution)
    fn_drop   = _make_fn(_shard_drop,   mesh, k, v, kv_lens, page_indices,
                         cu_q_lens, distribution)
    fn_pallas = _make_fn(_shard_pallas, mesh, k, v, kv_lens, page_indices,
                         cu_q_lens, distribution)

    print("  [correctness]", flush=True)
    out_fori   = fn_fori(kv_cache)
    out_drop   = fn_drop(kv_cache)
    out_pallas = fn_pallas(kv_cache)
    jax.block_until_ready(out_fori)
    jax.block_until_ready(out_drop)
    jax.block_until_ready(out_pallas)

    diff_drop = float(jnp.max(jnp.abs(
        out_fori.astype(jnp.float32) - out_drop.astype(jnp.float32))))
    diff_pallas = float(jnp.max(jnp.abs(
        out_fori.astype(jnp.float32) - out_pallas.astype(jnp.float32))))
    print(f"  max|fori - drop|   = {diff_drop:.6f}  "
          f"({'MATCH' if diff_drop == 0 else 'MISMATCH !!!'})", flush=True)
    print(f"  max|fori - pallas| = {diff_pallas:.6f}  "
          f"({'MATCH' if diff_pallas == 0 else 'MISMATCH !!!'})", flush=True)
    print(flush=True)

    print("  [timing]", flush=True)
    time_fn("fori_loop (current)",        fn_fori,   kv_cache)
    time_fn("vectorized+drop (proposed)", fn_drop,   kv_cache)
    time_fn("pallas write_decode_kv",     fn_pallas, kv_cache)
    print(flush=True)


if __name__ == "__main__":
    main()
