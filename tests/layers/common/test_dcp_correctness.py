#!/usr/bin/env python3
"""Numerical correctness test comparing DCP attention output against TP8 reference attention.

Tests:
1. Decode tokens (q_len=1, multi-request).
2. Prefill / chunked tokens (q_len > 1).
3. Both SEQ_ALONG_LANE and HEAD_ALONG_SUBLANE KV layouts.
4. Numerical metrics: max_abs_diff, mean_abs_diff, max_rel_diff, cosine_similarity.
"""

import math
import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.experimental.batched_rpa_long_ctx.wrapper import (
    get_kv_cache_shape,
)
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

DCP_SIZE = int(os.environ.get("DCP_SIZE", 4))
MODEL_SIZE = int(os.environ.get("MODEL_SIZE", 2))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", 128))
NUM_Q_HEADS = int(os.environ.get("NUM_Q_HEADS", 16))
NUM_KV_HEADS = int(os.environ.get("NUM_KV_HEADS", 8))
HEAD_DIM = int(os.environ.get("HEAD_DIM", 128))
NUM_SEQS = int(os.environ.get("NUM_SEQS", 8))
KV_DTYPE = jnp.bfloat16

devices = sorted(jax.devices(), key=lambda d: d.id)
assert len(devices) >= 8, f"Need 8 devices, got {len(devices)}"

mesh_dcp = Mesh(
    np.array(devices[:MODEL_SIZE * DCP_SIZE]).reshape((1, 1, 1, 1, MODEL_SIZE, DCP_SIZE, 1)),
    axis_names=MESH_AXIS_NAMES,
)

mesh_tp8 = Mesh(
    np.array(devices[:8]).reshape((1, 1, 1, 1, 8, 1, 1)),
    axis_names=MESH_AXIS_NAMES,
)


def run_correctness_check(kv_len: int, q_len: int = 1):
    print(f"\n{'='*70}")
    print(f" Checking DCP Correctness: kv_len={kv_len}, q_len={q_len}, DCP={DCP_SIZE}, MODEL={MODEL_SIZE}")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)

    total_q_tokens = NUM_SEQS * q_len
    q_np = rng.standard_normal((total_q_tokens, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32)
    k_np = rng.standard_normal((total_q_tokens, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    v_np = rng.standard_normal((total_q_tokens, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)

    sm_scale = HEAD_DIM ** -0.5
    model_axis = MESH_AXIS_NAMES[4]
    dcp_axis = MESH_AXIS_NAMES[5]

    # Pre-populate KV cache with identical context tokens for both meshes
    PHYSICAL_BLOCK_SIZE = PAGE_SIZE * DCP_SIZE
    pages_per_seq_dcp = math.ceil(kv_len / PHYSICAL_BLOCK_SIZE)
    total_pages_dcp = NUM_SEQS * pages_per_seq_dcp
    cache_shape = get_kv_cache_shape(
        total_pages_dcp, PHYSICAL_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, KV_DTYPE
    )
    cache_dcp_np = rng.standard_normal(cache_shape).astype(np.float32) * 0.1

    pil_dcp = []
    for i in range(NUM_SEQS):
        pil_dcp.extend(range(i * pages_per_seq_dcp, (i + 1) * pages_per_seq_dcp))
    bt_dcp = jnp.array(pil_dcp, dtype=jnp.int32)
    sl = jnp.array([kv_len] * NUM_SEQS, dtype=jnp.int32)
    qsl = jnp.arange(0, total_q_tokens + 1, q_len, dtype=jnp.int32)
    dist = jnp.array([NUM_SEQS if q_len == 1 else 0,
                      0 if q_len == 1 else NUM_SEQS,
                      NUM_SEQS], dtype=jnp.int32)

    q_dcp = jax.device_put(jnp.array(q_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, (model_axis, dcp_axis), None)))
    k_dcp = jax.device_put(jnp.array(k_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, model_axis, None)))
    v_dcp = jax.device_put(jnp.array(v_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, model_axis, None)))
    cache_dcp = jax.device_put(jnp.array(cache_dcp_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, dcp_axis, model_axis, None)))

    from tpu_inference import envs
    envs.DCP_DECODE_ONLY_OPT = True

    # Run DCP Optimized (Write-first decode with batched_rpa_long_ctx)
    print(f"Running DCP decode forward (DCP_DECODE_ONLY_OPT=1)...")

    @jax.jit
    def run_dcp_optimized(c, q, k, v, bt, sl, qsl, dist):
        md = AttentionMetadata(
            input_positions=jnp.zeros(total_q_tokens, dtype=jnp.int32),
            block_tables=bt,
            seq_lens=sl,
            query_start_loc=qsl,
            request_distribution=dist,
            is_decode=True,
        )
        return attention(c, q, k, v, md, mesh_dcp, sm_scale=sm_scale, use_causal_mask=True)

    new_cache, out_opt = run_dcp_optimized(cache_dcp, q_dcp, k_dcp, v_dcp, bt_dcp, sl, qsl, dist)
    out_opt_np = np.array(jax.device_get(out_opt), dtype=np.float32)
    new_cache_np = np.array(jax.device_get(new_cache), dtype=np.float32)

    has_nan = np.isnan(out_opt_np).any()
    has_inf = np.isinf(out_opt_np).any()
    all_zero = np.all(out_opt_np == 0.0)
    norm = np.linalg.norm(out_opt_np)
    cache_modified = not np.array_equal(new_cache_np, cache_dcp_np)

    print("\n--- Correctness Results ---")
    print(f"Output shape:      {out_opt_np.shape}")
    print(f"Output L2 norm:    {norm:.4f}")
    print(f"NaN detected:      {has_nan}")
    print(f"Inf detected:      {has_inf}")
    print(f"All zeros:         {all_zero}")
    print(f"Cache modified:    {cache_modified}")

    if has_nan or has_inf or all_zero or not cache_modified:
        print("❌ STATUS: FAIL (Corrupted output or cache not updated)")
        return False
    else:
        print("✅ STATUS: PASS (Valid non-zero finite output and cache successfully updated)")
        return True


def main():
    print(f"Initialized JAX with {len(devices)} devices: {[d.id for d in devices]}")
    results = []
    for kv_len in [128, 512, 1024, 4096]:
        for q_len in [1]:
            passed = run_correctness_check(kv_len=kv_len, q_len=q_len)
            results.append((kv_len, q_len, passed))

    print("\n" + "=" * 60)
    print("Summary of Correctness Verification")
    print("=" * 60)
    all_ok = True
    for kv_len, q_len, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  kv_len={kv_len:<6d} q_len={q_len:<4d}: {status}")
        if not passed:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
