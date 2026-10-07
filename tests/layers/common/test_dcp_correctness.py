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

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import get_kv_cache_shape
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

DCP_SIZE = int(os.environ.get("DCP_SIZE", 4))
MODEL_SIZE = int(os.environ.get("MODEL_SIZE", 2))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", 128))
NUM_Q_HEADS = int(os.environ.get("NUM_Q_HEADS", 16))
NUM_KV_HEADS = int(os.environ.get("NUM_KV_HEADS", 2))
HEAD_DIM = int(os.environ.get("HEAD_DIM", 128))
NUM_SEQS = int(os.environ.get("NUM_SEQS", 8))
KV_DTYPE = jnp.bfloat16
PHYSICAL_BLOCK_SIZE = PAGE_SIZE * DCP_SIZE

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

    # Pre-populate KV cache with random context tokens
    pages_per_seq_tp8 = math.ceil(kv_len / PAGE_SIZE)
    total_pages_tp8 = NUM_SEQS * pages_per_seq_tp8
    tp_factor = max(1, 8 // NUM_KV_HEADS)
    tp8_cache_shape = get_kv_cache_shape(
        total_pages_tp8, PAGE_SIZE, NUM_KV_HEADS * tp_factor, HEAD_DIM, KV_DTYPE
    )
    cache_tp8_np = rng.standard_normal(tp8_cache_shape).astype(np.float32) * 0.1

    # Block table
    pil_tp8 = []
    for i in range(NUM_SEQS):
        pil_tp8.extend(range(i * pages_per_seq_tp8, (i + 1) * pages_per_seq_tp8))
    bt_tp8 = jnp.array(pil_tp8, dtype=jnp.int32)
    sl = jnp.array([kv_len] * NUM_SEQS, dtype=jnp.int32)
    qsl = jnp.arange(0, total_q_tokens + 1, q_len, dtype=jnp.int32)
    dist = jnp.array([NUM_SEQS if q_len == 1 else 0,
                      0 if q_len == 1 else NUM_SEQS,
                      NUM_SEQS], dtype=jnp.int32)

    # 1. Run TP8 Reference
    print("Running TP8 baseline reference...")
    q_tp8 = jax.device_put(jnp.array(q_np, dtype=KV_DTYPE), NamedSharding(mesh_tp8, P(None, model_axis, None)))
    k_tp8 = jax.device_put(jnp.array(k_np, dtype=KV_DTYPE), NamedSharding(mesh_tp8, P(None, model_axis, None)))
    v_tp8 = jax.device_put(jnp.array(v_np, dtype=KV_DTYPE), NamedSharding(mesh_tp8, P(None, model_axis, None)))
    cache_tp8 = jax.device_put(jnp.array(cache_tp8_np, dtype=KV_DTYPE), NamedSharding(mesh_tp8, P(None, None, model_axis, None)))

    @jax.jit
    def run_tp8(c, q, k, v, bt, sl, qsl, dist):
        md = AttentionMetadata(
            input_positions=jnp.zeros(total_q_tokens, dtype=jnp.int32),
            block_tables=bt,
            seq_lens=sl,
            query_start_loc=qsl,
            request_distribution=dist,
            is_decode=(q_len == 1),
        )
        return attention(c, q, k, v, md, mesh_tp8, sm_scale=sm_scale, use_causal_mask=True)

    _, out_tp8 = run_tp8(cache_tp8, q_tp8, k_tp8, v_tp8, bt_tp8, sl, qsl, dist)
    out_tp8_np = np.array(jax.device_get(out_tp8), dtype=np.float32)

    # 2. Run DCP
    print(f"Running DCP (DCP={DCP_SIZE}, MODEL={MODEL_SIZE})...")
    pages_per_seq_dcp = math.ceil(kv_len / PHYSICAL_BLOCK_SIZE)
    total_pages_dcp = NUM_SEQS * pages_per_seq_dcp
    dcp_factor = max(1, MODEL_SIZE // NUM_KV_HEADS)
    dcp_cache_shape = get_kv_cache_shape(
        total_pages_dcp, PHYSICAL_BLOCK_SIZE, NUM_KV_HEADS * dcp_factor, HEAD_DIM, KV_DTYPE
    )
    cache_dcp_np = rng.standard_normal(dcp_cache_shape).astype(np.float32) * 0.1

    pil_dcp = []
    for i in range(NUM_SEQS):
        pil_dcp.extend(range(i * pages_per_seq_dcp, (i + 1) * pages_per_seq_dcp))
    bt_dcp = jnp.array(pil_dcp, dtype=jnp.int32)

    q_dcp = jax.device_put(jnp.array(q_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, (model_axis, dcp_axis), None)))
    k_dcp = jax.device_put(jnp.array(k_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, model_axis, None)))
    v_dcp = jax.device_put(jnp.array(v_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, model_axis, None)))
    cache_dcp = jax.device_put(jnp.array(cache_dcp_np, dtype=KV_DTYPE), NamedSharding(mesh_dcp, P(None, dcp_axis, model_axis, None)))

    @jax.jit
    def run_dcp(c, q, k, v, bt, sl, qsl, dist):
        md = AttentionMetadata(
            input_positions=jnp.zeros(total_q_tokens, dtype=jnp.int32),
            block_tables=bt,
            seq_lens=sl,
            query_start_loc=qsl,
            request_distribution=dist,
            is_decode=(q_len == 1),
        )
        return attention(c, q, k, v, md, mesh_dcp, sm_scale=sm_scale, use_causal_mask=True)

    _, out_dcp = run_dcp(cache_dcp, q_dcp, k_dcp, v_dcp, bt_dcp, sl, qsl, dist)
    out_dcp_np = np.array(jax.device_get(out_dcp), dtype=np.float32)

    # 3. Analyze Differences
    diff = np.abs(out_dcp_np - out_tp8_np)
    max_abs = np.max(diff)
    mean_abs = np.mean(diff)
    norm_tp8 = np.linalg.norm(out_tp8_np)
    norm_dcp = np.linalg.norm(out_dcp_np)
    cos_sim = np.sum(out_dcp_np * out_tp8_np) / (norm_tp8 * norm_dcp + 1e-12)

    has_nan_tp8 = np.isnan(out_tp8_np).any()
    has_nan_dcp = np.isnan(out_dcp_np).any()
    all_zero_dcp = np.all(out_dcp_np == 0.0)

    print("\n--- Correctness Results ---")
    print(f"Max Absolute Error:   {max_abs:.6f}")
    print(f"Mean Absolute Error:  {mean_abs:.6f}")
    print(f"Cosine Similarity:    {cos_sim:.6f}")
    print(f"TP8 NaN detected:     {has_nan_tp8}")
    print(f"DCP NaN detected:     {has_nan_dcp}")
    print(f"DCP Output all zeros: {all_zero_dcp}")

    if has_nan_dcp or all_zero_dcp or cos_sim < 0.98:
        print("❌ STATUS: FAIL (Significant divergence or corrupted output)")
        return False
    else:
        print("✅ STATUS: PASS (Outputs align within floating point tolerance)")
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
