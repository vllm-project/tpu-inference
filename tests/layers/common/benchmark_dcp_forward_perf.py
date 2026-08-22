#!/usr/bin/env python3

"""
Benchmark comparing DCP decode strategies against a pure-TP8 baseline.
Both configs use 8 devices total; the mesh determines the behavior.
The KV cache is allocated once outside the jit and passed via donate_argnums=(0,)
so the same physical buffer is reused each step via input_output_aliases in the
pallas_call.  Allocating inside the jit would cause XLA to emit a broadcast_in_dim
(zero-fill) of the multi-GB cache on every call, dominating benchmark latency.

When VLLM_DCP_PROJ_REPLICATE=1, an additional benchmark variant runs that includes
explicit q_proj / kv_proj matmuls (simulating replicated projection weights across
DCP ranks).  This measures the full cost of the replicate strategy: redundant
projection compute avoids the K/V all-gather inside shard_map.
"""

import math, os, time
os.environ["NEW_MODEL_DESIGN"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import get_kv_cache_shape
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
DCP_SIZE      = int(os.environ.get("DCP_SIZE",       2))
MODEL_SIZE    = int(os.environ.get("MODEL_SIZE",      4))  # TP within DCP mesh
PAGE_SIZE     = int(os.environ.get("PAGE_SIZE",     128))  # local tokens/page
NUM_Q_HEADS   = int(os.environ.get("NUM_Q_HEADS",    32))
NUM_KV_HEADS  = int(os.environ.get("NUM_KV_HEADS",    4))
HEAD_DIM      = int(os.environ.get("HEAD_DIM",      128))
D_MODEL       = int(os.environ.get("D_MODEL",      4096))  # hidden dim for proj
NUM_SEQS_REAL = int(os.environ.get("NUM_SEQS_REAL",  64))
MAX_NUM_SEQS  = int(os.environ.get("MAX_NUM_SEQS",  64))
WARMUP         = int(os.environ.get("WARMUP",           5))
BENCH          = int(os.environ.get("BENCH",          100))
PROFILE_STEPS  = int(os.environ.get("PROFILE_STEPS",    5))
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", "profiles/decode_bench")
KV_DTYPE = jnp.bfloat16
PHYSICAL_BLOCK_SIZE = PAGE_SIZE * DCP_SIZE  # global page size, sharded by CONTEXT=dcp
TP_MODEL = int(os.environ.get("TP_MODEL", 8))
_KV_LENS_K   = os.environ.get("KV_LENS_K", "4,32,128,256,512")
DECODE_KV_LENS = [int(k) * 1024 for k in _KV_LENS_K.split(",")]
PROJ_REPLICATE = os.environ.get("VLLM_DCP_PROJ_REPLICATE", "0") == "1"

# ── Meshes ────────────────────────────────────────────────────────────────────
devices = sorted(jax.devices(), key=lambda d: d.id)
assert len(devices) >= max(MODEL_SIZE * DCP_SIZE, TP_MODEL), (
    f"Need {max(MODEL_SIZE * DCP_SIZE, TP_MODEL)} devices, got {len(devices)}"
)

# DCP: attention() sees dcp>1 → routes to forward_decode
mesh = Mesh(
    np.array(devices[:MODEL_SIZE * DCP_SIZE]).reshape((1, 1, 1, 1, MODEL_SIZE, DCP_SIZE, 1)),
    axis_names=MESH_AXIS_NAMES,
)

# TP baseline: attention() sees dcp=1 → routes to sharded_ragged_paged_attention
mesh_tp8 = Mesh(
    np.array(devices[:TP_MODEL]).reshape((1, 1, 1, 1, TP_MODEL, 1, 1)),
    axis_names=MESH_AXIS_NAMES,
)

# ── Pre-sharding Helpers ───────────────────────────────────────────────────────
def preshard_inputs(q, k, v, bt, sl, qsl, dist, current_mesh):
    """Puts Q, K, V and metadata on the mesh with explicit sharding."""
    model_axis = MESH_AXIS_NAMES[4]
    dcp_axis   = MESH_AXIS_NAMES[5]
    put = lambda x, s: jax.device_put(x, NamedSharding(current_mesh, s))
    tp_size = current_mesh.shape['model'] * current_mesh.shape['dcp']
    if tp_size > 1:
        num_kv_heads = k.shape[1]
    if num_kv_heads < tp_size:
        if tp_size % num_kv_heads != 0:
            raise ValueError(f"For GQA/MQA, tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}")
        factor = tp_size // num_kv_heads
        k = jnp.repeat(k, factor, axis=1)
        v = jnp.repeat(v, factor, axis=1)

    # Q is sharded by both model and dcp axes to match production (QKV projection
    # output is sharded by ATTN_HEAD = ('model', 'expert', 'dcp')).  The context
    # phase needs Q all-gathered across dcp before attending to L/2 context; that
    # all-gather is emitted automatically by shard_map when the in_spec (KV_HEAD,
    # model-only) is coarser than this sharding.  K/V are decode new tokens and
    # stay replicated across dcp (local slice in shard_map, no collective needed).
    q_s = put(q, P(None, (model_axis, dcp_axis), None))
    kv_s = P(None, model_axis, None)
    k_s = put(k, kv_s)
    v_s = put(v, kv_s)

    rep = P()
    bt_s   = put(bt, rep)
    sl_s   = put(sl, rep)
    qsl_s  = put(qsl, rep)
    dist_s = put(dist, rep)

    return q_s, k_s, v_s, bt_s, sl_s, qsl_s, dist_s


def preshard_proj_inputs(bt, sl, qsl, dist, current_mesh):
    """Puts projection weights and hidden states on the DCP mesh for PROJ_REPLICATE.

    With PROJ_REPLICATE, Q and K/V projections are replicated across DCP so that
    every DCP rank has the full model-shard head slice after the matmul.  This
    avoids the K/V all-gather inside shard_map at the cost of redundant compute.
    """
    model_axis = MESH_AXIS_NAMES[4]
    put = lambda x, s: jax.device_put(x, NamedSharding(current_mesh, s))

    # Synthetic hidden states: (tokens, d_model), replicated across all devices.
    hidden = jnp.zeros((MAX_NUM_SEQS, D_MODEL), dtype=KV_DTYPE)
    hidden_s = put(hidden, P(None, None))

    # Projection weights sharded by model only (replicated across DCP).
    # Shape convention: (d_model, num_heads, head_dim) — heads axis sharded.
    q_w = jnp.zeros((D_MODEL, NUM_Q_HEADS,  HEAD_DIM), dtype=KV_DTYPE)
    k_w = jnp.zeros((D_MODEL, NUM_KV_HEADS, HEAD_DIM), dtype=KV_DTYPE)
    v_w = jnp.zeros((D_MODEL, NUM_KV_HEADS, HEAD_DIM), dtype=KV_DTYPE)
    q_w_s = put(q_w, P(None, model_axis, None))
    k_w_s = put(k_w, P(None, model_axis, None))
    v_w_s = put(v_w, P(None, model_axis, None))

    rep = P()
    bt_s   = put(bt, rep)
    sl_s   = put(sl, rep)
    qsl_s  = put(qsl, rep)
    dist_s = put(dist, rep)

    return hidden_s, q_w_s, k_w_s, v_w_s, bt_s, sl_s, qsl_s, dist_s


# ── Inputs Generator ──────────────────────────────────────────────────────────
def _make_inputs_raw(global_kv_len, page_size):
    num_tokens = MAX_NUM_SEQS
    q = jnp.zeros((num_tokens, NUM_Q_HEADS, HEAD_DIM), dtype=KV_DTYPE)
    k = jnp.zeros((num_tokens, NUM_KV_HEADS, HEAD_DIM), dtype=KV_DTYPE)
    v = jnp.zeros((num_tokens, NUM_KV_HEADS, HEAD_DIM), dtype=KV_DTYPE)

    pages_per_seq = math.ceil(global_kv_len / page_size)
    pil = []
    for i in range(MAX_NUM_SEQS):
        pil.extend(range(i * pages_per_seq, (i + 1) * pages_per_seq))
    block_tables = jnp.array(pil, dtype=jnp.int32)

    seq_lens = jnp.array(
        [global_kv_len] * NUM_SEQS_REAL + [0] * (MAX_NUM_SEQS - NUM_SEQS_REAL),
        dtype=jnp.int32,
    )
    query_start_loc = jnp.arange(MAX_NUM_SEQS + 1, dtype=jnp.int32)
    request_distribution = jnp.array(
        [NUM_SEQS_REAL, NUM_SEQS_REAL, NUM_SEQS_REAL], dtype=jnp.int32
    )
    return q, k, v, block_tables, seq_lens, query_start_loc, request_distribution

# ── Benchmark Engine ──────────────────────────────────────────────────────────
def _bench(label: str, fn, cache, args, width: int = 62):
    # Warmup
    for _ in range(1 + WARMUP):
        cache, out = fn(cache, *args)
        cache.block_until_ready()
        out.block_until_ready()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH):
        cache, out = fn(cache, *args)
    cache.block_until_ready()
    out.block_until_ready()
    us = (time.perf_counter() - t0) * 1e6 / BENCH
    print(f"  {label:<{width}s}  {us:8.1f} us  ({us / 1e3:.3f} ms)", flush=True)
    return cache

def _profile_fn(label: str, fn, cache, args, trace_dir: str):
    if PROFILE_STEPS <= 0:
        return cache
    os.makedirs(trace_dir, exist_ok=True)

    # Warmup prior to tracing
    cache, out = fn(cache, *args)
    cache.block_until_ready()
    out.block_until_ready()

    jax.profiler.start_trace(trace_dir)
    for i in range(PROFILE_STEPS):
        with jax.profiler.TraceAnnotation(f"{label}_step{i}"):
            cache, out = fn(cache, *args)
            cache.block_until_ready()
            out.block_until_ready()
    jax.profiler.stop_trace()
    print(f"  xprof → {trace_dir}", flush=True)
    return cache

# ── Bench Core ────────────────────────────────────────────────────────────────
def bench_kv_len(global_kv_len: int):
    kv_k = global_kv_len // 1024
    sm_scale = HEAD_DIM ** -0.5
    model_axis = MESH_AXIS_NAMES[4]
    dcp_axis = MESH_AXIS_NAMES[5]
    print(f"\n  ── KV={kv_k}K  DCP: {kv_k // DCP_SIZE}K/shard (no dup)  TP{TP_MODEL}: {kv_k}K/dev ──")
    print(">>")

    # ==========================================================================
    #  1. TP8 JIT Definition & Run
    # ==========================================================================
    pages_per_seq_tp8 = math.ceil(global_kv_len / PAGE_SIZE)
    total_pages_tp8 = MAX_NUM_SEQS * pages_per_seq_tp8
    factor = max(1, TP_MODEL // NUM_KV_HEADS)
    tp8_cache_shape = get_kv_cache_shape(
        total_pages_tp8, PAGE_SIZE, NUM_KV_HEADS * factor, HEAD_DIM, KV_DTYPE
    )
    cache_sharding_tp8 = NamedSharding(mesh_tp8, P(None, None, model_axis, None))

    @jax.jit
    def init_tp8_cache():
        cache = jnp.zeros(tp8_cache_shape, KV_DTYPE)
        return jax.lax.with_sharding_constraint(cache, cache_sharding_tp8)

    tp8_cache = init_tp8_cache()
    jax.block_until_ready(tp8_cache)

    q8_raw, k8_raw, v8_raw, bt8_raw, sl8_raw, qsl8_raw, dist8_raw = _make_inputs_raw(global_kv_len, PAGE_SIZE)
    tp8_args = preshard_inputs(q8_raw, k8_raw, v8_raw, bt8_raw, sl8_raw, qsl8_raw, dist8_raw, mesh_tp8)

    @jax.jit(donate_argnums=(0,))
    def tp8_fn(cache, q, k, v, bt, sl, qsl, dist):
        md = AttentionMetadata(
            input_positions=jnp.zeros(MAX_NUM_SEQS, dtype=jnp.int32),
            block_tables=bt,
            seq_lens=sl,
            query_start_loc=qsl,
            request_distribution=dist,
            # is_decode=False,
        )
        updated_cache, out = attention(cache, q, k, v, md, mesh_tp8, sm_scale=sm_scale, use_causal_mask=True)
        return updated_cache, out

    tp8_cache = _bench(f"tp{TP_MODEL}  model={TP_MODEL},dcp=1", tp8_fn, tp8_cache, tp8_args)
    tp8_cache = _profile_fn("tp8", tp8_fn, tp8_cache, tp8_args, f"{OUTPUT_DIR}/kv{kv_k}K/tp8")

    # Explicitly free TP8 memory before allocating DCP cache to avoid OOM.
    tp8_cache.delete()
    del tp8_cache, tp8_args

    # ==========================================================================
    #  2. DCP JIT Definition & Run
    # ==========================================================================
    pages_per_seq_dcp = math.ceil(global_kv_len / PHYSICAL_BLOCK_SIZE)
    total_pages_dcp = MAX_NUM_SEQS * pages_per_seq_dcp
    dcp_cache_shape = get_kv_cache_shape(
        total_pages_dcp, PHYSICAL_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, KV_DTYPE
    )
    cache_sharding_dcp = NamedSharding(mesh, P(None, dcp_axis, model_axis, None))

    @jax.jit
    def init_dcp_cache():
        cache = jnp.zeros(dcp_cache_shape, KV_DTYPE)
        return jax.lax.with_sharding_constraint(cache, cache_sharding_dcp)

    dcp_cache = init_dcp_cache()
    jax.block_until_ready(dcp_cache)

    q_raw, k_raw, v_raw, bt_raw, sl_raw, qsl_raw, dist_raw = _make_inputs_raw(global_kv_len, PHYSICAL_BLOCK_SIZE)
    dcp_args = preshard_inputs(q_raw, k_raw, v_raw, bt_raw, sl_raw, qsl_raw, dist_raw, mesh)

    def make_dcp_fn(decode_only_opt: bool):
        os.environ["DCP_DECODE_ONLY_OPT"] = "1" if decode_only_opt else ""
        os.environ.pop("VLLM_DCP_PROJ_REPLICATE", None)  # ensure regular DCP path
        @jax.jit(donate_argnums=(0,))
        def fn(cache, q, k, v, bt, sl, qsl, dist):
            md = AttentionMetadata(
                input_positions=jnp.zeros(MAX_NUM_SEQS, dtype=jnp.int32),
                block_tables=bt,
                seq_lens=sl,
                query_start_loc=qsl,
                request_distribution=dist,
                is_decode=True,
            )
            updated_cache, out = attention(cache, q, k, v, md, mesh, sm_scale=sm_scale, use_causal_mask=True)
            return updated_cache, out
        return fn

    for decode_only_opt, label, tag in [
        (True,  "DCP decode_only_opt  (write-first)", "decode_only"),
        # (False, "DCP pure dcp_forward (two-phase)",   "dcp_forward"),
    ]:
        dcp_fn = make_dcp_fn(decode_only_opt)
        dcp_cache = _bench(label, dcp_fn, dcp_cache, dcp_args)
        dcp_cache = _profile_fn(tag, dcp_fn, dcp_cache, dcp_args, f"{OUTPUT_DIR}/kv{kv_k}K/{tag}")

    # ==========================================================================
    #  3. DCP + PROJ_REPLICATE: q_proj / kv_proj step included in timing
    #     Each DCP rank holds the full model-shard heads after projection →
    #     no all-gather needed inside shard_map for K/V or Q.
    # ==========================================================================
    if PROJ_REPLICATE:
        proj_args = preshard_proj_inputs(bt_raw, sl_raw, qsl_raw, dist_raw, mesh)

        os.environ["DCP_DECODE_ONLY_OPT"] = "1"
        os.environ["VLLM_DCP_PROJ_REPLICATE"] = "1"

        @jax.jit(donate_argnums=(0,))
        def dcp_proj_fn(cache, hidden, q_w, k_w, v_w, bt, sl, qsl, dist):
            # q_proj step: (tokens, d_model) × (d_model, q_heads/model, head_dim)
            q = jnp.einsum('td,dhf->thf', hidden, q_w)
            # kv_proj step: (tokens, d_model) × (d_model, kv_heads/model, head_dim)
            k = jnp.einsum('td,dhf->thf', hidden, k_w)
            v = jnp.einsum('td,dhf->thf', hidden, v_w)
            md = AttentionMetadata(
                input_positions=jnp.zeros(MAX_NUM_SEQS, dtype=jnp.int32),
                block_tables=bt,
                seq_lens=sl,
                query_start_loc=qsl,
                request_distribution=dist,
                is_decode=True,
            )
            updated_cache, out = attention(cache, q, k, v, md, mesh, sm_scale=sm_scale, use_causal_mask=True)
            return updated_cache, out

        label = f"DCP proj_replicate  (q_proj+kv_proj+attn, no kv-all-gather)"
        dcp_cache = _bench(label, dcp_proj_fn, dcp_cache, proj_args)
        dcp_cache = _profile_fn("proj_replicate", dcp_proj_fn, dcp_cache, proj_args,
                                f"{OUTPUT_DIR}/kv{kv_k}K/proj_replicate")

        # Restore flag so subsequent kv_len iterations behave correctly.
        os.environ.pop("VLLM_DCP_PROJ_REPLICATE", None)

    # Free DCP cache before the next kv_len iteration to avoid OOM.
    dcp_cache.delete()
    del dcp_cache, dcp_args

def main():
    print("=" * 80, flush=True)
    print(f"  forward_decode  —  DCP vs TP{TP_MODEL} [No-Init-Overhead]")
    print("=" * 80, flush=True)
    print(f"  q_heads={NUM_Q_HEADS}  kv_heads={NUM_KV_HEADS}  head_dim={HEAD_DIM}"
          f"  DCP: phys_block={PHYSICAL_BLOCK_SIZE}  TP{TP_MODEL}: page={PAGE_SIZE}", flush=True)
    print(f"  real_seqs={NUM_SEQS_REAL}  padded={MAX_NUM_SEQS}  warmup={WARMUP}  bench={BENCH}", flush=True)
    if PROJ_REPLICATE:
        print(f"  VLLM_DCP_PROJ_REPLICATE=1  d_model={D_MODEL}  (proj+attn timing for DCP)", flush=True)
    print(flush=True)

    for kv_len in DECODE_KV_LENS:
        print(kv_len)
        bench_kv_len(kv_len)
    print(flush=True)

if __name__ == "__main__":
    main()
