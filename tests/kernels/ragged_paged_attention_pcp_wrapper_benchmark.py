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
"""PCP prefill TTFT benchmark through the REAL attention wrapper.

Unlike ragged_paged_attention_pcp_ttft_benchmark.py (which calls the kernel
directly and re-implements the wrapper's orchestration), this drives
`attention_interface.pcp_ragged_paged_attention` itself, on the production
sharding: a 2D mesh (pcp x model), q/k/v sharded on ATTN_DATA/ATTN_HEAD, and the
KV cache sharded on (BATCH, KV_CONTEXT, KV_HEAD) -- note KV_CONTEXT shards the
*page_size* dim, so a physical block holds `block_size * pcp` tokens and each
rank owns `block_size`.

TTFT(N) = sum over chunks i of the attention latency of chunk i (queries
[i*CH,(i+1)*CH) attending KV [0,(i+1)*CH)). Latency is measured at every CH step
with a max-size cache and a dynamic `kv_lens`, so there is one compile per config.

Usage:
  python tests/kernels/ragged_paged_attention_pcp_wrapper_benchmark.py [MAXCTX] [CH]
"""
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.attention_interface import (
    pcp_ragged_paged_attention, ragged_paged_attention)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

# Force the N-D axis names (which carry `pcp`) regardless of NEW_MODEL_DESIGN.
sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

NQ, NKV, HD = 16, 4, 128
PAGE = 256  # per-rank block_size (the GLOBAL page_size dim is PAGE * pcp)
DTYPE = jnp.bfloat16
SM = HD**-0.5
KVP = get_dtype_packing(DTYPE)
NKV2 = align_to(2 * NKV, KVP)
MAX_SEQ = 4
WARMUP, ITERS = 1, 3
_rng = np.random.default_rng(0)
_rand = lambda s: jnp.array(_rng.random(size=s, dtype=np.float32)).astype(DTYPE)


def _bench(fn, *a):
    jax.block_until_ready(fn(*a))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / ITERS * 1e3


def _mesh(pcp, tp):
    shape = tuple(
        pcp if a == "pcp" else tp if a == "model" else 1
        for a in MESH_AXIS_NAMES)
    n = pcp * tp
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), MESH_AXIS_NAMES)


def make_pcp_wrapper(pcp, tp, chunk, max_ctx):
    """Drive attention_interface.pcp_ragged_paged_attention on the real mesh."""
    if jax.device_count() < pcp * tp:
        return lambda ctx: float("nan")
    mesh = _mesh(pcp, tp)
    two_p, C = 2 * pcp, chunk // (2 * pcp)
    # KV_CONTEXT shards the page_size dim -> global page holds PAGE*pcp tokens.
    gpage = PAGE * pcp
    npages = max(cdiv(max_ctx, gpage), 1)

    q = _rand((chunk, NQ, HD))
    k = _rand((chunk, NKV, HD))
    v = _rand((chunk, NKV, HD))
    cache = jnp.zeros((npages, gpage, NKV2 // KVP, KVP, HD), DTYPE)

    put = lambda x, s: jax.device_put(x, NamedSharding(mesh, s))
    q = put(q, P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None))
    k = put(k, P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None))
    v = put(v, P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None))
    cache = put(
        cache,
        P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
          ShardingAxisName.KV_HEAD, None, None))

    # seq0 and seq1 of the fused current phase are the SAME request; the kernel
    # indexes page_indices as `seq_idx * pages_per_seq` and the WRITING seq is
    # the tail (seq1), so seq1 needs a copy of the request's pages, not zeros.
    _pg = jnp.arange(npages, dtype=jnp.int32)
    pi = jnp.zeros((MAX_SEQ * npages, ), jnp.int32).at[:2 * npages].set(
        jnp.concatenate([_pg, _pg]))
    cu = jnp.zeros((MAX_SEQ + 1, ), jnp.int32).at[1:].set(C)
    dist = jnp.array([0, 0, 1], jnp.int32)      # cache phase: 1 seq
    pcp_dist = jnp.array([0, 0, 2], jnp.int32)  # current phase: head+tail

    # Per-rank fused current-phase metadata (mirrors _prepare_inputs).
    pcp_cu = np.zeros((pcp, MAX_SEQ + 1), np.int32)
    pcp_qp = np.zeros((pcp, MAX_SEQ), np.int32)
    for r in range(pcp):
        toff = (two_p - 1 - r) * C
        treal = int(np.clip(chunk - toff, 0, C))
        pcp_cu[r, 1] = C
        pcp_cu[r, 2:] = C + treal
        pcp_qp[r, 0] = r * C
        pcp_qp[r, 1] = toff
    pcp_spec = P(ShardingAxisName.PREFILL_CONTEXT, None)
    pcp_cu = put(jnp.asarray(pcp_cu), pcp_spec)
    pcp_qp = put(jnp.asarray(pcp_qp), pcp_spec)

    @jax.jit
    def fn(q, k, v, kvl, kvcl):
        cache = jnp.zeros((npages, gpage, NKV2 // KVP, KVP, HD), DTYPE)
        out, _ = pcp_ragged_paged_attention(mesh, q, k, v, cache, kvl, pi, cu,
                                            dist, kvcl, pcp_cu, pcp_qp,
                                            pcp_dist, sm_scale=SM,
                                            update_kv_cache=True,
                                            use_causal_mask=True)
        return out

    def measure(ctx):
        kvl = jnp.zeros((MAX_SEQ, ), jnp.int32).at[:2].set(ctx)
        kvcl = jnp.zeros((MAX_SEQ, ), jnp.int32).at[:2].set(ctx - chunk)
        return _bench(fn, q, k, v, kvl, kvcl)

    return measure


def make_tp(tp, chunk, max_ctx):
    """TP baseline: heads sharded /tp, full context. No cross-head collective,
    so per-device latency IS the config's latency -> measure on one device."""
    nq, nkv = NQ // tp, max(1, NKV // tp)
    nkv2 = align_to(2 * nkv, KVP)
    npages = max(cdiv(max_ctx, PAGE), 1)
    q, k, v = (_rand((chunk, nq, HD)), _rand((chunk, nkv, HD)),
               _rand((chunk, nkv, HD)))
    pi = jnp.arange(npages, dtype=jnp.int32)

    @jax.jit
    def fn(q, k, v, ctx):
        cache = jnp.zeros((npages, PAGE, nkv2 // KVP, KVP, HD), DTYPE)
        out, _ = ragged_paged_attention(q, k, v, cache, ctx.reshape(1), pi,
                                        jnp.array([0, chunk], jnp.int32),
                                        jnp.array([0, 0, 1], jnp.int32),
                                        sm_scale=SM, use_causal_mask=True,
                                        update_kv_cache=True)
        return out

    return lambda ctx: _bench(fn, q, k, v, jnp.array(ctx, jnp.int32))


def human(n):
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n / 1024 / 1024:g}M"


if __name__ == "__main__":
    max_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2 * 1024 * 1024
    CH = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    steps = list(range(CH, max_ctx + 1, CH))
    report = [n for n in [16, 32, 64, 128, 256, 512, 1024, 2048]
              if CH <= n * 1024 <= max_ctx]
    print(f"# TTFT (ms) via attention_interface.pcp_ragged_paged_attention, "
          f"CH={human(CH)}, NQ={NQ} NKV={NKV} HD={HD} {DTYPE.__name__}")
    m = {
        "tp4": make_tp(4, CH, max_ctx),
        "tp8": make_tp(8, CH, max_ctx),
        "pcp2_tp4": make_pcp_wrapper(2, 4, CH, max_ctx),
        "pcp8_tp1": make_pcp_wrapper(8, 1, CH, max_ctx),
    }
    lat = {}
    for name, f in m.items():
        lat[name] = {c: f(c) for c in steps}
        print(f"  measured {name}", flush=True)
    print("\n{:>8} | {:>10} {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8}".format(
        "context", "tp4(ms)", "tp8(ms)", "pcp2_tp4", "pcp8_tp1", "tp4/pcp2",
        "tp4/pcp8", "tp4/tp8"))
    for n in report:
        N = n * 1024
        row = {k: sum(lat[k][c] for c in steps if c <= N) for k in m}
        print("{:>8} | {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} | "
              "{:>7.2f}x {:>7.2f}x {:>7.2f}x".format(
                  human(N), row["tp4"], row["tp8"], row["pcp2_tp4"],
                  row["pcp8_tp1"], row["tp4"] / row["pcp2_tp4"],
                  row["tp4"] / row["pcp8_tp1"], row["tp4"] / row["tp8"]),
              flush=True)
