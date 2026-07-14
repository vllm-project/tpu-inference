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
"""PCP-vs-TP prefill TTFT microbenchmark (kernel level).

TTFT(N) for a prompt of length N prefilled in chunks of `CH` = sum over chunks i
of the attention latency of chunk i (queries [i*CH,(i+1)*CH) attending KV
[0,(i+1)*CH)). We measure latency(ctx) at every CH step and cumulative-sum.

Configs (model has NQ=8 attention heads):
  * tp<T>     : heads sharded /T, full context. TP attention has no cross-head
                collective, so per-device latency (NQ/T heads, full ctx) IS the
                config's per-chunk latency. Measured on ONE device.
  * pcp2_tp<T>: prefill context parallel over pcp=2 AND heads /T. Per device:
                NQ/T heads, HALF the context (pcp-strided cache) + head-tail
                current with an all-gather. Measured on a pcp=2 sub-mesh so the
                pcp-axis collectives (current all-gather + LSE all-reduce) count;
                the tp axis only replicates across head-groups (no added latency).

Usage:
  python tests/kernels/ragged_paged_attention_pcp_ttft_benchmark.py context [MAXCTX] [CH]
  python tests/kernels/ragged_paged_attention_pcp_ttft_benchmark.py chunksweep [CTX]
"""
import os
import sys
import time
import types
from functools import partial

ROOT = "/home/wenxindong_google_com/work/tpu-inference"
for pkg, rel in [
    ("tpu_inference", "tpu_inference"),
    ("tpu_inference.kernels", "tpu_inference/kernels"),
    ("tpu_inference.kernels.experimental",
     "tpu_inference/kernels/experimental"),
    ("tpu_inference.kernels.experimental.rpa_v3_cp",
     "tpu_inference/kernels/experimental/rpa_v3_cp"),
    ("tpu_inference.kernels.ragged_paged_attention",
     "tpu_inference/kernels/ragged_paged_attention"),
    ("tpu_inference.kernels.ragged_paged_attention.v3",
     "tpu_inference/kernels/ragged_paged_attention/v3"),
]:
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(ROOT, rel)]
    m.__package__ = pkg
    sys.modules[pkg] = m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as PS  # noqa: E402

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import \
    ragged_paged_attention  # noqa: E402
from tpu_inference.kernels.ragged_paged_attention.v3.util import (  # noqa: E402
    align_to, cdiv, get_dtype_packing)

# 16 attention heads, GQA with 4 kv heads (total). NQ=16 (not 8) so tp8 lands on
# 2 q-heads/device: the kernel pads num_q_heads_per_kv_head up to q_packing (=2
# for bf16), so any config with 1 q-head/device (NQ=8 at tp8) silently does 2x
# the necessary work and misreads as "tp8 == tp4".
NQ, NKV, HD = 16, 4, 128
PAGE = 256
DTYPE = jnp.bfloat16
SM = HD**-0.5
KVP = get_dtype_packing(DTYPE)
WARMUP, ITERS = 1, 3
_rng = np.random.default_rng(0)


def heads(tp):
    """Per-device (q, kv) head counts after TP head-sharding. When NKV < tp the
    kv heads are replicated (>=1 per device), matching vLLM GQA/MQA sharding."""
    return NQ // tp, max(1, NKV // tp)


def _rand(shape):
    return jnp.array(_rng.random(size=shape, dtype=np.float32)).astype(DTYPE)


def _bench(fn, *args):
    r = fn(*args)
    jax.block_until_ready(r)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms


def _i32(xs):
    return jnp.array(xs, jnp.int32)


def _lse_all_reduce(o, lse, axis):
    """Merge per-rank partial attention (o, lse) across `axis` via LSE.

    Verbatim mirror of attention_interface._lse_all_reduce: each pcp rank's cache
    phase attends only its strided 1/pcp shard of the prev cache, so the partial
    softmaxes must be merged across ranks. 3 collectives: pmax + 2x psum.
    """
    m = jax.lax.pmax(lse, axis)
    m_safe = jnp.where(jnp.isinf(m), 0.0, m)
    w = jnp.exp(lse - m_safe)
    denom = jax.lax.psum(w, axis)
    o_merged = (jax.lax.psum(o * w[..., None], axis) /
                jnp.where(denom == 0.0, 1.0, denom)[..., None])
    lse_merged = jnp.where(denom == 0.0, -jnp.inf, m_safe + jnp.log(denom))
    return o_merged, lse_merged


# ------------------------------- TP latency ---------------------------------
# Build the jit ONCE per config with a max-size (zeros) cache; vary the context
# via a dynamic `ctx` scalar (kv_lens) so latency scales with the KV loop bound
# and no recompile is needed per step. Content is irrelevant for timing.
# update_kv_cache=False -> attention-only (the O(chunk) write is negligible vs
# the O(ctx) attention and would need per-step page bookkeeping).
def make_tp(tp, chunk, max_ctx):
    nq_local, nkv_local = heads(tp)
    nkv2 = align_to(2 * nkv_local, KVP)
    max_np = cdiv(max_ctx, PAGE)
    mesh = Mesh(np.array(jax.devices()[:1]), ("x", ))
    pi = jnp.arange(max_np, dtype=jnp.int32)
    q = _rand((chunk, nq_local, HD))
    kcur = _rand((chunk, nkv_local, HD))
    vcur = _rand((chunk, nkv_local, HD))

    @partial(shard_map,
             mesh=mesh,
             in_specs=(PS(), PS(), PS(), PS()),
             out_specs=PS(),
             check_rep=False)
    def fn(q, kcur, vcur, ctx):
        cache = jnp.zeros((max_np, PAGE, nkv2 // KVP, KVP, HD), DTYPE)
        out, _ = ragged_paged_attention(q,
                                        kcur,
                                        vcur,
                                        cache,
                                        ctx.reshape(1),
                                        pi,
                                        _i32([0, chunk]),
                                        _i32([0, 0, 1]),
                                        sm_scale=SM,
                                        use_causal_mask=True,
                                        update_kv_cache=False)
        return out

    jfn = jax.jit(fn)
    return lambda ctx: _bench(jfn, q, kcur, vcur, jnp.array(ctx, jnp.int32))


def make_pcp(pcp, tp, chunk, max_ctx):
    if jax.device_count() < pcp:
        return lambda ctx: float("nan")
    nq_local, nkv_local = heads(tp)  # tp shards heads; pcp shards context
    nkv2 = align_to(2 * nkv_local, KVP)
    two_p = 2 * pcp
    C = max(chunk // two_p, 1)
    max_prev_local = cdiv(max_ctx, pcp)
    max_np = max(cdiv(max_prev_local, PAGE), 1)
    npages_cur = max(cdiv(chunk, PAGE), 1)
    mesh = Mesh(np.array(jax.devices()[:pcp]), ("x", ))
    ql = _rand((pcp, 2, C, nq_local, HD))
    kl = _rand((pcp, 2, C, nkv_local, HD))
    vl = _rand((pcp, 2, C, nkv_local, HD))
    qsp = PS("x", None, None, None, None)
    pi_prev = jnp.arange(max_np, dtype=jnp.int32)
    pi_cur = jnp.arange(npages_cur, dtype=jnp.int32)

    @partial(shard_map,
             mesh=mesh,
             in_specs=(qsp, qsp, qsp, PS()),
             out_specs=qsp,
             check_rep=False)
    def fn(ql, kl, vl, ctx):
        r = jax.lax.axis_index("x")
        cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
        prev = ctx - chunk
        ag = lambda x: jax.lax.all_gather(x, "x", axis=0, tiled=True)
        kcur = ag(kl[0].reshape(2 * C, nkv_local, HD))
        vcur = ag(vl[0].reshape(2 * C, nkv_local, HD))
        cache = jnp.zeros((max_np, PAGE, nkv2 // KVP, KVP, HD), DTYPE)
        head_off, tail_off = r * C, (two_p - 1 - r) * C

        # --- cache phase: ONE launch (non-causal -> all queries equivalent) ---
        ag_q = ag(ql[0].reshape(2 * C, nq_local, HD))  # [pcp*2C], rank-ordered
        o1, _, l1 = ragged_paged_attention(ag_q,
                                           kcur,
                                           vcur,
                                           cache,
                                           ctx.reshape(1),
                                           pi_prev,
                                           _i32([0, two_p * C]),
                                           _i32([0, 0, 1]),
                                           skip_current_attn=True,
                                           use_causal_mask=False,
                                           update_kv_cache=False,
                                           cp_rank=cp_rank,
                                           cp_group_size=pcp,
                                           return_lse=True,
                                           sm_scale=SM,
                                           kv_cache_lens=prev.reshape(1))
        o1, l1 = _lse_all_reduce(o1, l1, "x")
        o1 = jax.lax.dynamic_slice_in_dim(o1, r * 2 * C, 2 * C, 0)
        l1 = jax.lax.dynamic_slice_in_dim(l1, r * 2 * C, 2 * C, 0)

        # --- current phase: ONE ragged launch, head+tail as 2 seqs ---
        q_both = ql[0].reshape(2 * C, nq_local, HD)
        o2, _, l2 = ragged_paged_attention(
            q_both,
            kcur,
            vcur,
            cache,
            jnp.stack([ctx, ctx]).astype(jnp.int32),
            jnp.concatenate([pi_cur, pi_cur]),
            _i32([0, C, 2 * C]),
            _i32([0, 0, 2]),
            q_pos_offsets=jnp.stack([head_off, tail_off]).astype(jnp.int32),
            skip_cache_attn=True,
            use_causal_mask=True,
            update_kv_cache=False,
            cp_rank=cp_rank,
            cp_group_size=pcp,
            return_lse=True,
            sm_scale=SM,
            kv_cache_lens=jnp.stack([prev, prev]).astype(jnp.int32))

        m = jnp.maximum(l1, l2)
        e1, e2 = jnp.exp(l1 - m), jnp.exp(l2 - m)
        o = (o1 * e1[..., None] + o2 * e2[..., None]) / (e1 + e2)[..., None]
        return o.reshape(2, C, nq_local, HD)[None]

    jfn = jax.jit(fn)
    return lambda ctx: _bench(jfn, ql, kl, vl, jnp.array(ctx, jnp.int32))


# --------------------------------- sweeps -----------------------------------
def human(n):
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n / 1024 / 1024:g}M"


def context_sweep(max_ctx, CH):
    steps = list(range(CH, max_ctx + 1, CH))
    # Only report context points that are >= the chunk size (a context below CH
    # has no chunks, so its cumulative TTFT is an empty sum -> 0.00/nan).
    report = [
        n for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if CH <= n * 1024 <= max_ctx
    ]
    makers = [("tp4", make_tp(4, CH, max_ctx)),
              ("tp8", make_tp(8, CH, max_ctx)),
              ("pcp2_tp4", make_pcp(2, 4, CH, max_ctx))]
    print(
        f"# TTFT (ms) via chunked prefill, CH={human(CH)}, NQ={NQ} NKV={NKV} "
        f"HD={HD} {DTYPE.__name__}, up to {human(max_ctx)}")
    lat = {name: {} for name, _ in makers}
    for name, measure in makers:
        for ctx in steps:
            lat[name][ctx] = measure(ctx)
        print(f"  measured {name}", flush=True)
    print("\n{:>8} | {:>10} {:>10} {:>10} | {:>8} {:>8}".format(
        "context", "tp4(ms)", "tp8(ms)", "pcp2_tp4", "tp4/pcp", "tp4/tp8"))
    for n in report:
        N = n * 1024
        row = {
            name: sum(lat[name][c] for c in steps if c <= N)
            for name, _ in makers
        }
        sp = row["tp4"] / row["pcp2_tp4"] if row["pcp2_tp4"] else float("nan")
        s8 = row["tp4"] / row["tp8"] if row["tp8"] else float("nan")
        print("{:>8} | {:>10.2f} {:>10.2f} {:>10.2f} | {:>7.2f}x {:>7.2f}x".
              format(human(N), row["tp4"], row["tp8"], row["pcp2_tp4"], sp,
                     s8),
              flush=True)


def chunk_sweep(ctx):
    chunks = [4096, 8192, 16384, 32768, 65536]
    print(f"# TTFT (ms) at context={human(ctx)} for varying chunk size, "
          f"NQ={NQ} NKV={NKV} HD={HD} {DTYPE.__name__}")
    print("{:>8} | {:>10} {:>10} {:>10} | {:>8} {:>8}".format(
        "chunk", "tp4(ms)", "tp8(ms)", "pcp2_tp4", "tp4/pcp", "tp4/tp8"))
    for CH in chunks:
        steps = list(range(CH, ctx + 1, CH))
        m4, m8, mp = (make_tp(4, CH,
                              ctx), make_tp(8, CH,
                                            ctx), make_pcp(2, 4, CH, ctx))
        t4 = sum(m4(c) for c in steps)
        t8 = sum(m8(c) for c in steps)
        pc = sum(mp(c) for c in steps)
        sp = t4 / pc if pc else float("nan")
        s8 = t4 / t8 if t8 else float("nan")
        print("{:>8} | {:>10.2f} {:>10.2f} {:>10.2f} | {:>7.2f}x {:>7.2f}x".
              format(human(CH), t4, t8, pc, sp, s8),
              flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "context"
    if mode == "context":
        max_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 128 * 1024
        CH = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
        context_sweep(max_ctx, CH)
    elif mode == "chunksweep":
        ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 128 * 1024
        chunk_sweep(ctx)
