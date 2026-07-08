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
"""Performance comparison: PCP vs TP for the RPA v3 kernel on long context.

Two modes, selected by the optional 4th arg ``prev_len``:

* **Pure prefill** (prev_len=0, default): a single causal prefill of ``S`` tokens.
  - TP  : shard heads; every device does full-S causal over ``NQ/TP`` heads.
  - PCP : all-gather the current KV; every device owns its head-tail query chunks
          (``S/PCP`` tokens) and all heads. 2 kernel launches/device (head, tail).

* **Chunked prefill** (prev_len>0): a current chunk of ``S`` tokens attends a
  previous KV cache of ``prev_len`` tokens plus itself causally (total context
  ``prev_len + S``).
  - TP  : one launch (cache holds prev; current is written + attended), heads split.
  - PCP : **4 launches/device** = {head, tail} x {current (causal, all-gather),
          prev (non-causal, from cache)}, the two passes merged per chunk via LSE.

Both paths consume the *same* random q/k/v so we also verify TP and PCP outputs
are numerically equivalent (they compute the same attention, sharded differently).

Timing note: all inputs are committed to their target sharding with
``jax.device_put`` BEFORE the timed loop, so ``bench`` measures only kernel work
(plus, for the all-gather variant, the in-kernel collective).

Run (uses the working-tree kernel via lightweight namespace stubs so the PCP
changes on this branch are picked up rather than the conda editable install):

    python tests/kernels/ragged_paged_attention_pcp_benchmark.py [S] [pcp] [tp] [prev_len]

Examples:
    ... 65536 4 4            # 64k pure prefill, pcp=4 vs tp=4
    ... 65536 8 4            # 64k pure prefill, pcp=8 vs tp=4 (TP head-limited)
    ... 65536 4 4 65536      # 128k total: 64k current chunk + 64k prev cache
        
    
"""

import os
import sys
import time
import types
from functools import partial

# --- Import the working-tree RPA v3 kernel (see testing-kernels-dual-checkout).
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _pkg, _rel in [
    ("tpu_inference", "tpu_inference"),
    ("tpu_inference.kernels", "tpu_inference/kernels"),
    ("tpu_inference.kernels.ragged_paged_attention",
     "tpu_inference/kernels/ragged_paged_attention"),
    ("tpu_inference.kernels.ragged_paged_attention.v3",
     "tpu_inference/kernels/ragged_paged_attention/v3"),
]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(ROOT, _rel)]
    _m.__package__ = _pkg
    sys.modules[_pkg] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as PS  # noqa: E402

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (  # noqa: E402
    merge_kv, ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (  # noqa: E402
    align_to, cdiv, get_dtype_packing)

# ------------------------------- config --------------------------------------
S = int(sys.argv[1]) if len(sys.argv) > 1 else 64 * 1024   # current tokens
PCP = int(sys.argv[2]) if len(sys.argv) > 2 else 4         # context-parallel size
TP = int(sys.argv[3]) if len(sys.argv) > 3 else 4          # tensor-parallel size
PREV = int(sys.argv[4]) if len(sys.argv) > 4 else 0        # prev cache tokens

NQ, NKV, HD = 16, 4, 256        # q heads, kv heads, head dim (GQA ratio 4)
PAGE = 32                       # kv-cache page size
DTYPE = jnp.bfloat16
SM_SCALE = HD**-0.5

CUR = S                                 # current chunk length
NUM_PAGES_CUR = cdiv(CUR, PAGE)         # pages for the current all-gathered KV
NUM_PAGES_PREV = cdiv(PREV, PAGE) if PREV else 0
NUM_PAGES_TP = cdiv(PREV + CUR, PAGE)   # TP chunked cache: prev + written current
KVP = get_dtype_packing(DTYPE)
# Head-tail layout: device r owns chunk r and chunk (2*PCP-1-r), each of size C.
C = CUR // (2 * PCP)

WARMUP = 3
ITERS = 10


def gen_random(shape, dtype, rng):
    """Uniform-random tensor, generated like the v3 kernel tests' gen_random."""
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)


def make_cache(nkv_local, npages):
    """Empty paged KV cache for `nkv_local` heads across `npages` pages."""
    nkv2 = align_to(2 * nkv_local, KVP)
    return jnp.zeros((npages, PAGE, nkv2 // KVP, KVP, HD), DTYPE)


def make_prev_cache(prev_k, prev_v, npages):
    """Paged cache holding the `PREV` prev tokens (from prev_k/prev_v) in the
    first pages of an `npages`-page buffer; remaining pages zero (current-KV
    scratch for TP's in-kernel write). Local head count inferred from prev_k."""
    kv = merge_kv(prev_k, prev_v)  # [PREV, nkv2//kvp, kvp, HD]
    h1, h2 = kv.shape[1], kv.shape[2]  # capture before reshape reassigns kv
    pages_prev = cdiv(PREV, PAGE)
    pad = pages_prev * PAGE - PREV
    kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0))).reshape(
        pages_prev, PAGE, h1, h2, HD)
    cache = jnp.zeros((npages, PAGE, h1, h2, HD), DTYPE)
    return cache.at[:pages_prev].set(kv)


def merge_lse(o1, l1, o2, l2):
    """Online-softmax merge of two partial attention outputs via their LSEs."""
    m = jnp.maximum(l1, l2)
    e1 = jnp.exp(l1 - m)
    e2 = jnp.exp(l2 - m)
    o = (o1 * e1[..., None] + o2 * e2[..., None]) / (e1 + e2)[..., None]
    return o, m + jnp.log(e1 + e2)


def put(mesh, spec, arr):
    """Commit `arr` to `spec` on `mesh` so the reshard is out of the timed loop."""
    return jax.device_put(arr, NamedSharding(mesh, spec))


def bench(fn, *args):
    """Return (avg latency ms, last output) with warmup + block_until_ready."""
    r = fn(*args)
    jax.block_until_ready(r)
    for _ in range(WARMUP):
        r = fn(*args)
        jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        r = fn(*args)
        jax.block_until_ready(r)
    return (time.perf_counter() - t0) / ITERS * 1e3, r  # ms, output


def i32(xs):
    return jnp.array(xs, jnp.int32)


def build_head_tail_q(q_full):
    """Reshape [CUR, NQ, HD] into per-device head-tail chunks.

    Device r owns chunk r (head) and chunk 2*PCP-1-r (tail) -> [PCP, 2, C, NQ, HD].
    """
    chunks = []
    for r in range(PCP):
        for ch in (r, 2 * PCP - 1 - r):
            chunks.append(q_full[ch * C:ch * C + C])
    return jnp.stack(chunks).reshape(PCP, 2, C, NQ, HD)


def reassemble_head_tail(out_global):
    """Inverse of build_head_tail_q for [PCP, 2, C, NQ, HD] outputs."""
    out = jnp.zeros((CUR, NQ, HD), DTYPE)
    for r in range(PCP):
        for i, ch in enumerate((r, 2 * PCP - 1 - r)):
            out = out.at[ch * C:ch * C + C].set(out_global[r, i])
    return out


# =========================== pure prefill ====================================
def run_tp(q, k, v):
    """TP: shard heads; every device does full-S causal over NQ/TP heads."""
    mesh = Mesh(np.array(jax.devices()[:TP]), ("x", ))
    hs = PS(None, "x", None)
    q, k, v = put(mesh, hs, q), put(mesh, hs, k), put(mesh, hs, v)

    @partial(shard_map, mesh=mesh,
             in_specs=(hs, hs, hs, PS(), PS(), PS(), PS()),
             out_specs=hs, check_rep=False)
    def fn(q, k, v, kv_lens, page_indices, cu_q_lens, distribution):
        cache = make_cache(k.shape[1], NUM_PAGES_CUR)
        out, _ = ragged_paged_attention(
            q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
            sm_scale=SM_SCALE, use_causal_mask=True, update_kv_cache=True)
        return out

    args = (i32([CUR]), jnp.arange(NUM_PAGES_CUR, dtype=jnp.int32),
            i32([0, CUR]), i32([0, 0, 1]))
    return bench(jax.jit(fn), q, k, v, *args)


def _pcp_body(qc, k_full, v_full, kv_lens, page_indices, cu_q_lens,
              distribution):
    """This device's two head-tail chunks vs the full (all-gathered) KV.

    Query buffer is padded to KV length CUR (kernel requires q.shape[0]==k.shape[0]);
    only C rows are valid. Returns [1, 2, C, NQ, HD] to keep the device axis.
    """
    r = jax.lax.axis_index("x")
    cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
    cache = make_cache(NKV, NUM_PAGES_CUR)  # empty; current read from HBM
    offsets = (r * C, (2 * PCP - 1 - r) * C)
    outs = []
    for i in range(2):
        qpos = jax.lax.reshape(offsets[i], (1, )).astype(jnp.int32)
        q_buf = jnp.zeros((CUR, NQ, HD), DTYPE).at[:C].set(qc[i])
        out, _ = ragged_paged_attention(
            q_buf, k_full, v_full, cache, kv_lens, page_indices, cu_q_lens,
            distribution, cp_rank=cp_rank, cp_group_size=PCP,
            all_gather_kv=True, q_pos_offsets=qpos, sm_scale=SM_SCALE,
            update_kv_cache=False, use_causal_mask=True)
        outs.append(out[:C])
    return jnp.stack(outs)[None]


def run_pcp(q2, k_full, v_full, gather):
    """PCP pure prefill. gather=False: KV replicated (kernel only). gather=True:
    KV sequence-sharded + all-gathered in the body (full pipeline)."""
    mesh = Mesh(np.array(jax.devices()[:PCP]), ("x", ))
    qspec = PS("x", None, None, None, None)
    kspec = PS("x", None, None) if gather else PS()
    q2 = put(mesh, qspec, q2)
    k_full, v_full = put(mesh, kspec, k_full), put(mesh, kspec, v_full)

    @partial(shard_map, mesh=mesh,
             in_specs=(qspec, kspec, kspec, PS(), PS(), PS(), PS()),
             out_specs=PS("x", None, None, None, None), check_rep=False)
    def fn(q2, k, v, kv_lens, page_indices, cu_q_lens, distribution):
        if gather:
            k = jax.lax.all_gather(k, "x", axis=0, tiled=True)
            v = jax.lax.all_gather(v, "x", axis=0, tiled=True)
        return _pcp_body(q2[0], k, v, kv_lens, page_indices, cu_q_lens,
                         distribution)

    args = (i32([CUR]), jnp.arange(NUM_PAGES_CUR, dtype=jnp.int32),
            i32([0, C]), i32([0, 0, 1]))
    return bench(jax.jit(fn), q2, k_full, v_full, *args)


# =========================== chunked prefill =================================
def run_tp_chunked(q, cur_k, cur_v, prev_k, prev_v):
    """TP chunked prefill: one launch, cache holds prev + written current."""
    mesh = Mesh(np.array(jax.devices()[:TP]), ("x", ))
    hs = PS(None, "x", None)
    q = put(mesh, hs, q)
    cur_k, cur_v = put(mesh, hs, cur_k), put(mesh, hs, cur_v)
    prev_k, prev_v = put(mesh, hs, prev_k), put(mesh, hs, prev_v)

    @partial(shard_map, mesh=mesh,
             in_specs=(hs, hs, hs, hs, hs, PS(), PS(), PS(), PS()),
             out_specs=hs, check_rep=False)
    def fn(q, cur_k, cur_v, prev_k, prev_v, kv_lens, page_indices, cu_q_lens,
           distribution):
        cache = make_prev_cache(prev_k, prev_v, NUM_PAGES_TP)
        out, _ = ragged_paged_attention(
            q, cur_k, cur_v, cache, kv_lens, page_indices, cu_q_lens,
            distribution, sm_scale=SM_SCALE, use_causal_mask=True,
            update_kv_cache=True)
        return out

    args = (i32([PREV + CUR]), jnp.arange(NUM_PAGES_TP, dtype=jnp.int32),
            i32([0, CUR]), i32([0, 0, 1]))
    return bench(jax.jit(fn), q, cur_k, cur_v, prev_k, prev_v, *args)


def _pcp_chunk_body(qc, cur_k, cur_v, prev_k, prev_v):
    """4 launches/device: {head, tail} x {current causal, prev non-causal}.

    The current pass (all-gathered current KV, causal, LSE) and the prev pass
    (prev cache, non-causal, LSE) are merged per chunk with online softmax.
    """
    r = jax.lax.axis_index("x")
    cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
    cur_cache = make_cache(NKV, NUM_PAGES_CUR)  # empty; current read from HBM
    prev_cache = make_prev_cache(prev_k, prev_v, NUM_PAGES_PREV)
    pi_cur = jnp.arange(NUM_PAGES_CUR, dtype=jnp.int32)
    pi_prev = jnp.arange(NUM_PAGES_PREV, dtype=jnp.int32)
    cu = i32([0, C])
    dist = i32([0, 0, 1])
    offsets = (r * C, (2 * PCP - 1 - r) * C)
    outs = []
    for i in range(2):
        qpos = jax.lax.reshape(offsets[i], (1, )).astype(jnp.int32)
        q_buf = jnp.zeros((CUR, NQ, HD), DTYPE).at[:C].set(qc[i])
        # Current: causal attention over the all-gathered current KV.
        o_cur, _, l_cur = ragged_paged_attention(
            q_buf, cur_k, cur_v, cur_cache, i32([CUR]), pi_cur, cu, dist,
            cp_rank=cp_rank, cp_group_size=PCP, all_gather_kv=True,
            q_pos_offsets=qpos, sm_scale=SM_SCALE, update_kv_cache=False,
            use_causal_mask=True, return_lse=True)
        # Prev: non-causal attention over the previous cache (cur_k/v ignored).
        o_prev, _, l_prev = ragged_paged_attention(
            q_buf, cur_k, cur_v, prev_cache, i32([PREV]), pi_prev, cu, dist,
            sm_scale=SM_SCALE, update_kv_cache=False, use_causal_mask=False,
            return_lse=True)
        o, _ = merge_lse(o_prev[:C], l_prev[:C], o_cur[:C], l_cur[:C])
        outs.append(o)
    return jnp.stack(outs)[None]


def run_pcp_chunked(q2, cur_k, cur_v, prev_k, prev_v, gather):
    """PCP chunked prefill (4 launches/device). gather toggles the all-gather."""
    mesh = Mesh(np.array(jax.devices()[:PCP]), ("x", ))
    qspec = PS("x", None, None, None, None)
    kspec = PS("x", None, None) if gather else PS()
    q2 = put(mesh, qspec, q2)
    cur_k, cur_v = put(mesh, kspec, cur_k), put(mesh, kspec, cur_v)
    prev_k, prev_v = put(mesh, PS(), prev_k), put(mesh, PS(), prev_v)

    @partial(shard_map, mesh=mesh,
             in_specs=(qspec, kspec, kspec, PS(), PS()), out_specs=qspec,
             check_rep=False)
    def fn(q2, cur_k, cur_v, prev_k, prev_v):
        if gather:
            cur_k = jax.lax.all_gather(cur_k, "x", axis=0, tiled=True)
            cur_v = jax.lax.all_gather(cur_v, "x", axis=0, tiled=True)
        return _pcp_chunk_body(q2[0], cur_k, cur_v, prev_k, prev_v)

    return bench(jax.jit(fn), q2, cur_k, cur_v, prev_k, prev_v)


def check_kv_update():
    """Verify the PCP strided KV write against the reference merged current KV.

    Self-contained at head_dim=128 (the strided cache-write path only supports
    hd=128; it is independent of the perf config's HD). Each rank writes its
    1/PCP strided share of the all-gathered current KV via a non-causal
    ``update_kv_cache=True`` launch. DCP-strided layout: global token g lives on
    rank ``g % PCP`` at local slot ``g // PCP`` (identity page indices). De-stride
    all ranks' caches and compare to ``merge_kv`` -- a pure copy, so bit-exact.
    """
    P = PCP
    hd, nq, nkv, page, C_kv = 128, 8, 2, 16, 64
    Skv = 2 * P * C_kv          # gathered current KV length = 2*P*C
    npages = cdiv(Skv, page)
    nkv2 = align_to(2 * nkv, KVP)
    rng = np.random.default_rng(7)
    k = gen_random((Skv, nkv, hd), DTYPE, rng)
    v = gen_random((Skv, nkv, hd), DTYPE, rng)
    ref = np.asarray(merge_kv(k, v))  # [Skv, nkv2//kvp, kvp, hd]

    mesh = Mesh(np.array(jax.devices()[:P]), ("x", ))
    k, v = put(mesh, PS(), k), put(mesh, PS(), v)

    @partial(shard_map, mesh=mesh, in_specs=(PS(), PS()),
             out_specs=PS("x", None, None, None, None, None), check_rep=False)
    def fn(k, v):
        r = jax.lax.axis_index("x")
        cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
        cache = jnp.zeros((npages, page, nkv2 // KVP, KVP, hd), DTYPE)
        q = jnp.zeros((Skv, nq, hd), DTYPE)
        _, new_cache = ragged_paged_attention(
            q, k, v, cache, i32([Skv]), jnp.arange(npages, dtype=jnp.int32),
            i32([0, C_kv]), i32([0, 0, 1]), cp_rank=cp_rank, cp_group_size=P,
            all_gather_kv=True, update_kv_cache=True, use_causal_mask=False)
        return new_cache[None]

    caches = np.asarray(jax.jit(fn)(k, v))  # [P, npages, page, nkv2//kvp, kvp, hd]
    flat = caches.reshape(P, -1, caches.shape[3], caches.shape[4], hd)
    g = np.arange(Skv)
    recon = flat[g % P, g // P]  # gather token g from its owning rank/slot
    exact = bool(np.array_equal(recon, ref))
    mism = int((recon != ref).sum())
    # Guard: the written cache must be real data, not left all-zero. Compute
    # stats in float32 -- averaging bf16 in-place loses precision badly.
    rf = recon.astype(np.float32)
    mx = float(np.abs(rf).max())
    mean_abs = float(np.abs(rf).mean())
    zfrac = float((rf == 0).mean())
    nnan = int(np.isnan(rf).sum())
    nondeg = mx > 0.0 and nnan == 0 and zfrac < 0.5
    print(f"  PCP strided KV write vs merged current KV (hd=128): exact={exact}  "
          f"mismatched-elems={mism}/{ref.size}")
    print(f"  written cache non-degenerate: max|x|={mx:.4e}  mean|x|={mean_abs:.4e}"
          f"  zero-frac={zfrac:.3f}  nan={nnan}  ok={nondeg}")
    return exact and nondeg


def run_allgather_only(k, v):
    """Standalone cost of the current-KV all-gather (both K and V)."""
    mesh = Mesh(np.array(jax.devices()[:PCP]), ("x", ))
    ksp = PS("x", None, None)
    k, v = put(mesh, ksp, k), put(mesh, ksp, v)

    @partial(shard_map, mesh=mesh, in_specs=(ksp, ksp), out_specs=(PS(), PS()),
             check_rep=False)
    def fn(k, v):
        return (jax.lax.all_gather(k, "x", axis=0, tiled=True),
                jax.lax.all_gather(v, "x", axis=0, tiled=True))

    return bench(jax.jit(fn), k, v)


def describe(out, label):
    """Confirm an output is non-degenerate: finite, not all-zero, real spread.

    A trivially-passing allclose (e.g. both sides all-zero or all-NaN) would be
    meaningless, so we surface the actual magnitude/finiteness here.
    """
    x = np.asarray(out, dtype=np.float32)
    n = x.size
    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    frac_zero = float((x == 0).sum()) / n
    mx = float(np.abs(x).max())
    ok = (n_nan == 0 and n_inf == 0 and mx > 0.0 and frac_zero < 0.5)
    print(f"  {label:<18}: max|x|={mx:.4e}  mean|x|={float(np.abs(x).mean()):.4e}"
          f"  nan={n_nan}  inf={n_inf}  zero-frac={frac_zero:.3f}  "
          f"non-degenerate={ok}")
    return ok


def compare(out_tp, out_pcp, label):
    """Report numeric agreement of two [CUR, NQ, HD] outputs (pulled to host:
    TP and PCP outputs live on different meshes so JAX can't combine them)."""
    a = np.asarray(out_tp, dtype=np.float32)
    b = np.asarray(out_pcp, dtype=np.float32)
    diff = np.abs(a - b)
    scale = float(np.abs(a).max()) + 1e-9
    close = bool(np.allclose(a, b, atol=2e-2, rtol=2e-2))
    print(f"  {label:<22}: max|Δ|={float(diff.max()):.4e}  "
          f"mean|Δ|={float(diff.mean()):.4e}  "
          f"max|Δ|/scale={float(diff.max())/scale:.4e}  allclose={close}")
    return close


def _header(mode):
    print("=" * 72)
    print(f"RPA v3 kernel: PCP vs TP  [{mode}]  perf + numerics")
    print("=" * 72)
    total = PREV + CUR
    print(f"  current / prev / total : {CUR} / {PREV} / {total} "
          f"({total // 1024}k)")
    print(f"  q/kv heads, head dim   : {NQ}/{NKV}, {HD}")
    print(f"  dtype, page size       : {DTYPE.__name__}, {PAGE}")
    print(f"  PCP size / TP size     : {PCP} / {TP}")
    # Per-device causal-attention FLOPs. pairs = current-query x key pairs.
    pairs = CUR * PREV + CUR * CUR / 2  # prev (all) + causal current
    tp_f = 4 * pairs * HD * (NQ / TP)
    pcp_f = 4 * (pairs / PCP) * HD * NQ
    print(f"  per-device FLOPs       : TP={tp_f/1e12:.2f} TF  "
          f"PCP={pcp_f/1e12:.2f} TF  (theory ratio {pcp_f/tp_f:.2f} = TP/PCP)")
    print("=" * 72)


def main():
    rng = np.random.default_rng(1234)
    q_cur = gen_random((CUR, NQ, HD), DTYPE, rng)
    cur_k = gen_random((CUR, NKV, HD), DTYPE, rng)
    cur_v = gen_random((CUR, NKV, HD), DTYPE, rng)
    q2 = build_head_tail_q(q_cur)

    if PREV == 0:
        _header("pure prefill")
        tp_ms, out_tp = run_tp(q_cur, cur_k, cur_v)
        pcp_ms, out_pcp_g = run_pcp(q2, cur_k, cur_v, gather=False)
        pcp_ag_ms, out_pcp_ag_g = run_pcp(q2, cur_k, cur_v, gather=True)
    else:
        _header("chunked prefill (4 PCP launches/device)")
        prev_k = gen_random((PREV, NKV, HD), DTYPE, rng)
        prev_v = gen_random((PREV, NKV, HD), DTYPE, rng)
        tp_ms, out_tp = run_tp_chunked(q_cur, cur_k, cur_v, prev_k, prev_v)
        pcp_ms, out_pcp_g = run_pcp_chunked(q2, cur_k, cur_v, prev_k, prev_v,
                                            gather=False)
        pcp_ag_ms, out_pcp_ag_g = run_pcp_chunked(q2, cur_k, cur_v, prev_k,
                                                  prev_v, gather=True)
    ag_ms, _ = run_allgather_only(cur_k, cur_v)

    out_pcp = reassemble_head_tail(out_pcp_g)
    out_pcp_ag = reassemble_head_tail(out_pcp_ag_g)

    print("timing (per-device wall clock; all devices run in parallel)")
    print(f"  TP                                : {tp_ms:8.3f} ms")
    print(f"  PCP kernel only (KV pre-gathered) : {pcp_ms:8.3f} ms")
    print(f"  PCP all-gather KV (standalone)    : {ag_ms:8.3f} ms")
    print(f"  PCP kernel + all-gather (full)    : {pcp_ag_ms:8.3f} ms")
    print(f"  marginal gather (full - kernel)   : {pcp_ag_ms - pcp_ms:8.3f} ms")
    print(f"  PCP-kernel / TP ratio             : {pcp_ms / tp_ms:8.3f}")
    print(f"  PCP-full   / TP ratio             : {pcp_ag_ms / tp_ms:8.3f}")
    print("-" * 72)
    print("output sanity (guards against a trivially-passing all-zero/NaN match)")
    d0 = describe(out_tp, "TP")
    d1 = describe(out_pcp, "PCP-kernel")
    d2 = describe(out_pcp_ag, "PCP-allgather")
    print("numerics (same random q/k/v; TP vs PCP output over current tokens)")
    ok1 = compare(out_tp, out_pcp, "TP vs PCP-kernel")
    ok2 = compare(out_tp, out_pcp_ag, "TP vs PCP-allgather")
    print("kv-cache update (PCP strided write correctness)")
    ok3 = check_kv_update()
    print(f"  EQUIVALENT: {ok1 and ok2 and d0 and d1 and d2 and ok3} "
          f"(equivalent AND non-degenerate AND kv-write correct)")
    print("-" * 72)
    print("On the gather: there is a strict data dependency (the kernel reads the")
    print("gathered KV), so it does NOT overlap the attention compute. 'marginal")
    print("gather' is small because the standalone number also writes the full")
    print("gathered KV to HBM -- a materialization the kernel needs anyway; the")
    print("truly-extra part is just the ICI communication.")
    print("=" * 72)


if __name__ == "__main__":
    main()
