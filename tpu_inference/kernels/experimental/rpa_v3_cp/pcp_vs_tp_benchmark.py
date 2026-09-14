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
"""Prefill TTFT benchmark: PCP (rpa_v3_cp) vs TP (rpa_v3) attention.

For every model config and device count, measures the attention latency of
each prefill chunk (queries [i*CH, (i+1)*CH) attending KV [0, (i+1)*CH)) and
reports cumulative TTFT(N) = sum over chunks, as a table of context lengths x
parallelism layouts, each cell relative to the all-device TP baseline.

  * ``tp{N}``          - rpa_v3 (ragged_paged_attention) with heads sharded
                         over all N devices. The baseline.
  * ``tp{N/2}``        - same on half the devices.
  * ``pcp{P}xtp{N/P}`` - rpa_v3_cp through the production PCP wrapper
                         (``cp_attention.pcp_forward``): a (pcp, model) mesh,
                         Q/K/V head-tail sharded over pcp, heads over model,
                         KV cache page-striped over pcp. In-kernel ring for the
                         cache phase.
  * ``bpcp{P}xtp{N/P}``- the same layout and the same in-kernel ring, on the
                         batched RPA kernel
                         (``cp_attention.pcp_forward_batched``). Needs the PCP
                         chunk size to be a multiple of the page size, so that
                         the all-gathered current KV can be addressed by page
                         in its natural rank order.

Attention alone understates the difference between the layouts. Per the
sharding rules (``ShardingAxisName``): o_proj is row-parallel over the model
axis, so under TP a layer all-reduces the whole chunk's [tokens, hidden]
o_proj output over all N devices, while under pcp{P}xtp{N/P} that all-reduce
covers 1/P of the tokens over N/P devices -- but PCP then all-gathers the
token-sharded activation over the pcp axis into the MLP's replicated layout
(``activation_ffw_td``). The MLP itself is tensor-sharded over every axis
(``MLP_TENSOR``), so its all-reduce is identical in both layouts and left out.
``--no-collectives`` selects which of the two things a step measures:

  * default: attention + the layer collectives around it (the o_proj
    all-reduce over the model axis and, for PCP, the all-gather over the pcp
    axis between the token-sharded attention layout and the replicated MLP
    layout), data dependent on the attention output;
  * ``--no-collectives``: pure attention, no collective outside the kernel
    (the PCP kernel's own ring DMAs and current-KV all-gather stay, they are
    the attention).

Matmuls are not modelled (they cost the same per device in every layout).

TP attention has no in-op collective, so it is measured on a shard_map over
the tp devices and blocked on all of them: it pays the same cross-device
dispatch/sync the PCP columns do. Meshes over different device subsets cannot
coexist in one TPU process, so every variant runs in its own subprocess.

Usage:
  python pcp_vs_tp_benchmark.py                       # all models, all layouts
  python pcp_vs_tp_benchmark.py --models Qwen3.5 --max-context 262144
  python pcp_vs_tp_benchmark.py --kv-dtype bfloat16 --pcp-sizes 8
  python pcp_vs_tp_benchmark.py --no-collectives                # pure attention
  python pcp_vs_tp_benchmark.py --profile-dir gs://bucket/path   # + xprof traces

With ``--profile-dir`` every layout also records an xprof trace of the step at
each ``--profile-contexts`` length (warm-up and timed iterations included)
under ``<profile-dir>/<model>_<N>dev_<layout>/ctx<len>/``; a ``gs://`` path
is uploaded with gsutil.
"""
import argparse
import dataclasses
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


@dataclasses.dataclass(frozen=True)
class ModelParams:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_devices: list[int]
    hidden_size: int  # width of the all-reduced o_proj output


MODEL_CONFIGS = {
    # google3/third_party/py/tpu_kernel_testbench/qwen35_config_defaults.json
    'Qwen3.5':
    ModelParams(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        num_devices=[8],
        hidden_size=4096,
    ),
    # https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/config.json
    'Qwen3-Coder-480B':
    ModelParams(
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[8],
        hidden_size=6144,
    ),
    # https://huggingface.co/google/gemma-4-31b-it/raw/main/config.json
    'Gemma4-31B':
    ModelParams(
        num_q_heads=32,
        num_kv_heads=16,
        head_dim=256,
        num_devices=[8],
        hidden_size=5376,
    ),
}

MAX_SEQ = 4


def _human(n):
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n // (1024 * 1024)}M"


def _boundaries(n, ch):
    """Chunk end-points for a prompt of n tokens: ch, 2ch, ..., n."""
    out, cur = [], 0
    while cur < n:
        cur = min(cur + ch, n)
        out.append(cur)
    return out


def _ladder(max_ctx):
    return [c for c in (1 << i for i in range(10, 31)) if c <= max_ctx]


# --------------------------------------------------------------------------
# Worker: one variant, one process (jax is imported here only).
# --------------------------------------------------------------------------
def _run_variant(mp,
                 variant,
                 chunk,
                 max_ctx,
                 kv_dtype_name,
                 page,
                 slack,
                 warmup,
                 iters,
                 collectives,
                 profile_dir,
                 profile_contexts,
                 n,
                 num_reqs=1,
                 check_chunks=0,
                 kv_layout_name="head_along_sublane"):
    # Raiden's engine extension must be loaded before jaxlib's XLA copy or the
    # two collide in static initializers and the process segfaults.
    # `tpu_inference.__init__` tries to do this, but by then its own
    # `env_override` has pulled in vLLM, whose platform-plugin discovery
    # imports tpu_inference.platforms -> jax. So preload it here, first.
    try:
        import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
    except ImportError:
        pass
    import tpu_inference  # noqa: F401  isort: skip
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from tpu_inference.kernels.experimental.batched_rpa import \
        configs as batched_rpa_configs
    from tpu_inference.kernels.experimental.batched_rpa import \
        wrapper as batched_rpa
    from tpu_inference.kernels.experimental.rpa_v3_cp import \
        kernel as rpa_v3_cp
    from tpu_inference.kernels.ragged_paged_attention.v3 import \
        kernel as rpa_v3
    from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv
    from tpu_inference.layers.common import sharding as sharding_mod
    from tpu_inference.layers.common.attention_interface import \
        ragged_paged_attention
    from tpu_inference.layers.common.attention_metadata import (
        AttentionMetadata, PCPMetadata)
    from tpu_inference.layers.common.cp_attention import (pcp_forward,
                                                          pcp_forward_batched)
    from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                      ShardingAxisName,
                                                      ShardingAxisNameBase)

    # The N-D axis names carry `pcp`; select them regardless of
    # NEW_MODEL_DESIGN so the benchmark does not depend on the env.
    sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

    seq_along_lane = kv_layout_name == "seq_along_lane"
    kv_layout = (batched_rpa_configs.KVLayout.SEQ_ALONG_LANE if seq_along_lane
                 else batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE)

    NQ, NKV, HD = mp.num_q_heads, mp.num_kv_heads, mp.head_dim
    dtype = jnp.bfloat16
    kv_dtype = getattr(jnp, kv_dtype_name)
    sm_scale = HD**-0.5
    rng = np.random.default_rng(0)

    def rand(shape):
        return jnp.asarray(rng.random(shape, np.float32)).astype(dtype)

    def layer_collectives(o, axis, gather_axis=None):
        """The layer collectives around attention: the o_proj output
        ([tokens, hidden] bf16) all-reduced over the model axis `axis` and,
        when `gather_axis` is given (PCP), all-gathered over it between the
        token-sharded attention layout and the replicated MLP layout. Made
        data dependent on the attention output `o` so they are timed after
        it. Returns `o` unchanged with --no-collectives."""
        if not collectives:
            return o
        tokens = o.shape[0]
        act = jnp.broadcast_to(
            o.reshape(tokens, -1)[:, :1].astype(dtype),
            (tokens, mp.hidden_size))
        act = jax.lax.psum(act, axis)
        if gather_axis is not None:
            act = jax.lax.all_gather(act, gather_axis, axis=0, tiled=True)
        return o + act[:tokens, :1].astype(o.dtype).reshape((tokens, ) +
                                                            (1, ) *
                                                            (o.ndim - 1))

    def bench_cache(fn, cache, *args):
        # fn(cache, *args) -> (out, cache), cache donated and threaded so the
        # in-place cache write is timed and nothing is re-materialized per
        # step. Every iteration is queued and the host blocks once, so the
        # per-step time reflects device work, not `iters` host round trips.
        out, cache = fn(cache, *args)
        jax.block_until_ready(out)
        for _ in range(warmup):
            out, cache = fn(cache, *args)
            jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(iters):
            out, cache = fn(cache, *args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / iters * 1e3

    def make_tp(tp):
        """rpa_v3 with heads sharded over `tp` devices (KV heads replicated
        when tp > num_kv_heads, as the model does)."""
        nq, nkv = NQ // tp, max(1, NKV // tp)
        npages = max(cdiv(max_ctx, page), 1) * slack
        mesh = Mesh(np.array(jax.devices()[:tp]).reshape(tp), ("x", ))
        sharding = NamedSharding(mesh, P("x"))

        def put(x):
            return jax.device_put(x, sharding)

        q = put(jnp.broadcast_to(rand((chunk, nq, HD)), (tp, chunk, nq, HD)))
        k = put(
            jnp.broadcast_to(
                rand((chunk, nkv, HD)).astype(kv_dtype), (tp, chunk, nkv, HD)))
        v = put(
            jnp.broadcast_to(
                rand((chunk, nkv, HD)).astype(kv_dtype), (tp, chunk, nkv, HD)))
        # Per-device layout (packing of K/V heads per 32-bit word for the KV
        # dtype) is the kernel's own; the leading page dim is stacked over tp.
        per_dev = rpa_v3.get_kv_cache_shape(npages, page, nkv, HD, kv_dtype)
        cache_shape = (tp * per_dev[0], ) + tuple(per_dev[1:])
        pi = jnp.arange(npages, dtype=jnp.int32)
        cu = jnp.array([0, chunk], jnp.int32)
        dist = jnp.array([0, 0, 1], jnp.int32)

        def per_device(cache, q1, k1, v1, ctx1):
            out, cache = ragged_paged_attention(q1[0],
                                                k1[0],
                                                v1[0],
                                                cache,
                                                ctx1.reshape(1),
                                                pi,
                                                cu,
                                                dist,
                                                sm_scale=sm_scale,
                                                use_causal_mask=True,
                                                update_kv_cache=True)
            return layer_collectives(out, "x")[None], cache

        @functools.partial(jax.jit, donate_argnums=(0, ))
        def fn(cache, q, k, v, ctx):
            return jax.shard_map(per_device,
                                 mesh=mesh,
                                 in_specs=(P("x"), P("x"), P("x"), P("x"),
                                           P()),
                                 out_specs=(P("x"), P("x")),
                                 check_vma=False)(cache, q, k, v, ctx)

        def measure(ctx):
            # Donated and threaded, like the PCP path, so the cache write is
            # timed and no per-step cache materialization is.
            cache = jax.device_put(jnp.zeros(cache_shape, kv_dtype), sharding)
            return bench_cache(fn, cache, q, k, v, jnp.array(ctx, jnp.int32))

        return measure

    def make_pcp(pcp, tp, batched=False, num_reqs=1):
        """PCP through cp_attention on a (pcp, model) mesh.

        `batched` picks the batched RPA kernel (`pcp_forward_batched`) over
        rpa_v3_cp (`pcp_forward`). Both see the same inputs and the same cache
        layout; they differ in how the cache phase combines the per-rank
        partials (out-of-kernel LSE merge vs the in-kernel ring).

        With `num_reqs` > 1 the step's tokens are split evenly into that many
        requests, each cut into its own `2P` chunks so every request keeps its
        own head-tail balance. Request i owns virtual sequences 2i (head) and
        2i+1 (tail), and a rank holds them in request order. Only the batched
        kernel takes more than one request.

        Returns (measure, check): `measure(ctx)` times one chunk step at that
        context, `check(nchunk)` runs a chunked prefill and compares every
        request's output against a dense causal fp32 reference.
        """
        forward = pcp_forward_batched if batched else pcp_forward
        kv_cache_shape_fn = (batched_rpa.get_kv_cache_shape
                             if batched else rpa_v3_cp.get_kv_cache_shape)
        shape = tuple(pcp if a == "pcp" else tp if a == "model" else 1
                      for a in MESH_AXIS_NAMES)
        mesh = Mesh(
            np.array(jax.devices()[:pcp * tp]).reshape(shape), MESH_AXIS_NAMES)
        two_p = 2 * pcp
        if num_reqs > 1 and not batched:
            raise NotImplementedError(
                "multi-request PCP is only wired up for the batched kernel")
        if chunk % (num_reqs * two_p):
            raise NotImplementedError(
                f"chunk {chunk} does not split into {num_reqs} requests of "
                f"{two_p} chunks")
        # Per-request padded chunk piece, and tokens per request per step.
        C = chunk // (num_reqs * two_p)
        req_chunk = two_p * C
        # KV_CONTEXT shards the page dim: a global page holds page*pcp tokens.
        gpage = page * pcp
        if batched and C % page:
            raise NotImplementedError(
                f"PCP chunk size {C} must be a multiple of the page size "
                f"{page}")
        max_seq = max(MAX_SEQ, 2 * num_reqs)
        # Each request carries its own context and its own cache pages.
        req_max_ctx = max(max_ctx // num_reqs, req_chunk)
        pages_per_seq = max(cdiv(req_max_ctx, gpage), 1)
        npages = pages_per_seq * slack * num_reqs
        # Each rank holds `page` tokens of every global page (KV_CONTEXT shards
        # the page dim over pcp) and its own KV heads (KV_HEAD shards the
        # packed planes over tp); the per-rank layout is the CP kernel's own.
        if seq_along_lane and not batched:
            raise NotImplementedError(
                "SEQ_ALONG_LANE is only wired up for the batched kernel")
        layout_kw = {"kv_layout": kv_layout} if batched else {}
        per_rank = kv_cache_shape_fn(npages, page, NKV // tp, HD, kv_dtype,
                                     **layout_kw)
        if seq_along_lane:
            # (pages, kv_heads*2, head_dim//packing, packing, page): the
            # sequence is the lane dim, so KV_CONTEXT shards the last axis.
            cache_shape = (npages, per_rank[1] * tp, per_rank[2], per_rank[3],
                           gpage)
            cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_HEAD,
                           None, None, ShardingAxisName.KV_CONTEXT)
        else:
            cache_shape = (npages, gpage, per_rank[2] * tp) + tuple(
                per_rank[3:])
            cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                           ShardingAxisName.KV_HEAD, None, None)

        def put(x, s):
            return jax.device_put(x, NamedSharding(mesh, s))

        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
        kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
        q = put(rand((chunk, NQ, HD)), q_spec)
        k = put(rand((chunk, NKV, HD)).astype(kv_dtype), kv_spec)
        v = put(rand((chunk, NKV, HD)).astype(kv_dtype), kv_spec)
        # Virtual sequences 2i and 2i+1 are the head and tail of request i, so
        # they index the same pages.
        _pi = np.zeros((max_seq, pages_per_seq), np.int32)
        for i in range(num_reqs):
            _pi[2 * i] = _pi[2 * i + 1] = np.arange(i * pages_per_seq,
                                                    (i + 1) * pages_per_seq)
        pi = jnp.asarray(_pi.reshape(-1))
        dist = jnp.array([0, 0, 2 * num_reqs], jnp.int32)
        pcp_cu = np.zeros((pcp, max_seq + 1), np.int32)
        pcp_qp = np.zeros((pcp, max_seq), np.int32)
        for r in range(pcp):
            run = 0
            for i in range(num_reqs):
                toff = (two_p - 1 - r) * C
                treal = int(np.clip(req_chunk - toff, 0, C))
                # cu_q_lens defines the layout, so a piece occupies its full
                # stride; only the last one may be short, since nothing
                # follows it.
                run += C
                pcp_cu[r, 2 * i + 1] = run
                run += treal if i == num_reqs - 1 else C
                pcp_cu[r, 2 * i + 2:] = run
                pcp_qp[r, 2 * i] = r * C
                pcp_qp[r, 2 * i + 1] = toff
        pcp_spec = P(ShardingAxisName.PREFILL_CONTEXT, None)
        pcp_cu = put(jnp.asarray(pcp_cu), pcp_spec)
        pcp_qp = put(jnp.asarray(pcp_qp), pcp_spec)
        chunk_sizes = ({} if not batched else {
            "pcp_chunk_sizes": (C, ) * num_reqs,
            "kv_layout": kv_layout,
        })
        fns = {}

        def fn_for(cache_pages, with_collectives=True):
            # `cache_pages` is static metadata (one program per bucket), as in
            # the runner.
            key = (cache_pages, with_collectives)
            if key not in fns:

                @functools.partial(jax.jit, donate_argnums=(0, ))
                def fn(cache,
                       q,
                       k,
                       v,
                       kvl,
                       kvcl,
                       _cp=cache_pages,
                       _co=with_collectives):
                    md = AttentionMetadata(
                        input_positions=jnp.zeros(1, jnp.int32),
                        seq_lens=kvl,
                        block_tables=pi,
                        request_distribution=dist,
                        pcp=PCPMetadata(query_start_loc=pcp_cu,
                                        kv_cache_lens=kvcl,
                                        q_pos_offsets=pcp_qp,
                                        cache_pages=_cp),
                    )
                    cache, out = forward(mesh,
                                         q,
                                         k,
                                         v,
                                         cache,
                                         md,
                                         sm_scale=sm_scale,
                                         update_kv_cache=True,
                                         use_causal_mask=True,
                                         **chunk_sizes)
                    if _co:
                        # Heads are sharded over the model axis; all-reduce.
                        out = jax.shard_map(functools.partial(
                            layer_collectives,
                            axis=ShardingAxisName.ATTN_HEAD,
                            gather_axis=ShardingAxisName.PREFILL_CONTEXT),
                                            mesh=mesh,
                                            in_specs=q_spec,
                                            out_specs=q_spec,
                                            check_vma=False)(out)
                    return out, cache

                fns[key] = fn
            return fns[key]

        def cache_pages_for(computed):
            # Mirror the runner: live page count rounded up to a power of two.
            if computed <= 0:
                return 0
            live = cdiv(computed, gpage)
            return min(1 << max(live - 1, 0).bit_length(), npages)

        def seq_lens_for(req_ctx):
            """kv_lens / kv_cache_lens over the 2R virtual sequences."""
            kvl = np.zeros((max_seq, ), np.int32)
            kvcl = np.zeros((max_seq, ), np.int32)
            for i in range(num_reqs):
                kvl[2 * i:2 * i + 2] = req_ctx
                kvcl[2 * i:2 * i + 2] = max(req_ctx - req_chunk, 0)
            return jnp.asarray(kvl), jnp.asarray(kvcl)

        def measure(ctx):
            req_ctx = max(ctx // num_reqs, req_chunk)
            kvl, kvcl = seq_lens_for(req_ctx)
            cache = jax.device_put(jnp.zeros(cache_shape, kv_dtype),
                                   NamedSharding(mesh, cache_spec))
            return bench_cache(fn_for(cache_pages_for(req_ctx - req_chunk)),
                               cache, q, k, v, kvl, kvcl)

        # ---- correctness ---------------------------------------------------
        def to_rank_order(per_req):
            """[req][req_chunk, ...] token order -> the sharded rank order."""
            blocks = []
            for r in range(pcp):
                for x in per_req:
                    xc = np.asarray(x).reshape(two_p, C, *x.shape[1:])
                    blocks.append(xc[r])
                    blocks.append(xc[two_p - 1 - r])
            return np.concatenate(blocks, 0)

        def from_rank_order(y, i):
            """Pull request i's tokens back out, in its own token order."""
            local = 2 * C * num_reqs
            chunks = [None] * two_p
            for r in range(pcp):
                base = r * local + 2 * C * i
                chunks[r] = y[base:base + C]
                chunks[two_p - 1 - r] = y[base + C:base + 2 * C]
            return np.concatenate(chunks, 0)

        def check(nchunk=2):
            crng = np.random.default_rng(0)

            def cr(shape, dt):
                x = crng.standard_normal(shape, np.float32) * 0.5
                return jnp.asarray(x).astype(dt)

            qs = [[cr((req_chunk, NQ, HD), dtype) for _ in range(nchunk)]
                  for _ in range(num_reqs)]
            ks = [[cr((req_chunk, NKV, HD), kv_dtype) for _ in range(nchunk)]
                  for _ in range(num_reqs)]
            vs = [[cr((req_chunk, NKV, HD), kv_dtype) for _ in range(nchunk)]
                  for _ in range(num_reqs)]

            def reference(i, j):
                qq = np.asarray(qs[i][j], np.float32)
                kk = np.concatenate(
                    [np.asarray(x, np.float32) for x in ks[i][:j + 1]])
                vv = np.concatenate(
                    [np.asarray(x, np.float32) for x in vs[i][:j + 1]])
                g = NQ // NKV
                out = np.zeros_like(qq)
                pos_q = j * req_chunk + np.arange(req_chunk)[:, None]
                keep = np.arange(kk.shape[0])[None, :] <= pos_q
                for h in range(NQ):
                    s = (qq[:, h] @ kk[:, h // g].T) * sm_scale
                    s = np.where(keep, s, -np.inf)
                    s = s - s.max(-1, keepdims=True)
                    p = np.exp(s)
                    out[:, h] = (p / p.sum(-1, keepdims=True)) @ vv[:, h // g]
                return out

            cache = jax.device_put(jnp.zeros(cache_shape, kv_dtype),
                                   NamedSharding(mesh, cache_spec))
            worst = {}
            for j in range(nchunk):
                req_ctx = (j + 1) * req_chunk
                kvl, kvcl = seq_lens_for(req_ctx)
                fn = fn_for(cache_pages_for(j * req_chunk),
                            with_collectives=False)
                out, cache = fn(
                    cache,
                    put(jnp.asarray(to_rank_order([x[j] for x in qs])),
                        q_spec),
                    put(jnp.asarray(to_rank_order([x[j] for x in ks])),
                        kv_spec),
                    put(jnp.asarray(to_rank_order([x[j] for x in vs])),
                        kv_spec), kvl, kvcl)
                out = np.asarray(jax.block_until_ready(out), np.float32)
                for i in range(num_reqs):
                    ref = reference(i, j)
                    err = np.abs(from_rank_order(out, i) - ref).max()
                    rel = float(err / max(np.abs(ref).max(), 1e-6))
                    worst[f"req{i}_chunk{j}"] = rel
            return worst

        return measure, check

    check = None
    if variant.startswith("bpcp"):
        pcp, tp = (int(x) for x in variant[4:].split("xtp"))
        measure, check = make_pcp(pcp, tp, batched=True, num_reqs=num_reqs)
    elif variant.startswith("pcp"):
        pcp, tp = (int(x) for x in variant[3:].split("xtp"))
        measure, check = make_pcp(pcp, tp, num_reqs=num_reqs)
    else:
        if num_reqs > 1:
            raise NotImplementedError("multi-request is a PCP-only layout")
        measure = make_tp(int(variant[2:]))

    if check_chunks:
        if check is None:
            # TP has no PCP layout to verify; the PCP checks carry their own
            # dense reference, so there is nothing to compare it against.
            raise NotImplementedError("correctness check is PCP-only")
        rel = check(check_chunks)
        worst = max(rel.values())
        return {
            "check": "OK" if worst < 0.05 else "MISMATCH",
            "worst_rel": round(worst, 5),
            **{
                k: round(v, 5)
                for k, v in rel.items()
            },
        }

    ladder = _ladder(max_ctx)
    needed = sorted({b for m in ladder for b in _boundaries(m, chunk)})
    step = {c: measure(c) for c in needed}
    if profile_dir:
        _profile(measure, profile_dir, f"{mp_name(mp)}_{n}dev_{variant}",
                 [c for c in profile_contexts if c <= max_ctx])
    return {
        str(m): sum(step[b] for b in _boundaries(m, chunk))
        for m in ladder
    }


def mp_name(mp):
    return next(k for k, v in MODEL_CONFIGS.items() if v is mp)


def _profile(measure, profile_dir, name, contexts):
    """Record an xprof trace of `measure(ctx)` for each context under
    <profile_dir>/<name>/ctx<len>/ (uploaded with gsutil for gs:// paths)."""
    import jax
    local = tempfile.mkdtemp(prefix="pcp_vs_tp_prof_")
    for ctx in contexts:
        d = os.path.join(local, name, f"ctx{_human(ctx)}")
        os.makedirs(d, exist_ok=True)
        with jax.profiler.trace(d):
            measure(ctx)
    if profile_dir.startswith("gs://"):
        subprocess.check_call([
            "gsutil", "-q", "-m", "cp", "-r",
            os.path.join(local, name),
            profile_dir.rstrip("/") + "/"
        ])
        shutil.rmtree(local, ignore_errors=True)
    else:
        os.makedirs(profile_dir, exist_ok=True)
        dst = os.path.join(profile_dir, name)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(os.path.join(local, name), dst)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _variants(mp, n, pcp_sizes):
    out = [f"tp{n}"]
    if n >= 2 and mp.num_q_heads % (n // 2) == 0:
        out.append(f"tp{n // 2}")
    for p in pcp_sizes:
        if p <= 1 or n % p:
            continue
        tp = n // p
        if p % 2 or mp.num_q_heads % tp or mp.num_kv_heads % tp:
            # PCP needs an even pcp size and shards Q and KV heads over tp
            # (TP replicates KV heads when tp > num_kv_heads, PCP does not).
            continue
        out.append(f"pcp{p}xtp{tp}")
        out.append(f"bpcp{p}xtp{tp}")
    return out


def _cell(v, base):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    r = v / base
    if abs(r - 1) < 0.005:
        return f"{v:.2f} (=)"
    if r >= 1.5:
        return f"{v:.2f} ({r:.2f}x slower)"
    if r > 1:
        return f"{v:.2f} ({(r - 1) * 100:.1f}% slower)"
    return f"{v:.2f} ({(1 - r) * 100:.1f}% faster)"


def _box_table(header, rows):
    widths = [max(len(str(x)) for x in col) + 2 for col in zip(header, *rows)]

    def line(left, mid, right):
        return left + mid.join("─" * w for w in widths) + right

    def fmt(cells, center=False):
        parts = []
        for c, w in zip(cells, widths):
            c = str(c)
            parts.append(c.center(w) if center else " " + c.ljust(w - 1))
        return "│" + "│".join(parts) + "│"

    out = [line("┌", "┬", "┐"), fmt(header, center=True)]
    for i, row in enumerate(rows):
        out.append(line("├", "┼", "┤"))
        out.append(fmt(row))
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models",
                    nargs="*",
                    default=list(MODEL_CONFIGS),
                    choices=list(MODEL_CONFIGS))
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--max-context", type=int, default=1024 * 1024)
    ap.add_argument("--kv-dtype",
                    default="float8_e4m3fn",
                    choices=["bfloat16", "float8_e4m3fn"],
                    help="KV cache dtype (Q/K/V activations stay bf16)")
    ap.add_argument("--pcp-sizes",
                    nargs="*",
                    type=int,
                    default=[2, 4, 8],
                    help="pcp sizes to try (those dividing num_devices)")
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--cache-slack",
                    type=int,
                    default=1,
                    help="KV cache pages allocated = slack x request pages")
    ap.add_argument("--no-collectives",
                    action="store_true",
                    help="pure attention: skip the layer collectives (o_proj "
                    "all-reduce over the model axis and, for PCP, the "
                    "all-gather over the pcp axis into the MLP layout)")
    ap.add_argument("--profile-dir",
                    default=None,
                    help="also record xprof traces per layout here (local "
                    "dir or gs:// path)")
    ap.add_argument("--profile-contexts",
                    default="8192,65536,1048576",
                    help="comma-separated context lengths to profile")
    ap.add_argument("--retries",
                    type=int,
                    default=3,
                    help="re-run a layout whose worker failed (e.g. the TPU "
                    "was held by another process), 60s apart")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--requests",
                    type=int,
                    default=1,
                    help="split each step's tokens into this many PCP "
                    "requests (batched kernel only); each is cut into its "
                    "own 2P chunks")
    ap.add_argument("--check",
                    type=int,
                    default=0,
                    nargs="?",
                    const=2,
                    metavar="NCHUNK",
                    help="instead of timing, run NCHUNK chunks of prefill "
                    "and compare every request against a dense fp32 "
                    "reference (default 2 chunks)")
    ap.add_argument("--kv-layout",
                    default="head_along_sublane",
                    choices=["head_along_sublane", "seq_along_lane"],
                    help="KV cache layout (batched kernel only)")
    ap.add_argument("--worker",
                    nargs=3,
                    metavar=("MODEL", "NUM_DEVICES", "VARIANT"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        model, n, variant = args.worker
        try:
            res = _run_variant(
                MODEL_CONFIGS[model],
                variant,
                args.chunk_size,
                args.max_context,
                args.kv_dtype,
                args.page_size,
                args.cache_slack,
                args.warmup,
                args.iters,
                not args.no_collectives,
                args.profile_dir,
                [int(c) for c in args.profile_contexts.split(",")],
                int(n),
                num_reqs=args.requests,
                check_chunks=args.check,
                kv_layout_name=args.kv_layout)
        except NotImplementedError:
            # The kernel does not support this head config (e.g. batched RPA
            # needs a sublane-aligned head group to return LSE); report it
            # instead of failing the sweep.
            res = {"error": "unsupported"}
        except Exception as e:  # noqa: BLE001
            if "vmem" not in str(e):
                raise
            # The kernel's default tiles do not fit VMEM for this head
            # config on one device; report it instead of failing the sweep.
            res = {"error": "vmem OOM"}
        with open(args.json, "w") as f:
            json.dump(res, f)
        return

    tables = []
    for model in args.models:
        mp = MODEL_CONFIGS[model]
        for n in mp.num_devices:
            variants = _variants(mp, n, args.pcp_sizes)
            results = {}
            for variant in variants:
                with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                    cmd = [
                        sys.executable, __file__, "--worker", model,
                        str(n), variant, "--json", tf.name, "--chunk-size",
                        str(args.chunk_size), "--max-context",
                        str(args.max_context), "--kv-dtype", args.kv_dtype,
                        "--page-size",
                        str(args.page_size), "--cache-slack",
                        str(args.cache_slack), "--warmup",
                        str(args.warmup), "--iters",
                        str(args.iters), "--requests",
                        str(args.requests), "--kv-layout", args.kv_layout
                    ] + (["--check", str(args.check)] if args.check else
                         []) + (["--no-collectives"]
                                if args.no_collectives else []) + ([
                                    "--profile-dir", args.profile_dir,
                                    "--profile-contexts", args.profile_contexts
                                ] if args.profile_dir else [])
                    t0 = time.time()
                    for attempt in range(args.retries + 1):
                        if attempt:
                            time.sleep(60)
                        rc = subprocess.call(cmd)
                        # The worker writes its results before exiting; trust
                        # them even if the runtime's teardown returns non-zero.
                        try:
                            results[variant] = json.load(open(tf.name))
                        except (OSError, ValueError):
                            results[variant] = {}
                        if results[variant]:
                            break
                    status = "ok" if results[variant] else f"FAILED rc={rc}"
                    print(
                        f"  {model} {n} dev {variant}: {status} "
                        f"({time.time() - t0:.0f}s, {attempt + 1} attempt"
                        f"{'s' if attempt else ''})",
                        flush=True)
            if args.check:
                # Correctness mode: one row per layout, worst relative error
                # over every request and chunk.
                rows = []
                for v in variants:
                    r = results.get(v) or {"check": "FAILED"}
                    if "error" in r:
                        rows.append([v, r["error"], "-"])
                        continue
                    rows.append([
                        v,
                        r.get("check", "FAILED"),
                        f"{r.get('worst_rel', float('nan')):.5f}",
                    ])
                title = (f"{model}: {n} devices, {args.requests} request(s), "
                         f"CH={_human(args.chunk_size)}, KV {args.kv_dtype}, "
                         f"{args.kv_layout} -- {args.check}-chunk prefill vs "
                         f"dense fp32 reference")
                tables.append(
                    title + "\n\n" +
                    _box_table(["Layout", "Result", "Worst rel err"], rows))
                print("\n" + tables[-1] + "\n", flush=True)
                continue
            base = results[variants[0]]
            rows = []
            for ctx in _ladder(args.max_context):
                key = str(ctx)
                if key not in base:
                    continue
                rows.append([_human(ctx), f"{base[key]:.2f}"] + [
                    _cell(results[v].get("error") or results[v].get(key),
                          base[key]) for v in variants[1:]
                ])
            header = ["Context", f"{variants[0]} ({n} dev)"] + [
                f"{v} ({n // 2} dev)" if v == f"tp{n // 2}" else v
                for v in variants[1:]
            ]
            what = ("pure attention" if args.no_collectives else
                    f"attention + collectives: o_proj all-reduce, pcp "
                    f"all-gather (hidden={mp.hidden_size})")
            title = (f"{model}: NQ={mp.num_q_heads} NKV={mp.num_kv_heads} "
                     f"HD={mp.head_dim}, {n} devices, "
                     f"CH={_human(args.chunk_size)}, KV {args.kv_dtype}, "
                     f"{what} -- cumulative TTFT (ms), baseline "
                     f"{variants[0]}")
            tables.append(title + "\n\n" + _box_table(header, rows))
            print("\n" + tables[-1] + "\n", flush=True)

    if len(tables) > 1:
        print("\n\n".join(tables))


if __name__ == "__main__":
    main()
