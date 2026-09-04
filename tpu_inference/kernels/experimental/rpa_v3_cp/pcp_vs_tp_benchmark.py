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
def _run_variant(mp, variant, chunk, max_ctx, kv_dtype_name, page, slack,
                 warmup, iters, collectives, profile_dir, profile_contexts, n):
    # tpu_inference must be imported before jax (its __init__ loads the
    # engine first).
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    import tpu_inference  # noqa: F401
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
    from tpu_inference.layers.common.cp_attention import pcp_forward
    from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                      ShardingAxisName,
                                                      ShardingAxisNameBase)

    # The N-D axis names carry `pcp`; select them regardless of
    # NEW_MODEL_DESIGN so the benchmark does not depend on the env.
    sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

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

    def make_pcp(pcp, tp):
        """rpa_v3_cp through cp_attention.pcp_forward on a (pcp, model) mesh."""
        shape = tuple(pcp if a == "pcp" else tp if a == "model" else 1
                      for a in MESH_AXIS_NAMES)
        mesh = Mesh(
            np.array(jax.devices()[:pcp * tp]).reshape(shape), MESH_AXIS_NAMES)
        two_p, C = 2 * pcp, chunk // (2 * pcp)
        # KV_CONTEXT shards the page dim: a global page holds page*pcp tokens.
        gpage = page * pcp
        pages_per_seq = max(cdiv(max_ctx, gpage), 1)
        npages = pages_per_seq * slack
        # Each rank holds `page` tokens of every global page (KV_CONTEXT shards
        # the page dim over pcp) and its own KV heads (KV_HEAD shards the
        # packed planes over tp); the per-rank layout is the CP kernel's own.
        per_rank = rpa_v3_cp.get_kv_cache_shape(npages, page, NKV // tp, HD,
                                                kv_dtype)
        cache_shape = (npages, gpage, per_rank[2] * tp) + tuple(per_rank[3:])
        cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                       ShardingAxisName.KV_HEAD, None, None)

        def put(x, s):
            return jax.device_put(x, NamedSharding(mesh, s))

        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
        q = put(rand((chunk, NQ, HD)), q_spec)
        kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
        k = put(rand((chunk, NKV, HD)).astype(kv_dtype), kv_spec)
        v = put(rand((chunk, NKV, HD)).astype(kv_dtype), kv_spec)
        # Head and tail chunks of the request are two "sequences" of the same
        # request; both index the same pages.
        pg = jnp.arange(pages_per_seq, dtype=jnp.int32)
        pi = jnp.zeros(
            (MAX_SEQ * pages_per_seq, ),
            jnp.int32).at[:2 * pages_per_seq].set(jnp.concatenate([pg, pg]))
        dist = jnp.array([0, 0, 2], jnp.int32)
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
        # Single request: every seq's current-KV block starts at 0, and
        # kv_token_order maps token t to its slot in the rank-order buffer.
        row_perm = [c for r in range(pcp) for c in (r, two_p - 1 - r)]
        inv_row = np.empty(two_p, np.int64)
        inv_row[row_perm] = np.arange(two_p)
        kv_order = (inv_row[:, None] * C + np.arange(C)[None, :]).reshape(-1)
        kv_starts = put(jnp.zeros((MAX_SEQ, ), jnp.int32), P())
        kv_order = put(jnp.asarray(kv_order, jnp.int32), P())
        fns = {}

        def fn_for(has_cached_kv):
            # `has_cached_kv` is static metadata (one program per value), as
            # in the runner.
            if has_cached_kv not in fns:

                @functools.partial(jax.jit, donate_argnums=(0, ))
                def fn(cache, q, k, v, kvl, kvcl, _hc=has_cached_kv):
                    md = AttentionMetadata(
                        input_positions=jnp.zeros(1, jnp.int32),
                        seq_lens=kvl,
                        block_tables=pi,
                        request_distribution=dist,
                        pcp=PCPMetadata(query_start_loc=pcp_cu,
                                        kv_cache_lens=kvcl,
                                        q_pos_offsets=pcp_qp,
                                        kv_new_starts=kv_starts,
                                        kv_token_order=kv_order,
                                        has_cached_kv=_hc),
                    )
                    cache, out = pcp_forward(mesh,
                                             q,
                                             k,
                                             v,
                                             cache,
                                             md,
                                             sm_scale=sm_scale,
                                             update_kv_cache=True,
                                             use_causal_mask=True)
                    # Heads are sharded over the model axis; all-reduce there.
                    out = jax.shard_map(functools.partial(
                        layer_collectives,
                        axis=ShardingAxisName.ATTN_HEAD,
                        gather_axis=ShardingAxisName.PREFILL_CONTEXT),
                                        mesh=mesh,
                                        in_specs=q_spec,
                                        out_specs=q_spec,
                                        check_vma=False)(out)
                    return out, cache

                fns[has_cached_kv] = fn
            return fns[has_cached_kv]

        def measure(ctx):
            kvl = jnp.zeros((MAX_SEQ, ), jnp.int32).at[:2].set(ctx)
            kvcl = jnp.zeros((MAX_SEQ, ),
                             jnp.int32).at[:2].set(max(ctx - chunk, 0))
            cache = jax.device_put(jnp.zeros(cache_shape, kv_dtype),
                                   NamedSharding(mesh, cache_spec))
            return bench_cache(fn_for(ctx > chunk), cache, q, k, v, kvl, kvcl)

        return measure

    if variant.startswith("pcp"):
        pcp, tp = (int(x) for x in variant[3:].split("xtp"))
        measure = make_pcp(pcp, tp)
    else:
        measure = make_tp(int(variant[2:]))

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
                MODEL_CONFIGS[model], variant, args.chunk_size,
                args.max_context, args.kv_dtype, args.page_size,
                args.cache_slack, args.warmup, args.iters,
                not args.no_collectives, args.profile_dir,
                [int(c) for c in args.profile_contexts.split(",")], int(n))
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
                        str(args.iters)
                    ] + (["--no-collectives"]
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
