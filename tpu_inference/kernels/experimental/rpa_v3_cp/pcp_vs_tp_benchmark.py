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

TP attention has no in-op collective, so it is measured on a shard_map over
the tp devices and blocked on all of them: it pays the same cross-device
dispatch/sync the PCP columns do. Meshes over different device subsets cannot
coexist in one TPU process, so every variant runs in its own subprocess.

Usage:
  python pcp_vs_tp_benchmark.py                       # all models, all layouts
  python pcp_vs_tp_benchmark.py --models Qwen3.5 --max-context 262144
  python pcp_vs_tp_benchmark.py --kv-dtype float8_e4m3fn --pcp-sizes 8
"""
import argparse
import dataclasses
import functools
import json
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


MODEL_CONFIGS = {
    # https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json
    'Qwen3-32B':
    ModelParams(
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[2],
    ),
    # google3/third_party/py/tpu_kernel_testbench/qwen35_config_defaults.json
    'Qwen3.5':
    ModelParams(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        num_devices=[8],
    ),
    # https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/config.json
    'Qwen3-Coder-480B':
    ModelParams(
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[8],
    ),
    # https://huggingface.co/google/gemma-4-31b-it/raw/main/config.json
    'Gemma4-31B':
    ModelParams(
        num_q_heads=32,
        num_kv_heads=16,
        head_dim=256,
        num_devices=[8],
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
                 warmup, iters):
    # tpu_inference must be imported before jax (its __init__ loads the
    # engine first).
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    import tpu_inference  # noqa: F401
    from tpu_inference.kernels.ragged_paged_attention.v3.util import (
        align_to, cdiv, get_dtype_packing)
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
    kvp = get_dtype_packing(kv_dtype)
    rng = np.random.default_rng(0)

    def rand(shape):
        return jnp.asarray(rng.random(shape, np.float32)).astype(dtype)

    def bench(fn, *args):
        # Queue every iteration and block once: per-step time reflects device
        # work, not `iters` host<->device round trips.
        jax.block_until_ready(fn(*args))
        for _ in range(warmup):
            jax.block_until_ready(fn(*args))
        t0 = time.perf_counter()
        out = None
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / iters * 1e3

    def bench_cache(fn, cache, *args):
        # fn(cache, *args) -> (out, cache), cache donated and threaded so the
        # in-place cache write is timed too.
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
        nkv2 = align_to(2 * nkv, kvp)
        npages = max(cdiv(max_ctx, page), 1) * slack
        mesh = Mesh(np.array(jax.devices()[:tp]).reshape(tp), ("x", ))
        put = lambda x: jax.device_put(x, NamedSharding(mesh, P("x")))
        q = put(jnp.broadcast_to(rand((chunk, nq, HD)), (tp, chunk, nq, HD)))
        k = put(
            jnp.broadcast_to(
                rand((chunk, nkv, HD)).astype(kv_dtype), (tp, chunk, nkv, HD)))
        v = put(
            jnp.broadcast_to(
                rand((chunk, nkv, HD)).astype(kv_dtype), (tp, chunk, nkv, HD)))
        pi = jnp.arange(npages, dtype=jnp.int32)
        cu = jnp.array([0, chunk], jnp.int32)
        dist = jnp.array([0, 0, 1], jnp.int32)

        def per_device(q1, k1, v1, ctx1):
            cache = jnp.zeros((npages, page, nkv2 // kvp, kvp, HD), kv_dtype)
            out, _ = ragged_paged_attention(q1[0],
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
            return out[None]

        @jax.jit
        def fn(q, k, v, ctx):
            return jax.shard_map(per_device,
                                 mesh=mesh,
                                 in_specs=(P("x"), P("x"), P("x"), P()),
                                 out_specs=P("x"),
                                 check_vma=False)(q, k, v, ctx)

        return lambda ctx: bench(fn, q, k, v, jnp.array(ctx, jnp.int32))

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
        # Packed KV planes are laid out per TP shard.
        nkv2 = tp * align_to(2 * max(1, NKV // tp), kvp)
        cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                       ShardingAxisName.KV_HEAD, None, None)
        put = lambda x, s: jax.device_put(x, NamedSharding(mesh, s))
        q = put(
            rand((chunk, NQ, HD)),
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None))
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
        fns = {}

        def fn_for(cache_pages):
            # `cache_pages` is static metadata (one program per bucket), as in
            # the runner.
            if cache_pages not in fns:

                @functools.partial(jax.jit, donate_argnums=(0, ))
                def fn(cache, q, k, v, kvl, kvcl, _cp=cache_pages):
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
                    cache, out = pcp_forward(mesh,
                                             q,
                                             k,
                                             v,
                                             cache,
                                             md,
                                             sm_scale=sm_scale,
                                             update_kv_cache=True,
                                             use_causal_mask=True)
                    return out, cache

                fns[cache_pages] = fn
            return fns[cache_pages]

        def cache_pages_for(ctx):
            # Mirror the runner: live page count rounded up to a power of two.
            computed = ctx - chunk
            if computed <= 0:
                return 0
            live = cdiv(computed, gpage)
            return min(1 << max(live - 1, 0).bit_length(), npages)

        def measure(ctx):
            kvl = jnp.zeros((MAX_SEQ, ), jnp.int32).at[:2].set(ctx)
            kvcl = jnp.zeros((MAX_SEQ, ),
                             jnp.int32).at[:2].set(max(ctx - chunk, 0))
            cache = jax.device_put(
                jnp.zeros((npages, gpage, nkv2 // kvp, kvp, HD), kv_dtype),
                NamedSharding(mesh, cache_spec))
            return bench_cache(fn_for(cache_pages_for(ctx)), cache, q, k, v,
                               kvl, kvcl)

        return measure

    if variant.startswith("pcp"):
        pcp, tp = (int(x) for x in variant[3:].split("xtp"))
        measure = make_pcp(pcp, tp)
    else:
        measure = make_tp(int(variant[2:]))

    ladder = _ladder(max_ctx)
    needed = sorted({b for n in ladder for b in _boundaries(n, chunk)})
    step = {c: measure(c) for c in needed}
    return {
        str(n): sum(step[b] for b in _boundaries(n, chunk))
        for n in ladder
    }


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
        if p % 2 or mp.num_q_heads % tp:
            continue  # PCP needs an even pcp size and whole Q heads per shard
        out.append(f"pcp{p}xtp{tp}")
    return out


def _cell(v, base):
    if v is None:
        return "n/a"
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

    def line(l, m, r):
        return l + m.join("─" * w for w in widths) + r

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
                    default="bfloat16",
                    choices=["bfloat16", "float8_e4m3fn"])
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
        res = _run_variant(MODEL_CONFIGS[model], variant, args.chunk_size,
                           args.max_context, args.kv_dtype, args.page_size,
                           args.cache_slack, args.warmup, args.iters)
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
                    ]
                    t0 = time.time()
                    rc = subprocess.call(cmd)
                    print(
                        f"  {model} {n} dev {variant}: "
                        f"{'ok' if rc == 0 else f'FAILED rc={rc}'} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True)
                    results[variant] = json.load(open(
                        tf.name)) if rc == 0 else {}
            base = results[variants[0]]
            rows = []
            for ctx in _ladder(args.max_context):
                key = str(ctx)
                if key not in base:
                    continue
                rows.append([_human(ctx), f"{base[key]:.2f}"] + [
                    _cell(results[v].get(key), base[key]) for v in variants[1:]
                ])
            header = ["Context", f"{variants[0]} ({n} dev)"] + [
                f"{v} ({n // 2} dev)" if v == f"tp{n // 2}" else v
                for v in variants[1:]
            ]
            title = (
                f"{model}: NQ={mp.num_q_heads} NKV={mp.num_kv_heads} "
                f"HD={mp.head_dim}, {n} devices, CH={_human(args.chunk_size)}, "
                f"KV {args.kv_dtype} -- cumulative TTFT (ms), baseline "
                f"{variants[0]}")
            tables.append(title + "\n\n" + _box_table(header, rows))
            print("\n" + tables[-1] + "\n", flush=True)

    if len(tables) > 1:
        print("\n\n".join(tables))


if __name__ == "__main__":
    main()
