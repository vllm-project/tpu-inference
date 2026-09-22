#!/usr/bin/env python3
"""Repro for mamba prefix-caching corruption on TPU.

Three independent checks, each runnable with the fixes on or off.

  --check allocator
      Drives `create_mamba_cache` directly. Dirties HBM with NaN, frees it, then
      reallocates the mamba pool and looks at what came back. No TPU serving
      stack needed, runs in ~30 s.

        unfixed (jnp.empty) -> pool comes back full of NaN
        fixed   (jnp.zeros) -> pool is clean

      NaN in a slot that a request later resumes from is what makes the model
      emit token 0 (`!`) forever.

  --check contamination
      Talks to a running vLLM server. Fires N concurrent requests that share one
      cold English prefix -- the shape a GRPO group has -- and asks each a
      question with a single verifiable answer. A request answering with another
      request's content means it read recurrent state that was not its own.

      Needs a COLD prefix, so every trial uses a fresh nonce. A warm/settled
      prefix does not reproduce, nor does a single stream, nor sequential
      multi-turn.

  --check prefix_sweep
      The `!` reproducer. Populates the cache with a long passage, then asks the
      same question at a sweep of prefix lengths, each a cache hit. A length
      whose mamba checkpoint was never written resumes from a slot holding
      someone else's state (or, on a poisoned pool, NaN -> logits NaN -> argmax
      picks token 0, which is `!`).

Server-side switches, so each check can run with and without the fixes:

    MAMBA_ZERO_NEW_BLOCKS=0|1         per-allocation zeroing of mamba slots
    MAMBA_WRITTEN_BOUNDARY_CLAMP=0|1  refuse hits on unwritten boundaries

Both default to 1. To see the bug, launch the server with both set to 0; to
force the failure hard rather than waiting for a dirty slot, also poison the
pool (see `_get_mamba_cache_allocator` in tpu_inference/runner/kv_cache.py).

Usage:

    # allocator check, no server required
    python repro.py --check allocator

    # against a server on :8000
    python repro.py --check contamination --trials 3 --streams 16
    python repro.py --check prefix_sweep
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "Qwen/Qwen3.5-35B-A3B"

PARA = (
    "Section {n}. The history of printing begins with the development of "
    "woodblock techniques, in which an entire page was carved into a single "
    "block of wood and then inked and pressed onto paper. Movable type, in "
    "which each character is an individual piece that can be rearranged, was "
    "a considerable advance, because it allowed a page to be composed quickly "
    "and the same characters to be reused across many different pages. The "
    "spread of printing changed how knowledge was stored and transmitted, "
    "making books cheaper, more numerous, and far more widely available than "
    "manuscripts copied by hand had ever been. "
)

QA = [
    ("What is the capital city of France? Answer with one word.", "paris"),
    ("What is two plus two? Answer with one number.", "4"),
    ("What colour is fresh snow? Answer with one word.", "white"),
    ("How many days are in a week? Answer with one number.", "7"),
    ("What is the largest ocean on Earth? Answer with one word.", "pacific"),
    ("What gas do humans breathe in to survive? Answer with one word.", "oxygen"),
    ("What is the chemical symbol for water? Answer with one short formula.", "h2o"),
    ("How many legs does a spider have? Answer with one number.", "8"),
]


# ---------------------------------------------------------------- allocator --


def check_allocator(args) -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from tpu_inference.runner.kv_cache import create_mamba_cache

    devices = np.array(jax.devices())
    n = len(devices)
    mesh = Mesh(devices.reshape(n), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    print(f"jax {jax.__version__}  {n}x {devices[0].device_kind}")

    # Shapes mirror a Qwen3.5-35B mamba pool; rounded to divide by device count.
    blocks = 4272 // n * n
    conv_shape = (blocks, 3, 8192)
    rec_shape = (blocks, 32, 128, 128)

    def dirty(shape, fill=np.nan, reps=3):
        for _ in range(reps):
            x = jax.device_put(np.full(shape, fill, dtype=np.float32), sharding)
            x.block_until_ready()
            del x

    def report(label, arr):
        a = np.asarray(jax.device_get(arr))
        nan = int(np.isnan(a).sum())
        nz = int(np.count_nonzero(a))
        print(f"  {label:<30} nonzero={nz:>13,}  nan={nan:>13,}")
        return nz, nan

    print("\ndirtying HBM with NaN, then allocating the mamba pool:")
    dirty(conv_shape)
    conv = create_mamba_cache(conv_shape, jnp.bfloat16, sharding)
    rec = create_mamba_cache(rec_shape, jnp.float32, sharding)
    nz1, nan1 = report("conv_state  (bf16)", conv)
    nz2, nan2 = report("recurrent   (f32)", rec)

    dirty_total = nz1 + nan1 + nz2 + nan2
    print()
    if dirty_total:
        print("RESULT: DIRTY -- the pool contains uninitialised memory.")
        print("        A request resuming from an unwritten slot reads NaN,")
        print("        which drives the sampler to token 0 ('!').")
        return 1
    print("RESULT: CLEAN -- the pool is zeroed.")
    return 0


# ------------------------------------------------------------ contamination --


def post(url, prompt, max_tokens=10):
    body = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["text"]


def check_contamination(args) -> int:
    from transformers import AutoTokenizer

    url = f"http://{args.host}:{args.port}/v1/completions"
    tok = AutoTokenizer.from_pretrained(args.model)
    total_bad = 0

    for trial in range(1, args.trials + 1):
        # A fresh nonce keeps the prefix COLD, so the streams race to prefill it.
        nonce = f"Trial {trial} seed {args.seed + trial}. "
        text = nonce + "".join(PARA.format(n=i) for i in range(1, 600))
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) >= args.prefix_len, f"passage too short: {len(ids)}"
        base = tok.decode(ids[:args.prefix_len])

        work = []
        for i in range(args.streams):
            q, expect = QA[i % len(QA)]
            work.append((f"{base}\n\nQuestion: {q}\nAnswer:", expect, i))

        with ThreadPoolExecutor(max_workers=args.streams) as ex:
            results = list(ex.map(lambda w: (post(url, w[0]), w[1], w[2]), work))

        bad = []
        for out, expect, i in sorted(results, key=lambda r: r[2]):
            ans = out.strip().split("\n")[0].strip()
            # `<think>` is a legitimate Qwen3.5 continuation on a raw completion
            # prompt, not corruption.
            ok = expect in ans.lower() or ans.startswith("<think>")
            if not ok:
                bad.append((i, expect, ans))
        total_bad += len(bad)
        print(f"  trial {trial}: {len(bad)}/{args.streams} wrong")
        for i, expect, ans in bad[:4]:
            print(f"      stream {i:>2} expected {expect:<8} got {ans[:58]!r}")

    print()
    if total_bad:
        print(f"RESULT: CORRUPTED -- {total_bad} wrong answers across "
              f"{args.trials} trials.")
        print("        Answers carrying another stream's content mean a request")
        print("        read recurrent state it did not write.")
        return 1
    print(f"RESULT: CLEAN -- 0 wrong across {args.trials} trials "
          f"({args.trials * args.streams} requests).")
    return 0


# ------------------------------------------------------------- prefix sweep --


def check_prefix_sweep(args) -> int:
    from transformers import AutoTokenizer

    url = f"http://{args.host}:{args.port}/v1/completions"
    question = ("\n\nQuestion: What is the capital city of France? "
                "Answer with one word.\nAnswer:")
    tok = AutoTokenizer.from_pretrained(args.model)
    text = "".join(PARA.format(n=i) for i in range(1, 600))
    ids = tok.encode(text, add_special_tokens=False)

    lengths = [int(x) for x in args.lengths.split(",")]
    need = max(lengths)
    assert len(ids) >= need, f"passage too short: {len(ids)} < {need}"

    # Populate: one cold pass over the whole passage.
    post(url, tok.decode(ids[:need]) + question, max_tokens=8)

    bad = 0
    for length in lengths:
        out = post(url, tok.decode(ids[:length]) + question, max_tokens=8)
        ans = out.strip()
        ok = "paris" in ans.lower()
        bad += not ok
        print(f"  {'OK  ' if ok else 'BAD '} L={length:<6} "
              f"block=(L-1)//256={max(length - 1, 0) // 256:<4} -> {ans[:50]!r}")

    print()
    if bad:
        print(f"RESULT: CORRUPTED -- {bad}/{len(lengths)} prefix lengths gave a "
              "wrong answer.")
        print("        Each is a cache hit resuming from a mamba slot no "
              "forward pass checkpointed.")
        return 1
    print(f"RESULT: CLEAN -- all {len(lengths)} prefix lengths answered "
          "correctly.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check",
                    choices=["allocator", "contamination", "prefix_sweep"],
                    required=True)
    ap.add_argument("--lengths",
                    default="256,512,1024,1536,2048,2560,3072,3584,4096,6144")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    checks = {
        "allocator": check_allocator,
        "contamination": check_contamination,
        "prefix_sweep": check_prefix_sweep,
    }
    return checks[args.check](args)


if __name__ == "__main__":
    sys.exit(main())
