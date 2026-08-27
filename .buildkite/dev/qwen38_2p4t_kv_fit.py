"""Does the requested benchmark shape fit in the KV cache this server built?

Qwen3.8-2.4T is a hybrid model: 23 full-attention layers share a paged KV pool,
69 Gated DeltaNet layers each hold one fixed-size recurrent slot per request.
The pool is sized from whatever HBM is left after 2.4 TB of weights, so it is
small in block terms and it is not knowable until the server has started.

Overrun is not a clean failure. The scheduler admits requests until it runs out
of blocks and then preempts, and with prefix caching off a preempted request is
recomputed from scratch -- so an oversubscribed run still reports numbers, just
noisy ones inflated by re-prefill. That is the opposite of what a benchmark is
for, so this is a hard gate rather than a warning.

Usage:
  python qwen38_2p4t_kv_fit.py --tokens-per-request 9216 --concurrency 8 \
      --block-size 128 [--log /root/vllm_serve.log] [--warn-only]
"""
import argparse
import math
import re
import sys

# runner/kv_cache_manager.py:_log_kv_cache_init. Attention layers get the full
# per-tensor block count, mamba layers get one slot per possible request, so the
# two ends of this list are the two capacities that matter.
_NUM_BLOCKS_RE = re.compile(r"Init kv-cache \|.*?num_blocks=\[([0-9,\s]+)\]")


def read_capacity(path: str) -> tuple[int, int]:
    """(attention blocks, mamba slots) from the server log, or (0, 0)."""
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return 0, 0
    # Every rank logs it and Ray may collapse duplicates; they agree, so the
    # last one is as good as the first.
    matches = _NUM_BLOCKS_RE.findall(text)
    if not matches:
        return 0, 0
    blocks = [int(x) for x in matches[-1].split(",")]
    return max(blocks), min(blocks)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-per-request", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--log", default="/root/vllm_serve.log")
    p.add_argument("--warn-only", action="store_true")
    a = p.parse_args()

    attn_blocks, mamba_slots = read_capacity(a.log)
    if not attn_blocks:
        # Not fatal: the log is the head rank's and the format could drift.
        # Refusing to run the benchmark over a parse miss would be worse than
        # running it.
        print(f"[kv-fit] SKIPPED: no 'Init kv-cache' line in {a.log}")
        return 0

    per_req = math.ceil(a.tokens_per_request / a.block_size)
    need = per_req * a.concurrency
    print(f"[kv-fit] pool={attn_blocks} blocks x {a.block_size} tok = "
          f"{attn_blocks * a.block_size} tok | mamba_slots={mamba_slots}")
    print(f"[kv-fit] need={need} blocks ({a.concurrency} x {per_req}) for "
          f"{a.tokens_per_request} tok/request -> "
          f"{100 * need / attn_blocks:.0f}% of pool")

    problems = []
    if need > attn_blocks:
        problems.append(
            f"attention pool holds {attn_blocks // per_req} requests of "
            f"{a.tokens_per_request} tokens, not {a.concurrency}")
    if a.concurrency > mamba_slots:
        problems.append(f"only {mamba_slots} recurrent slots for "
                        f"{a.concurrency} concurrent requests")
    if not problems:
        print("[kv-fit] OK")
        return 0

    for msg in problems:
        print(f"[kv-fit] {'WARNING' if a.warn_only else 'FATAL'}: {msg}")
    if a.warn_only:
        return 0
    print("[kv-fit] Lower MAX_CONCURRENCY, shorten the shape, or raise the "
          "pool; a preempting run produces numbers that measure re-prefill.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
