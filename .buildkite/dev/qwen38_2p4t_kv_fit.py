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
# two ends of this list are the two capacities that matter. This is what the
# *worker* physically allocated.
_NUM_BLOCKS_RE = re.compile(r"Init kv-cache \|.*?num_blocks=\[([0-9,\s]+)\]")

# ...which is not necessarily what the *scheduler* will hand out. The block
# pool that governs admission is built in the engine-core process from vLLM's
# own get_kv_cache_config(), and on a hybrid model that count is derived from
# the combined attention+mamba page size -- as though every attention block
# needed a mamba slot beside it. The worker's mamba-aware sizing, and the
# `cache_config.num_gpu_blocks_override` it sets, live on the far side of an
# RPC boundary and do not reach it unless the launcher also passes
# `--num-gpu-blocks-override`.
#
# Build #960 is why this is parsed separately: the worker allocated 15,150
# attention blocks and this gate happily reported a 1,939,200-token pool, while
# the scheduler was admitting against 661 blocks / 83,822 tokens. The benchmark
# leg passed the gate and the eval leg then lost 28 of 198 questions to the
# preemption storm the gate exists to prevent. Trust the scheduler's number.
_SCHED_BLOCKS_RE = re.compile(r"kv_cache_config\.num_blocks=([0-9]+)")


def read_capacity(path: str) -> tuple[int, int, int]:
    """(attention blocks, mamba slots, scheduler blocks) from the server log.

    Any element is 0 if that line was not found.
    """
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return 0, 0, 0
    # Every rank logs it and Ray may collapse duplicates; they agree, so the
    # last one is as good as the first.
    sched = _SCHED_BLOCKS_RE.findall(text)
    sched_blocks = int(sched[-1]) if sched else 0
    matches = _NUM_BLOCKS_RE.findall(text)
    if not matches:
        return 0, 0, sched_blocks
    blocks = [int(x) for x in matches[-1].split(",")]
    return max(blocks), min(blocks), sched_blocks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-per-request", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--log", default="/root/vllm_serve.log")
    p.add_argument("--warn-only", action="store_true")
    a = p.parse_args()

    attn_blocks, mamba_slots, sched_blocks = read_capacity(a.log)
    if not attn_blocks:
        # Not fatal: the log is the head rank's and the format could drift.
        # Refusing to run the benchmark over a parse miss would be worse than
        # running it.
        print(f"[kv-fit] SKIPPED: no 'Init kv-cache' line in {a.log}")
        return 0

    # The scheduler's count is the one that gates admission. Fall back to the
    # worker's only if the layout line is missing, and say so, because that
    # fallback is exactly the blind spot that let #960 through.
    if sched_blocks:
        pool = sched_blocks
    else:
        pool = attn_blocks
        print("[kv-fit] WARNING: no 'kv_cache_config.num_blocks' line found; "
              "falling back to the worker's allocation, which can overstate "
              "the pool the scheduler actually admits against.")

    per_req = math.ceil(a.tokens_per_request / a.block_size)
    need = per_req * a.concurrency
    print(f"[kv-fit] pool={pool} blocks x {a.block_size} tok = "
          f"{pool * a.block_size} tok | mamba_slots={mamba_slots} | "
          f"worker_alloc={attn_blocks} blocks")
    print(f"[kv-fit] need={need} blocks ({a.concurrency} x {per_req}) for "
          f"{a.tokens_per_request} tok/request -> "
          f"{100 * need / pool:.0f}% of pool")

    problems = []
    if sched_blocks and attn_blocks and sched_blocks < attn_blocks:
        # Not fatal on its own -- the run is still correct, just smaller than
        # it looks -- but it means HBM was reserved and cannot be used.
        #
        # Since #3481 the worker derives its tensor sizing from
        # kv_cache_config.num_blocks, so this should no longer be reachable.
        # Kept as a regression detector: if it ever fires again, the two sides
        # have gone back to sizing the pool independently.
        print(f"[kv-fit] WARNING: scheduler admits against {sched_blocks} "
              f"blocks but the worker allocated {attn_blocks}; "
              f"{attn_blocks - sched_blocks} blocks "
              f"({(attn_blocks - sched_blocks) * a.block_size} tokens) are "
              f"reserved and unaddressable. Pass --num-gpu-blocks-override to "
              f"the server so the engine-core config matches the allocation.")
    if need > pool:
        problems.append(
            f"attention pool holds {pool // per_req} requests of "
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
