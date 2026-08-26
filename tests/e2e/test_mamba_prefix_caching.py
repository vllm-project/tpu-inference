# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching for linear-attention (mamba/GDN) layers.

Qwen3.5 interleaves full attention with gated-delta-net linear attention.
With prefix caching on, vLLM puts the linear-attention layers in
`mamba_cache_mode="align"`: the recurrent state is checkpointed into
prefix-cacheable blocks addressed by the mamba block table, so a request
that shares a prefix with an earlier one resumes from that request's state
instead of recomputing it.

The failure mode this guards against is silent: if the state slots are
misaddressed, a cache hit resumes from another request's recurrent state
and generation degrades into fluent nonsense rather than crashing. So the
check compares against a prefix-caching-off baseline, and asserts a
non-zero hit count so it cannot pass without exercising state reuse.

Each engine runs in its own subprocess: building a second `LLM` in a
process that already built one fails the engine-core handshake on TPU.
"""
import json
import os
import subprocess
import sys
import tempfile


MODEL_NAME = "Qwen/Qwen3.5-4B"

# Long enough that the shared part spans several mamba blocks, so a cache
# hit has to carry recurrent state across block boundaries rather than
# landing inside the first partial block.
SHARED_PREFIX = (
    "You are a meticulous technical editor. Below is an excerpt from a "
    "reference manual describing a distributed key-value store. The store "
    "partitions keys across a consistent hash ring, replicates each "
    "partition to three nodes, and serves reads from any replica while "
    "routing writes through a per-partition leader. Leaders are elected by "
    "a majority quorum and hold a lease that must be renewed before it "
    "expires. Clients cache the partition map and refresh it lazily when a "
    "request is rejected with a stale-map error. Compaction runs in the "
    "background and never blocks foreground traffic. Read the excerpt "
    "carefully and answer the question that follows, using only the "
    "information given above and nothing else. ") * 4

QUESTIONS = [
    "How many replicas does each partition have?",
    "What happens when a client's partition map is stale?",
    "Who serves writes for a partition?",
    "Does compaction block foreground traffic?",
    "How are leaders elected?",
    "What must a leader do before its lease expires?",
    "Where can reads be served from?",
    "How are keys distributed across nodes?",
]


def _generate(mode: str, out_path: str) -> None:
    """Generate the prompt set in this process and dump the result.

    Only called in the child process (see `__main__` below), so vllm is
    imported here rather than at module scope: the engine core is spawned,
    and a spawned child re-imports this file.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        enable_prefix_caching=(mode == "on"),
        # Qwen3.5 carries a vision tower, but these prompts are text-only.
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=False,  # get_metrics() needs the stat loggers
    )
    outputs = llm.generate([SHARED_PREFIX + q for q in QUESTIONS],
                           SamplingParams(temperature=0.0, max_tokens=32))

    metrics: dict[str, float] = {}
    for m in llm.get_metrics():
        if m.name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
            value = getattr(m, "value", None)
            if value is None:
                value = sum(getattr(m, "values", []) or [])
            metrics[m.name] = metrics.get(m.name, 0) + value

    with open(out_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
                "texts": [o.outputs[0].text for o in outputs],
            }, f)


def _run_case(mode: str) -> dict:
    """Run one engine in a subprocess and return its result."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, f"{mode}.json")
        env = dict(os.environ,
                   MODEL_IMPL_TYPE="vllm",
                   SKIP_JAX_PRECOMPILE="1",
                   VLLM_XLA_CHECK_RECOMPILATION="0")
        proc = subprocess.run([sys.executable, __file__, mode, out_path],
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=1800)
        if not os.path.exists(out_path):
            raise AssertionError(
                f"prefix-caching-{mode} run produced no output\n"
                f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}")
        with open(out_path) as f:
            return json.load(f)


def test_mamba_prefix_caching_matches_baseline():
    """Greedy output must not change when prefix caching is enabled.

    All prompts share a long prefix, so every request after the first hits
    the cache and resumes its linear-attention state from a block another
    request wrote. Any mis-addressing of those state blocks shows up as a
    token divergence here.
    """
    baseline = _run_case("off")
    cached = _run_case("on")

    queries = cached["metrics"].get("vllm:prefix_cache_queries", 0)
    hits = cached["metrics"].get("vllm:prefix_cache_hits", 0)
    print(f"  prefix cache: {hits:.0f}/{queries:.0f} tokens hit "
          f"({hits / queries if queries else 0:.1%})")

    # Without hits the comparison below would pass trivially, having
    # exercised none of the state-reuse path.
    assert hits > 0, (
        "no prefix cache hits: the shared prefix was never reused, so this "
        "test did not exercise mamba state reuse")

    mismatched = [
        i for i, (b, c) in enumerate(
            zip(baseline["token_ids"], cached["token_ids"])) if b != c
    ]
    for i in mismatched:
        print(f"Token mismatch for prompt {i} ({QUESTIONS[i]!r}):")
        print(f"  prefix caching off: {baseline['texts'][i]!r}")
        print(f"  prefix caching on:  {cached['texts'][i]!r}")

    assert not mismatched, (
        f"{len(mismatched)}/{len(QUESTIONS)} prompts changed when prefix "
        f"caching was enabled; linear-attention state is not being reused "
        f"correctly")


if __name__ == "__main__":
    # Child entry point for `_run_case`.
    _generate(sys.argv[1], sys.argv[2])
