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
import re
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


def _generate(mode: str,
              out_path: str,
              dp_size: int = 1,
              budget: int | None = None) -> None:
    """Generate the prompt set in this process and dump the result.

    Only called in the child process (see `__main__` below), so vllm is
    imported here rather than at module scope: the engine core is spawned,
    and a spawned child re-imports this file.
    """
    from vllm import LLM, SamplingParams

    additional_config = {}
    if dp_size > 1:
        additional_config["sharding"] = {
            "sharding_strategy": {
                "enable_dp_attention": True
            }
        }
    if budget is not None:
        additional_config["mamba_cache_checkpoint_budget"] = budget

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        tensor_parallel_size=dp_size,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        enable_prefix_caching=(mode == "on"),
        additional_config=additional_config,
        # Qwen3.5 carries a vision tower, but these prompts are text-only.
        limit_mm_per_prompt={
            "image": 0,
            "video": 0
        },
        disable_log_stats=False,  # get_metrics() needs the stat loggers
    )
    outputs = [
        llm.generate(SHARED_PREFIX + q,
                     SamplingParams(temperature=0.0, max_tokens=32))[0]
        for q in QUESTIONS
    ]

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


def _run_case(mode: str, dp_size: int = 1, budget: int | None = None) -> dict:
    """Run one engine in a subprocess and return its result."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, f"{mode}_dp{dp_size}.json")
        env = dict(os.environ,
                   MODEL_IMPL_TYPE="vllm",
                   SKIP_JAX_PRECOMPILE="1",
                   VLLM_XLA_CHECK_RECOMPILATION="0")
        cmd = [sys.executable, __file__, mode, out_path, str(dp_size)]
        if budget is not None:
            cmd.append(str(budget))
        proc = subprocess.run(cmd,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=1800)
        if not os.path.exists(out_path):
            raise AssertionError(
                f"prefix-caching-{mode} (dp={dp_size}) run produced no output\n"
                f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}")
        with open(out_path) as f:
            data = json.load(f)

        # Parse compact mamba KV cache log from stdout/stderr of EngineCore
        combined_logs = proc.stdout + proc.stderr
        match = re.search(
            r"Compact-mamba KV cache.*?num_gpu_blocks_override=(\d+).*?_mamba_num_blocks=(\d+)",
            combined_logs,
        )
        if match:
            data["cache_info"] = {
                "num_gpu_blocks_override": int(match.group(1)),
                "mamba_num_blocks": int(match.group(2)),
            }
        return data


def _verify_mamba_prefix_caching(dp_size: int = 1, budget: int | None = None):
    """Verify greedy output consistency with prefix caching on vs off."""
    baseline = _run_case("off", dp_size=dp_size, budget=budget)
    cached = _run_case("on", dp_size=dp_size, budget=budget)

    queries = cached["metrics"].get("vllm:prefix_cache_queries", 0)
    hits = cached["metrics"].get("vllm:prefix_cache_hits", 0)
    print(
        f"  prefix cache (dp={dp_size}, budget={budget}): {hits:.0f}/{queries:.0f} tokens hit "
        f"({hits / queries if queries else 0:.1%})")

    assert hits > 0, (
        f"no prefix cache hits (dp={dp_size}): the shared prefix was never "
        f"reused, so this test did not exercise mamba state reuse")

    # Verify decoupled dual KV cache pools
    if "cache_info" in cached:
        info = cached["cache_info"]
        attn_blocks = info.get("num_gpu_blocks_override")
        mamba_blocks = info.get("mamba_num_blocks")
        print(
            f"  KV Cache pools (dp={dp_size}): Attention={attn_blocks} blocks, "
            f"Mamba={mamba_blocks} blocks")
        if mamba_blocks is not None and attn_blocks is not None:
            assert attn_blocks > mamba_blocks, (
                f"Attention pool ({attn_blocks}) must be larger than compact Mamba pool ({mamba_blocks})"
            )
            if budget is not None:
                # With spec budget, mamba pool should be close to active + budget
                assert mamba_blocks <= 8 * 2 + budget + 8

    mismatched = [
        i for i, (
            b, c) in enumerate(zip(baseline["token_ids"], cached["token_ids"]))
        if b != c
    ]
    for i in mismatched:
        print(f"Token mismatch for prompt {i} ({QUESTIONS[i]!r}):")
        print(f"  prefix caching off: {baseline['texts'][i]!r}")
        print(f"  prefix caching on:  {cached['texts'][i]!r}")

    assert not mismatched, (
        f"{len(mismatched)}/{len(QUESTIONS)} prompts changed when prefix "
        f"caching was enabled (dp={dp_size}, budget={budget}); linear-attention "
        f"state is not being reused correctly")


def test_mamba_prefix_caching_matches_baseline():
    """Greedy output must not change when prefix caching is enabled (DP=1)."""
    _verify_mamba_prefix_caching(dp_size=1)


def test_mamba_prefix_caching_dp_matches_baseline():
    """Greedy output must not change when prefix caching is enabled under DP attention (DP=2)."""
    _verify_mamba_prefix_caching(dp_size=2)


def test_mamba_prefix_caching_with_tight_checkpoint_budget():
    """Verify that under a tight checkpoint budget, eviction triggers in Mamba,
    the coordinator reconciles min(L_attn, L_mamba), and generation remains bit-exact."""
    _verify_mamba_prefix_caching(dp_size=1, budget=16)


if __name__ == "__main__":
    # Child entry point for `_run_case`.
    dp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else None
    _generate(sys.argv[1], sys.argv[2], dp, budget=budget)
