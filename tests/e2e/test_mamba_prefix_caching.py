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

# Largest baseline top-1 vs top-2 logprob gap (nats) at which a speculative
# decoding run may pick the runner-up token (see `_verify_mamba_prefix_caching`).
_NEAR_TIE_NATS = 0.5

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
              multiplier: int | None = None,
              num_spec_tokens: int = 0) -> None:
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
    if multiplier is not None:
        additional_config["custom_mamba_cache_multiplier"] = multiplier

    speculative_config = None
    if num_spec_tokens > 0:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": num_spec_tokens,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 1,
        }

    extra_kwargs = {}
    if num_spec_tokens > 0:
        # Dense mamba checkpoint retention, so blocks completed while decoding
        # are cached too (vLLM's default only keeps replay boundaries).
        extra_kwargs["prefix_cache_retention_interval"] = None

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        tensor_parallel_size=dp_size,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        enable_prefix_caching=(mode == "on"),
        speculative_config=speculative_config,
        additional_config=additional_config,
        # Qwen3.5 carries a vision tower, but these prompts are text-only.
        limit_mm_per_prompt={
            "image": 0,
            "video": 0
        },
        disable_log_stats=False,  # get_metrics() needs the stat loggers
        **extra_kwargs,
    )
    # Top-2 logprobs let the parent tell a near-tie flip from a real error.
    greedy = SamplingParams(temperature=0.0, max_tokens=32, logprobs=2)
    outputs = [
        llm.generate(SHARED_PREFIX + q, greedy)[0]
        for q in QUESTIONS
    ]
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    prompt_lens = [len(o.prompt_token_ids) for o in outputs]
    # Reuse a mamba block checkpoint exported mid-generation (with spec
    # decode: by a speculative verify window): the prompt stops 4 tokens
    # short of a block boundary, so block [0, block_size) is completed
    # (and its checkpoint copied out of the runner's scratch slots) only
    # while decoding. A follow-up that extends prompt + output then hits
    # it when blocks completed while decoding are retained.
    from vllm.inputs import TokensPrompt
    prefix_ids = llm.get_tokenizer().encode(SHARED_PREFIX + QUESTIONS[0])
    assert len(prefix_ids) > block_size, (len(prefix_ids), block_size)
    short_ids = prefix_ids[:block_size - 4]
    boundary_out = llm.generate(
        TokensPrompt(prompt_token_ids=short_ids),
        greedy)[0]
    followup_ids = (short_ids + list(boundary_out.outputs[0].token_ids[:8]) +
                    prefix_ids[block_size - 4:block_size + 12])
    followup_out = llm.generate(
        TokensPrompt(prompt_token_ids=followup_ids),
        greedy)[0]
    outputs += [boundary_out, followup_out]
    prompt_lens += [len(short_ids), len(followup_ids)]

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
                "top2_logprobs": [[
                    sorted(((int(t), float(lp.logprob))
                            for t, lp in d.items()),
                           key=lambda x: -x[1])[:2]
                    for d in (o.outputs[0].logprobs or [])
                ] for o in outputs],
                "texts": [o.outputs[0].text for o in outputs],
                "num_cached_tokens":
                [o.num_cached_tokens or 0 for o in outputs],
                "prompt_lens": prompt_lens,
                "block_size": block_size,
            }, f)


def _run_case(mode: str,
              dp_size: int = 1,
              multiplier: int | None = None,
              num_spec_tokens: int = 0,
              batched_rpa: bool = False) -> dict:
    """Run one engine in a subprocess and return its result."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, f"{mode}_dp{dp_size}_spec{num_spec_tokens}.json")
        env = dict(os.environ,
                   MODEL_IMPL_TYPE="vllm",
                   SKIP_JAX_PRECOMPILE="1",
                   VLLM_XLA_CHECK_RECOMPILATION="0")
        if batched_rpa:
            # Speculative decoding with GDN layers needs verify windows in
            # the decode segment (see `InputBatch.max_decode_tokens`).
            env["USE_BATCHED_RPA_KERNEL"] = "1"
        cmd = [
            sys.executable,
            __file__,
            mode,
            out_path,
            str(dp_size),
            "" if multiplier is None else str(multiplier),
            str(num_spec_tokens),
        ]
        proc = subprocess.run(cmd,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=1800)
        if not os.path.exists(out_path):
            raise AssertionError(
                f"prefix-caching-{mode} (dp={dp_size}, spec={num_spec_tokens}) run produced no output\n"
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


def _verify_mamba_prefix_caching(dp_size: int = 1,
                                 multiplier: int | None = None,
                                 num_spec_tokens: int = 0):
    """Verify greedy output consistency with prefix caching on vs off.

    The baseline never runs speculative decoding, so a spec decode case also
    checks that rejected drafts do not leak into the recurrent state.
    """
    baseline = _run_case("off",
                         dp_size=dp_size,
                         multiplier=multiplier,
                         num_spec_tokens=0,
                         batched_rpa=num_spec_tokens > 0)
    cached = _run_case("on",
                       dp_size=dp_size,
                       multiplier=multiplier,
                       num_spec_tokens=num_spec_tokens,
                       batched_rpa=num_spec_tokens > 0)

    queries = cached["metrics"].get("vllm:prefix_cache_queries", 0)
    hits = cached["metrics"].get("vllm:prefix_cache_hits", 0)
    print(
        f"  prefix cache (dp={dp_size}, multiplier={multiplier}, spec={num_spec_tokens}): "
        f"{hits:.0f}/{queries:.0f} tokens hit "
        f"({hits / queries if queries else 0:.1%})")

    assert hits > 0, (
        f"no prefix cache hits (dp={dp_size}): the shared prefix was never "
        f"reused, so this test did not exercise mamba state reuse")
    if num_spec_tokens > 0:
        # The follow-up (last output) extends a prompt that stopped short of
        # the first block boundary, so it can only hit that block if the
        # checkpoint written by a speculative verify window was cached.
        block_size = cached["block_size"]
        followup_cached = cached["num_cached_tokens"][-1]
        assert followup_cached >= block_size, (
            f"follow-up hit {followup_cached} cached tokens, expected >= "
            f"{block_size}: the mamba block completed during speculative "
            f"decoding was not cached")

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
            if multiplier is not None:
                # `custom_mamba_cache_multiplier` blocks per request in the
                # batch, plus the null block (less, if the joint HBM budget
                # forced a re-split).
                assert mamba_blocks <= multiplier * 8 + 1

    mismatched = [
        i for i, (
            b, c) in enumerate(zip(baseline["token_ids"], cached["token_ids"]))
        if b != c
    ]
    for i in mismatched:
        label = (QUESTIONS[i] if i < len(QUESTIONS) else
                 ("boundary" if i == len(QUESTIONS) else "followup"))
        print(f"Token mismatch for prompt {i} ({label!r}):")
        print(f"  prefix caching off: {baseline['texts'][i]!r}")
        print(f"  prefix caching on:  {cached['texts'][i]!r}")

    if num_spec_tokens > 0:
        # The baseline runs without speculative decoding, whose verify
        # windows batch tokens differently, so a near-tie (top-2 within
        # `_NEAR_TIE_NATS` in the baseline) can legitimately flip. Anything
        # else means the recurrent state went wrong.
        def is_near_tie(i: int) -> bool:
            b, c = baseline["token_ids"][i], cached["token_ids"][i]
            k = next(j for j in range(min(len(b), len(c))) if b[j] != c[j])
            top2 = baseline["top2_logprobs"][i][k]
            print(f"  prompt {i} diverges at token {k}: baseline top-2 "
                  f"{top2}, cached picked {c[k]}")
            return (len(top2) == 2 and c[k] == top2[1][0]
                    and top2[0][1] - top2[1][1] < _NEAR_TIE_NATS)

        mismatched = [i for i in mismatched if not is_near_tie(i)]

    assert not mismatched, (
        f"{len(mismatched)}/{len(baseline['token_ids'])} prompts changed when prefix "
        f"caching was enabled (dp={dp_size}, multiplier={multiplier}, spec={num_spec_tokens}); "
        f"linear-attention state is not being reused correctly")


def test_mamba_prefix_caching_matches_baseline():
    """Greedy output must not change when prefix caching is enabled (DP=1)."""
    _verify_mamba_prefix_caching(dp_size=1)


def test_mamba_prefix_caching_dp_matches_baseline():
    """Greedy output must not change when prefix caching is enabled under DP attention (DP=2)."""
    _verify_mamba_prefix_caching(dp_size=2)


def test_mamba_prefix_caching_with_tight_checkpoint_budget():
    """Verify that with only the resident blocks per request (2 instead of the
    default 8), eviction triggers in Mamba, the coordinator reconciles
    min(L_attn, L_mamba), and generation remains bit-exact."""
    _verify_mamba_prefix_caching(dp_size=1, multiplier=2)


def test_mamba_prefix_caching_spec_decode_matches_baseline():
    """Greedy output must not change when prefix caching is enabled with N-gram speculative decoding."""
    _verify_mamba_prefix_caching(dp_size=1, num_spec_tokens=3)


def test_mamba_prefix_caching_spec_decode_dp_matches_baseline():
    """Same as above under DP attention (DP=2): each rank carves its verify
    window scratch groups out of the top of its own mamba state shard."""
    _verify_mamba_prefix_caching(dp_size=2, num_spec_tokens=3)


if __name__ == "__main__":
    # Child entry point for `_run_case`.
    dp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    multiplier = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    num_spec_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    _generate(sys.argv[1],
              sys.argv[2],
              dp,
              multiplier=multiplier,
              num_spec_tokens=num_spec_tokens)
