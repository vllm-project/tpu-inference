# SPDX-License-Identifier: Apache-2.0
"""Inference benchmark to model large-scale async RL training with GRPO in vLLM.

This script simulates a continuous stream of GRPO training requests:
- A prompt of 4k-16k tokens is shared across a group of g=16 streams.
- Each stream is a multi-turn conversation (10-100 turns).
- Each turn generates 200-2k tokens, followed by a simulated environment
  response of 10-100 tokens.
- All streams run concurrently and asynchronously, testing vLLM's ability
  to handle prefix caching and async scheduling of multi-turn conversations.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer


def make_client_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None,
                                      sock_read=None,
                                      sock_connect=30),
        connector=aiohttp.TCPConnector(force_close=True, limit=0),
    )


def get_percentile(data: List[float], percentile: float) -> float:
    """Calculates the percentile value of a list of numbers.

    Args:
        data: List of numbers.
        percentile: Percentile value between 0 and 100.

    Returns:
        float: Calculated percentile.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100.0
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    if upper < len(sorted_data):
        return sorted_data[lower] * (1.0 -
                                     weight) + sorted_data[upper] * weight
    return sorted_data[lower]


def build_shared_preamble(tokenizer: AutoTokenizer,
                          args: argparse.Namespace) -> str:
    """Builds the preamble that every rollout in every batch shares.

    Real agentic RL prompts open with a long block that is byte-identical
    across every rollout and every training batch -- system instructions plus
    tool definitions. That block is the only thing a *new* batch can hit in the
    prefix cache, so it is the only thing a weight-sync cache-salt rotation
    actually invalidates. Without it the batches share nothing (each group's
    prompt is random tokens) and salting is unmeasurable.

    Generated from a fixed seed so it is identical across processes and runs,
    independent of `--seed`.

    Args:
        tokenizer: Tokenizer to decode tokens.
        args: Parsed arguments.

    Returns:
        str: The shared preamble text, or "" when disabled.
    """
    if args.shared_preamble_len <= 0:
        return ""
    rng = random.Random(0xA9E17)
    token_ids = [
        rng.randint(1000, 50000) for _ in range(args.shared_preamble_len)
    ]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_initial_prompt(tokenizer: AutoTokenizer,
                            args: argparse.Namespace) -> str:
    """Generates a random initial prompt of a specific token length.

    Args:
        tokenizer: Tokenizer to decode tokens.
        args: Parsed command line arguments.

    Returns:
        str: Generated prompt text.
    """
    length = random.randint(args.initial_prompt_len_min,
                            args.initial_prompt_len_max)
    # Using safe token ID range to avoid special control characters
    token_ids = [random.randint(1000, 50000) for _ in range(length)]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


async def run_grpo_stream(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    stream_idx: int,
    group_idx: int,
    args: argparse.Namespace,
    shared_preamble: str = "",
    cache_salt: Optional[str] = None,
    batch_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Runs a single GRPO stream as a multi-turn conversation.

    Args:
        session: aiohttp ClientSession for making requests.
        url: URL of the OpenAI Chat Completion API.
        model: Model name to use in the API request.
        tokenizer: Tokenizer to count/generate tokens.
        initial_prompt: The long shared initial prompt.
        stream_idx: Index of the stream within the group.
        group_idx: Index of the group/request.
        args: Parsed command line arguments.

    Returns:
        List[Dict[str, Any]]: Statistics of each turn in the stream.
    """
    num_turns = random.randint(args.turns_min, args.turns_max)
    # The shared preamble must sit at the very front of the token sequence to
    # be a real prefix, so it goes in the system message ahead of everything
    # else. cache_salt is folded into the first block's hash, which chains
    # through the rest -- so rotating the salt makes exactly this preamble
    # (and any other shared prefix) unreachable to the new batch.
    system_content = "You are a helpful assistant."
    if shared_preamble:
        system_content = f"{shared_preamble}\n\n{system_content}"
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": initial_prompt
        },
    ]

    stats = []

    for turn in range(1, num_turns + 1):
        max_tokens = random.randint(args.output_len_min, args.output_len_max)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": args.temperature,
        }
        if args.ignore_eos:
            payload["ignore_eos"] = True
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        # Ask the server to report the token counts it actually processed.
        # Sent as a final chunk with an empty "choices" list.
        payload["stream_options"] = {"include_usage": True}

        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        start_time = time.perf_counter()
        ttft = None
        usage = None
        full_response_text = []

        try:
            async with session.post(url, json=payload,
                                    headers=headers) as response:
                if response.status != 200:
                    err_text = await response.text()
                    raise RuntimeError(f"Status {response.status}: {err_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if ttft is None:
                                ttft = ((time.perf_counter() - start_time) *
                                        1000.0)

                            if data.get("usage"):
                                usage = data["usage"]

                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response_text.append(content)
                        except Exception:
                            pass

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000.0

            assistant_response = "".join(full_response_text)

            # Token counts come from the server: they are what the engine
            # actually processed. Counting client-side instead is inaccurate
            # in both directions -- re-encoding detokenized text is not
            # round-trip identity, and any tokens the server reports outside
            # "content" are invisible here.
            if not usage:
                raise RuntimeError(
                    "server did not report usage; it must support "
                    "stream_options.include_usage")

            prompt_tokens = usage["prompt_tokens"]
            assistant_tokens = usage["completion_tokens"]
            # Blocks the engine actually reused. This is the measured prefix
            # cache hit rate; the Turn-1 TTFT split further down only infers
            # hits from latency and cannot see cross-batch reuse at all.
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens")

            if ttft is None:
                ttft = total_time_ms

            if assistant_tokens > 1:
                tpot = (total_time_ms - ttft) / (assistant_tokens - 1)
            else:
                tpot = total_time_ms

            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

            turn_stat = {
                "group_idx": group_idx,
                "stream_idx": stream_idx,
                "batch_idx": batch_idx,
                "cache_salt": cache_salt,
                "turn": turn,
                "num_turns": num_turns,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "total_time_ms": total_time_ms,
                "output_tokens": assistant_tokens,
                "input_history_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "success": True,
            }

            # Simulate environment response of 10-100 tokens
            env_len = random.randint(args.env_len_min, args.env_len_max)
            env_token_ids = [
                random.randint(1000, 50000) for _ in range(env_len)
            ]
            env_text = tokenizer.decode(env_token_ids,
                                        skip_special_tokens=True)

            messages.append({"role": "user", "content": env_text})
            turn_stat["env_tokens"] = len(tokenizer.encode(env_text))

            stats.append(turn_stat)

        except Exception as e:
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
            # Include the type: some exceptions (notably asyncio.TimeoutError)
            # have an empty str(), which would otherwise record the failure
            # with no reason at all.
            error_msg = f"{type(e).__name__}: {e}" if str(e) else \
                type(e).__name__
            stats.append({
                "group_idx": group_idx,
                "stream_idx": stream_idx,
                "batch_idx": batch_idx,
                "cache_salt": cache_salt,
                "turn": turn,
                "num_turns": num_turns,
                "ttft_ms": total_time_ms,
                "tpot_ms": 0.0,
                "total_time_ms": total_time_ms,
                "output_tokens": 0,
                "input_history_tokens": 0,
                "cached_tokens": None,
                "success": False,
                "error": error_msg,
            })
            # End conversation on error
            break

    return stats


async def run_group(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: AutoTokenizer,
    group_idx: int,
    args: argparse.Namespace,
    shared_preamble: str = "",
    cache_salt: Optional[str] = None,
    batch_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Runs a single GRPO group of G parallel streams.

    Args:
        session: aiohttp ClientSession.
        url: API URL.
        model: Model name.
        tokenizer: Model tokenizer.
        group_idx: Index of this group.
        args: Parsed arguments.

    Returns:
        List[Dict[str, Any]]: Accumulated stats of all streams in the group.
    """
    initial_prompt = generate_initial_prompt(tokenizer, args)
    initial_prompt_len = len(tokenizer.encode(initial_prompt))
    print(f"Group {group_idx}: Starting {args.group_size} streams with "
          f"shared initial prompt of {initial_prompt_len} tokens...")

    tasks = []
    for stream_idx in range(args.group_size):
        tasks.append(
            run_grpo_stream(
                session,
                url,
                model,
                tokenizer,
                initial_prompt,
                stream_idx,
                group_idx,
                args,
                shared_preamble,
                cache_salt,
                batch_idx,
            ))

    results = await asyncio.gather(*tasks)

    flat_results = []
    for stream_result in results:
        flat_results.extend(stream_result)
    return flat_results


def print_batch_cache_report(all_stats: List[Dict[str, Any]],
                             sync_events: List[Dict[str, Any]]) -> None:
    """Prints measured prefix-cache reuse per RL batch.

    Uses `cached_tokens` reported by the server, which is the number of prompt
    tokens actually served from cached blocks. This is the number that shows
    whether salt rotation did its job: the batch launched after a weight sync
    should lose its share of the cross-batch prefix while keeping its own
    within-group and within-stream reuse.
    """
    ok = [s for s in all_stats if s["success"]]
    if not ok or all(s.get("cached_tokens") is None for s in ok):
        print("\nPer-batch cache reuse: server did not report "
              "prompt_tokens_details.cached_tokens.")
        return

    batches = sorted({s.get("batch_idx", 0) for s in ok})
    print("-" * 80)
    print("PREFIX CACHE REUSE BY RL BATCH (measured)")
    print("-" * 80)
    print(f"{'batch':>6} {'salt':>14} {'turns':>8} {'prompt tok':>14} "
          f"{'cached tok':>14} {'hit rate':>10}")
    for b in batches:
        rows = [s for s in ok if s.get("batch_idx", 0) == b]
        prompt = sum(s["input_history_tokens"] for s in rows)
        cached = sum(s.get("cached_tokens") or 0 for s in rows)
        rate = (cached / prompt * 100.0) if prompt else 0.0
        salt = next((s.get("cache_salt") for s in rows if s.get("cache_salt")),
                    "-")
        print(f"{b:>6} {str(salt):>14} {len(rows):>8} {prompt:>14,} "
              f"{cached:>14,} {rate:>9.2f}%")

    turn1 = [s for s in ok if s["turn"] == 1]
    if turn1:
        print("\nTurn 1 only (first contact with the shared preamble):")
        for b in batches:
            rows = [s for s in turn1 if s.get("batch_idx", 0) == b]
            if not rows:
                continue
            prompt = sum(s["input_history_tokens"] for s in rows)
            cached = sum(s.get("cached_tokens") or 0 for s in rows)
            rate = (cached / prompt * 100.0) if prompt else 0.0
            print(f"  batch {b}: {cached:,}/{prompt:,} tokens cached "
                  f"({rate:.2f}%)")

    if sync_events:
        print("\nWeight sync windows:")
        for e in sync_events:
            state = "paused" if e["paused"] else "NOT PAUSED (dev mode off)"
            print(f"  -> {e['version']}: {e['wall_sec']:.2f}s, {state}")


def print_report(
    all_stats: List[Dict[str, Any]],
    total_duration_sec: float,
    args: argparse.Namespace,
    sync_events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Calculates and prints the benchmark performance report.

    Args:
        all_stats: List of all turn statistics collected.
        total_duration_sec: Total duration of the benchmark run.
        args: Parsed arguments.
        sync_events: Weight sync windows taken during the run, if any.
    """
    print("\n" + "=" * 80)
    print("GRPO BENCHMARK PERFORMANCE REPORT")
    print("=" * 80)

    # General execution info
    total_turns = len(all_stats)
    successful_turns = sum(1 for s in all_stats if s["success"])
    failed_turns = total_turns - successful_turns
    success_rate = ((successful_turns / total_turns) *
                    100.0 if total_turns > 0 else 0.0)

    total_groups = len(set(s["group_idx"] for s in all_stats))
    total_streams = len(
        set((s["group_idx"], s["stream_idx"]) for s in all_stats))

    total_input_tokens = sum(s["input_history_tokens"] for s in all_stats)
    total_output_tokens = sum(s["output_tokens"] for s in all_stats)
    total_tokens = total_input_tokens + total_output_tokens

    print(f"Total Benchmark Time:      {total_duration_sec:.2f} seconds")
    print(f"Simulated GRPO Groups:     {total_groups}")
    print(f"Simulated Streams (g):     {total_streams}")
    print(f"Total Conversational Turns:{total_turns} (Success: "
          f"{successful_turns}, Failed: {failed_turns}, "
          f"Rate: {success_rate:.2f}%)")
    if failed_turns:
        reasons = Counter(s["error"] for s in all_stats if not s["success"])
        print("Failure Breakdown:")
        for reason, count in reasons.most_common():
            print(f"  {count:6d} x {reason}")
    print(f"Total Input Tokens (Pref): {total_input_tokens:,}")
    print(f"Total Output Tokens (Dec): {total_output_tokens:,}")
    print(f"Total Tokens Processed:    {total_tokens:,}")

    # Throughput metrics
    groups_per_sec = total_groups / total_duration_sec
    streams_per_sec = total_streams / total_duration_sec
    turns_per_sec = total_turns / total_duration_sec
    input_tokens_per_sec = total_input_tokens / total_duration_sec
    output_tokens_per_sec = total_output_tokens / total_duration_sec
    total_tokens_per_sec = total_tokens / total_duration_sec

    print("-" * 80)
    print("THROUGHPUT METRICS")
    print("-" * 80)
    print(f"GRPO Groups / sec:         {groups_per_sec:.4f}")
    print(f"Streams (Rollouts) / sec:  {streams_per_sec:.4f}")
    print(f"Conversation Turns / sec:  {turns_per_sec:.4f}")
    print(f"Input Tokens / sec:        {input_tokens_per_sec:.2f}")
    print(f"Output Tokens / sec:       {output_tokens_per_sec:.2f}")
    print(f"Total Tokens / sec:        {total_tokens_per_sec:.2f}")

    # Divide Turn 1 TTFT into "Prefill/Miss" vs "Cached/Hit" streams
    # For each group, the stream with the maximum TTFT at Turn 1 is the Miss (Prefill).
    # The others are Hits.
    turn1_miss_ttft = []
    turn1_hit_ttft = []
    subsequent_ttft = []
    all_tpot = []

    # Group statistics by group index for turn 1
    groups_turn1: Dict[int, List[Dict[str, Any]]] = {}
    for stat in all_stats:
        if not stat["success"]:
            continue
        all_tpot.append(stat["tpot_ms"])
        if stat["turn"] == 1:
            g_idx = stat["group_idx"]
            if g_idx not in groups_turn1:
                groups_turn1[g_idx] = []
            groups_turn1[g_idx].append(stat)
        else:
            subsequent_ttft.append(stat["ttft_ms"])

    for g_idx, stats in groups_turn1.items():
        if not stats:
            continue
        # Find the one with highest TTFT (assumed to be the cache miss prefill)
        sorted_stats = sorted(stats, key=lambda x: x["ttft_ms"], reverse=True)
        turn1_miss_ttft.append(sorted_stats[0]["ttft_ms"])
        for rem in sorted_stats[1:]:
            turn1_hit_ttft.append(rem["ttft_ms"])

    print("-" * 80)
    print("LATENCY METRICS")
    print("-" * 80)

    def print_latency_row(label: str, latencies: List[float]):
        if not latencies:
            print(f"{label:<30} N/A")
            return
        avg = sum(latencies) / len(latencies)
        p50 = get_percentile(latencies, 50)
        p90 = get_percentile(latencies, 90)
        p99 = get_percentile(latencies, 99)
        print(f"{label:<30} Avg: {avg:8.2f}ms | p50: {p50:8.2f}ms | "
              f"p90: {p90:8.2f}ms | p99: {p99:8.2f}ms")

    print_latency_row("Turn 1 - Cache Miss (Prefill)", turn1_miss_ttft)
    print_latency_row("Turn 1 - Cache Hit (Cached)", turn1_hit_ttft)
    print_latency_row("Turns 2+ - TTFT (Multi-turn)", subsequent_ttft)
    print_latency_row("TPOT (Time per Output Token)", all_tpot)

    # Prefix Cache Hit Ratio Analysis
    total_turn1 = len(turn1_miss_ttft) + len(turn1_hit_ttft)
    if total_turn1 > 0:
        hit_ratio = (len(turn1_hit_ttft) / total_turn1) * 100.0
        print(
            f"\nTurn 1 Prefix Cache Hits:  {len(turn1_hit_ttft)}/{total_turn1} ({hit_ratio:.2f}%)"
        )
        if turn1_miss_ttft:
            miss_avg = sum(turn1_miss_ttft) / len(turn1_miss_ttft)
            hit_avg = sum(turn1_hit_ttft) / len(
                turn1_hit_ttft) if turn1_hit_ttft else 0
            speedup = miss_avg / hit_avg if hit_avg > 0 else 1.0
            print(
                f"Prefix Cache Speedup:      {speedup:.2f}x faster TTFT on hits!"
            )

    print_batch_cache_report(all_stats, sync_events or [])
    print("=" * 80 + "\n")


async def weight_sync_window(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    new_version: str,
) -> Dict[str, Any]:
    """Pauses the engine, simulates a weight transfer, and resumes.

    Mirrors the async-RL flow: pause with mode="keep" so in-flight rollouts
    freeze rather than restart, hold the engine for the duration of a weight
    transfer, commit the new weight version, then resume. Requests submitted
    after this carry the new cache salt and therefore cannot reach any block
    written under the old policy.

    `clear_cache` defaults off on purpose: clearing preempts every frozen
    request and drops its KV, which is exactly what mode="keep" exists to
    avoid. Salt rotation gives the invalidation without the preemption.

    The /pause, /resume and /update_weight_version routes only exist when the
    server runs with VLLM_SERVER_DEV_MODE=1. If they are missing we warn and
    still rotate the salt, so the cache-invalidation half of the experiment
    remains valid.

    Args:
        session: aiohttp session.
        args: Parsed arguments.
        new_version: Weight version to commit; also the new cache salt.

    Returns:
        Dict[str, Any]: Timing and status of the sync window.
    """
    base = f"http://{args.host}:{args.port}"
    event: Dict[str, Any] = {
        "version": new_version,
        "paused": False,
        "wall_sec": 0.0,
    }
    started = time.perf_counter()

    async def post(path: str, **kwargs) -> bool:
        try:
            async with session.post(f"{base}{path}", **kwargs) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  [weight-sync] POST {path} -> {resp.status}: "
                          f"{body[:200]}")
                    return False
                return True
        except Exception as e:
            print(f"  [weight-sync] POST {path} failed: "
                  f"{type(e).__name__}: {e}")
            return False

    print(f"\n[weight-sync] Pausing engine (mode={args.pause_mode}, "
          f"clear_cache={args.pause_clear_cache}) for version {new_version}...")
    paused = await post(
        "/pause",
        params={
            "mode": args.pause_mode,
            "clear_cache": str(args.pause_clear_cache).lower(),
        },
    )
    event["paused"] = paused
    if not paused:
        print("  [weight-sync] Pause unavailable (needs "
              "VLLM_SERVER_DEV_MODE=1); continuing with salt rotation only.")

    if args.weight_sync_seconds > 0:
        await asyncio.sleep(args.weight_sync_seconds)

    await post("/update_weight_version", json={"new_version": new_version})

    if paused:
        await post("/resume")
        print("[weight-sync] Engine resumed.")

    event["wall_sec"] = time.perf_counter() - started
    return event


async def run_rl_schedule(url: str, args: argparse.Namespace, worker):
    """Runs the groups on an off-policy RL schedule.

    `--rl-inflight-batches` batches are launched under the current policy.
    Thereafter, each time the oldest in-flight batch drains, we take a weight
    sync window and launch the next batch under a fresh cache salt. With
    48 groups, --rl-batch-size 16 and the default 2 in-flight batches this is
    exactly the three-batch pipeline: k and k+1 start together, k finishes,
    weights sync, k+2 starts on the new policy while k+1 keeps going on its
    existing (old-policy) KV.

    Args:
        url: Chat completions URL.
        args: Parsed arguments.
        worker: Coroutine factory (group_idx, session, cache_salt, batch_idx).

    Returns:
        Tuple of (per-group results in group order, sync event dicts).
    """
    batch_size = args.rl_batch_size
    num_batches = (args.num_groups + batch_size - 1) // batch_size
    inflight = max(1, min(args.rl_inflight_batches, num_batches))

    def batch_groups(batch_idx: int) -> List[int]:
        lo = batch_idx * batch_size + 1
        hi = min((batch_idx + 1) * batch_size, args.num_groups)
        return list(range(lo, hi + 1))

    def salt_for(version: int) -> Optional[str]:
        if args.no_salt_rotation:
            return None
        return f"{args.cache_salt_prefix}-v{version}"

    print(f"\nRL schedule: {num_batches} batches x {batch_size} groups, "
          f"{inflight} in flight before the first weight sync.")

    sync_events: List[Dict[str, Any]] = []
    tasks: Dict[int, asyncio.Task] = {}
    batch_tasks: Dict[int, List[asyncio.Task]] = {}
    version = 0

    async with make_client_session() as session:

        def launch(batch_idx: int) -> None:
            salt = salt_for(version)
            groups = batch_groups(batch_idx)
            print(f"[batch {batch_idx}] launching groups "
                  f"{groups[0]}..{groups[-1]} with cache_salt={salt}")
            batch_tasks[batch_idx] = []
            for g in groups:
                task = asyncio.create_task(worker(g, session, salt, batch_idx))
                tasks[g] = task
                batch_tasks[batch_idx].append(task)

        for b in range(inflight):
            launch(b)

        for b in range(inflight, num_batches):
            oldest = b - inflight
            await asyncio.gather(*batch_tasks[oldest])
            print(f"[batch {oldest}] drained.")
            version += 1
            sync_events.append(await weight_sync_window(
                session, args, f"{args.cache_salt_prefix}-v{version}"))
            launch(b)

        results = await asyncio.gather(*(tasks[g]
                                         for g in sorted(tasks)))

    return results, sync_events


async def main_async(args: argparse.Namespace):
    """Asynchronous entry point for running the GRPO benchmark.

    Args:
        args: Parsed command line arguments.
    """
    random.seed(args.seed)

    print(f"Loading tokenizer from: {args.model_path_or_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id,
                                              trust_remote_code=True)

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Connecting to vLLM server at {url}...")

    async with make_client_session() as session:
        try:
            async with session.get(
                    f"http://{args.host}:{args.port}/health") as resp:
                if resp.status != 200:
                    print(
                        f"Warning: Server health check returned status {resp.status}"
                    )
                else:
                    print("Server health check OK.")
        except Exception as e:
            print(f"Error connecting to server health endpoint: {e}")
            print("Please ensure vLLM serve was started before running.")
            sys.exit(1)

    shared_preamble = build_shared_preamble(tokenizer, args)
    if shared_preamble:
        print(f"Shared preamble: {args.shared_preamble_len} tokens, "
              "identical across every group and batch.")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(group_idx: int, session: aiohttp.ClientSession,
                     cache_salt: Optional[str], batch_idx: int):
        async with semaphore:
            return await run_group(session, url, args.model, tokenizer,
                                   group_idx, args, shared_preamble,
                                   cache_salt, batch_idx)

    start_time = time.perf_counter()

    if args.rl_batch_size > 0:
        results, sync_events = await run_rl_schedule(url, args, worker)
    else:
        sync_events = []
        async with make_client_session() as session:
            group_tasks = [
                worker(i, session, None, 0)
                for i in range(1, args.num_groups + 1)
            ]
            results = await asyncio.gather(*group_tasks)

    end_time = time.perf_counter()
    total_duration_sec = end_time - start_time

    all_stats = []
    for group_res in results:
        all_stats.extend(group_res)

    print_report(all_stats, total_duration_sec, args, sync_events)


def main():
    """Main parsing entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM for GRPO RL multi-turn streams.")
    parser.add_argument(
        "--model-path-or-id",
        type=str,
        default="Qwen/Qwen3-1.7B-base",
        help="Model path or Hugging Face ID for loading the tokenizer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B-base",
        help="Model name in OpenAI API requests.",
    )
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="vLLM server host.")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="vLLM server port.")
    parser.add_argument(
        "--num-groups",
        type=int,
        default=2,
        help="Number of GRPO groups (requests) to simulate.",
    )
    parser.add_argument(
        "--group-size",
        "-g",
        type=int,
        default=16,
        help="Group size (number of parallel streams per prompt).",
    )
    parser.add_argument(
        "--initial-prompt-len-min",
        type=int,
        default=4000,
        help="Minimum initial prompt length in tokens.",
    )
    parser.add_argument(
        "--initial-prompt-len-max",
        type=int,
        default=16000,
        help="Maximum initial prompt length in tokens.",
    )
    parser.add_argument(
        "--turns-min",
        type=int,
        default=10,
        help="Minimum conversation turns.",
    )
    parser.add_argument(
        "--turns-max",
        type=int,
        default=100,
        help="Maximum conversation turns.",
    )
    parser.add_argument(
        "--output-len-min",
        type=int,
        default=200,
        help="Minimum assistant output length in tokens per turn.",
    )
    parser.add_argument(
        "--output-len-max",
        type=int,
        default=2000,
        help="Maximum assistant output length in tokens per turn.",
    )
    parser.add_argument(
        "--env-len-min",
        type=int,
        default=10,
        help="Minimum simulated environment response length in tokens.",
    )
    parser.add_argument(
        "--env-len-max",
        type=int,
        default=100,
        help="Maximum simulated environment response length in tokens.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of GRPO groups running concurrently.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_false",
        dest="ignore_eos",
        help="Do not ignore EOS tokens during generation.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for generation.")

    rl = parser.add_argument_group(
        "off-policy RL simulation",
        "Splits the groups into training batches, takes a weight sync window "
        "between them, and rotates the KV cache salt so a new batch cannot "
        "reuse blocks computed under the previous policy.")
    rl.add_argument(
        "--shared-preamble-len",
        type=int,
        default=0,
        help="Tokens of fixed preamble (system prompt + tool defs) prepended "
        "to every group in every batch. This is the only prefix shared across "
        "batches, so it is what salt rotation invalidates. 0 disables.",
    )
    rl.add_argument(
        "--rl-batch-size",
        type=int,
        default=0,
        help="Groups per training batch. 0 (default) runs the plain schedule "
        "with no batching, weight syncs or salt rotation.",
    )
    rl.add_argument(
        "--rl-inflight-batches",
        type=int,
        default=2,
        help="Batches launched before the first weight sync. 2 gives the "
        "standard one-off pipelining overlap (batch k and k+1 together).",
    )
    rl.add_argument(
        "--weight-sync-seconds",
        type=float,
        default=0.0,
        help="Seconds to hold the engine paused, simulating the weight "
        "transfer itself.",
    )
    rl.add_argument(
        "--pause-mode",
        choices=["keep", "abort", "wait"],
        default="keep",
        help="vLLM pause mode. 'keep' freezes in-flight rollouts so they "
        "resume where they left off, which is what async RL wants.",
    )
    rl.add_argument(
        "--pause-clear-cache",
        action="store_true",
        help="Ask the engine to clear the prefix cache during the pause. This "
        "preempts every frozen request and drops its KV -- the thing salt "
        "rotation exists to avoid. Off by default.",
    )
    rl.add_argument(
        "--cache-salt-prefix",
        type=str,
        default="policy",
        help="Prefix for the per-batch cache salt / weight version.",
    )
    rl.add_argument(
        "--no-salt-rotation",
        action="store_true",
        help="Control run: keep the RL batch schedule and weight sync windows "
        "but send no cache_salt, so later batches can still hit earlier "
        "batches' blocks.",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
