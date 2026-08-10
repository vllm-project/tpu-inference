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
from typing import Any, Dict, List

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


def make_token_text(tokenizer: AutoTokenizer, num_tokens: int,
                    rng: random.Random) -> str:
    """Builds text that encodes to num_tokens tokens.

    decode() then encode() is not an identity, so the round trip is corrected
    until the length matches. Callers rely on the exact length because the
    prompt lengths replayed from a trace are the measured ones.

    Args:
        tokenizer: Tokenizer used to decode and re-encode.
        num_tokens: Target token count.
        rng: Random source, so shared spans can be reproduced exactly.

    Returns:
        str: Text of the requested token length.
    """
    if num_tokens <= 0:
        return ""
    # Safe token ID range, avoiding special control characters.
    ids = [rng.randint(1000, 50000) for _ in range(num_tokens)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for _ in range(8):
        cur = len(tokenizer.encode(text, add_special_tokens=False))
        if cur == num_tokens:
            break
        if cur > num_tokens:
            keep = tokenizer.encode(text,
                                    add_special_tokens=False)[:num_tokens]
            text = tokenizer.decode(keep, skip_special_tokens=True)
        else:
            extra = [rng.randint(1000, 50000) for _ in range(num_tokens - cur)]
            text += tokenizer.decode(extra, skip_special_tokens=True)
    return text


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
    return make_token_text(tokenizer, length, random)


def build_initial_prompt(tokenizer: AutoTokenizer, global_prefix: str,
                         total_len: int, group_idx: int) -> str:
    """Builds a group's initial prompt as global_prefix + group-specific text.

    Every group receives the identical global_prefix string, so all
    trajectories share it as a token prefix, while the remainder is shared
    only by the streams within one group. This reproduces the two-level
    sharing that drives prefix cache behaviour in real agent rollouts.

    Args:
        tokenizer: Tokenizer used to size the group-specific portion.
        global_prefix: Text shared by every trajectory (may be empty).
        total_len: Desired total prompt length in tokens.
        group_idx: Group index, seeding the group-specific portion.

    Returns:
        str: The group's initial prompt.
    """
    global_len = len(tokenizer.encode(
        global_prefix, add_special_tokens=False)) if global_prefix else 0
    remainder = max(0, total_len - global_len)
    group_text = make_token_text(tokenizer, remainder,
                                 random.Random(1_000_000 + group_idx))
    if not global_prefix:
        return group_text
    return global_prefix + "\n" + group_text


def load_trace(path: str) -> List[List[Dict[str, Any]]]:
    """Loads a rollout trace, returning per-group lists of trajectory specs.

    Args:
        path: Path to the JSONL trace.

    Returns:
        List[List[Dict[str, Any]]]: Trajectories grouped by "group", each
        inner list sorted by "stream".
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            groups.setdefault(rec["group"], []).append(rec)
    for specs in groups.values():
        specs.sort(key=lambda r: r.get("stream", 0))
    return [groups[k] for k in sorted(groups)]


async def run_grpo_stream(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    stream_idx: int,
    group_idx: int,
    args: argparse.Namespace,
    spec: Dict[str, Any] | None = None,
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
    out_lens = spec.get("out_lens") if spec else None
    obs_lens = spec.get("obs_lens") if spec else None
    num_turns = (len(out_lens) if out_lens is not None else random.randint(
        args.turns_min, args.turns_max))
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": initial_prompt
        },
    ]

    stats = []

    for turn in range(1, num_turns + 1):
        if out_lens is not None:
            max_tokens = out_lens[turn - 1]
        else:
            max_tokens = random.randint(args.output_len_min,
                                        args.output_len_max)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": args.temperature,
        }
        if args.ignore_eos:
            payload["ignore_eos"] = True
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
                "turn": turn,
                "num_turns": num_turns,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "total_time_ms": total_time_ms,
                "output_tokens": assistant_tokens,
                "input_history_tokens": prompt_tokens,
                "success": True,
            }

            # Simulate environment response of 10-100 tokens
            if obs_lens is not None:
                idx = turn - 1
                env_len = obs_lens[idx] if idx < len(obs_lens) else 0
            else:
                env_len = random.randint(args.env_len_min, args.env_len_max)
            env_text = make_token_text(tokenizer, env_len, random)

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
                "turn": turn,
                "num_turns": num_turns,
                "ttft_ms": total_time_ms,
                "tpot_ms": 0.0,
                "total_time_ms": total_time_ms,
                "output_tokens": 0,
                "input_history_tokens": 0,
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
    global_prefix: str = "",
    specs: List[Dict[str, Any]] | None = None,
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
    if specs:
        target_len = specs[0].get("prompt_len") or args.initial_prompt_len_max
        initial_prompt = build_initial_prompt(tokenizer, global_prefix,
                                              target_len, group_idx)
        num_streams = len(specs)
    elif global_prefix:
        target_len = random.randint(args.initial_prompt_len_min,
                                    args.initial_prompt_len_max)
        initial_prompt = build_initial_prompt(tokenizer, global_prefix,
                                              target_len, group_idx)
        num_streams = args.group_size
    else:
        initial_prompt = generate_initial_prompt(tokenizer, args)
        num_streams = args.group_size

    initial_prompt_len = len(tokenizer.encode(initial_prompt))
    print(f"Group {group_idx}: Starting {num_streams} streams with "
          f"shared initial prompt of {initial_prompt_len} tokens...")

    tasks = []
    for stream_idx in range(num_streams):
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
                specs[stream_idx] if specs else None,
            ))

    results = await asyncio.gather(*tasks)

    flat_results = []
    for stream_result in results:
        flat_results.extend(stream_result)
    return flat_results


def print_report(
    all_stats: List[Dict[str, Any]],
    total_duration_sec: float,
    args: argparse.Namespace,
) -> None:
    """Calculates and prints the benchmark performance report.

    Args:
        all_stats: List of all turn statistics collected.
        total_duration_sec: Total duration of the benchmark run.
        args: Parsed arguments.
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
    print("=" * 80 + "\n")


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

    trace_groups: List[List[Dict[str, Any]]] | None = None
    if args.trace_file:
        trace_groups = load_trace(args.trace_file)
        # Unset means replay the whole trace, rather than the synthetic
        # default of 2 groups, which would silently truncate it.
        if args.num_groups is not None:
            trace_groups = trace_groups[:args.num_groups]
        turns = sum(
            len(s.get("out_lens", [])) for g in trace_groups for s in g)
        streams = sum(len(g) for g in trace_groups)
        print(f"Replaying trace {args.trace_file}: {len(trace_groups)} groups,"
              f" {streams} streams, {turns:,} turns")

    global_prefix = ""
    if args.global_prefix_len > 0:
        global_prefix = make_token_text(tokenizer, args.global_prefix_len,
                                        random.Random(0))
        got = len(tokenizer.encode(global_prefix, add_special_tokens=False))
        print(f"Global prefix shared by every trajectory: {got} tokens")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(group_idx: int,
                     session: aiohttp.ClientSession,
                     specs: List[Dict[str, Any]] | None = None):
        async with semaphore:
            return await run_group(session, url, args.model, tokenizer,
                                   group_idx, args, global_prefix, specs)

    start_time = time.perf_counter()

    async with make_client_session() as session:
        if trace_groups is not None:
            group_tasks = [
                worker(i + 1, session, specs)
                for i, specs in enumerate(trace_groups)
            ]
        else:
            num_groups = 2 if args.num_groups is None else args.num_groups
            group_tasks = [
                worker(i, session) for i in range(1, num_groups + 1)
            ]
        results = await asyncio.gather(*group_tasks)

    end_time = time.perf_counter()
    total_duration_sec = end_time - start_time

    all_stats = []
    for group_res in results:
        all_stats.extend(group_res)

    print_report(all_stats, total_duration_sec, args)


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
        default=None,
        help="Number of GRPO groups (requests) to simulate. Defaults to 2, "
        "or to every group in --trace-file when replaying a trace.",
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
        "--trace-file",
        type=str,
        default=None,
        help="Replay a recorded rollout trace (JSONL) instead of sampling "
        "turn counts and lengths. Each row is one trajectory: "
        '{"group", "stream", "prompt_len", "out_lens", "obs_lens"}. '
        "Overrides --turns-*, --output-len-*, --env-len-* and "
        "--initial-prompt-len-*.",
    )
    parser.add_argument(
        "--global-prefix-len",
        type=int,
        default=0,
        help="Tokens of prompt shared by *every* trajectory, mimicking the "
        "system prompt and tool definitions of a real agent. The remainder "
        "of each initial prompt is shared only within a group. Real "
        "SWE-agent rollouts measure ~6476 tokens global and ~1100 "
        "per-group.",
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

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
