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
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer


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


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Loads a prepared multi-turn dataset produced by prepare_r2e_dataset.py.

    Args:
        path: Path to the prepared JSONL file.

    Returns:
        List[Dict[str, Any]]: Dataset records.
    """
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def generate_env_response(tokenizer: AutoTokenizer, args: argparse.Namespace,
                          record: Optional[Dict[str, Any]], turn: int) -> str:
    """Produces the environment observation appended after an assistant turn.

    Args:
        tokenizer: Tokenizer to decode tokens.
        args: Parsed command line arguments.
        record: Dataset record for this group, or None for random mode.
        turn: 1-based index of the turn just completed.

    Returns:
        str: Environment response text.
    """
    if record is not None:
        env_turns = record["env_turns"]
        # Streams in a group may run past the scripted turn count when
        # --turns-max exceeds the record's; cycle rather than truncate so the
        # conversation keeps growing.
        return env_turns[(turn - 1) % len(env_turns)]

    env_len = random.randint(args.env_len_min, args.env_len_max)
    env_token_ids = [random.randint(1000, 50000) for _ in range(env_len)]
    return tokenizer.decode(env_token_ids, skip_special_tokens=True)


async def run_grpo_stream(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    stream_idx: int,
    group_idx: int,
    args: argparse.Namespace,
    record: Optional[Dict[str, Any]] = None,
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
        record: Dataset record backing this group, or None for random mode.

    Returns:
        List[Dict[str, Any]]: Statistics of each turn in the stream.
    """
    if record is not None:
        num_turns = record["num_turns"]
        system_prompt = record["system_prompt"]
    else:
        num_turns = random.randint(args.turns_min, args.turns_max)
        system_prompt = "You are a helpful assistant."

    messages = [
        {
            "role": "system",
            "content": system_prompt
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

        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        start_time = time.perf_counter()
        ttft = None
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
            assistant_tokens = len(tokenizer.encode(assistant_response))

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
                "input_history_tokens":
                len(tokenizer.encode(str(messages[:-1]))),
                "success": True,
            }

            # Simulate the environment's reply: real recorded tool output in
            # dataset mode, otherwise 10-100 random tokens.
            env_text = generate_env_response(tokenizer, args, record, turn)

            messages.append({"role": "user", "content": env_text})
            turn_stat["env_tokens"] = len(tokenizer.encode(env_text))

            stats.append(turn_stat)

        except Exception as e:
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
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
                "error": str(e),
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
    dataset: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Runs a single GRPO group of G parallel streams.

    Args:
        session: aiohttp ClientSession.
        url: API URL.
        model: Model name.
        tokenizer: Model tokenizer.
        group_idx: Index of this group.
        args: Parsed arguments.
        dataset: Prepared multi-turn records, or None for random mode.

    Returns:
        List[Dict[str, Any]]: Accumulated stats of all streams in the group.
    """
    if dataset is not None:
        # One task instance per group, matching real RL where a group of
        # rollouts shares a single task and therefore a single prefix.
        record = dataset[(group_idx - 1) % len(dataset)]
        initial_prompt = record["initial_prompt"]
        label = f" [{record['instance_id']}]"
    else:
        record = None
        initial_prompt = generate_initial_prompt(tokenizer, args)
        label = ""

    initial_prompt_len = len(tokenizer.encode(initial_prompt))
    print(f"Group {group_idx}{label}: Starting {args.group_size} streams with "
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
                record,
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

    dataset = None
    if args.dataset:
        dataset = load_dataset(args.dataset)
        print(
            f"Loaded {len(dataset)} task instances from {args.dataset}. "
            f"Prompt lengths and turn counts come from the dataset; "
            f"--initial-prompt-len-*, --turns-* and --env-len-* are ignored.")

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Connecting to vLLM server at {url}...")

    async with aiohttp.ClientSession() as session:
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

    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(group_idx: int, session: aiohttp.ClientSession):
        async with semaphore:
            return await run_group(session, url, args.model, tokenizer,
                                   group_idx, args, dataset)

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        group_tasks = [
            worker(i, session) for i in range(1, args.num_groups + 1)
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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Prepared multi-turn JSONL from prepare_r2e_dataset.py. Drives "
        "prompts and environment turns from real SWE task content instead of "
        "random tokens. Omit for the original random-token workload.",
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

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
