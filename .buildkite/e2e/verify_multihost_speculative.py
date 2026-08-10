#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verify multi-host n-gram speculative decoding through the vLLM API."""

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


# Keep this set small and deterministic. The repeated prompt exercises the
# n-gram proposer, while the other prompts guard ordinary greedy decoding.
PROMPTS = [
    "Keep repeating: " + "a " * 20,
    "The capital of France is",
    "Complete this sequence: 2, 4, 6, 8,",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "speculative", "wait"),
                        required=True)
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--speculative-output", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--resource-wait-seconds", type=int, default=300)
    return parser.parse_args()


def complete(args: argparse.Namespace, prompt: str) -> dict[str, str]:
    payload = json.dumps({
        "model": args.model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "ignore_eos": True,
    }).encode("utf-8")
    request = Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            body = json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error

    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"Unexpected completions response: {body}")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise AssertionError(f"Completion response has no text: {body}")
    return {"prompt": prompt, "text": text}


def generate_all(args: argparse.Namespace) -> list[dict[str, str]]:
    print(f"--- Sending {len(PROMPTS)} deterministic completion requests via {args.url}",
          flush=True)
    return [complete(args, prompt) for prompt in PROMPTS]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def compare_outputs(baseline: list[dict[str, str]],
                    speculative: list[dict[str, str]]) -> None:
    if len(baseline) != len(speculative):
        raise AssertionError(
            f"Response count differs: baseline={len(baseline)}, "
            f"speculative={len(speculative)}")
    mismatches = []
    for index, (baseline_output, speculative_output) in enumerate(
            zip(baseline, speculative)):
        if baseline_output != speculative_output:
            mismatches.append({
                "index": index,
                "prompt": baseline_output.get("prompt"),
                "baseline": baseline_output.get("text"),
                "speculative": speculative_output.get("text"),
            })
    if mismatches:
        raise AssertionError(
            "Speculative output differs from the baseline:\n" +
            json.dumps(mismatches, indent=2, ensure_ascii=False))


def wait_for_full_tpu_availability(args: argparse.Namespace) -> None:
    if args.tensor_parallel_size is None:
        raise ValueError("--tensor-parallel-size is required for --phase wait")
    import ray

    ray.init(address="auto", ignore_reinit_error=True)
    deadline = time.monotonic() + args.resource_wait_seconds
    try:
        while time.monotonic() < deadline:
            available_tpu = ray.available_resources().get("TPU", 0.0)
            print(
                "--- Waiting for baseline resources to release: "
                f"TPU available={available_tpu}/{args.tensor_parallel_size}",
                flush=True)
            if available_tpu >= args.tensor_parallel_size:
                return
            time.sleep(5)
    finally:
        ray.shutdown()
    raise TimeoutError("Baseline server did not release all TPU resources.")


def main() -> None:
    args = parse_args()
    if args.phase == "wait":
        wait_for_full_tpu_availability(args)
        return
    if None in (args.url, args.model, args.max_tokens, args.seed,
                args.baseline_output, args.speculative_output):
        raise ValueError(
            "API verification requires URL, model, sampling, and output arguments")

    outputs = generate_all(args)
    if args.phase == "baseline":
        write_json(args.baseline_output, outputs)
        print(f"--- Baseline outputs written to {args.baseline_output}", flush=True)
        return

    baseline = json.loads(args.baseline_output.read_text(encoding="utf-8"))
    compare_outputs(baseline, outputs)
    write_json(args.speculative_output, outputs)
    print("--- Speculative API responses exactly match the baseline", flush=True)


if __name__ == "__main__":
    main()
