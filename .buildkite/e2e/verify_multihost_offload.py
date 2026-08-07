#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verify multi-host TPU KV offload through the vLLM completions API."""

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


# This is deliberately much longer than a vLLM KV block (normally 16 tokens).
# The second prompt changes only its suffix, so it must reuse many complete
# prefix blocks saved while serving the first request.
LONG_PREFIX = " ".join([
    "A reliable distributed system records each operation before continuing."
] * 96)
PROMPTS = [
    LONG_PREFIX + " Summarize this principle in one sentence.",
    LONG_PREFIX + " State one concrete benefit of this principle.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "offload", "wait"),
                        required=True)
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--offload-output", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--resource-wait-seconds", type=int, default=300)
    return parser.parse_args()


def complete(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
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
    return {"text": text}


def reset_local_prefix_cache(args: argparse.Namespace) -> None:
    reset_url = args.url.removesuffix("/v1/completions") + "/reset_prefix_cache"
    request = Request(reset_url, data=b"", method="POST")
    for _ in range(60):
        with urlopen(request, timeout=30) as response:
            if json.loads(response.read()).get("success") is True:
                print("--- Reset local prefix cache through the serving API",
                      flush=True)
                return
        time.sleep(5)
    raise TimeoutError("Serving API could not reset the local prefix cache.")


def generate_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    print(f"--- Sending {len(PROMPTS)} long completion requests via {args.url}",
          flush=True)
    print(f"--- Shared prefix length: {len(LONG_PREFIX)} characters", flush=True)
    first_output = complete(args, PROMPTS[0])
    if args.phase == "offload":
        # Keep the connector-managed CPU cache, but clear HBM prefix state so
        # the second request must reload the shared blocks through H2D.
        reset_local_prefix_cache(args)
    return [first_output, complete(args, PROMPTS[1])]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


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
                f"--- Waiting for server resources to release: "
                f"TPU available={available_tpu}/{args.tensor_parallel_size}",
                flush=True,
            )
            if available_tpu >= args.tensor_parallel_size:
                return
            time.sleep(5)
    finally:
        ray.shutdown()
    raise TimeoutError("vLLM server did not release all TPU resources in time.")


def main() -> None:
    args = parse_args()
    if args.phase == "wait":
        wait_for_full_tpu_availability(args)
        return
    if None in (args.url, args.model, args.max_tokens, args.seed,
                args.baseline_output, args.offload_output):
        raise ValueError(
            "API verification requires URL, model, sampling, and output arguments")
    outputs = generate_all(args)
    if args.phase == "baseline":
        write_json(args.baseline_output, outputs)
        print(f"--- Baseline output written to {args.baseline_output}", flush=True)
        return

    baseline = json.loads(args.baseline_output.read_text(encoding="utf-8"))
    if outputs != baseline:
        raise AssertionError("Offload server output differs from the baseline server.")
    write_json(args.offload_output, outputs)
    print("--- Offload API responses exactly match the baseline", flush=True)


if __name__ == "__main__":
    main()
