#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verify multi-host TPU KV offload without overlapping full-slice engines.

The controller launches baseline and offload workloads in separate child
processes. Each child shuts down its engine before exiting. Between the two
workloads, the controller waits until the single existing Ray cluster reports
all TPU resources available, then starts the offload workload.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


PROMPTS = [
    "Google is a ",
    ("You are a helpful, harmless, and highly capable language model. "
     "Explain clearly and accurately: what are the colors of a rainbow?"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("controller", "baseline", "offload"),
                        default="controller")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--load-format", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--offload-output", type=Path, required=True)
    parser.add_argument("--resource-wait-seconds", type=int, default=300)
    return parser.parse_args()


def offload_config() -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="TPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_module_path="tpu_inference.offload.tpu_offload_connector",
    )


def serialize_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    return [{
        "text": output.outputs[0].text,
        "token_ids": output.outputs[0].token_ids,
    } for output in outputs]


def generate(args: argparse.Namespace, use_offload: bool) -> list[list[dict[str, Any]]]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=True,
    )
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "seed": args.seed,
        "trust_remote_code": True,
    }
    if use_offload:
        llm_kwargs["kv_transfer_config"] = offload_config()

    llm = LLM(**llm_kwargs)
    try:
        print("--- Generating pass 1", flush=True)
        first_pass = serialize_outputs(llm.generate(PROMPTS, sampling_params))
        if not use_offload:
            return [first_pass]

        # Allow asynchronous saves to complete, then remove only the HBM
        # prefix cache so pass 2 must reload matching blocks from host memory.
        time.sleep(5)
        print("--- Resetting prefix cache before offload reload", flush=True)
        llm.llm_engine.engine_core.reset_prefix_cache()
        time.sleep(1)
        print("--- Generating pass 2", flush=True)
        return [first_pass,
                serialize_outputs(llm.generate(PROMPTS, sampling_params))]
    finally:
        print("--- Shutting down LLM engine and releasing its Ray placement group", flush=True)
        llm.llm_engine.engine_core.shutdown()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_baseline(args: argparse.Namespace) -> None:
    outputs = generate(args, use_offload=False)
    write_json(args.baseline_output, outputs[0])
    print(f"--- Baseline output written to {args.baseline_output}", flush=True)


def run_offload(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline_output.read_text(encoding="utf-8"))
    offload_passes = generate(args, use_offload=True)
    for pass_index, outputs in enumerate(offload_passes, start=1):
        if outputs != baseline:
            raise AssertionError(
                f"Offload pass {pass_index} differs from the no-offload baseline.")
    write_json(args.offload_output, offload_passes)
    print("--- Both offload passes exactly match the baseline", flush=True)


def wait_for_full_tpu_availability(args: argparse.Namespace) -> None:
    import ray

    ray.init(address="auto", ignore_reinit_error=True)
    deadline = time.monotonic() + args.resource_wait_seconds
    try:
        while time.monotonic() < deadline:
            available_tpu = ray.available_resources().get("TPU", 0.0)
            print(
                f"--- Waiting for baseline resources to release: "
                f"TPU available={available_tpu}/{args.tensor_parallel_size}",
                flush=True,
            )
            if available_tpu >= args.tensor_parallel_size:
                return
            time.sleep(5)
    finally:
        ray.shutdown()
    raise TimeoutError(
        "Baseline Ray placement group did not release all TPU resources within "
        f"{args.resource_wait_seconds}s.")


def run_controller(args: argparse.Namespace) -> None:
    child_args = [arg for arg in sys.argv[1:] if not arg.startswith("--phase")]
    script = str(Path(__file__).resolve())
    subprocess.run([sys.executable, script, "--phase=baseline", *child_args],
                   check=True)
    wait_for_full_tpu_availability(args)
    subprocess.run([sys.executable, script, "--phase=offload", *child_args],
                   check=True)


def main() -> None:
    args = parse_args()
    if args.phase == "controller":
        run_controller(args)
    elif args.phase == "baseline":
        run_baseline(args)
    else:
        run_offload(args)


if __name__ == "__main__":
    main()
