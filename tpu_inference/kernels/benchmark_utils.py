# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark helpers for TPU kernels."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def get_device_name() -> str:
    return jax.devices()[0].device_kind


def get_peak_mem_bw_gbs() -> float | None:
    return pltpu.get_tpu_info().mem_bw_bytes_per_second / 1e9


def _block_until_ready(result: Any) -> None:
    if isinstance(result, tuple):
        for item in result:
            jax.block_until_ready(item)
    else:
        jax.block_until_ready(result)


def _extract_trace_latency_ms(trace_dir: str, event_name: str) -> float:
    trace_path = Path(trace_dir)
    candidates = sorted(trace_path.rglob("*.trace.json.gz"))
    if not candidates:
        candidates = sorted(trace_path.rglob("*.json.gz"))
    if not candidates:
        raise FileNotFoundError(f"No trace JSON found in {trace_dir}")

    with gzip.open(candidates[0], "rt") as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])
    tpu_pids: set[int] = set()
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        if "/device:TPU:" in event.get("args", {}).get("name", ""):
            tpu_pids.add(event["pid"])

    if not tpu_pids:
        raise RuntimeError("No TPU devices found in trace")

    first_pid = min(tpu_pids)
    durations_us = [
        event["dur"]
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") == first_pid
        and event.get("name", "").startswith(event_name)
        and event.get("dur", 0) > 0
    ]
    if not durations_us:
        raise RuntimeError(f"No events named '{event_name}' found in trace")

    return (sum(durations_us) / len(durations_us)) / 1000.0


def benchmark(
    kernel_fn: Callable[[], Any],
    *,
    iters: int,
    trace_dir: str | None = None,
    event_name: str | None = None,
) -> float:
    """Runs a JAX kernel benchmark and returns mean latency in milliseconds."""
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if event_name is None:
        raise ValueError("event_name is required")

    for _ in range(5):
        result = kernel_fn()
    _block_until_ready(result)

    if trace_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        profile_dir = tmp_dir.name
    else:
        tmp_dir = None
        profile_dir = trace_dir

    os.makedirs(profile_dir, exist_ok=True)
    profile_options = jax.profiler.ProfileOptions()
    profile_options.python_tracer_level = 1
    jax.profiler.start_trace(profile_dir, profiler_options=profile_options)
    for _ in range(iters):
        result = kernel_fn()
    _block_until_ready(result)
    jax.profiler.stop_trace()

    latency_ms = _extract_trace_latency_ms(profile_dir, event_name)
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return latency_ms
