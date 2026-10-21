# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Golden full-request microbenchmark for stacked RPA.

This benchmark models a burst of equal-length requests from arrival through
completion. The host-side scheduler is intentionally small and deterministic:

* all requests arrive at step 1;
* up to ``decode_q_len`` tokens are scheduled for every eligible decoder before
  prompt work;
* the remaining token budget is filled with FCFS chunked prefills;
* a completed prompt samples output token 1, so an ``O``-token response needs
  ``ceil((O - 1) / decode_q_len)`` decode attention calls; and
* no EOS, preemption, or stochastic speculative acceptance are modeled;
  ``--prefix-tokens`` optionally marks a uniform cached portion of each prompt,
  and ``decode_q_len > 1`` assumes perfect acceptance.

Workload, kernel geometry, sampling, and warmup values must be specified
explicitly. Runtime metadata places decode rows first, followed by chunked
prefill rows, with ``kv_lens`` including the query tokens scheduled in the
current step.

``--phase`` projects that complete serving timeline onto the selected kernel
region. ``decode`` keeps generated-token rows plus one-token prompt tails,
``prefill`` keeps the remaining PREFILL rows, and ``both`` preserves the full
step. Steps with no work in the selected region are omitted, while retained
steps keep their original serving-step index and KV lengths. ``--sample-every``
is applied after that projection and keeps every Nth surviving step, starting at
position zero. Each selected attention call runs once in the measured trace. The
stride is stored in ``config.sample_every``; per-step output keeps each original
serving index in ``steps[].step``. Selected later steps do not replay their
predecessors: the benchmark uses the preserved KV lengths with its synthetically
initialized cache, which is sufficient for kernel timing.

On TPU, warmup compiles ``attention_step`` across power-of-two static T shapes
from 16 through ``max_batch_tokens`` before tracing starts. Each measured step
builds its RPA schedules once outside the measured per-layer attention call. All
steps are then executed in order while threading the KV cache. Device latency is
extracted from the profiler's outer ``jit_attention_step`` spans, excluding
schedule dispatches, ``barrier-cores``, and trailing output copies. HBM bandwidth
utilization is estimated only for ``--phase decode`` because prefill/MIXED traffic
depends on q-blocking and causal visibility, not just one KV scan per request.
Compute utilization is estimated only for ``--phase prefill`` from useful causal
QK and PV FLOPs against the device's peak BF16 throughput.

Examples::

    # Validate and print a schedule without importing JAX.
    python tpu_inference/kernels/experimental/stacked_rpa/bench_stacked_rpa_golden.py \
        --schedule-only --num-requests 4 --input-tokens 128 --output-tokens 8 \
        --prefix-tokens 0 --decode-q-len 1 --phase both --sample-every 1 \
        --max-batch-tokens 128 --max-num-seqs 4 --max-model-len 256 \
        --page-size 128 --num-q-heads 8 --num-kv-heads 1 --head-dim 64 \
        --kv-dtype fp8 --full-context --warmup 1 --progress-every 1

    # Run the full benchmark on a TPU host.
    python -m tpu_inference.kernels.experimental.stacked_rpa.bench_stacked_rpa_golden \
        --num-requests 4 --input-tokens 128 --output-tokens 8 \
        --prefix-tokens 0 --decode-q-len 1 --phase both --sample-every 1 \
        --max-batch-tokens 128 --max-num-seqs 4 --max-model-len 256 \
        --page-size 128 --num-q-heads 8 --num-kv-heads 1 --head-dim 64 \
        --kv-dtype fp8 --full-context --warmup 1 --progress-every 1 \
        --output-json /tmp/stacked-rpa.json \
        --trace-dir /tmp/stacked-rpa-traces
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import json
import math
import shutil
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclasses.dataclass(frozen=True)
class WorkloadConfig:
    """Inputs to the deterministic serving scheduler."""

    num_requests: int
    input_tokens: int
    output_tokens: int
    max_batch_tokens: int
    prefix_tokens: int

    def validate(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.input_tokens <= 0:
            raise ValueError("input_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if self.prefix_tokens >= self.input_tokens:
            raise ValueError("prefix_tokens must be smaller than input_tokens")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")


@dataclasses.dataclass(frozen=True)
class ScheduledRequest:
    """One request's work in one scheduler step."""

    request_id: int
    query_tokens: int
    kv_tokens: int
    is_prompt: bool

    @property
    def is_kernel_decode(self) -> bool:
        """Generated rows and every one-token prompt tail route through DECODE."""
        return not self.is_prompt or self.query_tokens == 1


@dataclasses.dataclass(frozen=True)
class ServingStep:
    """Decode-first work submitted in one inference step."""

    index: int
    requests: tuple[ScheduledRequest, ...]

    @property
    def total_query_tokens(self) -> int:
        return sum(request.query_tokens for request in self.requests)

    @property
    def decode_count(self) -> int:
        return sum(request.is_kernel_decode for request in self.requests)

    @property
    def prefill_count(self) -> int:
        return len(self.requests) - self.decode_count


@dataclasses.dataclass(frozen=True)
class StepMetadata:
    """Host form of the four metadata arrays consumed by stacked RPA."""

    kv_lens: tuple[int, ...]
    page_indices: tuple[int, ...]
    cu_q_lens: tuple[int, ...]
    distribution: tuple[int, int, int]


@dataclasses.dataclass
class _RequestState:
    computed_tokens: int
    generated_tokens: int


PHASES = ("decode", "prefill", "both")


def _next_power_of_2(value: int) -> int:
    """Return the smallest power of two greater than or equal to value."""
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _iter_trace_events(trace_dir: str):
    for path in sorted(Path(trace_dir).rglob("*.trace.json.gz")):
        with gzip.open(path, "rt") as trace_file:
            yield json.load(trace_file).get("traceEvents", [])


def _tensorcore_pids(events: Sequence[dict[str, Any]]) -> set[int]:
    tensorcore_pids = set()
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_name = str(event.get("args", {}).get("name", ""))
            if "/device:TPU:" in process_name and "SparseCore" not in process_name:
                tensorcore_pids.add(event.get("pid"))
    return tensorcore_pids


def _device_kernel_ms_per_dispatch_from_trace(
    trace_dir: str,
    *,
    jit_name_prefix: str,
) -> list[float]:
    """Extract device latency for each matching top-level JIT dispatch."""
    matching_by_pid: dict[int, list[tuple[float, float, float]]] = {}
    ops_by_pid: dict[int, list[tuple[float, float, float, str]]] = {}
    for events in _iter_trace_events(trace_dir):
        tensorcore_pids = _tensorcore_pids(events)
        for event in events:
            pid = event.get("pid")
            if (event.get("ph") != "X" or not event.get("dur")
                    or pid not in tensorcore_pids):
                continue
            name = str(event.get("name", ""))
            start_us = float(event["ts"])
            duration_us = float(event["dur"])
            end_us = start_us + duration_us
            if name.startswith(jit_name_prefix):
                matching_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us))
            elif not name.startswith("jit_"):
                ops_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us, name))

    if not matching_by_pid:
        return []
    pid, dispatches = max(
        matching_by_pid.items(),
        key=lambda item: len(item[1]),
    )
    ops = ops_by_pid.get(pid, [])
    latencies_ms = []
    for start_us, end_us, duration_us in sorted(dispatches):
        children = [op for op in ops if start_us <= op[0] < end_us]
        barrier_us = sum(op[2] for op in children if op[3] == "barrier-cores")
        copy_us = 0.0
        if children:
            last = max(children, key=lambda op: op[1])
            if last[3].startswith("copy"):
                copy_us = last[2]
        latencies_ms.append(
            max(duration_us - barrier_us - copy_us, 0.0) / 1000.0)
    return latencies_ms


def _request_is_in_phase(request: ScheduledRequest, phase: str) -> bool:
    if phase == "both":
        return True
    if phase == "decode":
        return request.is_kernel_decode
    if phase == "prefill":
        return not request.is_kernel_decode
    raise ValueError(f"unsupported phase: {phase!r}")


def simulate_greedy_schedule(config: WorkloadConfig, *,
                             decode_q_len: int) -> list[ServingStep]:
    """Simulate decode-priority, FCFS chunked-prefill serving to completion."""
    config.validate()
    if decode_q_len <= 0:
        raise ValueError("decode_q_len must be positive")
    states = [
        _RequestState(
            computed_tokens=config.prefix_tokens,
            generated_tokens=0,
        ) for _ in range(config.num_requests)
    ]
    steps: list[ServingStep] = []

    while any(state.generated_tokens < config.output_tokens
              for state in states):
        budget = config.max_batch_tokens
        scheduled: list[ScheduledRequest] = []

        # Decode work is eligible only after prompt completion in a prior step.
        for request_id, state in enumerate(states):
            if budget == 0:
                break
            if (state.computed_tokens < config.input_tokens
                    or state.generated_tokens >= config.output_tokens):
                continue
            query_tokens = min(
                decode_q_len,
                config.output_tokens - state.generated_tokens,
                budget,
            )
            state.computed_tokens += query_tokens
            state.generated_tokens += query_tokens
            budget -= query_tokens
            scheduled.append(
                ScheduledRequest(
                    request_id=request_id,
                    query_tokens=query_tokens,
                    kv_tokens=state.computed_tokens,
                    is_prompt=False,
                ))

        # Spend the remaining budget on prompts in request-arrival order.
        for request_id, state in enumerate(states):
            if budget == 0:
                break
            prompt_remaining = config.input_tokens - state.computed_tokens
            if prompt_remaining <= 0:
                continue
            query_tokens = min(prompt_remaining, budget)
            state.computed_tokens += query_tokens
            budget -= query_tokens
            if state.computed_tokens == config.input_tokens:
                # The prompt's final hidden state samples the first output token.
                state.generated_tokens = 1
            scheduled.append(
                ScheduledRequest(
                    request_id=request_id,
                    query_tokens=query_tokens,
                    kv_tokens=state.computed_tokens,
                    is_prompt=True,
                ))

        if not scheduled:
            raise RuntimeError("scheduler made no progress")

        # Keep generated rows and one-token prompt tails contiguous in the
        # decode segment.
        scheduled.sort(key=lambda request: not request.is_kernel_decode)
        steps.append(
            ServingStep(index=len(steps) + 1, requests=tuple(scheduled)))

    return steps


def select_phase_steps(steps: Sequence[ServingStep],
                       phase: str) -> list[ServingStep]:
    """Project real serving steps onto one or both RPA kernel regions."""
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase!r}")

    selected = []
    for step in steps:
        requests = tuple(request for request in step.requests
                         if _request_is_in_phase(request, phase))
        if requests:
            selected.append(dataclasses.replace(step, requests=requests))
    if not selected:
        raise ValueError(f"phase {phase!r} has no kernel-active requests")
    return selected


def sample_steps(steps: Sequence[ServingStep],
                 sample_every: int) -> list[ServingStep]:
    """Keep every Nth phase-filtered call, starting at position zero."""
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    return list(steps[::sample_every])


def warmup_token_shapes(max_batch_tokens: int) -> list[int]:
    """Power-of-two static T shapes compiled before the measured trace."""
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")
    shapes = []
    token_shape = 16
    while token_shape <= max_batch_tokens:
        shapes.append(token_shape)
        token_shape *= 2
    return shapes


def _prefill_warmup_requests(
    config: WorkloadConfig,
    token_shape: int,
    *,
    first_request_id: int,
) -> list[ScheduledRequest]:
    requests = []
    remaining = token_shape
    max_query_tokens = config.input_tokens - config.prefix_tokens
    if max_query_tokens <= 1:
        return requests

    for request_id in range(first_request_id, config.num_requests):
        if remaining <= 1:
            break
        query_tokens = min(max_query_tokens, remaining)
        if query_tokens <= 1:
            break
        requests.append(
            ScheduledRequest(
                request_id=request_id,
                query_tokens=query_tokens,
                kv_tokens=config.prefix_tokens + query_tokens,
                is_prompt=True,
            ))
        remaining -= query_tokens
    return requests


def _synthetic_warmup_step(config: WorkloadConfig, token_shape: int,
                           phase: str) -> ServingStep:
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase!r}")
    if token_shape <= 0:
        raise ValueError("token_shape must be positive")

    requests = []
    request_context = config.input_tokens + config.output_tokens - 1
    decode_kv_tokens = max(1, min(request_context, config.input_tokens + 1))
    if phase == "decode":
        for request_id in range(min(config.num_requests, token_shape)):
            requests.append(
                ScheduledRequest(
                    request_id=request_id,
                    query_tokens=1,
                    kv_tokens=decode_kv_tokens,
                    is_prompt=False,
                ))
    elif phase == "prefill":
        requests.extend(
            _prefill_warmup_requests(
                config,
                token_shape,
                first_request_id=0,
            ))
    else:
        first_prefill_request_id = 0
        remaining_tokens = token_shape
        if config.num_requests > 1 and config.output_tokens > 1:
            requests.append(
                ScheduledRequest(
                    request_id=0,
                    query_tokens=1,
                    kv_tokens=decode_kv_tokens,
                    is_prompt=False,
                ))
            first_prefill_request_id = 1
            remaining_tokens -= 1
        requests.extend(
            _prefill_warmup_requests(
                config,
                remaining_tokens,
                first_request_id=first_prefill_request_id,
            ))

    if not requests:
        raise ValueError(f"cannot build a synthetic {phase!r} warmup step")
    return ServingStep(index=0, requests=tuple(requests))


def build_warmup_steps(
    steps: Sequence[ServingStep],
    token_shapes: Sequence[int],
    *,
    phase: str,
    config: WorkloadConfig,
) -> dict[int, ServingStep]:
    """Choose representative metadata for each static warmup T shape."""
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase!r}")

    warmup_steps = {}
    for token_shape in token_shapes:
        candidates = [
            step for step in steps if step.total_query_tokens <= token_shape
        ]
        representative = (max(
            candidates,
            key=lambda step: step.total_query_tokens,
        ) if candidates else None)
        if representative is None:
            representative = _synthetic_warmup_step(config, token_shape, phase)
        warmup_steps[token_shape] = representative
    return warmup_steps


def estimate_decode_attention_hbm_bytes(
    step: ServingStep,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype_bytes: int,
    kv_dtype_bytes: int,
    sliding_window: int | None,
) -> int:
    """Estimate HBM traffic touched by one decode-only attention dispatch."""
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive when set")
    if any(not request.is_kernel_decode for request in step.requests):
        raise ValueError(
            "HBM bandwidth estimation is only supported for decode steps")

    q_elements = step.total_query_tokens * num_q_heads * head_dim
    new_kv_elements = step.total_query_tokens * num_kv_heads * head_dim * 2
    cache_read_tokens = sum(
        (min(request.kv_tokens, sliding_window
             ) if sliding_window is not None else request.kv_tokens)
        for request in step.requests)
    cache_read_elements = cache_read_tokens * num_kv_heads * head_dim * 2
    # Count projected K/V once as input traffic and once as KV-cache writeback.
    return (q_elements * q_dtype_bytes + new_kv_elements * kv_dtype_bytes +
            new_kv_elements * kv_dtype_bytes +
            cache_read_elements * kv_dtype_bytes + q_elements * q_dtype_bytes)


def estimate_prefill_attention_flops(
    step: ServingStep,
    *,
    num_q_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> int:
    """Estimate useful QK and PV FLOPs for one prefill-only dispatch."""
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive when set")
    if any(request.is_kernel_decode for request in step.requests):
        raise ValueError("FLOP estimation is only supported for prefill steps")

    attention_pairs = 0
    for request in step.requests:
        query_tokens = request.query_tokens
        prefix_tokens = request.kv_tokens - query_tokens
        if prefix_tokens < 0:
            raise ValueError("kv_tokens must include all query tokens")

        if sliding_window is None:
            visible_pairs = (query_tokens * prefix_tokens + query_tokens *
                             (query_tokens + 1) // 2)
        else:
            ramp_tokens = min(query_tokens,
                              max(sliding_window - prefix_tokens, 0))
            visible_pairs = (ramp_tokens *
                             (2 * prefix_tokens + ramp_tokens + 1) // 2 +
                             (query_tokens - ramp_tokens) * sliding_window)
        attention_pairs += visible_pairs

    # QK and PV each perform one multiply-add per visible query/key pair.
    return 4 * num_q_heads * head_dim * attention_pairs


def build_step_metadata(
    step: ServingStep,
    *,
    max_num_seqs: int,
    pages_per_seq: int,
    allocated_pages_per_request: int,
) -> StepMetadata:
    """Materialize decode-first metadata for one simulated step."""
    if max_num_seqs < len(step.requests):
        raise ValueError(
            "max_num_seqs is smaller than this step's request count")
    if allocated_pages_per_request <= 0:
        raise ValueError("allocated_pages_per_request must be positive")
    if pages_per_seq < allocated_pages_per_request:
        raise ValueError(
            "pages_per_seq is smaller than the allocated page count")

    kv_lens = [0] * max_num_seqs
    cu_q_lens = [0] * (max_num_seqs + 1)
    page_indices = [0] * (max_num_seqs * pages_per_seq)
    for row, request in enumerate(step.requests):
        kv_lens[row] = request.kv_tokens
        cu_q_lens[row + 1] = cu_q_lens[row] + request.query_tokens
        page_start = request.request_id * allocated_pages_per_request
        row_start = row * pages_per_seq
        page_indices[row_start:row_start +
                     allocated_pages_per_request] = range(
                         page_start, page_start + allocated_pages_per_request)
    cu_q_lens[len(step.requests) +
              1:] = [step.total_query_tokens
                     ] * (max_num_seqs - len(step.requests))
    return StepMetadata(
        kv_lens=tuple(kv_lens),
        page_indices=tuple(page_indices),
        cu_q_lens=tuple(cu_q_lens),
        distribution=(step.decode_count, step.decode_count,
                      len(step.requests)),
    )


def summarize_schedule(
    config: WorkloadConfig,
    steps: Sequence[ServingStep],
    *,
    phase: str,
    padded_tokens: Callable[[int], int] | None,
) -> dict[str, Any]:
    """Build JSON-friendly schedule totals."""
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase!r}")
    actual_tokens = [step.total_query_tokens for step in steps]
    expected_query_tokens = sum(request.query_tokens for step in steps
                                for request in step.requests
                                if _request_is_in_phase(request, phase))
    summary: dict[str, Any] = {
        "num_steps":
        len(steps),
        "total_query_tokens":
        sum(actual_tokens),
        "expected_query_tokens":
        expected_query_tokens,
        "decode_steps":
        sum(step.decode_count > 0 for step in steps),
        "prefill_steps":
        sum(step.prefill_count > 0 for step in steps),
        "prefill_decode_steps":
        sum(step.decode_count > 0 and step.prefill_count > 0
            for step in steps),
        "saturated_steps":
        sum(tokens == config.max_batch_tokens for tokens in actual_tokens),
        "max_step_query_tokens":
        max(actual_tokens),
    }
    if padded_tokens is not None:
        padded = [padded_tokens(tokens) for tokens in actual_tokens]
        summary["total_padded_query_tokens"] = sum(padded)
        summary["padded_token_shapes"] = {
            str(shape): count
            for shape, count in sorted(Counter(padded).items())
        }
    return summary


def _print_schedule(
    config: WorkloadConfig,
    steps: Sequence[ServingStep],
    phase: str,
    sample_every: int,
    decode_q_len: int,
) -> None:
    summary = summarize_schedule(
        config,
        steps,
        phase=phase,
        padded_tokens=None,
    )
    print(
        "workload: "
        f"requests={config.num_requests} input={config.input_tokens} "
        f"output={config.output_tokens} prefix={config.prefix_tokens} "
        f"budget={config.max_batch_tokens} "
        f"phase={phase} decode_q_len={decode_q_len} sample_every={sample_every}"
    )
    print(
        "schedule: "
        f"steps={summary['num_steps']} query_tokens={summary['total_query_tokens']:,} "
        f"decode_steps={summary['decode_steps']} prefill_steps={summary['prefill_steps']} "
        f"saturated_steps={summary['saturated_steps']}")
    selected = list(steps[:10])
    if len(steps) > 15:
        selected.extend(steps[-5:])
    print(f"{'step':>6} {'q':>6} {'decode':>7} {'prefill':>7}  q_lens")
    previous_index = 0
    for step in selected:
        if previous_index and step.index > previous_index + 1:
            print("     ...")
        q_lens = ",".join(
            str(request.query_tokens) for request in step.requests)
        print(f"{step.index:>6} {step.total_query_tokens:>6} "
              f"{step.decode_count:>7} {step.prefill_count:>7}  {q_lens}")
        previous_index = step.index


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_runtime_args(args: argparse.Namespace,
                           config: WorkloadConfig) -> None:
    config.validate()
    if args.max_num_seqs < config.num_requests:
        raise ValueError("max_num_seqs must be >= num_requests")
    required_context = config.input_tokens + config.output_tokens - 1
    if args.max_model_len < required_context:
        raise ValueError(
            f"max_model_len must be >= {required_context} for this workload")
    if (args.page_size <= 0 or args.page_size % 128
            or args.page_size & (args.page_size - 1)):
        raise ValueError("page_size must be a power-of-two multiple of 128")
    if args.num_q_heads <= 0 or args.num_kv_heads <= 0 or args.head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if args.num_q_heads % args.num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if args.sliding_window is not None and args.sliding_window <= 0:
        raise ValueError("sliding_window must be positive when set")
    if args.decode_q_len <= 0:
        raise ValueError("decode_q_len must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")


def _run_tpu(
    args: argparse.Namespace,
    config: WorkloadConfig,
    steps: Sequence[ServingStep],
    warmup_source_steps: Sequence[ServingStep],
) -> dict[str, Any]:
    # Keep --schedule-only usable on a host without the TPU/JAX environment.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    from tpu_inference.kernels.experimental.stacked_rpa import wrapper
    from tpu_inference.kernels.experimental.stacked_rpa.utils import cdiv

    kv_dtype = {
        "bf16": jnp.bfloat16,
        "fp8": jnp.float8_e4m3fn,
    }[args.kv_dtype]
    sm_scale = args.head_dim**-0.5

    def padded_tokens(actual_tokens: int) -> int:
        return _next_power_of_2(actual_tokens)

    padded_shapes = sorted(
        {padded_tokens(step.total_query_tokens)
         for step in steps})
    compile_shapes = sorted(
        set(padded_shapes) | set(warmup_token_shapes(config.max_batch_tokens)))

    def input_array(shape, dtype, *, scale):
        values = np.full(shape, scale, dtype=np.float32)
        return jnp.asarray(values).astype(dtype)

    token_inputs = {}
    for token_shape in compile_shapes:
        token_inputs[token_shape] = (
            input_array((token_shape, args.num_q_heads, args.head_dim),
                        jnp.bfloat16,
                        scale=1.0),
            input_array((token_shape, args.num_kv_heads, args.head_dim),
                        kv_dtype,
                        scale=1.0),
            input_array((token_shape, args.num_kv_heads, args.head_dim),
                        kv_dtype,
                        scale=1.0),
        )

    request_context = config.input_tokens + config.output_tokens - 1
    allocated_pages_per_request = cdiv(request_context, args.page_size)
    pages_per_seq = cdiv(args.max_model_len, args.page_size)
    total_pages = config.num_requests * allocated_pages_per_request + 1
    kv_shape = wrapper.get_kv_cache_shape(
        total_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        kv_dtype,
    )

    def fresh_cache():
        return input_array(kv_shape, kv_dtype, scale=1e-3)

    def metadata_for(step: ServingStep):
        metadata = build_step_metadata(
            step,
            max_num_seqs=args.max_num_seqs,
            pages_per_seq=pages_per_seq,
            allocated_pages_per_request=allocated_pages_per_request,
        )
        return (
            jnp.asarray(metadata.kv_lens, dtype=jnp.int32),
            jnp.asarray(metadata.page_indices, dtype=jnp.int32),
            jnp.asarray(metadata.cu_q_lens, dtype=jnp.int32),
            jnp.asarray(metadata.distribution, dtype=jnp.int32),
        )

    @jax.jit
    def schedule_step(queries, keys, values, kv_cache, kv_lens, pages, cu_q,
                      dist):
        return wrapper.build_schedules(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            pages,
            cu_q,
            dist,
            sm_scale=sm_scale,
            sliding_window=args.sliding_window,
            decode_q_len=args.decode_q_len,
        )

    @functools.partial(jax.jit, donate_argnums=(3, ))
    def attention_step(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        pages,
        cu_q,
        dist,
        decode_schedule,
        prefill_schedule,
    ):
        return wrapper.ragged_paged_attention(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            pages,
            cu_q,
            dist,
            sm_scale=sm_scale,
            sliding_window=args.sliding_window,
            decode_q_len=args.decode_q_len,
            precomputed_schedules=(decode_schedule, prefill_schedule),
        )

    warmup_steps = build_warmup_steps(
        warmup_source_steps,
        compile_shapes,
        phase=args.phase,
        config=config,
    )

    print(
        f"device: {jax.devices()[0].device_kind}; q=bf16 kv={args.kv_dtype}; "
        f"heads={args.num_q_heads}/{args.num_kv_heads}/{args.head_dim}; "
        f"page={args.page_size}; max_model_len={args.max_model_len}; "
        f"max_num_seqs={args.max_num_seqs}; decode_q_len={args.decode_q_len}")
    print(
        f"compiling/warming token shapes: {compile_shapes}",
        flush=True,
    )
    warm_cache = fresh_cache()
    for token_shape, step in warmup_steps.items():
        queries, keys, values = token_inputs[token_shape]
        metadata = metadata_for(step)
        for _ in range(args.warmup):
            schedules = schedule_step(queries, keys, values, warm_cache,
                                      *metadata)
            _, warm_cache = attention_step(
                queries,
                keys,
                values,
                warm_cache,
                *metadata,
                *schedules,
            )
        jax.block_until_ready(warm_cache)

    # A level-2 trace of schedule generation is orders of magnitude larger than
    # the attention trace. Materialize schedules and metadata first, then trace
    # only the per-layer attention dispatches. Empty mode schedules are identical
    # for a given token shape, so reuse them across decode-only/prefill-only steps.
    print(f"precomputing {len(steps)} per-step schedules", flush=True)
    prepared_steps = []
    schedule_cache = fresh_cache()
    empty_decode_schedules = {}
    empty_prefill_schedules = {}
    for offset, step in enumerate(steps, start=1):
        token_shape = padded_tokens(step.total_query_tokens)
        queries, keys, values = token_inputs[token_shape]
        metadata = metadata_for(step)
        decode_schedule, prefill_schedule = schedule_step(
            queries, keys, values, schedule_cache, *metadata)
        jax.block_until_ready((decode_schedule, prefill_schedule))
        if step.decode_count == 0:
            decode_schedule = empty_decode_schedules.setdefault(
                token_shape, decode_schedule)
        if step.prefill_count == 0:
            prefill_schedule = empty_prefill_schedules.setdefault(
                token_shape, prefill_schedule)
        prepared_steps.append(
            (token_shape, metadata, (decode_schedule, prefill_schedule)))
        if args.progress_every and (offset == len(steps)
                                    or offset % args.progress_every == 0):
            print(
                f"  schedules {offset:>4}/{len(steps)}",
                flush=True,
            )
    del schedule_cache

    cache = fresh_cache()
    jax.block_until_ready(cache)
    temp_trace = None
    measured_trace_dir = None
    if args.trace_dir is None:
        temp_trace = tempfile.TemporaryDirectory(prefix="stacked_rpa_serving_")
        measured_trace_dir = Path(temp_trace.name)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        measured_trace_dir = args.trace_dir / f"stacked-serving-{stamp}"
        shutil.rmtree(measured_trace_dir, ignore_errors=True)
        measured_trace_dir.mkdir(parents=True)

    profile_options = jax.profiler.ProfileOptions()
    profile_options.python_tracer_level = 0
    profile_options.device_tracer_level = 2
    jax.profiler.start_trace(str(measured_trace_dir),
                             profiler_options=profile_options)

    for offset, step in enumerate(steps, start=1):
        token_shape, metadata, schedules = prepared_steps[offset - 1]
        queries, keys, values = token_inputs[token_shape]

        _, cache = attention_step(
            queries,
            keys,
            values,
            cache,
            *metadata,
            *schedules,
        )
        jax.block_until_ready(cache)
        if args.progress_every and (offset == 1 or offset == len(steps)
                                    or offset % args.progress_every == 0):
            print(
                f"  step {offset:>4}/{len(steps)}: "
                f"q={step.total_query_tokens:>4} padded={token_shape:>4} "
                f"decode={step.decode_count:>3} prefill={step.prefill_count:>2}",
                flush=True,
            )
    jax.profiler.stop_trace()
    prepared_steps = None

    device_ms = _device_kernel_ms_per_dispatch_from_trace(
        str(measured_trace_dir), jit_name_prefix="jit_attention_step")
    if len(device_ms) != len(steps):
        raise RuntimeError(
            "profiler trace contained "
            f"{len(device_ms)} jit_attention_step dispatches; expected {len(steps)}"
        )

    schedule_summary = summarize_schedule(config,
                                          steps,
                                          phase=args.phase,
                                          padded_tokens=padded_tokens)
    peak_hbm_bandwidth_gbs = pltpu.get_tpu_info().mem_bw_bytes_per_second / 1e9
    if args.phase == "decode":
        q_dtype_bytes = 2
        kv_dtype_bytes = jnp.dtype(kv_dtype).itemsize
        estimated_hbm_bytes = [
            estimate_decode_attention_hbm_bytes(
                step,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                q_dtype_bytes=q_dtype_bytes,
                kv_dtype_bytes=kv_dtype_bytes,
                sliding_window=args.sliding_window,
            ) for step in steps
        ]
        estimated_total_hbm_bytes = sum(estimated_hbm_bytes)
        estimated_hbm_bandwidth_gbs = (estimated_total_hbm_bytes /
                                       (sum(device_ms) / 1000.0) /
                                       1e9 if device_ms else 0.0)
        estimated_hbm_bandwidth_utilization_percent = (
            estimated_hbm_bandwidth_gbs / peak_hbm_bandwidth_gbs *
            100.0 if peak_hbm_bandwidth_gbs else None)
        estimated_hbm_scope = "decode"
    else:
        estimated_hbm_bytes = [None] * len(steps)
        estimated_total_hbm_bytes = None
        estimated_hbm_bandwidth_gbs = None
        estimated_hbm_bandwidth_utilization_percent = None
        estimated_hbm_scope = None
    peak_compute_tflops = pltpu.get_tpu_info().bf16_ops_per_second / 1e12
    if args.phase == "prefill":
        estimated_attention_flops = [
            estimate_prefill_attention_flops(
                step,
                num_q_heads=args.num_q_heads,
                head_dim=args.head_dim,
                sliding_window=args.sliding_window,
            ) for step in steps
        ]
        estimated_attention_total_flops = sum(estimated_attention_flops)
        achieved_attention_tflops = (estimated_attention_total_flops /
                                     (sum(device_ms) / 1000.0) /
                                     1e12 if device_ms else 0.0)
        estimated_attention_compute_utilization_percent = (
            achieved_attention_tflops / peak_compute_tflops *
            100.0 if peak_compute_tflops else None)
        estimated_attention_scope = "prefill"
    else:
        estimated_attention_flops = [None] * len(steps)
        estimated_attention_total_flops = None
        estimated_attention_compute_utilization_percent = None
        estimated_attention_scope = None
    runtime_summary: dict[str, Any] = {
        "device_attention_total_ms":
        sum(device_ms),
        "device_attention_mean_ms":
        statistics.mean(device_ms),
        "device_attention_p50_ms":
        statistics.median(device_ms),
        "device_attention_p90_ms":
        _percentile(device_ms, 90),
        "device_attention_p99_ms":
        _percentile(device_ms, 99),
        "estimated_hbm_total_bytes":
        estimated_total_hbm_bytes,
        "estimated_hbm_bandwidth_gbs":
        estimated_hbm_bandwidth_gbs,
        "peak_hbm_bandwidth_gbs":
        peak_hbm_bandwidth_gbs,
        "estimated_hbm_bandwidth_utilization_percent":
        (estimated_hbm_bandwidth_utilization_percent),
        "estimated_hbm_scope":
        estimated_hbm_scope,
        "estimated_attention_total_flops":
        estimated_attention_total_flops,
        "peak_compute_tflops":
        peak_compute_tflops,
        "estimated_attention_compute_utilization_percent":
        (estimated_attention_compute_utilization_percent),
        "estimated_attention_scope":
        estimated_attention_scope,
    }

    step_results = []
    for index, step in enumerate(steps):
        kv_tokens = [request.kv_tokens for request in step.requests]
        step_hbm_bytes = estimated_hbm_bytes[index]
        if step_hbm_bytes is None:
            step_hbm_bandwidth_gbs = None
            step_hbm_bandwidth_utilization_percent = None
        else:
            step_hbm_bandwidth_gbs = (step_hbm_bytes /
                                      (device_ms[index] / 1000.0) /
                                      1e9 if device_ms[index] else 0.0)
            step_hbm_bandwidth_utilization_percent = (
                step_hbm_bandwidth_gbs / peak_hbm_bandwidth_gbs *
                100.0 if peak_hbm_bandwidth_gbs else None)
        step_attention_flops = estimated_attention_flops[index]
        if step_attention_flops is None:
            step_attention_compute_utilization_percent = None
        else:
            step_achieved_attention_tflops = (step_attention_flops /
                                              (device_ms[index] / 1000.0) /
                                              1e12
                                              if device_ms[index] else 0.0)
            step_attention_compute_utilization_percent = (
                step_achieved_attention_tflops / peak_compute_tflops *
                100.0 if peak_compute_tflops else None)
        record = {
            "step":
            step.index,
            "query_tokens":
            step.total_query_tokens,
            "padded_query_tokens":
            padded_tokens(step.total_query_tokens),
            "decode_requests":
            step.decode_count,
            "prefill_requests":
            step.prefill_count,
            "min_kv_tokens":
            min(kv_tokens),
            "max_kv_tokens":
            max(kv_tokens),
            "device_attention_ms":
            device_ms[index],
            "estimated_hbm_bytes":
            step_hbm_bytes,
            "estimated_hbm_bandwidth_gbs":
            step_hbm_bandwidth_gbs,
            "estimated_hbm_bandwidth_utilization_percent":
            (step_hbm_bandwidth_utilization_percent),
            "estimated_attention_flops":
            step_attention_flops,
            "estimated_attention_compute_utilization_percent":
            (step_attention_compute_utilization_percent),
        }
        step_results.append(record)

    result = {
        "config": {
            **dataclasses.asdict(config),
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "page_size": args.page_size,
            "q_dtype": "bf16",
            "kv_dtype": args.kv_dtype,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "sliding_window": args.sliding_window,
            "phase": args.phase,
            "decode_q_len": args.decode_q_len,
            "sample_every": args.sample_every,
        },
        "device":
        jax.devices()[0].device_kind,
        "schedule":
        schedule_summary,
        "runtime":
        runtime_summary,
        "trace_dir": (str(measured_trace_dir) if args.trace_dir is not None
                      and measured_trace_dir is not None else None),
        "steps":
        step_results,
    }
    if temp_trace is not None:
        temp_trace.cleanup()
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        required=True,
        help="Cached tokens at the start of every prompt",
    )
    parser.add_argument(
        "--decode-q-len",
        type=int,
        required=True,
        help="Maximum generated query tokens per sequence",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        required=True,
        help="RPA request region to execute",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        required=True,
        help="Keep every Nth phase-filtered step",
    )
    parser.add_argument(
        "--max-batch-tokens",
        "--max-num-batched-tokens",
        dest="max_batch_tokens",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        required=True,
        help="Static request-slot capacity",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        required=True,
        help="Static page-table context capacity",
    )
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument(
        "--kv-dtype",
        choices=("bf16", "fp8"),
        required=True,
    )
    attention = parser.add_mutually_exclusive_group(required=True)
    attention.add_argument(
        "--sliding-window",
        type=int,
        help="Sliding-window attention size",
    )
    attention.add_argument(
        "--full-context",
        dest="sliding_window",
        action="store_const",
        const=None,
        help="Use full-context attention",
    )
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--progress-every", type=int, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Validate/print the host schedule without importing JAX",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None) -> int:
    args = _parse_args(argv)
    config = WorkloadConfig(
        num_requests=args.num_requests,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        prefix_tokens=args.prefix_tokens,
        max_batch_tokens=args.max_batch_tokens,
    )
    _validate_runtime_args(args, config)
    full_steps = simulate_greedy_schedule(config,
                                          decode_q_len=args.decode_q_len)
    phase_steps = select_phase_steps(full_steps, args.phase)
    steps = sample_steps(phase_steps, args.sample_every)
    _print_schedule(config, steps, args.phase, args.sample_every,
                    args.decode_q_len)

    if args.schedule_only:
        result = {
            "config": {
                **dataclasses.asdict(config),
                "max_num_seqs": args.max_num_seqs,
                "max_model_len": args.max_model_len,
                "page_size": args.page_size,
                "sliding_window": args.sliding_window,
                "phase": args.phase,
                "decode_q_len": args.decode_q_len,
                "sample_every": args.sample_every,
            },
            "schedule":
            summarize_schedule(
                config,
                steps,
                phase=args.phase,
                padded_tokens=None,
            ),
        }
    else:
        result = _run_tpu(args, config, steps, phase_steps)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")
    print("RESULT " + json.dumps(
        {
            "config": result["config"],
            "schedule": result["schedule"],
            "runtime": result.get("runtime"),
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
