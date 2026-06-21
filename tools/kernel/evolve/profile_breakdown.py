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
"""Profile-time breakdown for an evolve target.

Roofline tells you the regime (HBM/MXU/scalar bound). This tool drills
down to *which operations actually consume the time* by capturing a
``jax.profiler`` trace, parsing the Chrome trace JSON, and ranking
events by total wall time.

Output is a markdown table suitable for injecting into a mutator
prompt as "here are the top 10 hot operations — your mutation should
target one of these or it cannot win on this kernel". This is the
profile-guided-mutation lever the system needs to escape the
+1-5% ceiling that single-line edits hit on dispatch-bound fixtures.

Usage::

    python -m tools.kernel.evolve.profile_breakdown \\
        --target fused_moe_v1 --iters 20
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import jax

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OpProfile:
    name: str
    total_us: float
    call_count: int
    avg_us: float
    fraction: float  # of total measured time


def capture_profile(fn,
                    *,
                    iters: int = 20,
                    trace_dir: Path | None = None) -> Path:
    """Run ``fn`` ``iters`` times under jax.profiler.trace; return trace dir."""
    if trace_dir is None:
        trace_dir = Path(tempfile.mkdtemp(prefix="evolve_prof_"))
    # Warm
    out = fn()
    jax.block_until_ready(out)
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        for _ in range(iters):
            out = fn()
        jax.block_until_ready(out)
    return trace_dir


def _find_trace_json(trace_dir: Path) -> Path:
    for root, _, files in os.walk(trace_dir):
        for f in files:
            if f.endswith("trace.json.gz"):
                return Path(root) / f
    raise FileNotFoundError(f"no trace.json.gz under {trace_dir}")


def parse_breakdown(trace_dir: Path,
                    *,
                    top_k: int = 25,
                    min_total_us: float = 5.0) -> list[OpProfile]:
    """Parse the Chrome trace; return top-K operations by total time."""
    trace = _find_trace_json(trace_dir)
    with gzip.open(trace, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    by_name: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if "dur" in e and "name" in e:
            by_name[e["name"]].append(float(e["dur"]))
    # Aggregate
    rows = []
    for name, durs in by_name.items():
        total = sum(durs)
        if total < min_total_us:
            continue
        rows.append((total, len(durs), name))
    rows.sort(reverse=True)
    # Filter out the profiler's own overhead lines (they dominate the
    # raw top of the trace and aren't useful for kernel attribution).
    skip_prefixes = ("$profiler.py", "$contextlib.py", "$api.py:2551",
                     "$api.py:2562")
    filtered = []
    for total, count, name in rows:
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        filtered.append((total, count, name))
    grand = sum(t for t, _, _ in filtered) or 1.0
    out = []
    for total, count, name in filtered[:top_k]:
        out.append(
            OpProfile(name=name,
                      total_us=total / 1.0,
                      call_count=count,
                      avg_us=total / count,
                      fraction=total / grand))
    return out


def render_md(ops: list[OpProfile]) -> str:
    """Format the top hot operations as a markdown table for prompt
    injection."""
    lines = [
        "## Per-operation time breakdown (jax.profiler trace)\n",
        "Top operations by total time. **Your mutation should target one "
        "of these; mutations to lines that don't show up here cannot "
        "produce a measurable speedup.**\n",
        "| rank | total μs | count | avg μs | fraction | operation |",
        "|---|---|---|---|---|---|",
    ]
    for i, op in enumerate(ops, start=1):
        name = op.name if len(op.name) <= 80 else op.name[:77] + "..."
        lines.append(f"| {i} | {op.total_us:.0f} | {op.call_count} | "
                     f"{op.avg_us:.1f} | {op.fraction*100:.1f}% | `{name}` |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target",
                   required=True,
                   choices=("rpa_v3", "fused_moe_v1", "mla_v2",
                            "quantized_matmul"))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-md",
                   type=Path,
                   default=Path("/tmp/profile_breakdown.md"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")

    from tools.kernel.evolve.worktree import import_candidate_module
    if args.target == "rpa_v3":
        from tools.kernel.evolve.examples.rpa_v3_evolve import RpaV3Host
        from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
        from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner
        rc = RunConfig(case_set_id="prof",
                       run_id="r0",
                       case_set_desc="profile_breakdown",
                       tpu_version="tpu6e",
                       tpu_cores=1,
                       tpu_queue_multi="tpu_v6e_queue",
                       run_locally=True,
                       max_execution_minutes=10)
        host = RpaV3Host(RpaV3KernelTuner(run_config=rc))
    elif args.target == "fused_moe_v1":
        from tools.kernel.evolve.examples.kernel_hosts import \
            make_fused_moe_v1_host
        host = make_fused_moe_v1_host(num_tokens=512,
                                      hidden_size=2048,
                                      intermediate_size=4096,
                                      num_experts=8,
                                      topk=2)
    elif args.target == "mla_v2":
        from tools.kernel.evolve.examples.kernel_hosts import make_mla_v2_host
        host = make_mla_v2_host()
    elif args.target == "quantized_matmul":
        from tools.kernel.evolve.examples.kernel_hosts import \
            make_quantized_matmul_host
        host = make_quantized_matmul_host(n_batch=1024, n_in=4096, n_out=4096)
    else:
        raise NotImplementedError(args.target)

    src = host.read_baseline_source()
    with import_candidate_module(src, name_hint=f"prof_{args.target}") as mod:
        fn = host.build_kernel_fn(mod)
        t0 = time.perf_counter()
        trace_dir = capture_profile(fn, iters=args.iters)
        capture_wall = time.perf_counter() - t0
    print(
        f"Captured {args.iters} iters of {args.target} in "
        f"{capture_wall:.1f}s; trace at {trace_dir}",
        file=sys.stderr)
    ops = parse_breakdown(trace_dir, top_k=args.top_k)
    md = render_md(ops)
    args.out_md.write_text(md)
    print(md)
    print(f"\nFull MD written to {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
