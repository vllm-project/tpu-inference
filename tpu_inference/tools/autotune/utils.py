# Copyright 2025 Google LLC
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

import contextlib
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.syntax import Syntax

# Global console instance
console = Console()


@dataclass
class Measurement:
    """Standardized measurement result for a kernel configuration."""
    mean_latency_ns: float
    std_latency_ns: float
    compile_time_ms: Optional[float] = None
    throughput_tok_s: Optional[float] = None


def get_registry_file_name(tpu_version_slug: str) -> str:
    """Standardizes registry file naming (e.g. 'tpu_v6e', 'tpu_v5e')."""
    return tpu_version_slug.lower().replace(" ", "_")


def block_until_ready(outputs: Any):
    """Robustly blocks until JAX outputs are ready, handling tuples/lists."""
    if hasattr(outputs, "block_until_ready"):
        outputs.block_until_ready()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if hasattr(o, "block_until_ready"):
                o.block_until_ready()


def setup_logging(level: str = "INFO"):
    """Configures Rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)])


def setup_progress(transient: bool = False) -> Progress:
    """Returns a configured Rich Progress instance."""
    return Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=transient)


def write_results_json(path: str, data: Dict[str, Any]):
    """Writes data to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Results written to {path}[/green]")


def print_json_snippet(data: Dict[str, Any], label: str = "Tuning Results"):
    """Prints a copy-pasteable JSON snippet to the console."""
    json_str = json.dumps(data, indent=4)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    console.print(f"\n[bold]{label}:[/bold]")
    console.print(syntax)


def deep_update(source: Dict[str, Any],
                overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def update_json_registry(path: str, new_data: Dict[str, Any]):
    """Updates an existing JSON registry with new data."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            console.print(
                f"[red]Error decoding {path}. Creating new file.[/red]")
            existing_data = {}
    else:
        console.print(f"[yellow]Creating new registry at {path}[/yellow]")
        existing_data = {}

    # Merge
    merged_data = deep_update(existing_data, new_data)

    with open(path, 'w') as f:
        json.dump(merged_data, f, indent=2, sort_keys=True)
    console.print(f"[green]Successfully updated {path}[/green]")


# --- Profiling Utilities (Aligned with Tokamax) ---


class XprofProfileSession(contextlib.AbstractContextManager):
    """Hermetic JAX Profiler for measuring XLA Op time (Total Accelerator Time).
    
    Parses XPlane traces to extract true device execution time, excluding Python dispatch.
    Aligned with Tokamax `XprofProfileSession`.
    """

    def __init__(self):

        self._profile_tempdir = None
        self._profile = None

    @property
    def total_op_time(self) -> float:
        """Returns total device time of XLA operators in seconds."""
        if self._profile is None:
            raise ValueError("Profile session not started/stopped.")

        # Parse XPlanes (Tokamax Logic)
        xla_lines = []
        for xplane in self._profile.planes:
            if xplane.name.startswith('/device:'):
                for xline in xplane.lines:
                    # Heuristic: Match 'XLA Ops' or capture all device lines
                    if 'XLA Ops' in xline.name or 'Unknown' in xline.name:
                        xla_lines.append(xline)

        all_events = sum([list(x.events) for x in xla_lines], [])
        if not all_events:
            return 0.0

        # Calculate duration from bounds (naive sum might double count parallel ops,
        # but for single kernel benchmarks, max-min is usually correct for wall-time on device)
        t_starts = [e.start_ns for e in all_events]
        t_ends = [e.start_ns + e.duration_ns for e in all_events]
        if not t_starts:
            return 0.0

        duration_ns = max(t_ends) - min(t_starts)
        return duration_ns / 1e9

    def __enter__(self):
        try:
            self._profile_tempdir = tempfile.TemporaryDirectory(
                prefix='tpu_tune_profile_')
            jax.profiler.start_trace(self._profile_tempdir.name)
        except Exception as e:
            # Fallback or strict fail?
            # For now strict fail to warn user profiling isn't working
            raise RuntimeError(f"Failed to start JAX profiler: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profile_tempdir:
            jax.profiler.stop_trace()
            try:
                # Load profile data
                path = pathlib.Path(self._profile_tempdir.name)
                files = list(path.glob('**/*.xplane.pb'))
                if not files:
                    raise FileNotFoundError("No profile data found.")

                # Check if ProfileData class exists (JAX version dependency)
                if hasattr(jax.profiler, 'ProfileData'):
                    self._profile = jax.profiler.ProfileData.from_serialized_xspace(
                        files[0].read_bytes())
                else:
                    # Fallback for older JAX or missing utility
                    logging.warning(
                        "jax.profiler.ProfileData not found. Cannot parse XProf."
                    )
            finally:
                self._profile_tempdir.cleanup()
                self._profile_tempdir = None


def amortized_wrapper(fn, n_iter: int = 100):
    """Wraps a JAX function in a lax.fori_loop to run n_iter times on device.
    
    Used for amortizing python dispatch overhead when XProf is unavailable.
    """

    def looped_fn(*args, **kwargs):
        # Run once to get initial value and ensure invariant shape for loop carry
        init_val = fn(*args, **kwargs)

        def body(i, val):
            # We ignore the incoming `val` (carry) and re-run fn to get new output
            # (or simple dependencies if needed, but for benchmark reuse inputs is fine)
            return fn(*args, **kwargs)

        # Run n_iter more times
        return jax.lax.fori_loop(0, n_iter, body, init_val)

    return looped_fn


# --- Sharding Utilities ---


def apply_tp_scaling(
    global_val: int,
    tp_size: int,
    name: str = "value",
    printer: Optional[Console] = None,
) -> int:
    """Calculates the local sharded value for Tensor Parallelism.

    Args:
        global_val: The global dimension size (e.g. num_heads, features).
        tp_size: Tensor Parallelism degree.
        name: Name of the dimension for logging.
        printer: Optional Rich console for logging.

    Returns:
        The local dimension size (global // tp) or original if not divisible.
    """
    if tp_size <= 1:
        return global_val

    if printer is None:
        printer = console

    if global_val % tp_size == 0:
        local_val = global_val // tp_size
        printer.print(
            f"  Scaling {name}: {global_val} -> {local_val} (TP={tp_size})")
        return local_val

    # Validation/Warning logic
    if global_val == 1:
        printer.print(
            f"  Keeping {name}: {global_val} (Value=1 < TP, assuming replication)"
        )
        return global_val

    printer.print(
        f"[yellow]Warning: {name} {global_val} not divisible by TP {tp_size}. Keeping global value.[/yellow]"
    )
    return global_val
