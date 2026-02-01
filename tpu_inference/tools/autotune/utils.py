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
import csv
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
        with open(path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error decoding {path}: {e}[/red]")
                raise
    else:
        console.print(f"[yellow]Creating new registry at {path}[/yellow]")
        existing_data = {}

    # Recursive merge with safety check
    def safe_merge(target: Dict[str, Any],
                   source: Dict[str, Any],
                   path_str: str = "") -> Tuple[int, int]:
        """Recursively merges source into target with latency checks."""
        updated = 0
        skipped = 0

        for k, v in source.items():
            if k not in target:
                target[k] = v
                updated += 1
                continue

            # If both have 'stats' and 'latency_avg_ns', compare
            if isinstance(
                    v,
                    dict) and "stats" in v and "latency_avg_ns" in v["stats"]:
                if isinstance(target[k], dict) and "stats" in target[k]:
                    try:
                        new_lat = float(v["stats"].get("latency_avg_ns",
                                                       float('inf')))
                        old_lat = float(target[k]["stats"].get(
                            "latency_avg_ns", float('inf')))

                        if new_lat < old_lat:
                            target[k] = v
                            updated += 1
                        else:
                            skipped += 1
                            # Optional: verbose logging
                            # console.print(f"[dim]Skip {path_str}.{k}: {new_lat:.0f} >= {old_lat:.0f}[/dim]")
                    except (ValueError, TypeError):
                        # Fallback for Malformed data
                        target[k] = v
                        updated += 1
                else:
                    # Target has no stats, overwrite
                    target[k] = v
                    updated += 1

            elif isinstance(v, dict) and isinstance(target[k], dict):
                # Recurse
                u, s = safe_merge(target[k], v, f"{path_str}.{k}")
                updated += u
                skipped += s
            else:
                # Leaf node rewrite (non-stats)
                target[k] = v
                updated += 1

        return updated, skipped

    updated_count, skipped_count = safe_merge(existing_data, new_data)

    with open(path, 'w') as f:
        json.dump(existing_data, f, indent=2, sort_keys=True)
    console.print(
        f"[green]Registry Updated: {updated_count} accepted, {skipped_count} skipped (slower).[/green]"
    )


class RunContext:
    """Manages experiment directory and logging for a tuning run."""

    def __init__(self,
                 run_name: Optional[str] = None,
                 output_dir: str = "tuning_runs",
                 no_save: bool = False):
        self.no_save = no_save
        self.output_dir = pathlib.Path(output_dir)

        # Determine Run Name
        if not run_name or run_name == "auto":
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"tune_{timestamp}"

        self.run_name = run_name
        self.run_dir = self.output_dir / self.run_name

        self.csv_file = None
        self.csv_writer = None

        if not self.no_save:
            self._setup_directories()

    def _setup_directories(self):
        """Creates the run directory."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Initialized Run: {self.run_dir}[/green]")
        except OSError as e:
            console.print(
                f"[red]Failed to create run directory {self.run_dir}: {e}[/red]"
            )
            # Fallback to no_save if filesystem fails? Or crash?
            # For now, just warn and disable saving to prevent crash loop
            self.no_save = True

    @contextlib.contextmanager
    def open_csv(self, fieldnames: list[str]):
        """Context manager for the trials.csv log."""
        if self.no_save:
            yield None
            return

        csv_path = self.run_dir / "trials.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                yield writer
        except IOError as e:
            console.print(f"[red]Failed to open CSV log {csv_path}: {e}[/red]")
            yield None

    def save_results(self, results: Dict[str, Any]):
        """Saves the best results to results.json."""
        if self.no_save:
            return

        path = self.run_dir / "results.json"
        try:
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            console.print(f"[dim]Saved results to {path}[/dim]")
        except IOError as e:
            console.print(f"[red]Failed to save results.json: {e}[/red]")

    def save_metadata(self, metadata: Dict[str, Any]):
        """Saves run metadata (CLI args, hardware info) to run_metadata.json."""
        if self.no_save:
            return

        path = self.run_dir / "run_metadata.json"
        try:
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except IOError as e:
            console.print(f"[red]Failed to save run_metadata.json: {e}[/red]")

    def print_summary_table(self, results: Dict[str, Any], limit: int = 5):
        """Prints a rich summary table of the top results."""
        from rich.table import Table

        table = Table(
            title=f"Top Results ({min(len(results), limit)}/{len(results)})")
        table.add_column("Config Key", style="cyan", no_wrap=True)
        table.add_column("Latency", style="magenta")
        table.add_column("Compile Time", style="green")

        # Convert simple mapping to list for sorting
        # Key format varies (RPA vs Matmul), but value has 'stats'
        scored_results = []
        for key, entry in results.items():
            stats = entry.get('stats', {})
            latency = stats.get('latency_avg_ns', float('inf'))
            compile_time = stats.get('compile_time_s', 0.0)
            scored_results.append({
                "key": key,
                "latency": latency,
                "compile": compile_time
            })

        # Sort by latency
        scored_results.sort(key=lambda x: x["latency"])

        for item in scored_results[:limit]:
            lat_str = f"{item['latency'] / 1000:.2f} µs"
            if item['latency'] > 1e6:
                lat_str = f"{item['latency'] / 1e6:.2f} ms"

            table.add_row(str(item['key']), lat_str,
                          f"{item['compile']:.2f} s")

        console.print(table)
        if not self.no_save:
            console.print(
                f"\n[bold green]Run Saved to: {self.run_dir}[/bold green]")
            console.print(
                f"Apply results: [on black] tpu-tune apply {self.run_dir}/results.json [/on black]"
            )


# --- Profiling Utilities (Aligned with Tokamax) ---


class XprofProfileSession(contextlib.AbstractContextManager):
    """Hermetic JAX Profiler for measuring XLA Op time (Total Accelerator Time).
    
    Parses XPlane traces to extract true device execution time, excluding Python dispatch.
    Reference: https://github.com/google-deepmind/tokamax/blob/main/tokamax/profiling/xprof.py
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
        return float(duration_ns)

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
                    raise ImportError(
                        "jax.profiler.ProfileData not found. Please upgrade JAX."
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
