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
"""Telemetry analysis CLI: aggregate wins per kernel × shape × model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.kernel.evolve.telemetry.writer import load_events, summarize


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", nargs="+", type=Path)
    p.add_argument("--format", choices=("json", "table"), default="table")
    args = p.parse_args(argv)

    events = []
    for path in args.path:
        events.extend(load_events(path))
    if not events:
        print("(no events)", file=sys.stderr)
        return 1

    summary = summarize(events)
    if args.format == "json":
        print(json.dumps(summary, indent=2, default=str))
        return 0

    # Table format.
    print(f"{'kernel':24s} {'shape':36s} {'verified/total':>15s} "
          f"{'best_us':>9s} {'status counts':<30s}")
    print("-" * 120)
    for kernel, shapes in sorted(summary["kernels"].items()):
        for shape, info in sorted(shapes.items()):
            best = (f"{info['best_fitness_ns']/1e3:.2f}"
                    if info.get("best_fitness_ns") else "-")
            counts = ",".join(f"{k}={v}"
                              for k, v in sorted(info["by_status"].items()))
            print(f"{kernel:24s} {shape:36s} "
                  f"{info['verified']}/{info['total']:>10}  {best:>9s}  "
                  f"{counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
