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
"""Recipe: sub-optimal entry sweep for ragged_paged_attention v3 + Qwen3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(*, out_dir: Path) -> dict:
    """Run the sub-optimal sweep on Qwen3-0.6B. Return a CI summary."""
    diff_path = out_dir / "rpa_v3_auto_pr.diff"
    summary_path = out_dir / "rpa_v3_summary.json"
    telemetry_path = out_dir / "rpa_v3_telemetry.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "tools.kernel.evolve.sweep.suboptimal_entries",
        "--models",
        "Qwen3-0.6B",
        "--context-lengths",
        "1024",
        "--candidates",
        "4:32,8:32,16:32,8:16",
        "--max-tokens",
        "64",
        "--min-win-margin",
        "1.01",
        "--out-diff",
        str(diff_path),
        "--out-summary",
        str(summary_path),
        "--out-telemetry",
        str(telemetry_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    if proc.returncode != 0:
        return {
            "target_kernel": "ragged_paged_attention/v3",
            "wins_count": 0,
            "error": proc.stderr[-1500:]
        }
    try:
        summary = json.loads(summary_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {"wins": []}
    return {
        "target_kernel": "ragged_paged_attention/v3",
        "wins_count": len(summary.get("wins", [])),
        "diff_path": str(diff_path),
        "summary_path": str(summary_path),
        "telemetry_path": str(telemetry_path),
    }
