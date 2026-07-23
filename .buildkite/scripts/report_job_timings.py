#!/usr/bin/env python3
"""Render Buildkite job timestamps as a Markdown timing table.

Usage:
  bk job list --pipeline ORG/PIPELINE --build NUMBER --no-limit --json \
    | python3 .buildkite/scripts/report_job_timings.py
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Any


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def elapsed(start: str | None, end: str | None) -> float | None:
    start_time = parse_timestamp(start)
    end_time = parse_timestamp(end)
    if start_time is None or end_time is None:
        return None
    return max(0.0, (end_time - start_time).total_seconds())


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def cell(value: Any) -> str:
    return str(value or "—").replace("|", "\\|")


def main() -> int:
    parser = ArgumentParser(
        description="Render `bk job list --json` input as a Markdown timing table."
    )
    parser.parse_args()

    try:
        jobs = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        parser.error(f"stdin is not valid JSON: {error}")
    if not isinstance(jobs, list):
        parser.error("expected a JSON array from `bk job list --json`")

    print(
        "| Step | State | Dependency/gate | Queue/provision | Execution | End-to-end |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for job in jobs:
        if job.get("type") != "script":
            continue
        created = job.get("created_at")
        runnable = job.get("runnable_at")
        started = job.get("started_at")
        finished = job.get("finished_at")
        name = job.get("step_key") or job.get("name") or job.get("id")
        print(
            "| "
            + " | ".join(
                (
                    cell(name),
                    cell(job.get("state")),
                    format_duration(elapsed(created, runnable)),
                    format_duration(elapsed(runnable, started)),
                    format_duration(elapsed(started, finished)),
                    format_duration(elapsed(created, finished)),
                )
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
