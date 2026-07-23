#!/usr/bin/env python3
"""Run the explicit CPU-safe pytest allowlist without collecting TPU tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

_MARKER_TEXT = "pytest.mark.cpu_safe"


def load_manifest(path: Path, repo_root: Path) -> list[Path]:
    """Validate and return repo-relative paths from the CPU-safe manifest."""
    entries: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        entry = raw_line.split("#", 1)[0].strip()
        if not entry:
            continue
        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{path}:{line_number}: path must stay in repo: {entry}")
        candidate = (repo_root / relative).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: path escapes repository: {entry}") from error
        if not candidate.is_file():
            raise ValueError(f"{path}:{line_number}: test file does not exist: {entry}")
        if _MARKER_TEXT not in candidate.read_text():
            raise ValueError(
                f"{path}:{line_number}: {entry} is missing {_MARKER_TEXT}")
        if relative in seen:
            raise ValueError(f"{path}:{line_number}: duplicate test file: {entry}")
        seen.add(relative)
        entries.append(relative)
    if not entries:
        raise ValueError(f"{path}: CPU-safe manifest is empty")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".buildkite/cpu_safe_tests.txt"),
        help="repo-relative allowlist (default: %(default)s)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="validate the allowlist and ask pytest to collect without running",
    )
    args, pytest_args = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    try:
        tests = load_manifest(manifest, repo_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment.setdefault("TPU_NAME", "cpu-safe")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "cpu_safe",
        *map(str, tests),
    ]
    if args.collect_only:
        command.append("--collect-only")
    command.extend(pytest_args)
    print("CPU-safe test command:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=repo_root, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
