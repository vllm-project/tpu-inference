#!/usr/bin/env python3
"""Emit a conservative, deterministic Kubernetes test plan from changed paths."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

DEFAULT_OWNERSHIP = Path(".buildkite/kubernetes/test_ownership.json")
CPU_STEP = "kube_cpu_safe_tests"
DEFAULT_TPU_STEPS = {
    "kube_e2e_mlperf_jax",
    "kube_e2e_mlperf_jax_vllm",
    "kube_e2e_speculative_decoding",
    "kube_jax_unit_tests_part1",
    "kube_jax_unit_tests_part2",
    "kube_jax_unit_tests_kernels",
    "kube_lora_unit_tests",
    "kube_runai_streamer_jax",
    "kube_runai_streamer_torchax",
    "kube_qwen2_5_vl_7b_accuracy",
    "kube_lora_e2e_multi_chip",
    "kube_lora_unit_tests_multi_chip",
    "kube_runai_streamer_torchax_ray",
    "kube_disagg_single_host",
    "kube_mpmd_data_parallelism",
}
FULL_ONLY_STEPS = {
    "kube_e2e_mlperf_quantized",
    "kube_e2e_mlperf_new_models",
    "kube_e2e_mlperf_llama4",
    "kube_jax_unit_tests_collectives",
    "kube_lora_e2e_single_chip",
    "kube_lora_adapter_e2e_single_chip",
    "kube_e2e_mlperf_jax_vllm_multi_chip",
}
DOCUMENTATION_PATTERNS = ("*.md", "docs/**", "**/*.md")


def load_ownership(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError("ownership file requires a rules list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("patterns"), list):
            raise ValueError(f"ownership rule {index} requires patterns")
        if not rule.get("full") and not isinstance(rule.get("steps"), list):
            raise ValueError(f"ownership rule {index} requires steps or full=true")
    return rules


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def select_tests(
    changed_files: Iterable[str],
    rules: list[dict[str, Any]],
    *,
    event: str,
    force_full: bool = False,
) -> dict[str, Any]:
    changed = sorted(set(path.strip() for path in changed_files if path.strip()))
    code_changes = [
        path for path in changed if not _matches(path, DOCUMENTATION_PATTERNS)
    ]
    full_reasons: list[str] = []
    reasons: dict[str, set[str]] = {}
    unowned: list[str] = []

    if force_full:
        full_reasons.append("explicit full-matrix request")
    if event in {"main", "nightly", "release"}:
        full_reasons.append(f"{event} policy")

    for path in code_changes:
        matching_rules = [rule for rule in rules if _matches(path, rule["patterns"])]
        if not matching_rules:
            unowned.append(path)
            continue
        for rule in matching_rules:
            reason = f"{path}: {rule.get('reason', 'ownership rule')}"
            if rule.get("full"):
                full_reasons.append(reason)
            for step in rule.get("steps", []):
                reasons.setdefault(step, set()).add(reason)

    mode = "docs-only" if not code_changes else "targeted"
    selected = set(reasons)
    if full_reasons:
        mode = "full"
        selected = DEFAULT_TPU_STEPS | FULL_ONLY_STEPS
    elif unowned:
        # Unknown code must widen, never silently reduce coverage.
        mode = "default-fallback"
        selected |= DEFAULT_TPU_STEPS
        full_reasons.append("unowned code path widened to the default matrix")

    return {
        "schema_version": 1,
        "mode": mode,
        "shadow": True,
        "changed_files": changed,
        "unowned_files": unowned,
        "cpu_steps": [CPU_STEP] if code_changes else [],
        "tpu_steps": sorted(selected),
        "full_matrix": mode == "full",
        "reasons": {
            step: sorted(step_reasons) for step, step_reasons in sorted(reasons.items())
        },
        "policy_reasons": sorted(set(full_reasons)),
    }


def changed_files_from_git(base: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return output.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, default=DEFAULT_OWNERSHIP)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument(
        "--event",
        choices=("pull_request", "main", "nightly", "release"),
        default="pull_request",
    )
    parser.add_argument("--force-full", action="store_true")
    args = parser.parse_args()
    try:
        rules = load_ownership(args.ownership)
        changed = args.changed_file or changed_files_from_git(args.base)
        plan = select_tests(changed, rules, event=args.event, force_full=args.force_full)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"test selection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
