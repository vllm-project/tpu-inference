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
"""
CI/CD Status Monitoring Script

This script monitors the CI/CD pipeline status and provides detailed
information about failing workflows.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional


def run_gh_command(command: List[str]) -> Optional[str]:
    """Run a GitHub CLI command and return the output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}", file=sys.stderr)
        print(f"Error: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(
            "GitHub CLI (gh) not found. Please install it: https://cli.github.com/",
            file=sys.stderr,
        )
        return None


def get_workflow_runs(
    owner: str,
    repo: str,
    limit: int = 20,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get recent workflow runs from GitHub."""
    command = [
        "gh",
        "run",
        "list",
        "--repo",
        f"{owner}/{repo}",
        "--limit",
        str(limit),
        "--json",
        "databaseId,name,status,conclusion,headBranch,createdAt,event,displayTitle",
    ]

    if status:
        command.extend(["--status", status])

    output = run_gh_command(command)
    if output:
        return json.loads(output)
    return []


def get_failed_jobs(owner: str, repo: str,
                    run_id: int) -> List[Dict[str, Any]]:
    """Get failed jobs for a specific workflow run."""
    command = [
        "gh",
        "run",
        "view",
        str(run_id),
        "--repo",
        f"{owner}/{repo}",
        "--json",
        "jobs",
    ]

    output = run_gh_command(command)
    if output:
        data = json.loads(output)
        jobs = data.get("jobs", [])
        return [
            job for job in jobs
            if job.get("conclusion") in ["failure", "cancelled", "timed_out"]
        ]
    return []


def get_job_logs(owner: str, repo: str, run_id: int) -> Optional[str]:
    """Get logs for a failed workflow run."""
    command = [
        "gh", "run", "view",
        str(run_id), "--repo", f"{owner}/{repo}", "--log"
    ]

    output = run_gh_command(command)
    return output


def analyze_failure(run: Dict[str, Any], owner: str,
                    repo: str) -> Dict[str, Any]:
    """Analyze a failed workflow run and extract useful information."""
    run_id = run["databaseId"]
    failed_jobs = get_failed_jobs(owner, repo, run_id)

    # Categorize the failure
    failure_type = "unknown"
    if "pre-commit" in run["name"].lower():
        failure_type = "formatting"
    elif "ready" in run["name"].lower() or "label" in run["name"].lower():
        failure_type = "label_check"
    elif "test" in run["name"].lower():
        failure_type = "test"
    elif "build" in run["name"].lower():
        failure_type = "build"

    return {
        "run_id":
        run_id,
        "name":
        run["name"],
        "branch":
        run.get("headBranch", "unknown"),
        "conclusion":
        run.get("conclusion", "unknown"),
        "created_at":
        run.get("createdAt", ""),
        "event":
        run.get("event", "unknown"),
        "failure_type":
        failure_type,
        "failed_jobs": [{
            "name": job["name"],
            "conclusion": job["conclusion"]
        } for job in failed_jobs],
        "display_title":
        run.get("displayTitle", ""),
    }


def format_report(
    failures: List[Dict[str, Any]],
    detailed: bool = False,
) -> str:
    """Format the CI/CD status report."""
    if not failures:
        return "✅ No recent CI/CD failures detected!"

    report = [f"🔴 Found {len(failures)} recent CI/CD failures:\n"]

    # Group by failure type
    by_type = {}
    for failure in failures:
        failure_type = failure["failure_type"]
        if failure_type not in by_type:
            by_type[failure_type] = []
        by_type[failure_type].append(failure)

    for failure_type, items in sorted(by_type.items()):
        report.append(f"\n## {failure_type.upper()} Failures ({len(items)})")

        for item in items:
            report.append(f"\n### Run #{item['run_id']}: {item['name']}")
            report.append(f"- Branch: {item['branch']}")
            report.append(f"- Status: {item['conclusion']}")
            report.append(f"- Created: {item['created_at']}")
            report.append(f"- Title: {item['display_title']}")

            if item["failed_jobs"]:
                report.append("- Failed Jobs:")
                for job in item["failed_jobs"]:
                    report.append(f"  - {job['name']}: {job['conclusion']}")

            # Add specific recommendations
            if failure_type == "formatting":
                report.append(
                    "\n**Fix:** Run `pre-commit run --all-files` locally")
                report.append(
                    "  or comment `/fix-precommit` on the PR to auto-fix")
            elif failure_type == "label_check":
                report.append("\n**Fix:** Add the 'ready' label to the PR")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CI/CD pipeline status")
    parser.add_argument(
        "--owner",
        default="vllm-project",
        help="GitHub repository owner",
    )
    parser.add_argument(
        "--repo",
        default="tpu-inference",
        help="GitHub repository name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of recent runs to check",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Only show failed runs",
    )

    args = parser.parse_args()

    # Get workflow runs
    status = "failure" if args.only_failures else None
    runs = get_workflow_runs(args.owner, args.repo, args.limit, status)

    # Filter for failures and analyze
    failures = []
    for run in runs:
        if run.get("conclusion") in [
                "failure",
                "cancelled",
                "action_required",
        ]:
            failure_info = analyze_failure(run, args.owner, args.repo)
            failures.append(failure_info)

    # Output results
    if args.json:
        print(json.dumps(failures, indent=2))
    else:
        print(format_report(failures, args.detailed))

    # Exit with error code if there are failures
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
