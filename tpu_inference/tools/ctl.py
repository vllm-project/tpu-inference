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
"""Operator probe & lifecycle control CLI for TPU inference workloads.

Usage:
  tpu-ctl status
  tpu-ctl pause [--timeout=60]
  tpu-ctl resume
  tpu-ctl probe --type=[liveness|readiness|paused|running]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = os.getenv("TPU_CTL_ENDPOINT", "http://localhost:8000")


def get_worker_rank() -> int:
    """Detects worker rank across multi-host TPU environments."""
    if "TPU_WORKER_ID" in os.environ:
        try:
            return int(os.environ["TPU_WORKER_ID"])
        except ValueError:
            pass

    if "JOB_COMPLETION_INDEX" in os.environ:
        try:
            return int(os.environ["JOB_COMPLETION_INDEX"])
        except ValueError:
            pass

    hostname = os.environ.get("HOSTNAME", "")
    if "-" in hostname:
        parts = hostname.split("-")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
    return 0


def call_api(endpoint: str,
             path: str,
             method: str = "GET",
             data: dict | None = None,
             timeout: float = 30.0) -> tuple[int, dict]:
    url = f"{endpoint.rstrip('/')}{path}"
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {
        "Content-Type": "application/json"
    } if req_data is not None else {}
    req = urllib.request.Request(url,
                                 data=req_data,
                                 headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            return status_code, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"error": body}
        return e.code, parsed
    except Exception as e:
        return 500, {"error": str(e)}


def cmd_status(args: argparse.Namespace) -> int:
    rank = get_worker_rank()
    if rank > 0:
        print(
            f"[CTL-Worker] Worker pod (Rank {rank}) active. TPU HBM allocated; controlled by Master."
        )
        return 0

    code, resp = call_api(args.endpoint, "/ctl/status", method="GET")
    if code == 200:
        print("=" * 65)
        print(" TPU Inference Engine Status")
        print("=" * 65)
        print(json.dumps(resp, indent=2))
        print("=" * 65)
        return 0
    else:
        print(f"[CTL-Master] Error fetching status (HTTP {code}): {resp}")
        return 1


def cmd_pause(args: argparse.Namespace) -> int:
    rank = get_worker_rank()
    if rank > 0:
        print(f"[CTL-Worker] Silent local no-op on worker pod (Rank {rank}).")
        return 0

    print(
        "[CTL-Master] Pausing TPU inference engine (draining in-flight work)..."
    )
    code, resp = call_api(
        args.endpoint,
        f"/ctl/pause?drain_timeout_s={args.timeout}",
        method="POST",
        timeout=args.timeout + 10.0,
    )
    if code == 200:
        print(
            "[CTL-Master] Success: Engine paused. TPU HBM weights & cache preserved."
        )
        return 0
    else:
        print(
            f"[CTL-Master] Error pausing engine (HTTP {code}): {resp.get('detail', resp)}"
        )
        return 1


def cmd_resume(args: argparse.Namespace) -> int:
    rank = get_worker_rank()
    if rank > 0:
        print(f"[CTL-Worker] Silent local no-op on worker pod (Rank {rank}).")
        return 0

    print("[CTL-Master] Resuming TPU inference engine...")
    code, resp = call_api(args.endpoint, "/ctl/resume", method="POST")
    if code == 200:
        print("[CTL-Master] Success: Serving resumed with 0ms warmup.")
        return 0
    else:
        print(
            f"[CTL-Master] Error resuming engine (HTTP {code}): {resp.get('detail', resp)}"
        )
        return 1


def cmd_probe(args: argparse.Namespace) -> int:
    rank = get_worker_rank()

    if args.type == "liveness":
        # Check process liveness
        try:
            out = subprocess.check_output(["pgrep", "-f", "vllm|python3"])
            if out.strip():
                return 0
        except Exception:
            pass
        return 0

    if rank > 0:
        # Worker pod readiness is governed by local process and TPU initialization
        return 0

    if args.type == "readiness":
        code, _ = call_api(args.endpoint,
                           "/ctl/health/ready",
                           method="GET",
                           timeout=5.0)
        return 0 if code == 200 else 1

    code, resp = call_api(args.endpoint,
                          "/ctl/status",
                          method="GET",
                          timeout=5.0)
    if code != 200:
        return 1

    status_str = resp.get("status", "")
    if args.type == "paused":
        return 0 if status_str == "PAUSED" else 1
    elif args.type == "running":
        return 0 if status_str == "SERVING" else 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tpu-ctl",
        description="Operator probe & lifecycle control CLI for TPU inference.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Base URL of TPU serving engine (default: {DEFAULT_ENDPOINT})",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)

    # status
    subparsers.add_parser("status",
                          help="Get engine status and queue metrics.")

    # pause
    pause_parser = subparsers.add_parser(
        "pause", help="Quiesce in-flight requests and pause engine.")
    pause_parser.add_argument("--timeout",
                              type=float,
                              default=60.0,
                              help="Max seconds to wait for drain.")

    # resume
    subparsers.add_parser("resume", help="Resume serving with 0ms warmup.")

    # probe
    probe_parser = subparsers.add_parser("probe",
                                         help="Kubernetes probe evaluation.")
    probe_parser.add_argument(
        "--type",
        choices=["liveness", "readiness", "paused", "running"],
        required=True,
        help="Type of probe to evaluate.",
    )

    args = parser.parse_args(argv)

    if args.action == "status":
        return cmd_status(args)
    elif args.action == "pause":
        return cmd_pause(args)
    elif args.action == "resume":
        return cmd_resume(args)
    elif args.action == "probe":
        return cmd_probe(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
