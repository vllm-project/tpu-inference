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
"""Applies pass/fail thresholds to the metrics.json of eval_running_server.sh.

Each flag gates one metric. A gated metric that is missing -- its leg failed or
never ran -- fails the check instead of being skipped.

Usage:
  python3 check_eval_thresholds.py metrics.json --min-gsm8k 0.96
  python3 check_eval_thresholds.py metrics.json --min-output-throughput 1045 \
      --min-total-throughput 9395 --max-failed-requests 0
"""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", help="metrics.json to check")
    parser.add_argument("--min-gsm8k",
                        type=float,
                        help="minimum GSM8K exact match, applied to both "
                        "flexible-extract and strict-match")
    parser.add_argument("--min-output-throughput",
                        type=float,
                        help="minimum output token throughput (tok/s)")
    parser.add_argument("--min-total-throughput",
                        type=float,
                        help="minimum total token throughput (tok/s)")
    parser.add_argument("--max-failed-requests",
                        type=int,
                        help="maximum number of failed benchmark requests")
    args = parser.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)
    gsm8k = metrics.get("gsm8k", {})
    bench = metrics.get("bench", {})

    # (leg, name, value, op, limit)
    checks = []
    if args.min_gsm8k is not None:
        for key in ("flexible_extract", "strict_match"):
            checks.append(
                (gsm8k, f"gsm8k {key}", gsm8k.get(key), ">=", args.min_gsm8k))
    if args.min_output_throughput is not None:
        checks.append((bench, "output tok/s", bench.get("output_throughput"),
                       ">=", args.min_output_throughput))
    if args.min_total_throughput is not None:
        checks.append(
            (bench, "total tok/s", bench.get("total_token_throughput"), ">=",
             args.min_total_throughput))
    if args.max_failed_requests is not None:
        checks.append((bench, "failed requests", bench.get("failed"), "<=",
                       args.max_failed_requests))
    if not checks:
        parser.error("no thresholds given")

    all_passed = True
    for leg, name, value, op, limit in checks:
        if value is None:
            all_passed = False
            print(
                f"MISSING  {name} (leg status: {leg.get('status', 'not run')})"
            )
            continue
        passed = value >= limit if op == ">=" else value <= limit
        all_passed &= passed
        print(f"{'PASS' if passed else 'FAIL':8} {name}: {value} "
              f"(required {op} {limit})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
