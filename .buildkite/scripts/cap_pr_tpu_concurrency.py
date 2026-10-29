#!/usr/bin/env python3
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
"""Put a pull-request build's TPU steps in its author's concurrency group.

Reads a pipeline as JSON (`buildkite-agent pipeline upload --dry-run --format
json`) on stdin and writes it back with every TPU step on the kube queue in
the given group, so all of one author's PR builds share one limit on TPU jobs
in flight:

    cap_pr_tpu_concurrency.py <group> <limit> < pipeline.json

A TPU step is one on queue `kube` whose SHAPE is not `cpu`; the cpu steps hold
no chips and are left out. `eager` lets any of the author's jobs take a free
slot rather than wait for the oldest. A step that already names a concurrency
group keeps its own.
"""

import json
import sys


def on_kube(step):
    agents = step.get("agents")
    if isinstance(agents, dict):
        return agents.get("queue") == "kube"
    if isinstance(agents, list):
        return "queue=kube" in agents
    return False


def holds_chips(step):
    return on_kube(step) and (step.get("env") or {}).get("SHAPE") != "cpu"


def cap(steps, group, limit):
    for step in steps:
        if not isinstance(step, dict):
            continue
        if isinstance(step.get("steps"), list):
            cap(step["steps"], group, limit)
        elif holds_chips(step) and "concurrency_group" not in step:
            step["concurrency_group"] = group
            step["concurrency"] = limit
            step["concurrency_method"] = "eager"


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <group> <limit> < pipeline.json")
    group, limit = sys.argv[1], int(sys.argv[2])
    pipeline = json.load(sys.stdin)
    cap(pipeline.get("steps") or [], group, limit)
    json.dump(pipeline, sys.stdout)


if __name__ == "__main__":
    main()
