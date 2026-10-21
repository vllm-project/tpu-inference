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
"""Fail unless tpu-inference-ci passed a full nightly-style run of a commit.

    ci_gate.py <commit sha>

Reads the commit's GitHub statuses, which Buildkite posts for every
tpu-inference-ci build of it, so no Buildkite token is needed on the release
agent. Passing means the newest buildkite/tpu-inference-ci status is success
and at least one nightly-only step passed on that commit: a PR-lane build
alone doesn't run the nightly suites, so it can't vouch for a release.
"""

import json
import sys
import urllib.error
import urllib.request

REPO = "vllm-project/tpu-inference"
CONTEXT = "buildkite/tpu-inference-ci"
# Nightly-only step status substrings posted by bare-metal and kube CI fleets.
NIGHTLY_MARKERS = (
    "nightly",
    "check-metadata-health",
    "upload-tests-for-models",
    "pypi-test-performance-benchmarks",
)


def statuses(sha):
    """Every status of the commit, newest first, across all pages."""
    out = []
    for page in range(1, 11):
        url = (f"https://api.github.com/repos/{REPO}/commits/{sha}"
               f"/statuses?per_page=100&page={page}")
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        out += batch
        if len(batch) < 100:
            break
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: ci_gate.py <commit sha>")
    sha = sys.argv[1]
    try:
        all_statuses = statuses(sha)
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: GitHub answered {e.code} for the statuses of {sha}.")
    latest = {}
    for s in all_statuses:
        latest.setdefault(s["context"], s)  # newest first, so the first wins
    build = latest.get(CONTEXT)
    if build is None:
        sys.exit(
            f"ERROR: no {CONTEXT} build of {sha}. Run a full build first: "
            f"tpu-inference-ci on this commit with NIGHTLY=1.")
    nightly = [
        c for c, s in latest.items() if c.startswith(CONTEXT + "/") and any(
            m in c for m in NIGHTLY_MARKERS) and s["state"] == "success"
    ]
    print(f"{CONTEXT}: {build['state']} ({build['target_url']})")
    print(f"nightly-only steps passed: {len(nightly)}")
    if build["state"] != "success":
        sys.exit(
            f"ERROR: the newest {CONTEXT} build of {sha} is {build['state']}, "
            "not success.")
    if not nightly:
        sys.exit(
            f"ERROR: {CONTEXT} passed on {sha}, but no nightly step ran. "
            "Run it again with NIGHTLY=1 so the release is tested by the full suite."
        )
    print(build["target_url"])


if __name__ == "__main__":
    main()
