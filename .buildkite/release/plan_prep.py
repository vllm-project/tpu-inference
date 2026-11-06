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
"""Print the next steps of a release prep build as a Buildkite pipeline (JSON).

    plan_prep.py | buildkite-agent pipeline upload --no-interpolation
    plan_prep.py --release | buildkite-agent pipeline upload --no-interpolation

After update_support_matrices.sh: if the matrices changed, an approval block
and the step that pushes the reviewed patch; otherwise the release steps for
the tested commit. With --release, after push_support_matrices.sh: the release
steps for the pushed commit. The release steps are an approval block and a
trigger of the release pipeline on that exact commit; nothing is pushed or
started without an approval.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
  RELEASE_PIPELINE=<slug>      the release pipeline (default: vllm-tpu-release)
  RELEASE_VERSION=vX.Y.Zrc1    passed to the release build
  RELEASE_PREP_QUEUE=<queue>   agent queue (default: cpu)
"""

import json
import os
import subprocess
import sys


def meta(key):
    return subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def with_teams(block):
    teams = [
        t for t in os.environ.get("RELEASE_APPROVER_TEAMS", "").split(",") if t
    ]
    if teams:
        block["allowed_teams"] = teams
    return block


def push_steps(branch, tested):
    return [
        with_teams({
            "block":
            f":memo: Push the support matrix update to {branch}?",
            "key":
            "approve_push",
            "prompt":
            (f"Review the release-prep/support_matrices.patch artifact of the "
             f"update step: support_matrices/release/ and the release tables "
             f"of README.md and docs/includes, from the NIGHTLY=1 "
             f"tpu-inference-ci runs of {tested[:12]}. Approving commits "
             f"exactly that patch to {branch} as the CI bot and pushes it."),
        }),
        {
            "label":
            ":github: Push the support matrix update",
            "key":
            "push_matrices",
            "depends_on":
            "approve_push",
            "agents": {
                "queue": os.environ.get("RELEASE_PREP_QUEUE", "cpu")
            },
            "secrets": ["GITHUB_CI_BOT_APP_ID", "GITHUB_CI_BOT_PEM"],
            "concurrency":
            1,
            "concurrency_group":
            f"release-prep-{branch.replace('/', '-')}",
            "commands": [
                "bash .buildkite/release/push_support_matrices.sh",
                "python3 .buildkite/release/plan_prep.py --release"
                " | buildkite-agent pipeline upload --no-interpolation",
            ],
        },
    ]


def release_steps(branch, commit):
    # TODO: confirm vllm-tpu-release is the release pipeline's Buildkite slug.
    pipeline = os.environ.get("RELEASE_PIPELINE", "vllm-tpu-release")
    version = os.environ.get("RELEASE_VERSION", "")
    build = {
        "branch": branch,
        "commit": commit,
        "message": f"Release {version or branch} at {commit[:12]}",
    }
    if version:
        build["env"] = {"RELEASE_VERSION": version}
    return [
        with_teams({
            "block":
            f":rocket: Start {pipeline} on {branch} at {commit[:12]}?",
            "key":
            "approve_release",
            "prompt":
            (f"Starts {pipeline} on {branch} at {commit}"
             f"{f' as {version}' if version else ''}. That build makes the "
             "vllm/vllm-tpu Docker image and the tpu-inference and vllm-tpu "
             "wheels from this commit, then waits for its own approval "
             "before publishing them to Docker Hub and PyPI."),
        }),
        {
            "label": f":rocket: Start {pipeline}",
            "key": "start_release",
            "depends_on": "approve_release",
            "trigger": pipeline,
            "async": True,
            "build": build,
        },
    ]


def main():
    branch = os.environ["BUILDKITE_BRANCH"]
    if sys.argv[1:] == ["--release"]:
        steps = release_steps(branch, meta("release-prep-release-commit"))
    elif meta("release-prep-changed") == "yes":
        steps = push_steps(branch, meta("release-prep-commit"))
    else:
        steps = release_steps(branch, meta("release-prep-commit"))
    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
