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
"""Print the second half of a release build as a Buildkite pipeline (JSON).

    plan_publish.py | buildkite-agent pipeline upload --no-interpolation

Runs once the image is staged: the approval block, the Docker Hub publish, and
the check that what users get is what was built. Uploaded rather than written
into pipeline.yml so the approval prompt can name the exact version and staged
digest, and so the team restriction is added only when it is set.
--no-interpolation keeps the $VAR references in the commands for the shell
that runs them.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
  RELEASE_QUEUE=<queue>        agent queue (default: cpu, as in pipeline.yml)
"""

import json
import os
import re
import subprocess


def meta(key):
    return subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


QUEUE = {"queue": os.environ.get("RELEASE_QUEUE", "cpu")}
# Every command step runs main's release tooling, pinned to the commit the
# pipeline was uploaded from (pipeline.yml has the same block).
TOOLS = "\n".join([
    'T="$(buildkite-agent meta-data get release-tools-commit)"',
    'git fetch -q origin "$T"',
    'RELEASE_TOOLS="$(mktemp -d)"',
    'git archive "$T" .buildkite/release | tar -x -C "$RELEASE_TOOLS"',
    'export RELEASE_TOOLS="$RELEASE_TOOLS/.buildkite"',
])


def command_step(label, key, depends_on, commands, **extra):
    step = {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "agents": QUEUE,
        "commands": [TOOLS] + commands,
    }
    step.update(extra)
    return step


def main():
    version = meta("release-version")
    staged = meta("release-image")
    prerelease = re.search(r"(a|b|rc|dev)[0-9]+",
                           version) is not None  # lib.sh is_prerelease
    branch = os.environ["BUILDKITE_BRANCH"]
    dockerhub_repo = os.environ.get("RELEASE_DOCKERHUB_REPO", "vllm/vllm-tpu")
    steps = []

    latest_options = [{"label": "No", "value": "no"}]
    if not prerelease:
        latest_options.insert(0, {"label": "Yes", "value": "yes"})
    block = {
        "block":
        f":rocket: Publish tpu-inference {version}?",
        "key":
        "approve_publish",
        "depends_on": ["plan_publish"],
        "prompt":
        (f"Publish tpu-inference {version} from {branch}: the staged image "
         f"{staged} to Docker Hub as {dockerhub_repo}:v{version}. "
         "Check that tpu-inference-ci passed on this commit first. "
         "PyPI is not handled by this pipeline yet."),
        "fields": [
            {
                # No default: the approver has to choose.
                "select":
                "Move latest on Docker Hub to this version?",
                "key":
                "dockerhub_move_latest",
                "required":
                True,
                "options":
                latest_options,
                "hint":
                "No for a patch to an older line. "
                "A pre-release never moves latest.",
            },
            {
                "select":
                f"Overwrite v{version} if it is already on Docker Hub?",
                "key":
                "dockerhub_overwrite",
                "required":
                True,
                "default":
                "no",
                "options": [
                    {
                        "label": "No",
                        "value": "no"
                    },
                    {
                        "label": "Yes",
                        "value": "yes"
                    },
                ],
                "hint":
                "Released tags should not change; "
                "Yes only to replace a broken image.",
            },
        ],
    }
    teams = [
        t for t in os.environ.get("RELEASE_APPROVER_TEAMS", "").split(",") if t
    ]
    if teams:
        block["allowed_teams"] = teams
    steps.append(block)

    steps.append(
        command_step(
            ":docker: Publish the image to Docker Hub",
            "publish_images",
            ["approve_publish"],
            ['bash "$RELEASE_TOOLS/release/publish_images.sh"'],
            secrets=["DOCKERHUB_TOKEN"],
            concurrency=1,
            concurrency_group="tpu-inference-dockerhub-publish",
        ))
    # Placeholder: PyPI is published by .github/workflows/release.yml today.
    steps.append({
        "label": ":package: Publish to PyPI (placeholder)",
        "key": "publish_pypi",
        "depends_on": ["approve_publish"],
        "agents": QUEUE,
        "command": 'echo "PyPI: not handled by this pipeline yet."',
    })
    steps.append(
        command_step(
            ":white_check_mark: Check what users get",
            "verify_published",
            ["publish_images", "publish_pypi"],
            ['bash "$RELEASE_TOOLS/release/verify_published.sh"'],
        ))

    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
