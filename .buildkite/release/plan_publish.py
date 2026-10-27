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
"""Print the approval and publish steps of a release build as a Buildkite
pipeline (JSON).

    plan_publish.py | buildkite-agent pipeline upload --no-interpolation

Runs after the image is staged. The steps are generated rather than static so
the approval prompt names the version and the staged image digest, and so
allowed_teams is set only when RELEASE_APPROVER_TEAMS is. --no-interpolation:
the output holds nothing for Buildkite to interpolate.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
  RELEASE_QUEUE=<queue>        agent queue (default: release-tpu-inference)
"""

import json
import os
import subprocess


def meta(key):
    return subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


QUEUE = {"queue": os.environ.get("RELEASE_QUEUE", "release-tpu-inference")}


def command_step(label, key, depends_on, commands, **extra):
    step = {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "agents": QUEUE,
        "commands": commands,
    }
    step.update(extra)
    return step


def main():
    version = meta("release-version")
    staged = meta("release-image")
    prerelease = meta("release-prerelease") == "yes"
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
         "Check that tpu-inference-ci passed on this commit first."),
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
            ["bash .buildkite/release/publish_images.sh"],
            secrets=["DOCKERHUB_TOKEN"],
            concurrency=1,
            concurrency_group="tpu-inference-dockerhub-publish",
        ))
    steps.append(
        command_step(
            ":white_check_mark: Check the published image",
            "verify_published",
            ["publish_images"],
            ["bash .buildkite/release/verify_published.sh"],
        ))

    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
