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

Runs once everything is built: the approval block, the publishes, and the
check that what users get is what was built. Uploaded rather than written into
pipeline.yml so the approval prompt can name the exact version, and so the team
restriction is added only when it is set. --no-interpolation: nothing in it is
for Buildkite to interpolate.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
  RELEASE_QUEUE=<queue>        agent queue (default: release-tpu-inference, as
                               in pipeline.yml)
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


def placeholder_step(label, key, depends_on, what):
    return {
        "label": f"{label} (placeholder)",
        "key": key,
        "depends_on": depends_on,
        "agents": QUEUE,
        "command": f'echo "{what}: not handled by this pipeline yet."',
    }


def main():
    version = meta("release-version")
    branch = os.environ["BUILDKITE_BRANCH"]
    steps = []

    block = {
        "block":
        f":rocket: Publish tpu-inference {version}?",
        "key":
        "approve_publish",
        "depends_on": ["plan_publish"],
        "prompt": (f"Publish tpu-inference {version} from {branch}: "
                   f"tpu-inference and vllm-tpu {version} to PyPI. "
                   "PyPI releases can't be replaced. "
                   "Check that tpu-inference-ci passed on this commit first."),
    }
    teams = [
        t for t in os.environ.get("RELEASE_APPROVER_TEAMS", "").split(",") if t
    ]
    if teams:
        block["allowed_teams"] = teams
    steps.append(block)

    # Placeholder: publishing the Docker image comes in a follow-up.
    steps.append(
        placeholder_step(":docker: Publish the image to Docker Hub",
                         "publish_images", ["approve_publish"],
                         "Docker image"))
    steps.append(
        command_step(
            ":package: Publish tpu-inference to PyPI",
            "publish_core",
            ["approve_publish"],
            ["bash .buildkite/scripts/publish_wheel.sh tpu-inference"],
            env={"PUBLISH_PYPI": "tpu-inference"},
            concurrency=1,
            concurrency_group="tpu-inference-publish",
        ))
    # The combined wheel pins tpu-inference==<version>, so it goes up only
    # once the core is installable.
    steps.append(
        command_step(
            ":package: Publish vllm-tpu to PyPI",
            "publish_vllm_tpu",
            ["publish_core"],
            ["bash .buildkite/scripts/publish_wheel.sh vllm-tpu"],
            env={"PUBLISH_PYPI": "vllm-tpu"},
            concurrency=1,
            concurrency_group="vllm-tpu-publish",
        ))
    # Placeholder: checking what users get comes in a follow-up. Each
    # publish_wheel.sh run already waits until its wheel installs from PyPI.
    steps.append(
        placeholder_step(":white_check_mark: Check what users get",
                         "verify_published",
                         ["publish_images", "publish_vllm_tpu"],
                         "Checking the published image and wheels"))

    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
