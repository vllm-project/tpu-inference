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
"""Uploads the steps of one tpu-inference release build.

The release pipeline runs this as its only configured step. It decides what
the build publishes and uploads just those steps, so everything a release runs
lives under .buildkite/release/, which has its own code owners.

RELEASE_MODE picks what to publish (default: nightly):
  nightly  vllm/vllm-tpu built from this commit and the vLLM LKG, pushed as
           :nightly and a dated tag. Builds of main only.
"""

import argparse
import json
import os
import subprocess
import sys

DOCKERHUB_USER = "tpuinference"

# The Docker Hub token is a cluster secret, and the two clusters that can run
# this pipeline name it differently: the TPU cluster's cpu queue reads
# DOCKERHUB_TOKEN, and TPU Release, whose agents run nothing but release
# pipelines, reads DOCKERHUB_TOKEN_TPU_INFERENCE. Keyed by the queue this
# bootstrap runs on, which is also where the publish step runs.
DOCKERHUB_SECRETS = {
    "cpu": "DOCKERHUB_TOKEN",
    "release-tpu-inference": "DOCKERHUB_TOKEN_TPU_INFERENCE",
}

MODES = ("nightly", )


class ReleaseError(Exception):
    pass


def build_steps(mode: str, branch: str, queue: str) -> dict:
    """Returns the pipeline to upload for one release build."""
    if mode not in MODES:
        raise ReleaseError(f"unknown RELEASE_MODE {mode!r}; expected one of "
                           f"{', '.join(MODES)}")
    if queue not in DOCKERHUB_SECRETS:
        raise ReleaseError(
            f"queue {queue!r} has no Docker Hub secret; expected one of "
            f"{', '.join(DOCKERHUB_SECRETS)}")
    if branch != "main":
        raise ReleaseError(
            f"the nightly image is published from main, not {branch!r}")

    secret = DOCKERHUB_SECRETS[queue]
    return {
        "steps": [{
            "label":
            ":docker: Publish vllm/vllm-tpu nightly image",
            "key":
            "publish_nightly_image",
            "agents": {
                "queue": queue
            },
            "secrets": [secret],
            "plugins": [{
                "docker-login#v3.0.0": {
                    "username": DOCKERHUB_USER,
                    "password-env": secret,
                }
            }],
            "command":
            "python3 .buildkite/release/publish_image.py nightly",
        }]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="print the pipeline instead of uploading it")
    args = parser.parse_args()

    try:
        pipeline = build_steps(
            mode=os.environ.get("RELEASE_MODE") or "nightly",
            branch=os.environ["BUILDKITE_BRANCH"],
            queue=os.environ["BUILDKITE_AGENT_META_DATA_QUEUE"],
        )
    except ReleaseError as e:
        print(f"bootstrap: {e}", file=sys.stderr)
        return 1

    text = json.dumps(pipeline, indent=2)
    if args.dry_run:
        print(text)
        return 0
    subprocess.run(["buildkite-agent", "pipeline", "upload"],
                   input=text,
                   text=True,
                   check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
