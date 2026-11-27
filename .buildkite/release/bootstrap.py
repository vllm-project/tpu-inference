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

The release pipeline runs this as its only configured step. It checks what the
build was asked to publish and uploads just that mode's steps, so everything a
release runs lives under .buildkite/release/, which has its own code owners.

RELEASE_MODE picks what to publish (default: nightly):
  nightly  vllm/vllm-tpu built from this commit and the vLLM LKG, pushed as
           :nightly and a dated tag (nightly_image.yml). Builds of main only.
"""

import argparse
import os
import pathlib
import subprocess
import sys
from typing import NamedTuple

RELEASE_DIR = pathlib.Path(__file__).resolve().parent

# Each mode's steps, in this directory.
MODES = {
    "nightly": "nightly_image.yml",
}

# The queues this pipeline may run on: cpu in the TPU cluster, and
# release-tpu-inference in TPU Release, whose agents run nothing but release
# pipelines. The publish step runs on the queue this bootstrap runs on, so the
# pipeline needs no setting to say which cluster it is in.
RELEASE_QUEUES = ("cpu", "release-tpu-inference")


class ReleaseError(Exception):
    pass


class Upload(NamedTuple):
    pipeline: pathlib.Path
    # Substituted into the pipeline file by `buildkite-agent pipeline upload`.
    env: dict


def plan(mode: str, branch: str, queue: str) -> Upload:
    """Returns the pipeline file to upload and the values it needs."""
    if mode not in MODES:
        raise ReleaseError(f"unknown RELEASE_MODE {mode!r}; expected one of "
                           f"{', '.join(MODES)}")
    if queue not in RELEASE_QUEUES:
        raise ReleaseError(f"queue {queue!r} doesn't run releases; expected "
                           f"one of {', '.join(RELEASE_QUEUES)}")
    if branch != "main":
        raise ReleaseError(
            f"the nightly image is published from main, not {branch!r}")

    return Upload(pipeline=RELEASE_DIR / MODES[mode],
                  env={"RELEASE_QUEUE": queue})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="print what would be uploaded instead")
    args = parser.parse_args()

    try:
        upload = plan(
            mode=os.environ.get("RELEASE_MODE") or "nightly",
            branch=os.environ["BUILDKITE_BRANCH"],
            queue=os.environ["BUILDKITE_AGENT_META_DATA_QUEUE"],
        )
    except ReleaseError as e:
        print(f"bootstrap: {e}", file=sys.stderr)
        return 1

    print(f"bootstrap: uploading {upload.pipeline.name} with "
          f"{' '.join(f'{k}={v}' for k, v in upload.env.items())}")
    if args.dry_run:
        return 0
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload",
         str(upload.pipeline)],
        env={
            **os.environ,
            **upload.env
        },
        check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
