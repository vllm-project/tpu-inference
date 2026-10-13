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
"""Builds vllm/vllm-tpu from the checked-out commit and pushes it.

nightly: built against the vLLM LKG in .buildkite/vllm_lkg.version and pushed
as nightly-<date>-<tpu-inference sha>-<vllm sha>, then as :nightly. The step
that runs this has already logged in to Docker Hub.
"""

import argparse
import datetime
import pathlib
import subprocess
import sys
from typing import Callable, Sequence

IMAGE = "vllm/vllm-tpu"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LKG_FILE = REPO_ROOT / ".buildkite" / "vllm_lkg.version"

Runner = Callable[[Sequence[str]], None]


def run(cmd: Sequence[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def vllm_commit(lkg_file: pathlib.Path = LKG_FILE) -> str:
    """The vLLM commit the image is built against.

    No fallback to vLLM's HEAD: a published image has to name the vLLM it was
    tested with, and the LKG is that commit.
    """
    commit = lkg_file.read_text().strip() if lkg_file.exists() else ""
    if not commit:
        raise SystemExit(f"publish_image: {lkg_file} is missing or empty")
    return commit


def nightly_tag(today: datetime.date, commit: str, vllm: str) -> str:
    return f"nightly-{today:%Y%m%d}-{commit[:7]}-{vllm[:7]}"


def publish_nightly(commit: str,
                    vllm: str,
                    today: datetime.date,
                    runner: Runner = run) -> None:
    dated = f"{IMAGE}:{nightly_tag(today, commit, vllm)}"
    moving = f"{IMAGE}:nightly"

    print(f"--- Building {dated} (vLLM {vllm})", flush=True)
    runner(["docker", "system", "prune", "-a", "-f"])
    runner([
        "docker", "build", "--build-arg", f"VLLM_COMMIT_HASH={vllm}",
        "--no-cache", "-f", "docker/Dockerfile", "-t", dated, "-t", moving, "."
    ])

    # The dated tag first, so :nightly only moves once its image is published
    # under a name that won't change.
    print("--- Pushing", flush=True)
    runner(["docker", "push", dated])
    runner(["docker", "push", moving])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["nightly"])
    parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=REPO_ROOT,
                            check=True,
                            capture_output=True,
                            text=True).stdout.strip()
    publish_nightly(commit=commit,
                    vllm=vllm_commit(),
                    today=datetime.datetime.now(datetime.timezone.utc).date())
    return 0


if __name__ == "__main__":
    sys.exit(main())
