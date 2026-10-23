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

import datetime

import pytest
from publish_image import nightly_tag, publish_nightly, vllm_commit

COMMIT = "c46f1556d0123456789abcdef0123456789abcd"
VLLM = "5ab4e4445bcbe2bf07351b82d1af160234c32273"
TODAY = datetime.date(2026, 10, 13)


def test_nightly_tag():
    assert nightly_tag(TODAY, COMMIT,
                       VLLM) == "nightly-20261013-c46f155-5ab4e44"


def test_publish_nightly_builds_then_pushes_dated_before_moving():
    commands = []
    publish_nightly(COMMIT, VLLM, TODAY, runner=commands.append)
    dated = "vllm/vllm-tpu:nightly-20261013-c46f155-5ab4e44"
    assert commands == [
        ["docker", "system", "prune", "-a", "-f"],
        [
            "docker", "build", "--build-arg", f"VLLM_COMMIT_HASH={VLLM}",
            "--no-cache", "-f", "docker/Dockerfile", "-t", dated, "-t",
            "vllm/vllm-tpu:nightly", "."
        ],
        ["docker", "push", dated],
        ["docker", "push", "vllm/vllm-tpu:nightly"],
    ]


def test_vllm_commit_reads_the_lkg(tmp_path):
    lkg = tmp_path / "vllm_lkg.version"
    lkg.write_text(VLLM + "\n")
    assert vllm_commit(lkg) == VLLM


@pytest.mark.parametrize("content", [None, "", "\n"])
def test_vllm_commit_refuses_a_missing_lkg(tmp_path, content):
    lkg = tmp_path / "vllm_lkg.version"
    if content is not None:
        lkg.write_text(content)
    with pytest.raises(SystemExit, match="missing or empty"):
        vllm_commit(lkg)
