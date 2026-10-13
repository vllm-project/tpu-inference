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

import json

import pytest
from bootstrap import ReleaseError, build_steps


def test_nightly_on_the_tpu_cluster_reads_dockerhub_token():
    (step, ) = build_steps("nightly", "main", "cpu")["steps"]
    assert step["agents"] == {"queue": "cpu"}
    assert step["secrets"] == ["DOCKERHUB_TOKEN"]
    login = step["plugins"][0]["docker-login#v3.0.0"]
    assert login == {
        "username": "tpuinference",
        "password-env": "DOCKERHUB_TOKEN"
    }
    assert step["command"] == (
        "python3 .buildkite/release/publish_image.py nightly")


def test_nightly_on_tpu_release_reads_its_own_secret():
    (step, ) = build_steps("nightly", "main", "release-tpu-inference")["steps"]
    assert step["agents"] == {"queue": "release-tpu-inference"}
    assert step["secrets"] == ["DOCKERHUB_TOKEN_TPU_INFERENCE"]
    login = step["plugins"][0]["docker-login#v3.0.0"]
    assert login["password-env"] == "DOCKERHUB_TOKEN_TPU_INFERENCE"


def test_pipeline_is_json():
    pipeline = build_steps("nightly", "main", "cpu")
    assert json.loads(json.dumps(pipeline)) == pipeline


@pytest.mark.parametrize(
    "mode, branch, queue, message",
    [
        ("nightly", "my-branch", "cpu", "from main"),
        ("release", "main", "cpu", "unknown RELEASE_MODE"),
        ("nightly", "main", "cpu_64_core", "no Docker Hub secret"),
    ],
)
def test_refuses(mode, branch, queue, message):
    with pytest.raises(ReleaseError, match=message):
        build_steps(mode, branch, queue)
