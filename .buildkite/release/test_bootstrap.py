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

import re

import pytest
from bootstrap import MODES, RELEASE_DIR, ReleaseError, plan


@pytest.mark.parametrize("queue", ["cpu", "release-tpu-inference"])
def test_nightly_publishes_on_the_queue_bootstrap_runs_on(queue):
    upload = plan("nightly", "main", queue)
    assert upload.pipeline == RELEASE_DIR / "nightly_image.yml"
    assert upload.env == {"RELEASE_QUEUE": queue}


@pytest.mark.parametrize("mode", MODES)
def test_pipeline_files_use_exactly_the_values_bootstrap_sets(mode):
    upload = plan(mode, "main", "cpu")
    text = upload.pipeline.read_text()
    assert set(re.findall(r"\$\{(\w+)\?\}", text)) == set(upload.env)
    # Anything else would be substituted from the agent's environment.
    assert re.findall(r"\$\{?\w+", text) == re.findall(r"\$\{\w+", text)


@pytest.mark.parametrize("mode", MODES)
def test_secrets_are_literal_names(mode):
    # The agent leaves variables in `secrets` unsubstituted.
    yaml = pytest.importorskip("yaml")
    pipeline = yaml.safe_load(plan(mode, "main", "cpu").pipeline.read_text())
    for step in pipeline["steps"]:
        for name in step.get("secrets", []):
            assert re.fullmatch(r"[A-Z0-9_]+", name), name


@pytest.mark.parametrize("mode", MODES)
def test_pipeline_files_parse(mode):
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(plan(mode, "main", "cpu").pipeline.read_text())
    assert steps["steps"]


@pytest.mark.parametrize(
    "mode, branch, queue, message",
    [
        ("nightly", "my-branch", "cpu", "from main"),
        ("release", "main", "cpu", "unknown RELEASE_MODE"),
        ("nightly", "main", "cpu_64_core", "doesn't run releases"),
    ],
)
def test_refuses(mode, branch, queue, message):
    with pytest.raises(ReleaseError, match=message):
        plan(mode, branch, queue)
