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
"""`build_model_args` in the accuracy harness.

A model whose engine configuration does not fit `key=value` -- anything with
a nested value, such as a sharding strategy -- has to reach lm_eval as a JSON
object instead, and the two forms are not interchangeable downstream: one
goes to `create_from_arg_string`, the other to `create_from_arg_obj`.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "vllm" / "integration"
# test_accuracy.py imports its sibling by bare name.
sys.path.insert(0, str(_SCRIPT_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "accuracy_harness_under_test", _SCRIPT_DIR / "test_accuracy.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_SHARDED = {
    "pretrained": "some-org/Some-Model",
    "tensor_parallel_size": 8,
    "trust_remote_code": True,
    "additional_config": {
        "sharding": {
            "sharding_strategy": {
                "tensor_parallelism": 1,
                "expert_parallelism": 8,
                "enable_dp_attention": True,
            }
        }
    },
}


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    """Every test states its own override, or the absence of one."""
    monkeypatch.delenv("ACCURACY_MODEL_ARGS_JSON", raising=False)


def test_default_is_the_key_value_string():
    assert _MODULE.build_model_args("some-org/Some-Model") == (
        "pretrained=some-org/Some-Model,max_model_len=4096")


def test_more_args_is_appended_to_the_string_form():
    assert _MODULE.build_model_args(
        "some-org/Some-Model",
        "kv_cache_dtype=fp8") == ("pretrained=some-org/Some-Model,"
                                  "max_model_len=4096,"
                                  "kv_cache_dtype=fp8")


def test_override_returns_the_parsed_object(monkeypatch):
    """The nested value survives, which is the whole point of the hook."""
    monkeypatch.setenv("ACCURACY_MODEL_ARGS_JSON", json.dumps(_SHARDED))

    model_args = _MODULE.build_model_args("some-org/Some-Model")

    assert model_args == _SHARDED
    assert model_args["additional_config"]["sharding"]["sharding_strategy"][
        "expert_parallelism"] == 8


def test_override_ignores_the_per_model_defaults(monkeypatch):
    """A model with a defaults entry does not get it merged in underneath."""
    monkeypatch.setenv("ACCURACY_MODEL_ARGS_JSON",
                       '{"pretrained":"Qwen/Qwen3-30B-A3B"}')

    assert _MODULE.build_model_args("Qwen/Qwen3-30B-A3B") == {
        "pretrained": "Qwen/Qwen3-30B-A3B"
    }


def test_override_does_not_get_more_args_concatenated(monkeypatch):
    """Appending comma syntax to a dict would corrupt both forms."""
    monkeypatch.setenv("ACCURACY_MODEL_ARGS_JSON", json.dumps(_SHARDED))

    assert _MODULE.build_model_args("some-org/Some-Model",
                                    "kv_cache_dtype=fp8") == _SHARDED


def test_malformed_json_fails_rather_than_falling_back(monkeypatch):
    """Quietly serving the defaults instead would look like a passing run."""
    monkeypatch.setenv("ACCURACY_MODEL_ARGS_JSON",
                       '{"pretrained": "some-org/Some-Model"')

    with pytest.raises(json.JSONDecodeError):
        _MODULE.build_model_args("some-org/Some-Model")
