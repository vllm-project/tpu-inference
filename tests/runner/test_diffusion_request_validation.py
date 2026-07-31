# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest


def _load_validation_with_fake_vllm(monkeypatch):

    class InputProcessor:

        def __init__(self, additional_config):
            self.vllm_config = SimpleNamespace(
                additional_config=additional_config)

        def process_inputs(self, *args, **kwargs):
            return args, kwargs

    modules = {
        "vllm":
        types.ModuleType("vllm"),
        "vllm.v1":
        types.ModuleType("vllm.v1"),
        "vllm.v1.engine":
        types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.input_processor":
        types.ModuleType("vllm.v1.engine.input_processor"),
    }
    modules["vllm.v1.engine.input_processor"].InputProcessor = InputProcessor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "runner" / "diffusion" / "request_validation.py")
    spec = importlib.util.spec_from_file_location(
        "diffusion_request_validation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, InputProcessor


def test_resumable_request_is_rejected_only_for_block_diffusion(monkeypatch):
    validation, InputProcessor = _load_validation_with_fake_vllm(monkeypatch)
    validation.patch_vllm_input_processor_for_block_diffusion()

    autoregressive = InputProcessor({})
    assert autoregressive.process_inputs(
        resumable=True)[1]["resumable"] is True

    diffusion = InputProcessor({"generation_strategy": "block_diffusion"})
    assert diffusion.process_inputs(resumable=False)[1]["resumable"] is False
    with pytest.raises(ValueError, match="resumable or streaming-input"):
        diffusion.process_inputs(resumable=True)


def test_resumable_positional_argument_and_patch_idempotence(monkeypatch):
    validation, InputProcessor = _load_validation_with_fake_vllm(monkeypatch)
    validation.patch_vllm_input_processor_for_block_diffusion()
    validation.patch_vllm_input_processor_for_block_diffusion()
    diffusion = InputProcessor({"generation_strategy": "block_diffusion"})

    positional = tuple(range(10)) + (True, )
    with pytest.raises(ValueError, match="resumable or streaming-input"):
        diffusion.process_inputs(*positional)
