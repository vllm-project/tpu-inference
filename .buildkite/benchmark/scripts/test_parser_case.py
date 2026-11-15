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

import sys
import unittest.mock as mock

from parser_case import (build_command, get_current_machine_type,
                         normalize_device_name)


def test_normalize_device_name():
    assert normalize_device_name("tpu7x-16") == "v7x-16"
    assert normalize_device_name("7x-16") == "v7x-16"
    assert normalize_device_name("v6e-8") == "v6e-8"
    assert normalize_device_name(None) is None
    assert normalize_device_name("") is None
    assert normalize_device_name("tpu6e-8") == "v6e-8"


@mock.patch.dict(sys.modules, {'tpu_info': mock.MagicMock()})
def test_get_current_machine_type_fallback():
    # Mock tpu_info to simulate v7x-16 locally but fallback says v7x-32
    tpu_info = sys.modules['tpu_info']
    mock_chip_type = mock.MagicMock()
    mock_chip_type.value.name = "7x"
    # Local chips = 16
    tpu_info.device.get_local_chips.return_value = (mock_chip_type, 16)

    assert get_current_machine_type("v7x-32") == "v7x-32"
    assert get_current_machine_type("v7x-16") == "v7x-16"
    assert get_current_machine_type(
        "v7x-8") == "v7x-16"  # local is larger, so local is used


@mock.patch.dict(sys.modules, {'tpu_info': mock.MagicMock()})
def test_get_current_machine_type_fallback_v6e():
    # Mock tpu_info to simulate v6e-4 locally but fallback says v6e-8
    tpu_info = sys.modules['tpu_info']
    mock_chip_type = mock.MagicMock()
    mock_chip_type.value.name = "v6e"
    mock_chip_type.value.devices_per_chip = 1
    # Local chips = 4
    tpu_info.device.get_local_chips.return_value = (mock_chip_type, 4)

    assert get_current_machine_type("v6e-8") == "v6e-8"


def test_get_current_machine_type_no_library():
    # When library is not present
    with mock.patch.dict('sys.modules', {'tpu_info': None}):
        assert get_current_machine_type("v6e-8") == "v6e-8"
        assert get_current_machine_type(None) is None


def test_build_command_model_mapping():
    # If key is "model-path" it should output "--model" instead.
    args_dict = {"model-path": "/my/model/path", "tensor-parallel-size": 4}
    cmd = build_command("vllm_serve", args_dict)
    assert "--model /my/model/path" in cmd
    assert "--model-path" not in cmd
    assert "--tensor-parallel-size 4" in cmd

    # If both "model" and "model-path" exist, "model" is skipped in favor of "model-path".
    args_dict_both = {
        "model": "meta-llama/Llama-2",
        "model-path": "/my/model/path"
    }
    cmd_both = build_command("vllm_serve", args_dict_both)
    assert "--model /my/model/path" in cmd_both
    assert "meta-llama/Llama-2" not in cmd_both


if __name__ == "__main__":
    print("Running tests without pytest...")
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            print(f"Executing {name}...")
            func()
    print("All tests passed successfully!")
