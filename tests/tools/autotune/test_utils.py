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
import os
import tempfile
from unittest.mock import MagicMock

from tpu_inference.tools.autotune.utils import (apply_tp_scaling,
                                                block_until_ready,
                                                get_registry_file_name,
                                                update_json_registry)


def test_get_registry_file_name():
    assert get_registry_file_name("TPU v5e") == "tpu_v5e"
    assert get_registry_file_name("TPU v7") == "tpu_v7"
    assert get_registry_file_name("TPU v5 Lite") == "tpu_v5_lite"
    assert get_registry_file_name("CPU") == "cpu"


def test_block_until_ready():
    # Test with single object
    mock_obj = MagicMock()
    block_until_ready(mock_obj)
    mock_obj.block_until_ready.assert_called_once()

    # Test with list
    mock_list = [MagicMock(), MagicMock()]
    block_until_ready(mock_list)
    for m in mock_list:
        m.block_until_ready.assert_called_once()

    # Test with object waiting for interface (no block_until_ready)
    # Should not crash
    block_until_ready("string")


def test_apply_tp_scaling():
    mock_console = MagicMock()

    # CASE 1: Divisible
    # 32 / 4 = 8
    assert apply_tp_scaling(32, 4, name="heads", printer=mock_console) == 8

    # CASE 2: Not Divisible
    # 33 / 4 != int -> Should keep 33
    assert apply_tp_scaling(33, 4, name="heads", printer=mock_console) == 33

    # CASE 3: Value is 1 (Replication)
    # 1 / 4 != int -> Should keep 1 (common for KV heads)
    assert apply_tp_scaling(1, 4, name="kv_heads", printer=mock_console) == 1

    # CASE 4: TP=1 (No change)
    assert apply_tp_scaling(100, 1, name="dims", printer=mock_console) == 100


def test_update_json_registry():
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json',
                                     delete=False) as tmp:
        # Initial data
        json.dump({"existing": 1, "nested": {"a": 1}}, tmp)
        tmp_path = tmp.name

    try:
        # Update with new data
        base_overrides = {"new": 2, "nested": {"b": 2}}
        update_json_registry(tmp_path, base_overrides)

        with open(tmp_path, 'r') as f:
            data = json.load(f)

        # Verify merge
        assert data["existing"] == 1
        assert data["new"] == 2
        assert data["nested"]["a"] == 1
        assert data["nested"]["b"] == 2

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_update_json_registry_new_file():

    # Actually let's use a directory
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "new_registry.json")

        new_data = {"created": True}
        update_json_registry(path, new_data)

        assert os.path.exists(path)
        with open(path, 'r') as f:
            data = json.load(f)
        assert data == new_data
