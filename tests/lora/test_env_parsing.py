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

from tpu_inference.lora.lora_manager import parse_lora_module_path_env


def test_parse_lora_module_path_env(monkeypatch):
    # Test case 1: Missing env var
    monkeypatch.delenv("LORA_MODULE_PATH", raising=False)
    assert parse_lora_module_path_env() is None

    # Test case 2: Standard format with mapping
    monkeypatch.setenv("LORA_MODULE_PATH",
                       "something/(query|key|value|out)/something")
    expected = sorted(["q_proj", "k_proj", "v_proj", "o_proj"])
    result = parse_lora_module_path_env()
    assert sorted(result) == expected

    # Test case 3: Fallback for unknown modules
    monkeypatch.setenv("LORA_MODULE_PATH", "(embed|lm_head|unknown)")
    expected = sorted(["embed_tokens", "lm_head", "unknown"])
    result = parse_lora_module_path_env()
    assert sorted(result) == expected

    # Test case 4: No matching groups, falls back to splitting
    monkeypatch.setenv("LORA_MODULE_PATH", "no_groups_here")
    assert parse_lora_module_path_env() == ["no_groups_here"]

    # Test case 5: Empty groups
    monkeypatch.setenv("LORA_MODULE_PATH", "()")
    assert parse_lora_module_path_env() is None

    # Test case 6: Comma separated list
    monkeypatch.setenv("LORA_MODULE_PATH", "q_proj,k_proj,v_proj")
    expected = sorted(["q_proj", "k_proj", "v_proj"])
    result = parse_lora_module_path_env()
    assert sorted(result) == expected

    # Test case 7: Pipe separated list without parentheses
    monkeypatch.setenv("LORA_MODULE_PATH", "query|key|value")
    expected = sorted(["q_proj", "k_proj", "v_proj"])
    result = parse_lora_module_path_env()
    assert sorted(result) == expected
