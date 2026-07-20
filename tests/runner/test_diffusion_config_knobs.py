# Copyright 2025 Google LLC
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
"""Checks the block-diffusion config knobs are parsed from additional_config.

``TPUModelRunner`` cannot be instantiated on CPU (it needs a TPU mesh and the
full vLLM config), so this test parses ``tpu_runner.py`` with ``ast`` and
asserts each knob is read from ``additional_config.get(<key>, <default>)`` with
the correct key and default. This runs on CPU with no heavy imports.
"""

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TPU_RUNNER = _REPO_ROOT / "tpu_inference" / "runner" / "tpu_runner.py"

# attr name -> (config key, expected default)
_EXPECTED = {
    "enable_diffusion_decode": ("enable_diffusion_decode", False),
    "diffusion_block_size": ("diffusion_block_size", 32),
    "diffusion_commit_threshold": ("diffusion_commit_threshold", 0.9),
    "diffusion_max_denoise_steps": ("diffusion_max_denoise_steps", 0),
}


def _collect_config_assignments() -> dict:
    """Map ``self.<attr>`` -> (key, default) for additional_config.get calls."""
    module = ast.parse(_TPU_RUNNER.read_text())
    found = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        # Target must be exactly `self.<attr>`.
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(
                target.value, ast.Name) and target.value.id == "self"):
            continue
        # Value must be a call to `...additional_config.get(<key>, <default>)`.
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(
                call.func, ast.Attribute) and call.func.attr == "get"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "additional_config"):
            continue
        if len(call.args) != 2:
            continue
        key_node, default_node = call.args
        if not (isinstance(key_node, ast.Constant)
                and isinstance(default_node, ast.Constant)):
            continue
        found[target.attr] = (key_node.value, default_node.value)
    return found


class TestDiffusionConfigKnobs:

    def test_all_knobs_parsed_with_correct_keys_and_defaults(self):
        found = _collect_config_assignments()
        for attr, (key, default) in _EXPECTED.items():
            assert attr in found, f"self.{attr} not assigned from additional_config"
            got_key, got_default = found[attr]
            assert got_key == key, (f"self.{attr} reads key {got_key!r}, "
                                    f"expected {key!r}")
            # Compare by type and value so False vs 0 and 0 vs 0.9 don't alias.
            same_type = type(got_default) is type(default)
            assert same_type and got_default == default, (
                f"self.{attr} default is {got_default!r}, expected {default!r}"
            )
