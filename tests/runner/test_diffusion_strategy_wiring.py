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

import ast
import pathlib

RUNNER = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
          "runner" / "tpu_runner.py")
STRATEGY = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "runner" / "diffusion" / "strategy.py")


def _method(name):
    module = ast.parse(RUNNER.read_text())
    runner_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner")
    for node in runner_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {name!r} not found")


def _strategy_method(name):
    module = ast.parse(STRATEGY.read_text())
    strategy_class = next(node for node in module.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "BlockDiffusionStrategy")
    for node in strategy_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Strategy method {name!r} not found")


def test_runner_resolves_generation_strategy_once_at_startup():
    init_source = ast.unparse(_method("__init__"))

    assert "resolve_generation_strategy(vllm_config)" in init_source
    assert "BlockDiffusionStrategy" in init_source


def test_diffusion_dispatch_precedes_autoregressive_phase_dispatch():
    execute_source = ast.unparse(_method("_execute_model"))
    diffusion_dispatch = execute_source.index(
        "self.block_diffusion_strategy.execute(scheduler_output)")
    autoregressive_dispatch = execute_source.index(
        "self.enable_continue_decode")

    assert diffusion_dispatch < autoregressive_dispatch


def test_finished_requests_are_cleaned_before_empty_cycle_return():
    execute_source = ast.unparse(_method("_execute_model"))
    cleanup = execute_source.index("on_scheduler_update")
    empty_cycle = execute_source.index(
        "if not scheduler_output.total_num_scheduled_tokens")

    assert cleanup < empty_cycle


def test_diffusion_precompile_uses_the_runtime_mesh_context():
    capture_source = ast.unparse(_method("capture_model"))

    assert "with jax.set_mesh(self.mesh)" in capture_source
    assert "self.block_diffusion_strategy.precompile()" in capture_source


def test_diffusion_forward_uses_nested_jit_safe_model_callable():
    forward_source = ast.unparse(_strategy_method("_model_forward"))

    assert "runner.model_fn_no_options" in forward_source
    assert "runner.model_fn(" not in forward_source
