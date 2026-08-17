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
    forward_source = ast.unparse(_strategy_method("_run_model"))

    assert "runner.model_fn_no_options" in forward_source
    assert "runner.model_fn(" not in forward_source


def test_dual_cache_has_distinct_full_partial_and_final_forwards():
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))
    partial_source = ast.unparse(
        _strategy_method("_model_forward_partial_subblock"))
    final_source = ast.unparse(_strategy_method("_model_forward_final"))

    assert "denoise_block_dual_cache" in denoise_source
    assert "dynamic_slice" in partial_source
    assert "sub_block_size" in partial_source
    assert "[:, -1, :]" in final_source


def test_diffusion_uses_configured_capacity_and_partial_cache_metadata():
    build_source = ast.unparse(_strategy_method("_build_batch"))
    precompile_source = ast.unparse(_strategy_method("precompile"))

    assert "select_diffusion_batch_size(num_active, capacity)" in build_source
    assert "batch_size = runner.max_num_reqs" not in build_source
    assert "replace_cached_kv=True" in build_source
    assert "partial_query_start_loc" in build_source
    assert "[0, num_active, num_active]" in build_source
    assert "rpa_static_query_len=block_size" in build_source
    assert "rpa_static_query_len=sub_block_size" in build_source
    assert "diffusion_batch_sizes(self.batch_size)" in precompile_source


def test_diffusion_rejects_cache_reuse_and_transfer():
    validation_source = ast.unparse(
        _strategy_method("_validate_runner_capabilities"))

    assert "enable_prefix_caching" in validation_source
    assert "kv_transfer_config" in validation_source
