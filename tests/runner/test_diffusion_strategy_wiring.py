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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
          "runner" / "tpu_runner.py")
STRATEGY = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "runner" / "experimental" / "diffusion" / "strategy.py")


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
        "self._execute_continue_decode(scheduler_output)")

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

    assert "runner.model.step_fn_no_options" in forward_source
    assert "runner.model_fn(" not in forward_source


def _load_execute_model(empty_output):
    methods = [_method("_execute_model")]
    try:
        methods.append(_method("_can_use_continue_decode"))
    except AssertionError:
        pass
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {
        "EMPTY_MODEL_RUNNER_OUTPUT": empty_output,
        "has_kv_transfer_group": lambda: False,
        "logger": Mock(),
        "np": np,
    }
    source = "from __future__ import annotations\n" + ast.unparse(module)
    exec(compile(source, str(RUNNER), "exec"), namespace)
    namespace["TPUModelRunner"] = SimpleNamespace(
        _can_use_continue_decode=namespace.get("_can_use_continue_decode"))
    return namespace["_execute_model"]


@pytest.mark.parametrize("scheduled_tokens", [0, 8])
def test_diffusion_dispatch_preserves_empty_cycle_cleanup(scheduled_tokens):
    empty_output = object()
    diffusion_output = object()
    events = []
    strategy = SimpleNamespace(
        on_scheduler_update=lambda finished: events.append(
            ("cleanup", finished)),
        execute=lambda output: events.append(
            ("diffusion", output)) or diffusion_output,
    )
    runner = SimpleNamespace(
        persistent_batch_manager=SimpleNamespace(
            update_states=lambda output, positions: events.append(
                ("states", output))),
        get_mrope_input_positions_fn=None,
        block_diffusion_strategy=strategy,
        aux_output_worker=SimpleNamespace(
            begin_step=lambda metadata: events.append(("aux", metadata))),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    output = SimpleNamespace(
        finished_req_ids={"finished-request"},
        aux_output_connector_metadata="step-metadata",
        total_num_scheduled_tokens=scheduled_tokens,
    )

    result = _load_execute_model(empty_output)(runner, output)

    assert result is (diffusion_output if scheduled_tokens else empty_output)
    expected = [("states", output), ("cleanup", {"finished-request"}),
                ("aux", "step-metadata")]
    if scheduled_tokens:
        expected.append(("diffusion", output))
    assert events == expected


def test_ordinary_decode_dispatch_is_preserved():
    autoregressive_output = object()
    events = []
    runner = SimpleNamespace(
        persistent_batch_manager=SimpleNamespace(
            update_states=lambda output, positions: events.append("states")),
        get_mrope_input_positions_fn=None,
        block_diffusion_strategy=None,
        aux_output_worker=None,
        input_batch=SimpleNamespace(
            request_distribution=[1],
            num_reqs=1,
            num_computed_tokens_cpu=np.array([1]),
            num_prompt_tokens=np.array([1]),
        ),
        enable_continue_decode=True,
        _execute_continue_decode=lambda output: events.append("autoregressive")
        or autoregressive_output,
    )
    output = SimpleNamespace(total_num_scheduled_tokens=1,
                             has_structured_output_requests=False)

    result = _load_execute_model(object())(runner, output)

    assert result is autoregressive_output
    assert events == ["states", "autoregressive"]
