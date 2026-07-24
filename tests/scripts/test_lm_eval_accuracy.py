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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = (Path(__file__).parents[2] / "scripts" / "vllm" /
                "integration" / "lm_eval_accuracy.py")
_SPEC = importlib.util.spec_from_file_location("lm_eval_accuracy",
                                               _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_evaluate_with_vllm_always_shuts_down(monkeypatch, evaluation_fails):
    shutdown_calls = []
    engine_core = types.SimpleNamespace(
        shutdown=lambda: shutdown_calls.append(True))
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(engine_core=engine_core)))
    monkeypatch.setattr(_MODULE, "_create_vllm_model",
                        lambda *args, **kwargs: model)

    fake_lm_eval = types.ModuleType("lm_eval")

    def simple_evaluate(**kwargs):
        if evaluation_fails:
            raise RuntimeError("evaluation failed")
        return {"results": {}}

    fake_lm_eval.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)

    if evaluation_fails:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            _MODULE.evaluate_with_vllm(model_args={}, tasks="task")
    else:
        assert _MODULE.evaluate_with_vllm(model_args={}, tasks="task") == {
            "results": {}
        }

    assert shutdown_calls == [True]


def test_main_parses_arguments(monkeypatch):
    captured_args = {}

    def fake_evaluate_with_vllm(**kwargs):
        captured_args.update(kwargs)
        return {"results": {}}

    monkeypatch.setattr(_MODULE, "evaluate_with_vllm", fake_evaluate_with_vllm)
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.utils = types.SimpleNamespace(
        setup_logging=lambda: None,
        make_table=lambda res: "table",
    )
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.utils", fake_lm_eval.utils)

    test_argv = [
        "lm_eval_accuracy.py",
        "--model_args",
        "pretrained=test-model",
        "--tasks",
        "gsm8k",
        "--apply_chat_template",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    _MODULE.main()

    assert captured_args.get("apply_chat_template") is True


def test_main_output_path_and_include_path(monkeypatch, tmp_path):
    shutdown_calls = []
    engine_core = types.SimpleNamespace(
        shutdown=lambda: shutdown_calls.append(True))
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(engine_core=engine_core)))
    monkeypatch.setattr(_MODULE, "_create_vllm_model",
                        lambda *args, **kwargs: model)

    received_kwargs = {}

    def simple_evaluate(**kwargs):
        received_kwargs.update(kwargs)
        return {"results": {"acc": 0.95}}

    included_paths = []
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = simple_evaluate
    fake_tasks = types.SimpleNamespace(
        include_path=lambda path: included_paths.append(path))
    fake_lm_eval.tasks = fake_tasks
    fake_lm_eval.utils = types.SimpleNamespace(
        setup_logging=lambda: None,
        make_table=lambda res: "table",
    )

    class FakeEvaluationTracker:

        def __init__(self, output_path, log_samples):
            self.output_path = output_path
            self.log_samples = log_samples

    fake_loggers = types.SimpleNamespace(
        EvaluationTracker=FakeEvaluationTracker)
    fake_lm_eval.loggers = fake_loggers

    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "lm_eval.utils", fake_lm_eval.utils)
    monkeypatch.setitem(sys.modules, "lm_eval.loggers", fake_loggers)

    out_json = tmp_path / "test_output.json"
    test_argv = [
        "lm_eval_accuracy.py",
        "--model_args",
        "pretrained=test-model",
        "--tasks",
        "gsm8k",
        "--apply_chat_template",
        "--output_path",
        str(out_json),
        "--include_path",
        str(tmp_path),
        "--log_samples",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    _MODULE.main()

    assert included_paths == [str(tmp_path)]
    assert "output_path" not in received_kwargs
    assert "include_path" not in received_kwargs
    tracker = received_kwargs.get("evaluation_tracker")
    assert tracker is not None
    assert tracker.output_path == str(out_json)
    assert tracker.log_samples is True
    assert shutdown_calls == [True]
