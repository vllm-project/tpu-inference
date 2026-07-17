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
