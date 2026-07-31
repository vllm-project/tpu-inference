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


def _load_utils_with_fake_vllm(monkeypatch):

    class Scheduler:

        def __init__(self, vllm_config):
            del vllm_config
            self.num_lookahead_tokens = 2

        def _update_request_with_output(self, request, new_token_ids):
            retained = list(new_token_ids[:request.retain_tokens])
            request.output_token_ids.extend(retained)
            return retained, len(retained) != len(new_token_ids)

    class AsyncScheduler:

        def _update_request_with_output(self, request, new_token_ids):
            request.num_output_placeholders -= len(new_token_ids)
            return list(new_token_ids), False

    modules = {
        "vllm":
        types.ModuleType("vllm"),
        "vllm.v1":
        types.ModuleType("vllm.v1"),
        "vllm.v1.core":
        types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched":
        types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler":
        types.ModuleType("vllm.v1.core.sched.scheduler"),
        "vllm.v1.core.sched.async_scheduler":
        types.ModuleType("vllm.v1.core.sched.async_scheduler"),
    }
    modules["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
    modules[
        "vllm.v1.core.sched.async_scheduler"].AsyncScheduler = AsyncScheduler
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "core" / "sched" / "utils.py")
    spec = importlib.util.spec_from_file_location("scheduler_utils_under_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, Scheduler, AsyncScheduler


def test_continue_decode_lookahead_is_preserved(monkeypatch):
    utils, Scheduler, _ = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_continue_decode()

    scheduler = Scheduler(
        SimpleNamespace(additional_config={
            "enable_continue_decode": True,
            "max_decode_steps": 10,
        }))

    assert scheduler.num_lookahead_tokens == 9


def test_explicit_multi_token_lookahead_takes_precedence(monkeypatch):
    utils, Scheduler, _ = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_multi_token_decode()

    scheduler = Scheduler(
        SimpleNamespace(
            additional_config={
                "enable_continue_decode": True,
                utils.MULTI_TOKEN_LOOKAHEAD_CONFIG: 31,
            }))

    assert scheduler.num_lookahead_tokens == 31


def test_output_accounting_uses_tokens_retained_after_eos(monkeypatch):
    utils, Scheduler, _ = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_multi_token_decode()
    scheduler = Scheduler(
        SimpleNamespace(additional_config={
            utils.MULTI_TOKEN_LOOKAHEAD_CONFIG: 3,
        }))
    request = SimpleNamespace(
        retain_tokens=2,
        output_token_ids=[],
        num_computed_tokens=1,
    )

    retained, stopped = scheduler._update_request_with_output(
        request, [10, 11, 12, 13])

    assert retained == [10, 11]
    assert stopped is True
    assert request.num_computed_tokens == 2


def test_async_placeholder_accounting_consumes_one_scheduled_step(monkeypatch):
    utils, _, AsyncScheduler = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_multi_token_decode()
    scheduler = AsyncScheduler()
    scheduler._multi_token_decode_enabled = True
    request = SimpleNamespace(num_output_placeholders=1)

    scheduler._update_request_with_output(request, [10, 11, 12, 13])

    assert request.num_output_placeholders == 0


def test_normal_scheduler_is_unchanged_after_class_patch(monkeypatch):
    utils, Scheduler, _ = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_multi_token_decode()
    scheduler = Scheduler(SimpleNamespace(additional_config={}))
    request = SimpleNamespace(
        retain_tokens=3,
        output_token_ids=[],
        num_computed_tokens=1,
    )

    scheduler._update_request_with_output(request, [1, 2, 3])

    assert request.num_computed_tokens == 1


def test_patch_is_idempotent(monkeypatch):
    utils, Scheduler, _ = _load_utils_with_fake_vllm(monkeypatch)
    utils.patch_vllm_scheduler_for_multi_token_decode()
    utils.patch_vllm_scheduler_for_continue_decode()
    scheduler = Scheduler(
        SimpleNamespace(additional_config={
            "enable_continue_decode": True,
        }))
    request = SimpleNamespace(
        retain_tokens=3,
        output_token_ids=[],
        num_computed_tokens=1,
    )

    scheduler._update_request_with_output(request, [1, 2, 3])

    assert request.num_computed_tokens == 3
