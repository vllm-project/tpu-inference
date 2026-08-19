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


class _Clock:

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _load_scheduler_with_fake_vllm(monkeypatch):

    class Request:

        def __init__(self, request_id):
            self.request_id = request_id

    class RequestStatus:
        pass

    class Scheduler:

        def __init__(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=None,
            mm_registry=None,
            include_finished_set=False,
            log_stats=False,
        ):
            del (
                vllm_config,
                kv_cache_config,
                structured_output_manager,
                block_size,
                hash_block_size,
                mm_registry,
                include_finished_set,
                log_stats,
            )
            self.admitted = []
            self.finished = []
            self.schedule_calls = 0
            self.base_unfinished = set()
            self.requests = {}

        def add_request(self, request):
            self.admitted.append(request.request_id)
            self.base_unfinished.add(request.request_id)
            self.requests[request.request_id] = request

        def schedule(self, *args, **kwargs):
            self.schedule_calls += 1
            return args, kwargs

        def finish_requests(self, request_ids, finished_status):
            self.finished.append((list(request_ids), finished_status))
            self.base_unfinished.difference_update(request_ids)
            for request_id in request_ids:
                self.requests.pop(request_id, None)

        def get_num_unfinished_requests(self):
            return len(self.base_unfinished)

        def has_unfinished_requests(self):
            return bool(self.base_unfinished)

    class Placeholder:
        pass

    def resolve_generation_strategy(vllm_config):
        max_wait_ms = vllm_config.additional_config["diffusion"][
            "cohort_max_wait_ms"]
        return SimpleNamespace(diffusion=SimpleNamespace(
            runtime=SimpleNamespace(cohort_max_wait_ms=max_wait_ms)))

    modules = {
        "vllm":
        types.ModuleType("vllm"),
        "vllm.config":
        types.ModuleType("vllm.config"),
        "vllm.multimodal":
        types.ModuleType("vllm.multimodal"),
        "vllm.v1":
        types.ModuleType("vllm.v1"),
        "vllm.v1.core":
        types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched":
        types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.output":
        types.ModuleType("vllm.v1.core.sched.output"),
        "vllm.v1.core.sched.scheduler":
        types.ModuleType("vllm.v1.core.sched.scheduler"),
        "vllm.v1.kv_cache_interface":
        types.ModuleType("vllm.v1.kv_cache_interface"),
        "vllm.v1.request":
        types.ModuleType("vllm.v1.request"),
        "vllm.v1.structured_output":
        types.ModuleType("vllm.v1.structured_output"),
        "tpu_inference":
        types.ModuleType("tpu_inference"),
        "tpu_inference.runner":
        types.ModuleType("tpu_inference.runner"),
        "tpu_inference.runner.diffusion":
        types.ModuleType("tpu_inference.runner.diffusion"),
        "tpu_inference.runner.diffusion.config":
        types.ModuleType("tpu_inference.runner.diffusion.config"),
    }
    modules["vllm.config"].VllmConfig = Placeholder
    modules["vllm.multimodal"].MULTIMODAL_REGISTRY = Placeholder()
    modules["vllm.multimodal"].MultiModalRegistry = Placeholder
    modules["vllm.v1.core.sched.output"].SchedulerOutput = Placeholder
    modules["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
    modules["vllm.v1.kv_cache_interface"].KVCacheConfig = Placeholder
    modules["vllm.v1.request"].Request = Request
    modules["vllm.v1.request"].RequestStatus = RequestStatus
    modules["vllm.v1.structured_output"].StructuredOutputManager = Placeholder
    modules[
        "tpu_inference.runner.diffusion.config"].resolve_generation_strategy = resolve_generation_strategy
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "core" / "sched" / "diffusion_cohort_scheduler.py")
    spec = importlib.util.spec_from_file_location(
        "diffusion_cohort_scheduler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, Request, RequestStatus


def _vllm_config(max_num_seqs=2, max_wait_ms=10.0):
    return SimpleNamespace(
        additional_config={
            "diffusion": {
                "cohort_max_wait_ms": max_wait_ms,
            },
        },
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _scheduler(module, vllm_config):
    return module.BlockDiffusionCohortScheduler(
        vllm_config=vllm_config,
        kv_cache_config=object(),
        structured_output_manager=object(),
        block_size=16,
    )


def test_cohort_queue_releases_fifo_at_capacity(monkeypatch):
    module, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    queue = module.CohortAdmissionQueue(max_size=3,
                                        max_wait_ms=10.0,
                                        clock=clock)

    for item in range(5):
        queue.add(item)

    assert queue.drain_ready() == [0, 1, 2]
    assert len(queue) == 2
    assert not queue.is_ready()

    clock.now = 0.01
    assert queue.drain_ready() == [3, 4]


def test_cohort_queue_deadline_tracks_oldest_remaining_item(monkeypatch):
    module, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    queue = module.CohortAdmissionQueue(max_size=3,
                                        max_wait_ms=10.0,
                                        clock=clock)
    queue.add("cancelled")
    clock.now = 0.004
    queue.add("remaining")

    assert queue.discard(lambda item: item == "cancelled") == ["cancelled"]
    assert queue.deadline == 0.014
    clock.now = 0.013
    assert not queue.is_ready()
    clock.now = 0.014
    assert queue.drain_ready() == ["remaining"]


def test_scheduler_keeps_running_work_moving_before_cohort_is_ready(
        monkeypatch):
    module, Request, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(module, _vllm_config())
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("new"))

    assert scheduler.schedule("step") == (("step", ), {})
    assert scheduler.schedule_calls == 1
    assert scheduler.admitted == []
    assert scheduler.get_num_unfinished_requests() == 2
    assert scheduler.has_unfinished_requests()

    clock.now = 0.01
    scheduler.schedule()
    assert scheduler.admitted == ["new"]


def test_scheduler_releases_full_cohort_and_cancels_held_requests(monkeypatch):
    module, Request, RequestStatus = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config())
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("cancelled"))

    scheduler.finish_requests(["cancelled", "running"], RequestStatus)

    assert scheduler.finished == [(["running"], RequestStatus)]
    assert scheduler.get_num_unfinished_requests() == 1
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.add_request(Request("second"))
    scheduler.schedule()
    assert scheduler.admitted == ["first", "second"]


def test_scheduler_cancel_all_removes_held_and_admitted_requests(monkeypatch):
    module, Request, RequestStatus = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config())
    scheduler.base_unfinished.add("running")
    scheduler.requests["running"] = Request("running")
    scheduler.add_request(Request("held"))

    scheduler.finish_requests(None, RequestStatus)

    assert scheduler.finished == [(["running"], RequestStatus)]
    assert scheduler.get_num_unfinished_requests() == 0
    assert not scheduler.has_unfinished_requests()
