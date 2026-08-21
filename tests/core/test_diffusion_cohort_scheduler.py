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
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace

import pytest


class _Clock:

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _load_scheduler_with_fake_vllm(monkeypatch):

    class PauseState(Enum):
        UNPAUSED = "unpaused"
        PAUSED_NEW = "paused_new"
        PAUSED_ALL = "paused_all"

    class RequestStatus(Enum):
        WAITING = "waiting"
        FINISHED_ABORTED = "finished_aborted"

        @staticmethod
        def is_finished(status):
            return status == RequestStatus.FINISHED_ABORTED

    class Request:

        def __init__(self, request_id, client_index=0):
            self.request_id = request_id
            self.client_index = client_index
            self.status = RequestStatus.WAITING

        def is_finished(self):
            return RequestStatus.is_finished(self.status)

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
                log_stats,
            )
            self.admitted = []
            self.finished = []
            self.schedule_calls = 0
            self.base_unfinished = set()
            self.base_running = self.base_unfinished
            self.requests = {}
            self.finished_req_ids = set()
            self.finished_req_ids_dict = (defaultdict(set)
                                          if include_finished_set else None)
            self._pause_state = PauseState.UNPAUSED

        def add_request(self, request):
            self.admitted.append(request.request_id)
            self.base_unfinished.add(request.request_id)
            self.requests[request.request_id] = request

        def schedule(self, *args, **kwargs):
            self.schedule_calls += 1
            return args, kwargs

        def update_from_output(self, request_ids):
            self.base_unfinished.difference_update(request_ids)
            self.base_running.difference_update(request_ids)
            for request_id in request_ids:
                self.requests.pop(request_id, None)
            return list(request_ids)

        def finish_requests(self, request_ids, finished_status):
            assert RequestStatus.is_finished(finished_status)
            self.finished.append((list(request_ids), finished_status))
            finished_requests = []
            for request_id in request_ids:
                request = self.requests.get(request_id)
                if request is not None and not request.is_finished():
                    request.status = finished_status
                    self.finished_req_ids.add(request_id)
                    if self.finished_req_ids_dict is not None:
                        self.finished_req_ids_dict[request.client_index].add(
                            request_id)
                    finished_requests.append(
                        (request.request_id, request.client_index))
            self.base_unfinished.difference_update(request_ids)
            self.base_running.difference_update(request_ids)
            for request_id in request_ids:
                self.requests.pop(request_id, None)
            return finished_requests

        def get_num_unfinished_requests(self):
            if self._pause_state == PauseState.PAUSED_ALL:
                return 0
            if self._pause_state == PauseState.PAUSED_NEW:
                return len(self.base_running)
            return len(self.base_unfinished)

        def get_request_counts(self):
            return (len(self.base_running),
                    len(self.base_unfinished - self.base_running))

        def has_unfinished_requests(self):
            return self.get_num_unfinished_requests() > 0

        def has_finished_requests(self):
            return bool(self.finished_req_ids)

        @property
        def pause_state(self):
            return self._pause_state

        def set_pause_state(self, pause_state):
            self._pause_state = pause_state

    class Placeholder:
        pass

    def resolve_generation_strategy(vllm_config):
        max_wait_ms = vllm_config.additional_config["diffusion"][
            "cohort_max_wait_ms"]
        quiet_wait_ms = vllm_config.additional_config["diffusion"].get(
            "cohort_quiet_wait_ms", 0.0)
        strict_waves = vllm_config.additional_config["diffusion"].get(
            "cohort_strict_waves", False)
        return SimpleNamespace(diffusion=SimpleNamespace(
            runtime=SimpleNamespace(cohort_max_wait_ms=max_wait_ms,
                                    cohort_quiet_wait_ms=quiet_wait_ms,
                                    cohort_strict_waves=strict_waves)))

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
        "vllm.v1.core.sched.interface":
        types.ModuleType("vllm.v1.core.sched.interface"),
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
    modules["vllm.v1.core.sched.interface"].PauseState = PauseState
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
    return module, Request, RequestStatus, PauseState


def _vllm_config(max_num_seqs=2,
                 max_wait_ms=10.0,
                 quiet_wait_ms=0.0,
                 strict_waves=False):
    return SimpleNamespace(
        additional_config={
            "diffusion": {
                "cohort_max_wait_ms": max_wait_ms,
                "cohort_quiet_wait_ms": quiet_wait_ms,
                "cohort_strict_waves": strict_waves,
            },
        },
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _scheduler(module, vllm_config, *, include_finished_set=False):
    return module.BlockDiffusionCohortScheduler(
        vllm_config=vllm_config,
        kv_cache_config=object(),
        structured_output_manager=object(),
        block_size=16,
        include_finished_set=include_finished_set,
    )


def test_cohort_queue_releases_fifo_at_capacity(monkeypatch):
    module, _, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
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
    module, _, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
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


def test_cohort_queue_releases_after_quiet_period_with_hard_cap(monkeypatch):
    module, _, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    queue = module.CohortAdmissionQueue(max_size=32,
                                        max_wait_ms=20.0,
                                        quiet_wait_ms=2.0,
                                        clock=clock)
    queue.add("first")
    clock.now = 0.004
    queue.add("second")

    assert queue.deadline == 0.006
    clock.now = 0.005
    assert not queue.is_ready()
    clock.now = 0.006
    assert queue.drain_ready() == ["first", "second"]

    clock.now = 1.0
    queue.add("hard-cap-first")
    for index, now in enumerate((1.005, 1.010, 1.015, 1.019)):
        clock.now = now
        queue.add(index)
    assert queue.deadline == 1.02


def test_scheduler_keeps_running_work_moving_before_cohort_is_ready(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
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


def test_strict_waves_keep_running_work_moving_without_admitting_full_cohort(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module, _vllm_config(strict_waves=True))
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))

    assert scheduler.schedule("step") == (("step", ), {})
    assert scheduler.schedule_calls == 1
    assert scheduler.admitted == []
    assert scheduler.get_num_unfinished_requests() == 3
    assert scheduler.get_request_counts() == (1, 2)


def test_strict_waves_admit_first_ready_wave_when_held_requests_are_unfinished(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module, _vllm_config(strict_waves=True))
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))

    assert scheduler.has_unfinished_requests()
    assert scheduler.get_num_unfinished_requests() == 2

    scheduler.schedule()

    assert scheduler.admitted == ["first", "second"]


def test_strict_waves_accumulate_continuous_arrivals_for_next_full_wave(
        monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(max_num_seqs=3, strict_waves=True))
    scheduler.base_unfinished.add("running")

    for request_id in ("first", "second", "third"):
        scheduler.add_request(Request(request_id))
        scheduler.schedule()
        assert scheduler.admitted == []

    scheduler.finish_requests("running", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    assert scheduler.admitted == ["first", "second", "third"]


def test_strict_waves_first_partial_cohort_uses_quiet_wait(monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=4,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    scheduler.add_request(Request("first"))

    clock.now = 0.001
    scheduler.schedule()
    assert scheduler.admitted == []

    clock.now = 0.002
    scheduler.schedule()
    assert scheduler.admitted == ["first"]
    assert not scheduler._last_admitted_cohort_was_full


def test_strict_waves_hold_partial_refill_past_quiet_wait(monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=3,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    first_wave = ["first", "second", "third"]
    for request_id in first_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()

    clock.now = 0.001
    scheduler.add_request(Request("refill"))
    clock.now = 0.003
    scheduler.update_from_output(first_wave)
    scheduler.schedule()

    assert scheduler.admitted == first_wave
    assert scheduler._last_admitted_cohort_was_full


def test_strict_waves_admit_full_refill_only_after_base_is_idle(monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(max_num_seqs=2, strict_waves=True))
    first_wave = ["first", "second"]
    refill_wave = ["third", "fourth"]
    for request_id in first_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()

    for request_id in refill_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()
    assert scheduler.admitted == first_wave

    scheduler.update_from_output(first_wave)
    scheduler.schedule()
    assert scheduler.admitted == first_wave + refill_wave
    assert scheduler._last_admitted_cohort_was_full


def test_strict_waves_hard_timeout_partial_refill_then_restore_quiet_wait(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=2,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    first_wave = ["first", "second"]
    for request_id in first_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()

    clock.now = 0.001
    scheduler.add_request(Request("timed-out-refill"))
    scheduler.update_from_output(first_wave)
    clock.now = 0.003
    scheduler.schedule()
    assert scheduler.admitted == first_wave

    clock.now = 0.011
    scheduler.schedule()
    assert scheduler.admitted == first_wave + ["timed-out-refill"]
    assert not scheduler._last_admitted_cohort_was_full

    scheduler.update_from_output(["timed-out-refill"])
    clock.now = 0.012
    scheduler.add_request(Request("next-partial"))
    clock.now = 0.014
    scheduler.schedule()
    assert scheduler.admitted == first_wave + [
        "timed-out-refill", "next-partial"
    ]


def test_strict_waves_full_refill_requirement_expires_while_idle(monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=2,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    first_wave = ["first", "second"]
    for request_id in first_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()

    clock.now = 0.001
    scheduler.update_from_output(first_wave)
    clock.now = 0.020
    scheduler.add_request(Request("new-first"))
    scheduler.schedule()
    assert scheduler.admitted == first_wave

    clock.now = 0.022
    scheduler.schedule()
    assert scheduler.admitted == first_wave + ["new-first"]
    assert not scheduler._last_admitted_cohort_was_full


def test_strict_waves_abort_all_resets_full_refill_requirement(monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=2,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))
    scheduler.schedule()
    scheduler.add_request(Request("held"))

    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    clock.now = 0.001
    scheduler.add_request(Request("new-first"))
    clock.now = 0.003
    scheduler.schedule()

    assert scheduler.admitted == ["first", "second", "new-first"]
    assert not scheduler._last_admitted_cohort_was_full


def test_strict_waves_cancelled_refill_uses_remaining_hard_deadline(
        monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=2,
                     max_wait_ms=10.0,
                     quiet_wait_ms=2.0,
                     strict_waves=True))
    first_wave = ["first", "second"]
    for request_id in first_wave:
        scheduler.add_request(Request(request_id))
    scheduler.schedule()

    clock.now = 0.001
    scheduler.add_request(Request("cancelled"))
    clock.now = 0.002
    scheduler.add_request(Request("remaining"))
    scheduler.update_from_output(first_wave)
    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)

    clock.now = 0.011
    scheduler.schedule()
    assert scheduler.admitted == first_wave

    clock.now = 0.012
    scheduler.schedule()
    assert scheduler.admitted == first_wave + ["remaining"]


def test_strict_waves_pass_continuation_for_admitted_request_to_base(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module, _vllm_config(strict_waves=True))
    scheduler.add_request(Request("streaming"))
    scheduler.add_request(Request("peer"))
    scheduler.schedule()

    scheduler.add_request(Request("streaming"))

    assert scheduler.admitted == ["streaming", "peer", "streaming"]
    assert scheduler._cohort_request_ids == set()
    assert scheduler.get_num_unfinished_requests() == 2


def test_strict_waves_release_timed_out_cohort_only_after_base_is_idle(
        monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=4, max_wait_ms=10.0, strict_waves=True))
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("timed-out"))

    clock.now = 0.01
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.finish_requests("running", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    assert scheduler.admitted == ["timed-out"]


def test_strict_waves_cancel_held_request_without_blocking_remaining_timeout(
        monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=2, max_wait_ms=10.0, strict_waves=True))
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("remaining"))
    scheduler.add_request(Request("cancelled"))

    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)
    clock.now = 0.01
    scheduler.finish_requests("running", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    assert scheduler.admitted == ["remaining"]
    assert scheduler.finished == [(["running"], RequestStatus.FINISHED_ABORTED)
                                  ]


def test_default_cohort_mode_admits_full_cohort_while_base_is_running(
        monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module, _vllm_config(strict_waves=False))
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))

    scheduler.schedule()

    assert scheduler.admitted == ["first", "second"]


def test_scheduler_uses_quiet_period_for_burst_admission(monkeypatch):
    module, Request, _, _ = _load_scheduler_with_fake_vllm(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(module, "monotonic", clock)
    scheduler = _scheduler(
        module,
        _vllm_config(max_num_seqs=32, max_wait_ms=20.0, quiet_wait_ms=2.0))
    scheduler.add_request(Request("first"))
    clock.now = 0.004
    scheduler.add_request(Request("second"))

    clock.now = 0.005
    scheduler.schedule()
    assert scheduler.admitted == []
    clock.now = 0.006
    scheduler.schedule()
    assert scheduler.admitted == ["first", "second"]


def test_scheduler_releases_full_cohort_and_cancels_held_requests(monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config())
    scheduler.base_unfinished.add("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("cancelled"))

    scheduler.finish_requests(["cancelled", "running"],
                              RequestStatus.FINISHED_ABORTED)

    assert scheduler.finished == [(["running"], RequestStatus.FINISHED_ABORTED)
                                  ]
    assert scheduler.get_num_unfinished_requests() == 1
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.add_request(Request("second"))
    scheduler.schedule()
    assert scheduler.admitted == ["first", "second"]


def test_scheduler_cancel_all_removes_held_and_admitted_requests(monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config())
    scheduler.base_unfinished.add("running")
    scheduler.requests["running"] = Request("running", client_index=1)
    scheduler.add_request(Request("held", client_index=2))

    finished_requests = scheduler.finish_requests(
        None, RequestStatus.FINISHED_ABORTED)

    assert finished_requests == [("held", 2), ("running", 1)]
    assert scheduler.finished == [(["running"], RequestStatus.FINISHED_ABORTED)
                                  ]
    assert scheduler.get_num_unfinished_requests() == 0
    assert not scheduler.has_unfinished_requests()


def test_paused_all_does_not_count_or_admit_held_requests(monkeypatch):
    module, Request, _, PauseState = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config(max_num_seqs=1))
    scheduler.add_request(Request("held"))

    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert scheduler.get_num_unfinished_requests() == 0
    assert not scheduler.has_unfinished_requests()
    assert scheduler.get_request_counts() == (0, 1)
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.schedule()
    assert scheduler.admitted == ["held"]


def test_paused_new_keeps_held_wave_out_of_running_work(monkeypatch):
    module, Request, RequestStatus, PauseState = \
        _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(max_num_seqs=2, strict_waves=True))
    scheduler.base_unfinished.add("running")
    scheduler.requests["running"] = Request("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))

    scheduler.set_pause_state(PauseState.PAUSED_NEW)

    assert scheduler.get_num_unfinished_requests() == 1
    assert scheduler.get_request_counts() == (1, 2)
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.finish_requests("running", RequestStatus.FINISHED_ABORTED)
    assert scheduler.get_num_unfinished_requests() == 0
    assert not scheduler.has_unfinished_requests()
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.schedule()
    assert scheduler.admitted == ["first", "second"]


def test_paused_new_base_waiting_request_is_not_physical_idle(monkeypatch):
    module, Request, _, PauseState = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(max_num_seqs=1, strict_waves=True))
    scheduler.base_running = set()
    scheduler.base_unfinished.add("waiting")
    scheduler.requests["waiting"] = Request("waiting")
    scheduler.add_request(Request("held"))

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.get_request_counts() == (0, 2)
    scheduler.schedule()
    assert scheduler.admitted == []

    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.schedule()
    assert scheduler.admitted == []


def test_paused_all_does_not_mix_held_and_existing_strict_waves(monkeypatch):
    module, Request, RequestStatus, PauseState = \
        _load_scheduler_with_fake_vllm(monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(max_num_seqs=2, strict_waves=True))
    scheduler.base_unfinished.add("running")
    scheduler.requests["running"] = Request("running")
    scheduler.add_request(Request("first"))
    scheduler.add_request(Request("second"))

    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    scheduler.schedule()
    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.schedule()

    assert scheduler.admitted == []

    scheduler.finish_requests("running", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    assert scheduler.admitted == ["first", "second"]


@pytest.mark.parametrize("include_finished_set", [False, True])
def test_explicit_abort_of_held_request_publishes_finished_ids(
        monkeypatch, include_finished_set):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module,
                           _vllm_config(),
                           include_finished_set=include_finished_set)
    held = Request("held", client_index=7)
    scheduler.add_request(held)

    finished = scheduler.finish_requests(["held", "held"],
                                         RequestStatus.FINISHED_ABORTED)

    assert finished == [("held", 7)]
    assert held.status == RequestStatus.FINISHED_ABORTED
    assert scheduler.finished_req_ids == {"held"}
    if include_finished_set:
        assert scheduler.finished_req_ids_dict == {7: {"held"}}
    else:
        assert scheduler.finished_req_ids_dict is None
    assert scheduler.has_finished_requests()
    assert scheduler.finish_requests("held",
                                     RequestStatus.FINISHED_ABORTED) == []


def test_abort_all_returns_held_request_without_deferred_publication(
        monkeypatch):
    module, Request, RequestStatus, _ = _load_scheduler_with_fake_vllm(
        monkeypatch)
    scheduler = _scheduler(module, _vllm_config(), include_finished_set=True)
    held = Request("held", client_index=7)
    scheduler.add_request(held)

    finished = scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    assert finished == [("held", 7)]
    assert held.status == RequestStatus.FINISHED_ABORTED
    assert scheduler.finished_req_ids == set()
    assert scheduler.finished_req_ids_dict == {}
