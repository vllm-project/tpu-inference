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

import inspect
from unittest import mock

import pytest

from tpu_inference.core.sched.utils import \
    patch_vllm_scheduler_for_continue_decode


@pytest.fixture(autouse=True)
def restore_scheduler_patch():
    """The continue_decode patch is a process-global, irreversible monkeypatch
    (including Scheduler.__init__, which forces num_lookahead_tokens >= 9).
    CI runs all of tests/ in one pytest process, so restore the originals after
    each test to avoid leaking the patch into unrelated tests."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    originals = (
        Scheduler._update_request_with_output,
        Scheduler.update_from_output,
        Scheduler.__init__,
        AsyncScheduler._update_request_with_output,
    )
    yield
    (
        Scheduler._update_request_with_output,
        Scheduler.update_from_output,
        Scheduler.__init__,
        AsyncScheduler._update_request_with_output,
    ) = originals
    for cls in (Scheduler, AsyncScheduler):
        if "_continue_decode_patched" in cls.__dict__:
            del cls._continue_decode_patched


def _make_request(num_computed_tokens=10, num_output_placeholders=1):
    request = mock.MagicMock()
    request.num_computed_tokens = num_computed_tokens
    request.num_output_placeholders = num_output_placeholders
    return request


def _make_scheduler(cls):
    # spec= so getattr(scheduler, "_cd_stale_in_flight", False) is False (a
    # bare MagicMock would auto-create a truthy attribute) and so the zero-arg
    # super() in the wrapped async original resolves (isinstance check).
    # Instance attributes read by the base implementation are set explicitly.
    scheduler = mock.MagicMock(spec=cls)
    scheduler.max_model_len = 1024
    return scheduler


class TestContinueDecodeSchedulerPatch:

    def test_patched_signatures_accept_is_stale(self):
        """vLLM passes is_stale=... as a kwarg from update_from_output; the
        patched wrappers must accept it or the EngineCore dies with a
        TypeError."""
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        for cls in (Scheduler, AsyncScheduler):
            params = inspect.signature(
                cls._update_request_with_output).parameters
            assert "is_stale" in params, (
                f"{cls.__name__}._update_request_with_output must accept "
                f"is_stale (got {list(params)})")
            assert any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in params.values()), (
                           f"{cls.__name__}._update_request_with_output must "
                           f"pass through future kwargs (**kwargs)")

    def test_sync_patch_advances_num_computed_tokens_unless_stale(self):
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler = _make_scheduler(Scheduler)
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        return_value=False):
            # Fresh output: 3 tokens returned for 1 scheduled step -> advance
            # num_computed_tokens by the extra 2 on-device tokens.
            request = _make_request(num_computed_tokens=10)
            Scheduler._update_request_with_output(scheduler,
                                                  request, [1, 2, 3],
                                                  is_stale=False)
            assert request.num_computed_tokens == 12

            # Stale output predates the preemption rollback -> no advance.
            request = _make_request(num_computed_tokens=10)
            Scheduler._update_request_with_output(scheduler,
                                                  request, [1, 2, 3],
                                                  is_stale=True)
            assert request.num_computed_tokens == 10

    def test_async_patch_stale_handling(self):
        """The async original calls super() WITHOUT forwarding is_stale, so
        staleness reaches the patched base via the _cd_stale_in_flight flag.
        Assert both counters end-to-end through the full async chain."""
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        scheduler = _make_scheduler(AsyncScheduler)
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        return_value=False):
            # Fresh output: placeholders pre-compensated by (N - 1) so the
            # original's -= N lands back at 0; num_computed_tokens advances.
            request = _make_request(num_computed_tokens=10,
                                    num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=False)
            assert request.num_output_placeholders == 0
            assert request.num_computed_tokens == 12

            # Stale output: placeholders were zeroed at preemption and
            # num_computed_tokens was rolled back; neither may move.
            request = _make_request(num_computed_tokens=10,
                                    num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=True)
            assert request.num_output_placeholders == 1
            assert request.num_computed_tokens == 10
            # The flag must not leak past the call.
            assert scheduler._cd_stale_in_flight is False
            assert scheduler._cd_in_async_update is False

    def test_async_patch_stop_trimming_preserves_placeholders(self):
        """When check_stop trims new_token_ids mid-burst (e.g., max_tokens hit on
        token 2 of 4), num_output_placeholders must still decrement by 1."""
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        scheduler = _make_scheduler(AsyncScheduler)
        # Stop on the 2nd token so [10, 20, 30, 40] trims to [10, 20].
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        side_effect=[False, True]):
            request = _make_request(num_computed_tokens=10,
                                    num_output_placeholders=2)
            res, stopped = AsyncScheduler._update_request_with_output(
                scheduler, request, [10, 20, 30, 40], is_stale=False)
            assert stopped is True
            assert res == [10, 20]
            assert request.num_output_placeholders == 1
            assert request.num_computed_tokens == 11

    def test_update_from_output_without_routed_experts_leaves_scheduled_tokens_untouched(
            self):
        """When routed_experts is None, num_scheduled_tokens is not modified
        and num_in_flight_tokens decrements by the scheduled step count (1)."""
        from vllm.v1.core.sched.scheduler import Scheduler

        request = _make_request(num_computed_tokens=10)
        request.num_in_flight_tokens = 1
        request.num_stale_output_tokens = 0

        observed_scheduled = {}

        def fake_orig_update(sched_self, sched_out, runner_out, *args,
                             **kwargs):
            for rid, count in sched_out.num_scheduled_tokens.items():
                observed_scheduled[rid] = count
                req = sched_self.requests.get(rid)
                if req is not None:
                    req.num_in_flight_tokens -= count
            return "ok"

        with mock.patch.object(Scheduler,
                               "update_from_output",
                               new=fake_orig_update):
            patch_vllm_scheduler_for_continue_decode()
            scheduler = _make_scheduler(Scheduler)
            scheduler.requests = {"req1": request}

            scheduler_output = mock.MagicMock()
            scheduler_output.num_scheduled_tokens = {"req1": 1}
            scheduler_output.scheduled_cached_reqs = mock.MagicMock(
                req_ids=["req1"], num_output_tokens=[3])
            scheduler_output.scheduled_spec_decode_tokens = {}

            model_runner_output = mock.MagicMock()
            model_runner_output.routed_experts = None
            model_runner_output.sampled_token_ids = [[10, 20, 30, 40, 50]]
            model_runner_output.req_id_to_index = {"req1": 0}

            res = Scheduler.update_from_output(scheduler, scheduler_output,
                                               model_runner_output)

        assert res == "ok"
        assert observed_scheduled == {"req1": 1}
        assert scheduler_output.num_scheduled_tokens == {"req1": 1}
        assert request.num_in_flight_tokens == 0

    def test_update_from_output_with_routed_experts_temporarily_expands_and_restores(
            self):
        """When routed_experts is present on a multi-token decode output,
        original_update_from_output sees num_scheduled_tokens[req_id] == N for
        slicing routed_experts, num_in_flight_tokens / num_stale_output_tokens
        still have a net decrement of 1, and num_scheduled_tokens is restored."""
        from vllm.v1.core.sched.scheduler import Scheduler

        req_fresh = _make_request(num_computed_tokens=10)
        req_fresh.num_in_flight_tokens = 1
        req_fresh.num_stale_output_tokens = 0

        req_stale = _make_request(num_computed_tokens=10)
        req_stale.num_in_flight_tokens = 1
        req_stale.num_stale_output_tokens = 1

        req_prefill = _make_request(num_computed_tokens=0)
        req_prefill.num_in_flight_tokens = 10
        req_prefill.num_stale_output_tokens = 0

        req_spec = _make_request(num_computed_tokens=10)
        req_spec.num_in_flight_tokens = 3
        req_spec.num_stale_output_tokens = 0

        observed_scheduled = {}

        def fake_orig_update(sched_self, sched_out, runner_out, *args,
                             **kwargs):
            for rid, count in sched_out.num_scheduled_tokens.items():
                observed_scheduled[rid] = count
                req = sched_self.requests.get(rid)
                if req is not None:
                    req.num_in_flight_tokens -= count
                    if req.num_stale_output_tokens > 0:
                        req.num_stale_output_tokens -= count
            return "ok"

        with mock.patch.object(Scheduler,
                               "update_from_output",
                               new=fake_orig_update):
            patch_vllm_scheduler_for_continue_decode()
            scheduler = _make_scheduler(Scheduler)
            scheduler.requests = {
                "req1": req_fresh,
                "req2": req_stale,
                "req_prefill": req_prefill,
                "req_spec": req_spec,
            }

            scheduler_output = mock.MagicMock()
            scheduler_output.num_scheduled_tokens = {
                "req1": 1,
                "req2": 1,
                "req_prefill": 10,
                "req_spec": 3,
            }
            scheduler_output.scheduled_cached_reqs = mock.MagicMock(
                req_ids=["req1", "req2", "req_prefill", "req_spec"],
                num_output_tokens=[3, 2, 0, 4],
            )
            scheduler_output.scheduled_spec_decode_tokens = {
                "req_spec": [101, 102]
            }

            model_runner_output = mock.MagicMock()
            model_runner_output.routed_experts = mock.MagicMock()
            model_runner_output.req_ids = [
                "req1", "req2", "req_prefill", "req_spec"
            ]
            model_runner_output.sampled_token_ids = [
                [10, 20, 30, 40, 50],
                [60, 70, 80, 90, 100],
                [11],
                [21, 22, 23, 24],
            ]
            model_runner_output.req_id_to_index = {
                "req1": 0,
                "req2": 1,
                "req_prefill": 2,
                "req_spec": 3,
            }

            res = Scheduler.update_from_output(scheduler, scheduler_output,
                                               model_runner_output)

        assert res == "ok"
        assert observed_scheduled == {
            "req1": 5,
            "req2": 5,
            "req_prefill": 10,
            "req_spec": 3,
        }
        assert scheduler_output.num_scheduled_tokens == {
            "req1": 1,
            "req2": 1,
            "req_prefill": 10,
            "req_spec": 3,
        }
        assert req_fresh.num_in_flight_tokens == 0
        assert req_fresh.num_stale_output_tokens == 0
        assert req_stale.num_in_flight_tokens == 0
        assert req_stale.num_stale_output_tokens == 0
        assert req_prefill.num_in_flight_tokens == 0
        assert req_spec.num_in_flight_tokens == 0

    def test_update_from_output_with_routed_experts_mid_burst_stop_slices_head(
            self):
        """When check_stop trims new_token_ids mid-burst (e.g. 2 kept out of 4
        sampled), out.routed_experts must receive the head [req_offset : req_offset + 2]
        rather than the tail [end - 2 : end] sliced by upstream vLLM."""
        import numpy as np
        from vllm.v1.core.sched.scheduler import Scheduler

        req = _make_request(num_computed_tokens=10)
        req.num_in_flight_tokens = 1
        req.num_stale_output_tokens = 0

        # 4 sampled tokens for req1 (rows 0..3): each row has distinct values
        routing_data = np.arange(4 * 2 * 2, dtype=np.int32).reshape(4, 2, 2)

        def fake_orig_update(sched_self, sched_out, runner_out, *args,
                             **kwargs):
            # Simulate upstream vLLM slicing the tail [end - len(new_token_ids) : end]
            end = sched_out.num_scheduled_tokens["req1"]
            out = mock.MagicMock()
            out.request_id = "req1"
            out.new_token_ids = [10, 20]  # trimmed from 4 to 2 by check_stop
            out.routed_experts = runner_out.routed_experts.routing_data[end -
                                                                        2:end]
            engine_outputs = mock.MagicMock()
            engine_outputs.outputs = [out]
            return {0: engine_outputs}

        with mock.patch.object(Scheduler,
                               "update_from_output",
                               new=fake_orig_update):
            patch_vllm_scheduler_for_continue_decode()
            scheduler = _make_scheduler(Scheduler)
            scheduler.enable_return_routed_experts = True
            scheduler.requests = {"req1": req}

            scheduler_output = mock.MagicMock()
            scheduler_output.num_scheduled_tokens = {"req1": 1}
            scheduler_output.scheduled_cached_reqs = mock.MagicMock(
                req_ids=["req1"], num_output_tokens=[3])
            scheduler_output.scheduled_spec_decode_tokens = {}

            model_runner_output = mock.MagicMock()
            model_runner_output.routed_experts = mock.MagicMock(
                routing_data=routing_data)
            model_runner_output.req_ids = ["req1"]
            model_runner_output.sampled_token_ids = [[10, 20, 30, 40]]
            model_runner_output.req_id_to_index = {"req1": 0}

            outputs = Scheduler.update_from_output(scheduler, scheduler_output,
                                                   model_runner_output)

        out = outputs[0].outputs[0]
        np.testing.assert_array_equal(out.routed_experts, routing_data[:2])
        assert scheduler_output.num_scheduled_tokens == {"req1": 1}
