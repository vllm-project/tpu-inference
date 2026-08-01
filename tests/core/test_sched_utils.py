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

    originals = (Scheduler._update_request_with_output, Scheduler.__init__,
                 AsyncScheduler._update_request_with_output)
    yield
    (Scheduler._update_request_with_output, Scheduler.__init__,
     AsyncScheduler._update_request_with_output) = originals
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
