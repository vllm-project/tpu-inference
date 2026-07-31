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

from tpu_inference.core.sched.utils import \
    patch_vllm_scheduler_for_continue_decode


def _make_request(num_computed_tokens=10, num_output_placeholders=1):
    request = mock.MagicMock()
    request.num_computed_tokens = num_computed_tokens
    request.num_output_placeholders = num_output_placeholders
    return request


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

    def test_sync_patch_advances_num_computed_tokens_unless_stale(self):
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler = mock.MagicMock()
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

    def test_async_patch_placeholder_compensation_respects_is_stale(self):
        patch_vllm_scheduler_for_continue_decode()
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        # spec= so the zero-arg super() in the wrapped original resolves
        # (isinstance check); instance attributes read by the base
        # implementation must be set explicitly.
        scheduler = mock.MagicMock(spec=AsyncScheduler)
        scheduler.max_model_len = 1024
        with mock.patch("vllm.v1.core.sched.scheduler.check_stop",
                        return_value=False):
            # Fresh output: pre-compensate placeholders by (N - 1) so the
            # original's -= N lands back at 0 without underflowing.
            request = _make_request(num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=False)
            assert request.num_output_placeholders == 0

            # Stale output: placeholders were zeroed at preemption; neither the
            # pre-compensation nor the original's decrement may touch them.
            request = _make_request(num_output_placeholders=1)
            AsyncScheduler._update_request_with_output(scheduler,
                                                       request, [1, 2, 3],
                                                       is_stale=True)
            assert request.num_output_placeholders == 1
