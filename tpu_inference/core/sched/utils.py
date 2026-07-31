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

DEFAULT_MAX_DECODE_STEPS = 10
MULTI_TOKEN_LOOKAHEAD_CONFIG = "multi_token_decode_lookahead"


def _get_multi_token_lookahead(vllm_config) -> int:
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    if MULTI_TOKEN_LOOKAHEAD_CONFIG in additional_config:
        lookahead = int(additional_config[MULTI_TOKEN_LOOKAHEAD_CONFIG])
        if lookahead < 0:
            raise ValueError(
                f"{MULTI_TOKEN_LOOKAHEAD_CONFIG} must be non-negative")
        return lookahead
    if additional_config.get("enable_continue_decode", False):
        max_decode_steps = int(
            additional_config.get("max_decode_steps",
                                  DEFAULT_MAX_DECODE_STEPS))
        return max(0, max_decode_steps - 1)
    return 0


def _is_multi_token_decode_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    return (MULTI_TOKEN_LOOKAHEAD_CONFIG in additional_config
            or bool(additional_config.get("enable_continue_decode", False)))


def patch_vllm_scheduler_for_multi_token_decode() -> None:
    """Patch vLLM scheduler accounting for multi-token model outputs.

    The host schedules one decode position while the model runner may return N
    tokens. The patch reserves a configured lookahead and reconciles the N - 1
    additional tokens after stop-token processing.

    This function applies three patches:
    1. patched_init: Forces KVCacheManager to reserve the configured lookahead
       during schedule().
    2. patched_update_base: Reconciles request.num_computed_tokens on host by
       adding the extra (N - 1) tokens generated on-device once model output returns.
    3. patched_async_update_request_with_output: Pre-compensates in-flight
       num_output_placeholders in AsyncScheduler by adding (N - 1) before
       subtracting N, preventing placeholder underflow in async mode.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(Scheduler, "_multi_token_decode_patched", False):
        original_update_base = Scheduler._update_request_with_output

        def patched_update_base(scheduler_self, request, new_token_ids):
            res_token_ids, stopped = original_update_base(
                scheduler_self, request, new_token_ids)

            if getattr(scheduler_self, "_multi_token_decode_enabled", False):
                diff = len(res_token_ids) - 1
                if diff > 0:
                    request.num_computed_tokens += diff

            return res_token_ids, stopped

        Scheduler._update_request_with_output = patched_update_base

        original_init = Scheduler.__init__

        def patched_init(scheduler_self, vllm_config, *args, **kwargs):
            original_init(scheduler_self, vllm_config, *args, **kwargs)

            scheduler_self._multi_token_decode_enabled = \
                _is_multi_token_decode_enabled(vllm_config)
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens,
                _get_multi_token_lookahead(vllm_config),
            )

        Scheduler.__init__ = patched_init

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if not getattr(AsyncScheduler, "_multi_token_decode_patched", False):
        original_async_update_req = AsyncScheduler._update_request_with_output

        def patched_async_update_request_with_output(scheduler_self, request,
                                                     new_token_ids):
            if (getattr(scheduler_self, "_multi_token_decode_enabled", False)
                    and len(new_token_ids) > 1):
                request.num_output_placeholders += (len(new_token_ids) - 1)
            return original_async_update_req(scheduler_self, request,
                                             new_token_ids)

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._multi_token_decode_patched = True
        AsyncScheduler._continue_decode_patched = True

    Scheduler._multi_token_decode_patched = True
    Scheduler._continue_decode_patched = True


def patch_vllm_scheduler_for_continue_decode() -> None:
    """Compatibility alias for existing continue-decode callers."""
    patch_vllm_scheduler_for_multi_token_decode()
