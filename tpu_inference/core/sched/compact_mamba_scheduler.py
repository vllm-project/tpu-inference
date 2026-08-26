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
"""Scheduler entry points that install compact Mamba pools in EngineCore."""

from typing import Any

from tpu_inference.core.compact_mamba_pool import (
    install_compact_mamba_pool,
    mamba_cache_positions,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler


def _split_at_next_mamba_cached_position(
    request: Any,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    cached_positions: frozenset[int] | None = None,
    *,
    num_computed_tokens: int | None = None,
) -> int:
    """Cap a prefill step at the next selected Mamba prefix boundary.

    ``cached_positions`` is the fully-composed eligible set from
    :func:`mamba_cache_positions` (auto over fixed, mid-placeholder positions
    already dropped), so this only finds the nearest boundary ahead of the
    computed prefix. ``None`` -> native chunking (no forced boundary).
    """
    if cached_positions is None or num_new_tokens <= 0:
        return num_new_tokens

    if num_computed_tokens is None:
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_local_computed_tokens +
                               num_external_computed_tokens)

    next_position = min(
        (position
         for position in cached_positions if position > num_computed_tokens),
        default=None,
    )
    if (next_position is not None
            and next_position < num_computed_tokens + num_new_tokens):
        return next_position - num_computed_tokens
    return num_new_tokens


class _SelectedMambaPositionSchedulerMixin:
    """Adds selected-position boundaries to vLLM's align-mode splitting."""

    def _try_schedule_encoder_inputs(
        self,
        request,
        num_computed_tokens,
        num_new_tokens,
        encoder_compute_budget,
        shift_computed_tokens=0,
    ):
        # Native vLLM schedules encoder work before applying its Mamba split.
        # Cap the range first so encoder inputs beyond the selected boundary are
        # not computed early or charged to this step's encoder budget.
        num_new_tokens = _split_at_next_mamba_cached_position(
            request,
            num_new_tokens,
            num_computed_tokens=num_computed_tokens,
            cached_positions=mamba_cache_positions(self, request),
        )
        return super()._try_schedule_encoder_inputs(
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens,
        )

    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        num_new_tokens = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        return _split_at_next_mamba_cached_position(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
            cached_positions=mamba_cache_positions(self, request),
        )


class CompactMambaScheduler(_SelectedMambaPositionSchedulerMixin, Scheduler):
    """Standard scheduler with TPU compact Mamba pool support installed."""

    def __init__(self, *args, **kwargs) -> None:
        install_compact_mamba_pool()
        super().__init__(*args, **kwargs)


class CompactMambaAsyncScheduler(_SelectedMambaPositionSchedulerMixin,
                                 AsyncScheduler):
    """Async scheduler with TPU compact Mamba pool support installed."""

    def __init__(self, *args, **kwargs) -> None:
        install_compact_mamba_pool()
        super().__init__(*args, **kwargs)
