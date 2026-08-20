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

import inspect
import math
from collections import deque
from collections.abc import Callable, Iterable
from time import monotonic
from typing import Any, Generic, TypeVar

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tpu_inference.runner.diffusion.config import resolve_generation_strategy

_ItemT = TypeVar("_ItemT")


class CohortAdmissionQueue(Generic[_ItemT]):

    def __init__(
        self,
        max_size: int,
        max_wait_ms: float,
        quiet_wait_ms: float = 0.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("Cohort max_size must be positive")
        if not math.isfinite(max_wait_ms) or max_wait_ms <= 0.0:
            raise ValueError("Cohort max_wait_ms must be finite and positive")
        if not math.isfinite(quiet_wait_ms) or quiet_wait_ms < 0.0:
            raise ValueError(
                "Cohort quiet_wait_ms must be finite and non-negative")
        if quiet_wait_ms > max_wait_ms:
            raise ValueError(
                "Cohort quiet_wait_ms must not exceed max_wait_ms")
        self.max_size = max_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.quiet_wait_seconds = quiet_wait_ms / 1000.0
        self._clock = monotonic if clock is None else clock
        self._pending: deque[tuple[_ItemT, float]] = deque()

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def deadline(self) -> float | None:
        if not self._pending:
            return None
        deadline = self._pending[0][1] + self.max_wait_seconds
        if self.quiet_wait_seconds:
            deadline = min(deadline,
                           self._pending[-1][1] + self.quiet_wait_seconds)
        return deadline

    def add(self, item: _ItemT) -> None:
        self._pending.append((item, self._clock()))

    def is_ready(self, now: float | None = None) -> bool:
        if not self._pending:
            return False
        if len(self._pending) >= self.max_size:
            return True
        deadline = self.deadline
        assert deadline is not None
        return (self._clock() if now is None else now) >= deadline

    def drain_ready(self, now: float | None = None) -> list[_ItemT]:
        if not self.is_ready(now):
            return []
        cohort_size = min(len(self._pending), self.max_size)
        return [self._pending.popleft()[0] for _ in range(cohort_size)]

    def discard(self, predicate: Callable[[_ItemT], bool]) -> list[_ItemT]:
        retained: deque[tuple[_ItemT, float]] = deque()
        discarded = []
        while self._pending:
            item = self._pending.popleft()
            if predicate(item[0]):
                discarded.append(item[0])
            else:
                retained.append(item)
        self._pending = retained
        return discarded


class BlockDiffusionCohortScheduler(Scheduler):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        scheduler_kwargs = dict(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        if "hash_block_size" in inspect.signature(Scheduler).parameters:
            scheduler_kwargs["hash_block_size"] = hash_block_size
        super().__init__(**scheduler_kwargs)
        generation_strategy = resolve_generation_strategy(vllm_config)
        assert generation_strategy.diffusion is not None
        max_wait_ms = generation_strategy.diffusion.runtime.cohort_max_wait_ms
        quiet_wait_ms = (
            generation_strategy.diffusion.runtime.cohort_quiet_wait_ms)
        self._strict_waves = (
            generation_strategy.diffusion.runtime.cohort_strict_waves)
        if max_wait_ms <= 0.0:
            raise ValueError(
                "BlockDiffusionCohortScheduler requires positive cohort_max_wait_ms"
            )
        self._cohort = CohortAdmissionQueue[Request](
            max_size=vllm_config.scheduler_config.max_num_seqs,
            max_wait_ms=max_wait_ms,
            quiet_wait_ms=quiet_wait_ms,
        )
        self._cohort_request_ids: set[str] = set()

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            super().add_request(request)
            return
        if request.request_id in self._cohort_request_ids:
            raise ValueError(
                f"Request {request.request_id!r} is already awaiting admission"
            )
        self._cohort.add(request)
        self._cohort_request_ids.add(request.request_id)

    def schedule(self, *args: Any, **kwargs: Any) -> SchedulerOutput:
        base_is_idle = super().get_num_unfinished_requests() == 0
        if not self._strict_waves or base_is_idle:
            for request in self._cohort.drain_ready():
                self._cohort_request_ids.remove(request.request_id)
                super().add_request(request)
        return super().schedule(*args, **kwargs)

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        if request_ids is None:
            normalized_ids = list(self._cohort_request_ids)
            normalized_ids.extend(self.requests)
        else:
            normalized_ids = ([request_ids] if isinstance(request_ids, str)
                              else list(request_ids))
        request_id_set = set(normalized_ids)
        discarded = self._cohort.discard(
            lambda request: request.request_id in request_id_set)
        discarded_ids = {request.request_id for request in discarded}
        self._cohort_request_ids.difference_update(discarded_ids)
        finished_requests = [(request.request_id, request.client_index)
                             for request in discarded]
        admitted_ids = [
            request_id for request_id in normalized_ids
            if request_id not in discarded_ids
        ]
        if admitted_ids:
            finished_requests.extend(super().finish_requests(
                admitted_ids, finished_status))
        return finished_requests

    def get_num_unfinished_requests(self) -> int:
        return super().get_num_unfinished_requests() + len(self._cohort)

    def get_request_counts(self) -> tuple[int, int]:
        running, waiting = super().get_request_counts()
        return running, waiting + len(self._cohort)

    def has_unfinished_requests(self) -> bool:
        return bool(self._cohort) or super().has_unfinished_requests()
