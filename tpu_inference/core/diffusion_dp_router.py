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

from __future__ import annotations

import inspect
from typing import Any

from tpu_inference.runner.diffusion.config import (GenerationStrategy,
                                                   resolve_generation_strategy)

_PATCH_MARKER = "_tpu_diffusion_round_robin_dp"
_EXPECTED_CLIENT_INIT_PARAMETERS = (
    "self",
    "vllm_config",
    "executor_class",
    "log_stats",
    "client_addresses",
    "client_count",
    "client_index",
)
_EXPECTED_ROUTER_PARAMETERS = ("self", "request")
_EXPECTED_FACTORY_PARAMETERS = (
    "vllm_config",
    "executor_class",
    "log_stats",
    "client_addresses",
    "client_count",
    "client_index",
)


def _require_parameters(callable_obj: Any, expected: tuple[str, ...],
                        name: str) -> None:
    actual = tuple(inspect.signature(callable_obj).parameters)
    if actual != expected:
        raise RuntimeError(
            f"Unsupported vLLM {name} signature: expected {expected}, got "
            f"{actual}")


def _round_robin_enabled(vllm_config: Any) -> bool:
    generation_strategy = resolve_generation_strategy(vllm_config)
    if generation_strategy.strategy is not GenerationStrategy.BLOCK_DIFFUSION:
        return False
    assert generation_strategy.diffusion is not None
    return generation_strategy.diffusion.runtime.cohort_round_robin_dp


def patch_vllm_dp_client_for_block_diffusion_round_robin() -> None:
    """Patch vLLM's DP client with opt-in cohort-block round robin."""
    from vllm.v1.engine import core_client
    from vllm.v1.pool.late_interaction import \
        get_late_interaction_engine_index

    original_client = core_client.DPLBAsyncMPClient
    if getattr(original_client, _PATCH_MARKER, False):
        return

    _require_parameters(original_client.__init__,
                        _EXPECTED_CLIENT_INIT_PARAMETERS,
                        "DPLBAsyncMPClient.__init__")
    _require_parameters(original_client.get_core_engine_for_request,
                        _EXPECTED_ROUTER_PARAMETERS,
                        "DPLBAsyncMPClient.get_core_engine_for_request")
    factory = core_client.EngineCoreClient.make_async_mp_client
    _require_parameters(factory, _EXPECTED_FACTORY_PARAMETERS,
                        "EngineCoreClient.make_async_mp_client")
    factory_impl = inspect.unwrap(factory)
    if factory_impl.__globals__.get(
            "DPLBAsyncMPClient") is not original_client:
        raise RuntimeError(
            "Unsupported vLLM DP client factory: DPLBAsyncMPClient is not "
            "resolved from the core_client module")

    class CohortRoundRobinDPLBAsyncMPClient(original_client):
        _tpu_diffusion_round_robin_dp = True

        def __init__(
            self,
            vllm_config: Any,
            executor_class: Any,
            log_stats: bool,
            client_addresses: dict[str, str] | None = None,
            client_count: int = 1,
            client_index: int = 0,
        ) -> None:
            round_robin_enabled = _round_robin_enabled(vllm_config)
            if round_robin_enabled and client_count != 1:
                raise ValueError(
                    "diffusion cohort_round_robin_dp requires client_count=1")
            cohort_size = 0
            if round_robin_enabled:
                cohort_size = vllm_config.scheduler_config.max_num_seqs
                if type(cohort_size) is not int or cohort_size < 1:
                    raise ValueError(
                        "diffusion cohort_round_robin_dp requires positive "
                        "scheduler_config.max_num_seqs")
            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                client_addresses,
                client_count,
                client_index,
            )
            self._cohort_round_robin_dp_enabled = round_robin_enabled
            self._cohort_round_robin_dp_cohort_size = cohort_size
            self._cohort_round_robin_dp_next_engine_index = \
                self.eng_start_index
            self._cohort_round_robin_dp_active_engine_index = None
            self._cohort_round_robin_dp_requests_in_cohort = 0

        def _start_next_cohort(self) -> int:
            num_engines = len(self.core_engines)
            start_index = (self._cohort_round_robin_dp_next_engine_index %
                           num_engines)
            engine_indexes = {
                engine: index
                for index, engine in enumerate(self.core_engines)
            }
            in_flight_counts = [0] * num_engines
            for engine in self.reqs_in_flight.values():
                engine_index = engine_indexes.get(engine)
                if engine_index is not None:
                    in_flight_counts[engine_index] += 1
            engine_index = min(
                ((start_index + offset) % num_engines
                 for offset in range(num_engines)),
                key=lambda index: (in_flight_counts[index],
                                   (index - start_index) % num_engines),
            )
            self._cohort_round_robin_dp_active_engine_index = engine_index
            self._cohort_round_robin_dp_next_engine_index = (engine_index +
                                                             1) % num_engines
            return engine_index

        def get_core_engine_for_request(self, request: Any) -> Any:
            if not self._cohort_round_robin_dp_enabled:
                return super().get_core_engine_for_request(request)
            if request.data_parallel_rank is not None:
                return super().get_core_engine_for_request(request)
            if get_late_interaction_engine_index(
                    request.pooling_params,
                    len(self.core_engines)) is not None:
                return super().get_core_engine_for_request(request)
            existing_engine = self.reqs_in_flight.get(request.request_id)
            if existing_engine is not None:
                return existing_engine

            if self._cohort_round_robin_dp_requests_in_cohort == 0:
                engine_index = self._start_next_cohort()
            else:
                engine_index = self._cohort_round_robin_dp_active_engine_index
                assert engine_index is not None
            self._cohort_round_robin_dp_requests_in_cohort += 1
            if (self._cohort_round_robin_dp_requests_in_cohort
                    >= self._cohort_round_robin_dp_cohort_size):
                self._cohort_round_robin_dp_requests_in_cohort = 0
                self._cohort_round_robin_dp_active_engine_index = None
            self.lb_engines[engine_index][0] += 1
            chosen_engine = self.core_engines[engine_index]
            self.reqs_in_flight[request.request_id] = chosen_engine
            return chosen_engine

    core_client.DPLBAsyncMPClient = CohortRoundRobinDPLBAsyncMPClient
