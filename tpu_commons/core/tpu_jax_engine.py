# Copyright 2024 Google LLC
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
"""Implementation of Engine API for MaxText."""

import threading
import warnings
from typing import Any, Tuple

import jax
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tpu_commons.runner.utils import LatencyTracker

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any
logger = init_logger(__name__)


class JaxEngine():
    """The computational core of the disaggregated inference engine.

    The class is _not_ thread safe. The caller is responsible for sycnrhonization.
    """

    def __init__(self, vllm_config, kv_cache_config, vllm_executor):
        self.model_runner = vllm_executor.driver_worker.model_runner
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=vllm_config.scheduler_config.max_model_len,
            enable_caching=False,
        )
        self.vllm_config = vllm_config
        self._max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Requests we are already processing.
        self._requests: list[Request] = []
        # Newly added requests.
        self._new_requests: list[Request] = []
        self._completed_requests: list[str] = []
        self._request_map: dict[str, Request] = {}
        self._pending_num_prefill_tokens = 0
        self._kv_cache_manager_lock = threading.Lock()

    def get_new_block_ids(self, vllm_request: Request,
                          num_tokens: int) -> tuple[list[int], ...]:
        with self._kv_cache_manager_lock:
            computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(
                vllm_request)
            _ = self.kv_cache_manager.allocate_slots(
                vllm_request, num_tokens, new_computed_blocks=computed_blocks)
            req_id = vllm_request.request_id
            return self.kv_cache_manager.get_block_ids(req_id)

    def get_block_ids(self, req_id: str) -> tuple[list[int], ...]:
        with self._kv_cache_manager_lock:
            return self.kv_cache_manager.get_block_ids(req_id)

    def has_more_capacity(self):
        """Returns True if we still have room for more requests.

        Most likely prefill will be gated by the number of tokens; generate will
        be gated by the number of requests.
        """
        return (self._pending_num_prefill_tokens < self._max_num_tokens
                and len(self._requests) + len(self._new_requests)
                < self._max_num_reqs and self.model_runner.input_batch.num_reqs
                < self.model_runner.max_num_reqs)

    def dump_stats(self) -> str:
        return (
            f"#prefill_tokens={self._pending_num_prefill_tokens},"
            f"#reqs={len(self._requests)}, #new_reqs={len(self._new_requests)},"
            f"has_more_cacacity={self.has_more_capacity()},"
            f"input_batch_size={self.model_runner.input_batch.num_reqs};"
            f"input_batch={self.model_runner.input_batch.req_id_to_index}")

    def is_prefill_idle(self) -> bool:
        return (self._pending_num_prefill_tokens <= 0
                and self.model_runner.input_batch.num_reqs <= 0)

    def is_generate_idle(self) -> bool:
        return (len(self._requests) + len(self._new_requests) <= 0
                and self.model_runner.input_batch.num_reqs <= 0)

    def add_request(self, req: Request, num_tokens: int):
        self._request_map[req.request_id] = req
        self._new_requests.append(req)
        self._pending_num_prefill_tokens += num_tokens

    def _schedule_prefill(self) -> SchedulerOutput:
        """Schedule the next batch to be processed.

        We currently prioritize oldest requests in the queue.
        """
        capacity_left = self._max_num_tokens
        cached_reqs = []
        num_scheduled_tokens: dict[str, int] = {}
        for req in self._requests:
            assert capacity_left > 0

            num_tokens_to_schedule = min(
                capacity_left, req.num_tokens - req.num_computed_tokens)
            new_block_ids = self.get_new_block_ids(req, num_tokens_to_schedule)

            new_token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens +
                                              num_tokens_to_schedule]
            req_data = CachedRequestData.from_request(req, False,
                                                      new_token_ids,
                                                      new_block_ids)
            cached_reqs.append(req_data)
            capacity_left -= num_tokens_to_schedule
            num_scheduled_tokens[req.request_id] = num_tokens_to_schedule

        new_reqs = []
        scheduled_new_reqs_list: set[Request] = set()
        for req in self._new_requests:
            assert capacity_left > 0

            num_tokens_to_schedule = min(
                capacity_left, req.num_tokens - req.num_computed_tokens)

            new_block_ids = self.get_new_block_ids(req, num_tokens_to_schedule)
            req_data = NewRequestData.from_request(req, new_block_ids)
            new_reqs.append(req_data)
            capacity_left -= num_tokens_to_schedule
            num_scheduled_tokens[req.request_id] = num_tokens_to_schedule
            scheduled_new_reqs_list.add(req)

        self._move_newly_scheduled_to_running(scheduled_new_reqs_list)

        return SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=(self._max_num_tokens - capacity_left),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(self._completed_requests),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

    def _move_newly_scheduled_to_running(self, scheduled_reqs: set[Request]):
        """Moves scheduled requests from _new_requests to _requests."""
        if not scheduled_reqs:
            return
        self._requests.extend(scheduled_reqs)
        self._new_requests = [
            r for r in self._new_requests if r not in scheduled_reqs
        ]

    # Public non-JIT prefill method that updates page state
    def prefill(self) -> Tuple[dict[str, list[jax.Array]], ModelRunnerOutput]:
        scheduler_output = self._schedule_prefill()
        self._pending_num_prefill_tokens -= scheduler_output.total_num_scheduled_tokens
        self._completed_requests.clear()

        logger.info(f"Scheduled output: {scheduler_output}")

        _, runner_output = self.model_runner._execute_model(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens <= 0:
            logger.warning("No active requests!")
            return {}, EMPTY_MODEL_RUNNER_OUTPUT

        logger.debug(f"Prefill result: {runner_output}")

        num_tokens_scheduled: list[int] = []
        kv_cache_map: dict[str, list[jax.Array]] = {}
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items(
        ):
            req = self._request_map[req_id]
            req.num_computed_tokens += num_tokens
            req.num_cached_tokens += num_tokens
            req.status = RequestStatus.RUNNING

            prefill_done = req.num_computed_tokens == req.num_prompt_tokens
            num_tokens_scheduled.append(num_tokens)

            if prefill_done:
                new_token_ids = runner_output.sampled_token_ids[
                    runner_output.req_id_to_index[req_id]]
                req.append_output_token_ids(new_token_ids)
                req.num_computed_tokens += 1
                req.num_cached_tokens += 1

                block_ids = self.get_block_ids(req_id)
                with LatencyTracker(f"ExtractKVCache-{len(block_ids[0])}"):
                    # Assume one KV cache group for now.
                    kv_cache_map[
                        req_id] = self.model_runner.get_kv_cache_for_block_ids(
                            block_ids[0])
                logger.debug(
                    f"prefill done: for {req_id} with {num_tokens} tokens")
                self._completed_requests.append(req_id)
                self._request_map.pop(req_id)

        self._requests = [
            r for r in self._requests if r.request_id in self._request_map
        ]

        return kv_cache_map, runner_output

    def _schedule_generate(self) -> SchedulerOutput:
        # Filter out requests that are already finished to prevent
        # them from being scheduled again.
        self._requests.extend(self._new_requests)
        self._new_requests.clear()

        active_requests = {
            req.request_id: req
            for req in self._requests if not req.is_finished()
        }
        logger.debug(
            f"scheduling generation... #active_reqs={len(active_requests)}")

        cached_reqs: list[CachedRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}
        req_to_new_block_ids = {}
        for request_id, request in active_requests.items():
            with self._kv_cache_manager_lock:
                new_blocks = self.kv_cache_manager.allocate_slots(request, 1)
            req_to_new_block_ids[request_id] = new_blocks.get_block_ids()
            num_computed_tokens = request.num_computed_tokens
            new_token_ids = request.all_token_ids[
                num_computed_tokens:num_computed_tokens + 1]
            req_data = CachedRequestData.from_request(
                request, False, new_token_ids,
                req_to_new_block_ids[request_id])

            cached_reqs.append(req_data)
            num_scheduled_tokens[request_id] = 1

        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=len(active_requests),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(self._completed_requests),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

    def generate(self) -> ModelRunnerOutput:
        """Public API for generate that updates page state outside JIT."""
        scheduler_output = self._schedule_generate()
        self._completed_requests.clear()

        _, runner_output = self.model_runner._execute_model(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens <= 0:
            logger.warning("No active requests!")
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = runner_output.sampled_token_ids
        for req_id in scheduler_output.num_scheduled_tokens.keys():
            request = self._request_map[req_id]
            req_index = runner_output.req_id_to_index[req_id]
            new_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []
            num_appended = 0
            for output_token_id in new_token_ids:
                request.append_output_token_ids(output_token_id)
                num_appended += 1
                stopped = check_stop(
                    request, self.vllm_config.scheduler_config.max_model_len)
                if stopped:
                    # The request is now finished. Mark it for removal.
                    self._completed_requests.append(req_id)
                    self._request_map.pop(req_id)
                    # Stop processing more tokens for this request in this step.
                    break
            request.num_computed_tokens += num_appended
            request.num_cached_tokens += num_appended
        self._requests = [
            r for r in self._requests if r.request_id in self._request_map
        ]
        logger.debug(
            f"generate done: {runner_output}; req to remove: {self._completed_requests}"
        )
        return runner_output

    def free_request(self, request: Request):
        """Frees the KV-cache blocks allocated for a request."""
        with self._kv_cache_manager_lock:
            self.kv_cache_manager.free(request)

    def get_prefix_destination_sharding(self) -> Any:
        return {
            "cache": self.model_runner.outputs_sharding,
            "next_tokens": self.model_runner.outputs_sharding,
            "running_indices": self.model_runner.outputs_sharding,
            "output_token_indices": self.model_runner.outputs_sharding,
            "kv_cache_write_indices": self.model_runner.outputs_sharding,
        }

    @property
    def max_concurrent_decodes(self) -> int:
        """Free slots."""
        return self.model_runner.input_batch.max_num_reqs
