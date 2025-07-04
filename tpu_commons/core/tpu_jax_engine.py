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

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import jax
from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput 
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tpu_commons.core.jetstream_commons.engine import engine_api
from tpu_commons.runner.jax.input_batch_jax import CachedRequestState

warnings.simplefilter("ignore", category=FutureWarning)
DecodeState = Any
Prefix = Any
PackedPrefix = Any
Params = Any
PRNGKeyType = Any
logger = init_logger(__name__)


class JaxEngine(engine_api.Engine):
    """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

    def __init__(self, vllm_config, kv_cache_manager, vllm_executor):
        self.model_runner = vllm_executor.driver_worker.model_runner
        self.kv_cache_manager = kv_cache_manager
        self.vllm_config = vllm_config

    def get_new_block_ids(self, vllm_request: Request):
        computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(
            vllm_request)
        _ = self.kv_cache_manager.allocate_slots(
            vllm_request,
            vllm_request.num_tokens,
            new_computed_blocks=computed_blocks)
        req_id = vllm_request.request_id
        return self.kv_cache_manager.get_block_ids(req_id)

    # Public non-JIT prefill method that updates page state
    def prefill(
        self,  # pytype: disable=signature-mismatch
        *,
        vllm_request: Optional[Request] = None,
    ) -> Tuple[Prefix, ModelRunnerOutput]:
        req_id = vllm_request.request_id
        new_block_ids = self.get_new_block_ids(vllm_request)
        request = NewRequestData.from_request(vllm_request, new_block_ids)

        num_scheduled_tokens = {req_id: vllm_request.num_tokens}
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=[],
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=vllm_request.num_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        metadata, runner_output = self.model_runner._execute_model(scheduler_output)

        num_scheduled_tokens_per_req = np.array([vllm_request.num_tokens], dtype=np.int32)
        kv_cache_slices = self.model_runner.get_kv_cache_for_requests(
            [req_id], metadata.kv_cache_write_indices, num_scheduled_tokens_per_req
        )

        kv_caches_slice = kv_cache_slices[req_id]

        logger.info(f"Prefill result: {runner_output}")
        new_token_ids = runner_output.sampled_token_ids[
            runner_output.req_id_to_index[req_id]]
        vllm_request.append_output_token_ids(new_token_ids)
        vllm_request.num_computed_tokens = vllm_request.num_prompt_tokens + 1
        vllm_request.num_cached_tokens = vllm_request.num_prompt_tokens + 1
        vllm_request.status = RequestStatus.RUNNING

        prefix = {
            "cache": kv_caches_slice,
            "request": vllm_request,
        }
        logger.info(f"prefill done: {runner_output} \nfor {vllm_request} with {vllm_request.num_tokens} tokens")
        return prefix, runner_output

    def generate(self, requests: dict[str, Request]) -> Tuple[ModelRunnerOutput, Any]:
        """Public API for generate that updates page state outside JIT."""
        # Filter out requests that are already finished to prevent
        # them from being scheduled again.
        active_requests = {
            req_id: req
            for req_id, req in requests.items()
            if not req.is_finished()
        }

        # If there are no active requests, return an empty output.
        if not active_requests:
            logger.warning("No active requests!")
            return EMPTY_MODEL_RUNNER_OUTPUT, []

        cached_reqs: list[CachedRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}
        req_to_new_block_ids = {}
        for request_id, request in active_requests.items():
            new_blocks = self.kv_cache_manager.allocate_slots(request, 1)
            req_to_new_block_ids[
                request_id] = new_blocks.get_block_ids()
            num_computed_tokens = request.num_computed_tokens
            new_token_ids = request.all_token_ids[
                num_computed_tokens:num_computed_tokens + 1]
            req_data = CachedRequestData.from_request(
                request, False, new_token_ids,
                req_to_new_block_ids[request_id])
            logger.info(f"Prepare generate: {req_data}")

            cached_reqs.append(req_data)
            num_scheduled_tokens[request_id] = 1

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=len(active_requests),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        _, runner_output = self.model_runner._execute_model(scheduler_output)

        sampled_token_ids = runner_output.sampled_token_ids
        reqs_to_remove: list[str] = []
        for req_id, request in active_requests.items():
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
                    reqs_to_remove.append(req_id)
                    # Stop processing more tokens for this request in this step.
                    break
            request.num_computed_tokens += num_appended
            request.num_cached_tokens += num_appended
        logger.info(f"generate done: {runner_output}; req to remove: {reqs_to_remove}")
        return runner_output, reqs_to_remove

    def insert(self, kv_cache: list[jax.Array]) -> None:
        """Non-JIT wrapper for inserting prefill cache."""
        pass

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
