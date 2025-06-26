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

import jax
from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.outputs import ModelRunnerOutput
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
        # self.config = config
        # input_batch = self.model_runner.input_batch

    # Public non-JIT prefill method that updates page state
    def prefill(
        self,  # pytype: disable=signature-mismatch
        *,
        vllm_request: Optional[Request] = None,
    ) -> Tuple[Prefix, ModelRunnerOutput, Request]:
        computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(
            vllm_request)
        _ = self.kv_cache_manager.allocate_slots(
            vllm_request,
            vllm_request.num_tokens,
            new_computed_blocks=computed_blocks)
        new_block_ids = self.kv_cache_manager.get_block_ids(
            vllm_request.request_id)
        request = NewRequestData.from_request(vllm_request, new_block_ids)
        #assume all tokens will get prefilled.
        request.num_computed_tokens = vllm_request.num_tokens
        input_batch = self.model_runner.input_batch
        request_to_add = CachedRequestState(
            req_id=request.req_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            generator=None,
            block_ids=request.block_ids,
            num_computed_tokens=vllm_request.num_tokens,
            output_token_ids=[],
            lora_request=request.lora_request,
        )
        input_batch.add_request(request_to_add, None)
        # logger.info("added request %s to input batch!!", request.req_id)
        self.model_runner.requests[request.req_id] = request_to_add
        inputs = self.model_runner._prepare_prefill([request])
        if inputs is not None:
            model_inputs, (running_indices, output_token_indices) = inputs
            # TODO change the model interface such that prefill returns
            self.model_runner.kv_caches, next_tokens, logits = self.model_runner.model_fn(
                *model_inputs)
            # logger.info(
            #     "finished model_fn %s; next token %s running_indices %s output_token_indices %s",
            #     request.req_id, next_tokens, running_indices,
            #     output_token_indices)
            # self.model_runner.output_cache = \
            # self.model_runner.write_outputs(self.model_runner.output_cache,
            #                                 next_tokens,
            #                                 running_indices,
            #                                 output_token_indices)

            prompt_logprobs_dict = {}
            running_indices = []
            output_token_indices = []

            # NOTE(pooyam): Unfinished prefills should not return anything to vLLM scheduler.
            # if not self._is_generating_new_token(scheduler_output, vllm_request):
            #     continue

            index = input_batch.req_id_to_index[request.req_id]
            output_token_index = max(
                input_batch.num_computed_tokens_cpu[index] -
                input_batch.num_prompt_tokens[index] + 1, 0)
            running_indices.append(index)
            output_token_indices.append(output_token_index)
            seq_len = max(input_batch.num_prompt_tokens[index],
                          input_batch.num_computed_tokens_cpu[index])
            input_batch.token_ids_cpu[index, seq_len] =\
            input_batch.token_ids_cpu[index, seq_len - 1] + 1  # Dummy
            input_batch.num_computed_tokens_cpu[
                index] = request.num_computed_tokens

            # TODO(pooyam): Figure out why all three of `num_tokens`, `num_prompt_tokens`, and 'num_computed_tokens_cpu` exist.
            prompt_logprobs_dict[request.req_id] = None

        # TODO(pooyam): device-to-host transfer step by step is inefficient. Should we execute for longer decoding steps?
        # Not sure yet how that would work with vLLM engine that calls `execute_model`
        sampled_token_ids = [[] for _ in range(input_batch.max_num_reqs)]
        sampled_token_ids[running_indices[0]] = [int(next_tokens[0])]
        # if running_indices:
        #   outputs = self.model_runner.output_cache.at[running_indices,
        #                                   output_token_indices].get()
        #   outputs = jax.device_get(outputs).tolist()
        #   # NOTE(pooyam): vLLM scheduler reads via `sampled_token_ids[req_index]` where sampled_token_ids is `list[list[int]]`.
        #   # Not sure why they didn't make it dictionary because not all running sequences will be scheduled at each iter and
        #   # we are sending pointless [] as the output of such requests. I think it's possible to optimize this if we just send a dict.
        #   for running_index, output in zip(running_indices, outputs):
        #       sampled_token_ids[running_index] = [output]
        runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index=input_batch.req_id_to_index,
            prompt_logprobs_dict=prompt_logprobs_dict,
            logprobs=None,
            spec_token_ids=None,
            sampled_token_ids=sampled_token_ids,
        )

        prefix = {
            "seq": vllm_request,
            "cache": self.model_runner.kv_caches,
            "next_tokens": next_tokens,
            "running_indices": running_indices,
            "output_token_indices": output_token_indices,
            "attention_metadata":
            model_inputs[4],  #Ask people to structurize this
        }
        new_token_ids = runner_output.sampled_token_ids[
            runner_output.req_id_to_index[vllm_request.request_id]]
        vllm_request.append_output_token_ids(new_token_ids)
        vllm_request.num_computed_tokens = vllm_request.num_prompt_tokens
        vllm_request.num_cached_tokens = vllm_request.num_prompt_tokens
        vllm_request.status = RequestStatus.RUNNING
        return prefix, runner_output, vllm_request

    def generate(self, all_requests) -> Tuple[Any, ModelRunnerOutput, Any]:
        """Public API for generate that updates page state outside JIT."""
        input_batch = self.model_runner.input_batch
        req_to_new_block_ids = {}
        cached_reqs = [
            all_requests[request_id]
            for request_id in input_batch.req_id_to_index
            if request_id in all_requests
        ]
        scheduled_cached_reqs = []
        for request in cached_reqs:
            # logger.info("Converting scheduling request %s to cached request", request.request_id)
            new_blocks = self.kv_cache_manager.allocate_slots(request, 1)
            req_to_new_block_ids[
                request.request_id] = new_blocks.get_block_ids()
            num_computed_tokens = request.num_computed_tokens
            new_token_ids = request.all_token_ids[
                num_computed_tokens:num_computed_tokens + 1]
            req_data = CachedRequestData.from_request(
                request, False, new_token_ids,
                req_to_new_block_ids[request.request_id])
            scheduled_cached_reqs.append(req_data)
            req_state = self.model_runner.requests[request.request_id]
            req_state.num_computed_tokens = req_data.num_computed_tokens
            for block_ids, new_block_ids in zip(req_state.block_ids,
                                                req_data.new_block_ids,
                                                strict=True):
                block_ids.extend(new_block_ids)
            req_index = input_batch.req_id_to_index.get(request.request_id)
            # if req_index is None:
            #     # The request is not in the persistent batch.
            #     # The request was either preempted and resumed later, or was not
            #     # scheduled in the previous step and needs to be added again.
            #     req_ids_to_add.append(request.request_id)
            #     continue
            input_batch.num_computed_tokens_cpu[
                req_index] = req_data.num_computed_tokens
            input_batch.block_table.append_row(req_data.new_block_ids,
                                               req_index)
        inputs = self.model_runner._prepare_decode(scheduled_cached_reqs)
        if inputs is not None:
            model_inputs, (running_indices, output_token_indices) = inputs
            self.model_runner.kv_caches, next_tokens, logits = self.model_runner.model_fn(
                *model_inputs)
            # logger.info(
            #     "finished model_fn %s; next token %s running_indices %s output_token_indices %s",
            #     input_batch.req_id_to_index, next_tokens, running_indices,
            #     output_token_indices)
            # logger.info("generated next tokens: %s", next_tokens)
            self.model_runner.output_cache = \
            self.model_runner.write_outputs(self.model_runner.output_cache,
                                            next_tokens,
                                            running_indices,
                                            output_token_indices)

            prompt_logprobs_dict = {}
            running_indices = []
            output_token_indices = []

            for i, seq in enumerate(scheduled_cached_reqs):
                # NOTE(pooyam): Unfinished prefills should not return anything to vLLM scheduler.
                # if not self.model_runner._is_generating_new_token(scheduler_output, seq):
                #     continue
                index = input_batch.req_id_to_index[seq.req_id]
                output_token_index = max(
                    input_batch.num_computed_tokens_cpu[index] -
                    input_batch.num_prompt_tokens[index] + 1, 0)
                running_indices.append(index)
                output_token_indices.append(output_token_index)
                seq_len = max(input_batch.num_prompt_tokens[index],
                              input_batch.num_computed_tokens_cpu[index])
                input_batch.token_ids_cpu[index, seq_len] = \
                input_batch.token_ids_cpu[index, seq_len - 1] + 1  # Dummy

                # TODO(pooyam): Figure out why all three of `num_tokens`, `num_prompt_tokens`, and 'num_computed_tokens_cpu` exist.
                prompt_logprobs_dict[seq.req_id] = None

                # TODO(pooyam): device-to-host transfer step by step is inefficient. Should we execute for longer decoding steps?
                # Not sure yet how that would work with vLLM engine that calls `execute_model`
            sampled_token_ids = [[] for _ in range(input_batch.num_reqs)]

            if running_indices:
                outputs = self.model_runner.output_cache.at[
                    running_indices, output_token_indices].get()
                outputs = jax.device_get(outputs).tolist()
                # NOTE(pooyam): vLLM scheduler reads via `sampled_token_ids[req_index]` where sampled_token_ids is `list[list[int]]`.
                # Not sure why they didn't make it dictionary because not all running sequences will be scheduled at each iter and
                # we are sending pointless [] as the output of such requests. I think it's possible to optimize this if we just send a dict.
                for running_index, output in zip(running_indices, outputs):
                    sampled_token_ids[running_index] = [output]
            reqs_to_remove = []
            for req_id in input_batch.req_id_to_index:
                new_token_ids = sampled_token_ids[
                    input_batch.req_id_to_index[req_id]]
                all_requests[req_id].append_output_token_ids(new_token_ids)
                all_requests[req_id].num_computed_tokens += 1
                all_requests[req_id].num_cached_tokens += 1
                stopped = check_stop(
                    all_requests[req_id],
                    self.vllm_config.model_config.max_model_len)
                if stopped:
                    reqs_to_remove.append(req_id)
            model_output_to_return = ModelRunnerOutput(
                req_ids=input_batch.req_ids,
                req_id_to_index=input_batch.req_id_to_index,
                prompt_logprobs_dict=prompt_logprobs_dict,
                logprobs=None,
                spec_token_ids=None,
                sampled_token_ids=sampled_token_ids,
            )

            return all_requests, model_output_to_return, reqs_to_remove

    def insert(self, prefix: Prefix) -> None:
        """Non-JIT wrapper for inserting prefill cache."""
        pass
        # slot = prefix["attention_metadata"].kv_cache_write_indices
        # prefill_cache = prefix["cache"]
        # kv cache is still full now
        # self.model_runner.kv_caches = prefill_cache

    def get_prefix_destination_sharding(self) -> Any:
        return None

    #     @functools.partial(jax.jit, out_shardings=None)
    #     def initialize():
    #         return jax.tree_util.tree_map(
    #             lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

    #     init_state = initialize()
    #     cache = init_state["cache"]

    #     def is_lp(k):
    #         return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

    #     self.kv_cache_annotations_named = jax.tree_util.tree_map(
    #         lambda x: tuple(x.names), cache, is_leaf=is_lp)
    #     zeroed = max_utils.unbox_logicallypartioned(init_state)
    #     return zeroed

    @property
    def max_concurrent_decodes(self) -> int:
        """Free slots."""
        return self.model_runner.input_batch.max_num_reqs

    @property
    def max_prefill_length(self) -> int:
        """Maximum prefill length."""
        return int(self.config.max_prefill_predict_length)

    @property
    def use_chunked_prefill(self) -> bool:
        """Whether to use chunked prefill."""
        return self.config.use_chunked_prefill

    @property
    def prefill_chunk_size(self) -> int:
        """Prefill chunk size."""
        return int(self.config.prefill_chunk_size)

    @property
    def samples_per_slot(self) -> int:
        """Number of samples per slot."""
        return 1

    @property
    def mesh(self) -> jax.sharding.Mesh:
        return self._mesh

    @property
    def colocated_cpus(self) -> None:
        """CPU devices colocated with the engine's accelerators."""
        raise NotImplementedError
