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

from collections import defaultdict
from typing import Any, List, Optional, Tuple, Callable
import functools
import os.path
import uuid
import warnings

from jax.experimental.layout import DeviceLocalLayout as DLL
from jax.experimental.layout import Layout
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp

# from flax import linen as nn
# from flax import struct
# from flax.linen import partitioning as nn_partitioning
# import flax

# from tpu_commons.runner.tpu_jax_runner_v2
from vllm.v1.request import Request
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.logger import init_logger

from tpu_commons.core.jetstream_commons.core import config_lib
from tpu_commons.core.jetstream_commons.engine import engine_api
from tpu_commons.core.jetstream_commons.engine import token_utils
from tpu_commons.core.jetstream_commons.engine import tokenizer_api
from tpu_commons.core.jetstream_commons.engine.tokenizer_pb2 import TokenizerParameters
from tpu_commons.core.jetstream_commons.engine.tokenizer_pb2 import TokenizerType

from tpu_commons.worker.input_batch_jax import CachedRequestState, InputBatch

# from MaxText import inference_utils
# from MaxText import max_utils
# from MaxText import maxtext_utils
# from MaxText import pyconfig
# from MaxText.common_types import MODEL_MODE_PREFILL, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE
# from MaxText.globals import PKG_DIR
# from MaxText.inference.page_manager import PageManager, PageState
# from MaxText.layers import models, quantizations
# from MaxText.utils import lora_utils


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
    self.req_id_to_req = {}
    self.kv_cache_manager = kv_cache_manager
    # self.config = config

  # Public non-JIT prefill method that updates page state
  def prefill(
      self,  # pytype: disable=signature-mismatch
      *,
      vllm_req_data: Optional[Request] = None,
  ) -> Tuple[Prefix, ModelRunnerOutput]:
    computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(vllm_req_data)
    new_blocks = self.kv_cache_manager.allocate_slots(vllm_req_data, 
                                                      vllm_req_data.num_tokens,
                                                      new_computed_blocks=computed_blocks)
    new_block_ids = self.kv_cache_manager.get_block_ids(vllm_req_data.request_id)
    request = NewRequestData.from_request(vllm_req_data, new_block_ids)
    logger.info("Finished allocating blocks for prefill req %s", request.__dict__)
    if self.req_id_to_req.get(request.req_id) == None:
      self.req_id_to_req[request.req_id] = vllm_req_data
    input_batch = self.model_runner.input_batch
    request_to_add = CachedRequestState(
      req_id=request.req_id,
      prompt_token_ids=request.prompt_token_ids,
      mm_inputs=request.mm_inputs,
      mm_positions=request.mm_positions,
      sampling_params=request.sampling_params,
      generator=None,
      block_ids=request.block_ids,
      num_computed_tokens=request.num_computed_tokens,
      output_token_ids=[],
      lora_request=request.lora_request,
    )
    input_batch.add_request(request_to_add, None)
    inputs = self.model_runner._prepare_prefill([request])
    if inputs is not None:
      model_inputs, (running_indices, output_token_indices) = inputs
      # TODO change the model interface such that prefill returns 
      kv_caches, next_tokens, logits = self.model_runner.model_fn(*model_inputs)
      self.model_runner.output_cache = \
      self.model_runner.write_outputs(self.model_runner.output_cache,
                                      next_tokens,
                                      running_indices,
                                      output_token_indices)

      prompt_logprobs_dict = {}
      running_indices = []
      output_token_indices = []

      # NOTE(pooyam): Unfinished prefills should not return anything to vLLM scheduler.
      # if not self._is_generating_new_token(scheduler_output, vllm_req_data):
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

      # TODO(pooyam): Figure out why all three of `num_tokens`, `num_prompt_tokens`, and 'num_computed_tokens_cpu` exist.
      prompt_logprobs_dict[request.req_id] = None

    # TODO(pooyam): device-to-host transfer step by step is inefficient. Should we execute for longer decoding steps?
    # Not sure yet how that would work with vLLM engine that calls `execute_model`
    sampled_token_ids = [[] for _ in range(input_batch.num_reqs)]

    if running_indices:
      outputs = self.model_runner.output_cache.at[running_indices,
                                      output_token_indices].get()
      outputs = jax.device_get(outputs).tolist()
      # NOTE(pooyam): vLLM scheduler reads via `sampled_token_ids[req_index]` where sampled_token_ids is `list[list[int]]`.
      # Not sure why they didn't make it dictionary because not all running sequences will be scheduled at each iter and
      # we are sending pointless [] as the output of such requests. I think it's possible to optimize this if we just send a dict.
      for running_index, output in zip(running_indices, outputs):
          sampled_token_ids[running_index] = [output]
    runner_output = ModelRunnerOutput(
      req_ids=input_batch.req_ids,
      req_id_to_index=input_batch.req_id_to_index,
      prompt_logprobs_dict=prompt_logprobs_dict,
      logprobs=None,
      spec_token_ids=None,
      sampled_token_ids=sampled_token_ids,
    )
    prefix = {
      "seq": vllm_req_data,
      "cache": kv_caches,
      "next_tokens": next_tokens,
      "running_indices": running_indices,
      "output_token_indices": output_token_indices, 
      "attention_metadata": model_inputs[5], #Ask people to structurize this
    }  
    return prefix, runner_output


  def generate(self) -> ModelRunnerOutput:
    """Public API for generate that updates page state outside JIT."""
    input_batch = self.model_runner.input_batch
    scheduled_cached_reqs = [
      CachedRequestData.from_request(self.req_id_to_req[req_id]) 
      for req_id in input_batch.req_id_to_index]
    inputs = self.model_runner._prepare_decode(scheduled_cached_reqs)
    if inputs is not None:
      model_inputs, (running_indices, output_token_indices) = inputs
      self.model_runner.kv_caches, next_tokens, logits = self.model_fn(*model_inputs)
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

        index = input_batch.req_id_to_index[vllm_req_data.request_id]
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
        prompt_logprobs_dict[vllm_req_data.request_id] = None

        # TODO(pooyam): device-to-host transfer step by step is inefficient. Should we execute for longer decoding steps?
        # Not sure yet how that would work with vLLM engine that calls `execute_model`
      sampled_token_ids = [[] for _ in range(input_batch.num_reqs)]

      if running_indices:
          outputs = self.model_runner.output_cache.at[running_indices,
                                          output_token_indices].get()
          outputs = jax.device_get(outputs).tolist()
          # NOTE(pooyam): vLLM scheduler reads via `sampled_token_ids[req_index]` where sampled_token_ids is `list[list[int]]`.
          # Not sure why they didn't make it dictionary because not all running sequences will be scheduled at each iter and
          # we are sending pointless [] as the output of such requests. I think it's possible to optimize this if we just send a dict.
          for running_index, output in zip(running_indices, outputs):
              sampled_token_ids[running_index] = [output]

      return ModelRunnerOutput(
          req_ids=input_batch.req_ids,
          req_id_to_index=input_batch.req_id_to_index,
          prompt_logprobs_dict=prompt_logprobs_dict,
          logprobs=None,
          spec_token_ids=None,
          sampled_token_ids=sampled_token_ids,
      )


  def insert(self, prefix: Prefix) -> None:
    """Non-JIT wrapper for inserting prefill cache."""
    slot = prefix["attention_metadata"].kv_cache_write_indices
    prefill_cache = prefix["cache"]
    # kv cache is still full now
    self.model_runner.kv_caches = prefill_cache

  def get_prefix_destination_sharding(self) -> Any:
    return None

    @functools.partial(jax.jit, out_shardings=shardings)
    def initialize():
      return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

    init_state = initialize()
    cache = init_state["cache"]

    def is_lp(k):
      return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

    self.kv_cache_annotations_named = jax.tree_util.tree_map(lambda x: tuple(x.names), cache, is_leaf=is_lp)
    zeroed = max_utils.unbox_logicallypartioned(init_state)
    return zeroed

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

