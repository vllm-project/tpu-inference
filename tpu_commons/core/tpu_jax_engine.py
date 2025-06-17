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

import tpu_commons.runner.tpu_jax_runner_v2
from vllm.v1.request import Request
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput


from jetstream.core import config_lib
from jetstream.engine import engine_api
from jetstream.engine import token_utils
from jetstream.engine import tokenizer_api
from jetstream.engine.tokenizer_pb2 import TokenizerParameters
from jetstream.engine.tokenizer_pb2 import TokenizerType

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


# TODO(yuyanpeng): Should import ExistingPrefix from jetstream.engine.engine_api
@struct.dataclass
class ExistingPrefix:
  """Represents a prefix that has already been processed.

  Attributes:
    cache: The kv-cache for the prefix get from model params cache.
    common_prefix_tokens: The tokens that have already been processed without padding.
  """

  cache: Any
  common_prefix_tokens: jax.Array


class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  TODO: evaluate the need for this given the restructured pyconfig.py
  """

  def __init__(self, keys):
    # self.keys = keys
    self.__dict__["keys"] = keys

  def __getattr__(self, attr):
    if attr not in self.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return self.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return self.keys


class JaxEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, vllm_executor):
    self.model_runner = vllm_executor.driver_worker.model_runner
    self.req_id_to_req = {}
    # self.config = config

  # Public non-JIT prefill method that updates page state
  def prefill(
      self,  # pytype: disable=signature-mismatch
      *,
      vllm_req_data: Optional[Request] = None,
  ) -> Tuple[Prefix, ModelRunnerOutput]:
    if self.req_id_to_req.get(vllm_req_data.req_id) == None:
      self.req_id_to_req[vllm_req_data.req_id] = vllm_req_data
    input_batch = self.model_runner.input_batch
    inputs = self.model_runner._prepare_prefill([vllm_req_data])
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

      index = input_batch.req_id_to_index[seq.req_id]
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
      prompt_logprobs_dict[seq.req_id] = None

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
      self.req_id_to_req[req_id] 
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
    return int(self.config.per_device_batch_size * jax.device_count())

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


def set_engine_vars_from_base_engine(
    engine: MaxEngine,
    base_engine: MaxEngine,
    rng: PRNGKeyType,
):
  """Set internal vars from base_engine, which has already loaded the checkpoint and has sharding,
  mesh, and kv cache related vars set.
  """
  if base_engine.model.quant:
    engine.model.quant.quant_mode = base_engine.model.quant.quant_mode
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations(engine.model, engine.config, rng, engine.mesh)  # pylint: disable=protected-access
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine.mesh, x),
      engine.kv_cache_annotations,  # pylint: disable=protected-access
  )


def create_engine_from_config_flags(
    maxengine_config_filepath, batch_size, max_prefill_predict_length, max_target_length, args_str
):
  """Create new MaxEngine instance with given batch_size, prefill and target lengths, and any config
  params provided through `args_str`.
  """
  # batch and cache related
  args = {
      "scan_layers": "false",
      "async_checkpointing": "false",
      "ici_fsdp_parallelism": "1",
      "ici_autoregressive_parallelism": "1",
      "ici_tensor_parallelism": "-1",
      "weight_dtype": "bfloat16",
      "attention": "dot_product",
      "max_prefill_predict_length": f"{max_prefill_predict_length}",
      "max_target_length": f"{max_target_length}",
      "per_device_batch_size": f"{batch_size}",
  }

  print(f"Command line args: {args_str}")
  cmd_args = args_str.split(" ")
  for cmd_arg in cmd_args:
    if cmd_arg:
      k, v = cmd_arg.split("=")
      args[k.strip()] = v.strip()
  assert "load_parameters_path" in args, "load_parameters_path must be defined"
  if maxengine_config_filepath is None:
    maxengine_config_filepath = os.path.join(PKG_DIR, "configs", "base.yml")
  updated_args = [os.path.join(PKG_DIR, "maxengine_server.py"), maxengine_config_filepath]
  for k, v in args.items():
    option = f"{k}={v}"
    updated_args.append(option)
  print(f"Invoking maxengine with args:\n \t{updated_args}")
  cfg = pyconfig.initialize(updated_args)
  engine = MaxEngine(cfg)
  return engine