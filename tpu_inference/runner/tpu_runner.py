# Copyright 2025 Google LLC
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

import copy
import functools
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import vllm.envs as vllm_envs
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import set_forward_context
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, KVConnectorOutput, LogprobsLists,
                             ModelRunnerOutput)
from vllm.v1.request import Request
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import \
    KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

import tpu_inference.envs as envs
from tpu_inference import utils as common_utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D,
                                                  ShardingAxisName,
                                                  ShardingConfigManager)
from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.layers.jax.sample.sampling import (compute_logprobs,
                                                      gather_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    shard_put, transfer_state_with_mappings)
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.compilation_manager import CompilationManager
from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.kv_cache_manager import KVCacheManager
from tpu_inference.runner.lora_utils import LoraUtils
from tpu_inference.runner.multimodal_manager import MultiModalManager
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager
from tpu_inference.runner.speculative_decoding_manager import (
    SpecDecodeMetadata, SpeculativeDecodingManager)
from tpu_inference.runner.structured_decoding_manager import \
    StructuredDecodingManager
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.utils import (device_array, make_optimized_mesh,
                                 time_function, to_jax_dtype, to_torch_dtype)

logger = init_logger(__name__)

logging.getLogger("torchax.tensor").setLevel(logging.ERROR)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8

DUMMY_METADATA = AttentionMetadata(
    input_positions=[],
    block_tables=[],
    request_distribution=[0, 0, 0],
)


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner.

    This class acts as a wrapper around the standard ModelRunnerOutput. Its
    primary purpose is to hold references to data still on the TPU device
    (like the `next_tokens` JAX array) without blocking the main thread.

    The `get_output()` method is called to resolve these async results,
    triggering the JAX device-to-host (CPU) data transfer and populating
    the final `ModelRunnerOutput` object.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        next_tokens: jax.Array,
        num_reqs: int,
        discard_sampled_tokens_req_indices: list[int],
        logits_indices_selector: Optional[List[int]] = None,
    ):
        self._model_runner_output = model_runner_output
        self._next_tokens = next_tokens
        self._num_reqs = num_reqs
        self._discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices
        self.logits_indices_selector: list[int] = logits_indices_selector

    def get_output(self) -> ModelRunnerOutput:
        next_tokens_cpu = np.asarray(jax.device_get(self._next_tokens))
        if self.logits_indices_selector is not None:
            next_tokens_cpu = next_tokens_cpu[self.logits_indices_selector]
        selected_token_ids = np.expand_dims(next_tokens_cpu[:self._num_reqs],
                                            1)
        valid_sampled_token_ids = selected_token_ids.tolist()
        for i in self._discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids
        return self._model_runner_output


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens: jax.Array
    request_seq_lens: list[tuple[int, CachedRequestState, int]]
    discard_sampled_tokens_req_indices: list[int]
    placeholder_req_id_to_index: dict[str, int]
    logits_indices_selector: Optional[List[int]] = None


@dataclass
class ExecuteModelState:
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "VllmSchedulerOutput"
    attn_metadata: AttentionMetadata
    input_ids: Optional[jax.Array]
    hidden_states: jax.Array
    logits: jax.Array
    aux_hidden_states: Optional[jax.Array]
    spec_decode_metadata: Optional[SpecDecodeMetadata]
    kv_connector_output: Optional[KVConnectorOutput]
    logits_indices_selector: Optional[List[int]] = None
    padded_num_reqs: Optional[int] = None


@functools.partial(jax.jit, donate_argnums=(0, 1, 2))
def _substitute_placeholder_token(
        input_ids: jax.Array, token_in_tpu_cur_input_indices: jax.Array,
        token_in_tpu_pre_next_tokens_indices: jax.Array,
        next_tokens: jax.Array, placeholder_num: int):
    """Substitute placeholder tokens from TPU for async scheduler

    Padding for parallelisation of the substitute_placeholder_token_fn
    [1, 3] => [1, 3, 0, 2, 4, 5, 6, 7, 8]
    The reason for such a special padding instead of padding with -1 is:
    An edge case when the end index needs to be updated and padding is required.
    If we pad the array with -1, the _substitute_placeholder_token_fn will repeatedly update the end element with the original value
    Although such a scenario is unlikely to happen in vLLM, it is best to eliminate any potential risks.

    Args:
        input_ids: possible input_ids size
        token_in_tpu_cur_input_indices: replace holder idx in input_ids. Length the same to input_ids.
        token_in_tpu_pre_next_tokens_indices: value idx in next_tokens. Length the same to input_ids.
        next_tokens: next tokens on the TPU from previous step.
        placeholder_num: number of placeholders. placeholder_num <= len(token_in_tpu_cur_input_indices)
    Return:
        input_ids after replace placeholder tokens
    """
    assert input_ids.shape == token_in_tpu_cur_input_indices.shape == token_in_tpu_pre_next_tokens_indices.shape, \
        f"Shape mismatch: input_ids and index arrays must have identical shapes due to precompilation assumptions. " \
        f"Got: {input_ids.shape=}, {token_in_tpu_cur_input_indices.shape=}, {token_in_tpu_pre_next_tokens_indices.shape=}"

    # updates the input_ids for all placeholders.
    mask = jnp.arange(input_ids.shape[0]) < placeholder_num
    new_token_values = next_tokens[token_in_tpu_pre_next_tokens_indices]
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = jnp.where(mask, new_token_values, original_values)
    return input_ids.at[token_in_tpu_cur_input_indices].set(update_values)


def _jax_logprobs_to_lists(logprobs_tensors,
                           logits_indices_selector=None,
                           cu_num_generated_tokens=None):
    """Convert JAX LogprobsTensors to LogprobsLists by converting JAX arrays to numpy."""
    log_token_ids_list = logprobs_tensors.logprob_token_ids.tolist()
    logprobs_list = logprobs_tensors.logprobs.tolist()
    selected_token_ranks_list = logprobs_tensors.selected_token_ranks.tolist()

    if logits_indices_selector is not None:
        log_token_ids_list = [
            log_token_ids_list[i] for i in logits_indices_selector
        ]
        logprobs_list = [logprobs_list[i] for i in logits_indices_selector]
        selected_token_ranks_list = [
            selected_token_ranks_list[i] for i in logits_indices_selector
        ]

    return LogprobsLists(
        logprob_token_ids=np.asarray(log_token_ids_list),
        logprobs=np.asarray(logprobs_list),
        sampled_token_ranks=np.asarray(selected_token_ranks_list),
        cu_num_generated_tokens=cu_num_generated_tokens,
    )


class TPUModelRunner(KVConnectorModelRunnerMixin, LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[Any],
        rank: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # TODO(jevinjiang): override block size based on RPA v3.
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        self.devices = devices
        self.dtype = self.model_config.dtype
        self.maybe_forbid_compile = runner_utils.ForbidCompile(
        ) if envs.VLLM_XLA_CHECK_RECOMPILATION else nullcontext()
        self.dp_size = self.vllm_config.sharding_config.total_dp_size
        self.rank = rank
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank

        self._init_random()
        self._init_mesh()
        self._init_phased_profiling()
        self._init_mm()
        self._init_inputs()
        self._init_speculative_decoding()

        # Delegate functions to specific manager classes.
        self.compilation_manager = CompilationManager(self)
        if self.is_last_rank:
            self.speculative_decoding_manager = SpeculativeDecodingManager(
                self)
            self.structured_decoding_manager = StructuredDecodingManager(self)
        self.kv_cache_manager = KVCacheManager(self)
        self.mm_manager = MultiModalManager(self)
        self.persistent_batch_manager = PersistentBatchManager(
            self.requests, self.input_batch, self.encoder_cache,
            self.uses_mrope, self.model_config, self.is_last_rank)
        self.lora_utils = LoraUtils(self)

        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype == "auto":
            cache_dtype = self.dtype
        self.kv_cache_dtype = to_torch_dtype(cache_dtype)

        self._pre_async_results: AsyncPreResults | None = None
        self._substitute_placeholder_token_fn = _substitute_placeholder_token
        self.execute_model_state: ExecuteModelState | None = None

        self.kv_caches: list[jax.Array] = []
        self.layer_name_to_kvcache_index: dict[str, int] = {}

    def _init_random(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        self.rng_key = jax.random.key(self.model_config.seed)

    def _init_mesh(self) -> None:
        if envs.NEW_MODEL_DESIGN:
            self.mesh = self._create_new_model_mesh()
        else:
            # NOTE(wenxindongwork): The new MoE kernel expects a 2D mesh, so we need
            # to create a 2D mesh for now. We should make the new_model_mesh as the default
            # in the future.
            self.mesh = self._create_2d_mesh()

        logger.info(f"Init mesh | mesh={self.mesh}")

    def _create_new_model_mesh(self) -> jax.sharding.Mesh:
        num_slices = envs.NUM_SLICES

        logger.info(f"Creating new model mesh | devices={len(self.devices)}, "
                    f"num_slices={num_slices}")

        if num_slices == 1:
            devices_array = self._create_single_slice_mesh()
        else:
            devices_array = self._create_multi_slice_mesh(num_slices)

        return jax.sharding.Mesh(devices_array, MESH_AXIS_NAMES)

    def _create_single_slice_mesh(self) -> jax.Array:
        sharding_strategy: ShardingConfigManager = self.vllm_config.sharding_config
        mesh_shape = (
            sharding_strategy.model_dp_size,
            sharding_strategy.attn_dp_size,
            sharding_strategy.expert_size,
            sharding_strategy.tp_size,
        )

        return mesh_utils.create_device_mesh(
            mesh_shape,
            self.devices,
            allow_split_physical_axes=True,
        )

    def _create_multi_slice_mesh(self, num_slices: int) -> jax.Array:
        sharding_strategy: ShardingConfigManager = self.vllm_config.sharding_config
        dp_inner = sharding_strategy.model_dp_size // num_slices

        # Splits data parallelism across multiple slices.
        ici_mesh_shape = (
            dp_inner,
            sharding_strategy.attn_dp_size,
            sharding_strategy.expert_size,
            sharding_strategy.tp_size,
        )
        dcn_mesh_shape = (num_slices, 1, 1, 1)

        return mesh_utils.create_hybrid_device_mesh(
            mesh_shape=ici_mesh_shape,
            dcn_mesh_shape=dcn_mesh_shape,
            devices=self.devices,
            allow_split_physical_axes=True,
        )

    def _create_2d_mesh(self) -> jax.sharding.Mesh:

        sharding_strategy: ShardingConfigManager = self.vllm_config.sharding_config
        mesh_shape = (
            sharding_strategy.model_dp_size,
            sharding_strategy.tp_size,
        )

        enforce_device_order = (
            self.vllm_config.sharding_config.device_indexes is not None
            and len(self.vllm_config.sharding_config.device_indexes) > 0)

        if enforce_device_order:
            return jax.make_mesh(mesh_shape,
                                 MESH_AXIS_NAMES_2D,
                                 devices=self.devices)
        else:
            return make_optimized_mesh(mesh_shape,
                                       MESH_AXIS_NAMES_2D,
                                       devices=self.devices)

    def _init_phased_profiling(self) -> None:
        self.phased_profiling_dir = envs.PHASED_PROFILING_DIR
        self.phase_based_profiler = None
        if self.phased_profiling_dir:
            self.phase_based_profiler = runner_utils.PhasedBasedProfiler(
                self.phased_profiling_dir)

    def _init_mm(self) -> None:
        self.is_multimodal_model = None
        self.uses_mrope = self.model_config.uses_mrope

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        if self.speculative_config:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "eagle3":
                self.drafter = Eagle3Proposer(self.vllm_config, self)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

    def _init_inputs(self) -> None:
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        # The total number of requests is dp_size * max_num_seqs
        self.max_num_reqs = max(self.dp_size * scheduler_config.max_num_seqs,
                                MIN_NUM_SEQS)
        # [16, 32, 64, 128, 256, 512, 1024, 2048]
        self.num_tokens_paddings = runner_utils.get_token_paddings(
            min_token_size=max(16, self.dp_size),
            max_token_size=scheduler_config.max_num_batched_tokens *
            self.dp_size,
            padding_gap=vllm_envs.VLLM_TPU_BUCKET_PADDING_GAP)
        self.num_tokens_paddings_per_dp = [
            padding // self.dp_size for padding in self.num_tokens_paddings
        ]
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, jax.Array] = {}
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            pin_memory=False,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
        )

        self.input_ids_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)
        self.positions_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)
        # Note: self.input_batch and self.block_tables_cpu are both initialized
        # with only 1 block_size. For hybrid kv cache, it will be re-init
        # in kv_cache_manager's maybe_reinitialize_input_batch.
        self.block_tables_cpu = [
            np.zeros((self.max_num_reqs, self.max_num_blocks_per_req),
                     dtype=np.int32)
        ]

        self.query_start_loc_cpu = np.zeros(self.max_num_reqs + self.dp_size,
                                            dtype=np.int32)
        self.seq_lens_cpu = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.logits_indices_cpu = np.zeros(self.max_num_reqs, dtype=np.int32)
        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_cpu = np.arange(self.max_num_tokens, dtype=np.int64)
        min_num_reqs = max(MIN_NUM_SEQS, self.dp_size)
        self.num_reqs_paddings = runner_utils.get_req_paddings(
            min_req_size=min_num_reqs, max_req_size=self.max_num_reqs)
        self.num_reqs_paddings_per_dp = [
            padding // self.dp_size for padding in self.num_reqs_paddings
        ]

        # Padding for logits. Without speculative decoding, each request has one position to select from.
        # With speculative decoding, each request has multiple positions to select from.
        max_logits_per_req = 1
        if self.speculative_config:
            max_logits_per_req = self.speculative_config.num_speculative_tokens + 1  # Including bonus token
            self.num_logits_paddings = runner_utils.get_token_paddings(
                min_token_size=MIN_NUM_SEQS,
                max_token_size=self.max_num_reqs * max_logits_per_req,
                padding_gap=0)
        else:
            self.num_logits_paddings = None

        self.temperatures_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ps_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ks_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)

        # tensors for structured decoding
        self.vocab_size = self.model_config.get_vocab_size()
        self.grammar_bitmask_cpu = np.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=np.int32,
        )
        self.require_structured_out_cpu = np.zeros(
            (self.max_num_reqs, 1),
            dtype=np.bool_,
        )
        self.structured_decode_arange = np.arange(0, 32, dtype=np.int32)

        # multi-modal support
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)

        # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
        # the modality of inputs. For text-only inputs, each dimension has
        # identical position IDs, making M-RoPE functionally equivalent to
        # 1D-RoPE.
        # See page 5 of https://arxiv.org/abs/2409.12191
        self.mrope_positions_cpu = np.zeros((3, self.max_num_tokens),
                                            dtype=np.int64)

    def load_model(self):
        self.model_fn, self.compute_logits_fn, self.combine_hidden_states_fn, multimodal_fns, self.state, self.lora_manager, self.model = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )

        multimodal_fns = multimodal_fns or {}
        self.precompile_vision_encoder_fn = multimodal_fns.get(
            "precompile_vision_encoder_fn", None)
        self.embed_multimodal_fn = multimodal_fns.get("embed_multimodal_fn",
                                                      None)
        self.embed_input_ids_fn = multimodal_fns.get("embed_input_ids_fn",
                                                     None)
        self.get_mrope_input_positions_fn = multimodal_fns.get(
            "get_mrope_input_positions_fn", None)

        if self.drafter is not None:
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.state)

        self.rng_params_for_sampling = nnx.Rngs(
            jax.random.key(self.model_config.seed)).params()
        self.is_multimodal_model = (
            self.model_config.is_multimodal_model
            and self.embed_multimodal_fn is not None and hasattr(
                self.model_config.hf_config, "architectures"
            )  #TODO: Remove Llama Guard 4 specific condition once the LG4 Vision portion is implemented
            and len(self.model_config.hf_config.architectures) >= 1
            and self.model_config.hf_config.architectures[0]
            != "Llama4ForConditionalGeneration")

        logger.info(f"Init model | "
                    f"hbm={common_utils.hbm_usage_gb(self.devices)}GiB")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate", )

    def get_kv_cache_spec(self):
        return self.kv_cache_manager.get_kv_cache_spec()

    def initialize_kv_cache(self,
                            kv_cache_config: KVCacheConfig,
                            topology_order_id: int = 0) -> None:
        self.topology_order_id = topology_order_id
        self.kv_cache_config = kv_cache_config
        self.use_hybrid_kvcache = len(kv_cache_config.kv_cache_groups) > 1
        self.kv_cache_manager.initialize_kv_cache(kv_cache_config)
        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self)

    def capture_model(self) -> None:
        self.compilation_manager.capture_model()

    @time_function
    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> ModelRunnerOutput | JaxIntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        _, output = self._execute_model(scheduler_output, intermediate_tensors)
        return output

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            # This can happen in pipeline parallel case.
            return EMPTY_MODEL_RUNNER_OUTPUT

        (scheduler_output, attn_metadata, input_ids, hidden_states, logits,
         aux_hidden_states, spec_decode_metadata, kv_connector_output,
         logits_indices_selector,
         padded_num_reqs) = (self.execute_model_state.scheduler_output,
                             self.execute_model_state.attn_metadata,
                             self.execute_model_state.input_ids,
                             self.execute_model_state.hidden_states,
                             self.execute_model_state.logits,
                             self.execute_model_state.aux_hidden_states,
                             self.execute_model_state.spec_decode_metadata,
                             self.execute_model_state.kv_connector_output,
                             self.execute_model_state.logits_indices_selector,
                             self.execute_model_state.padded_num_reqs)
        self.execute_model_state = None

        if grammar_output is not None:
            (
                require_struct_decoding, grammar_bitmask_padded, arange
            ) = self.structured_decoding_manager.prepare_structured_decoding_input(
                logits, grammar_output)
            logits = self.structured_decoding_manager.structured_decode_fn(
                require_struct_decoding,
                grammar_bitmask_padded,
                logits,
                arange,
            )
        return self._sample_from_logits(
            scheduler_output, attn_metadata, input_ids, hidden_states, logits,
            aux_hidden_states, spec_decode_metadata, kv_connector_output,
            logits_indices_selector, padded_num_reqs)

    def _modify_prev_results(self):
        # If copy to host has not been done, we just wait.
        # device_get should return immediately as we have scheduled it in previous function call.
        assert self._pre_async_results is not None, "When we call _modify_prev_results(), self._pre_async_results should already exist"
        pre_req_ids = self._pre_async_results.req_ids
        pre_next_tokens = self._pre_async_results.next_tokens
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices
        pre_logits_indices_selector = self._pre_async_results.logits_indices_selector

        next_tokens_cpu = np.asarray(jax.device_get(pre_next_tokens))
        if pre_logits_indices_selector is not None:
            next_tokens_cpu = next_tokens_cpu[pre_logits_indices_selector]
        selected_token_ids = np.expand_dims(next_tokens_cpu[:len(pre_req_ids)],
                                            1)
        valid_sampled_token_ids = selected_token_ids.tolist()

        # Mask out the sampled tokens that should not be sampled.
        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        # Append sampled tokens
        for pre_req_idx, req_state, _ in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if not sampled_ids:
                continue

            # If request not active in the *current* batch (e.g. finished or evicted), skip it.
            req_id = pre_req_ids[pre_req_idx]
            if req_id not in self.input_batch.req_id_to_index:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            assert req_state is self.requests[
                req_id], "The req_state should be valid and identical"

            # Updated on previous execute
            end_idx = self.input_batch.num_tokens_no_spec[req_idx]
            assert len(sampled_ids) == 1, "do not support spec decode yet"
            start_idx = end_idx - 1
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            # Replace previous placeholder
            req_state.output_token_ids[-1] = sampled_ids[-1]

    def _update_placeholder(self,
                            discard_sampled_tokens_req_indices,
                            request_seq_lens,
                            logits_indices_selector=None):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_sampled_tokens_req_indices_set = set(
            discard_sampled_tokens_req_indices)
        for req_idx, req_state, _ in request_seq_lens:
            if req_idx in discard_sampled_tokens_req_indices_set:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            # Not supporting spec decode yet, assume only 1 new token
            end_idx = start_idx + 1
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            # Update cpu tokens at next execute and prepare input from tpu
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            # For placeholder, should be update on next execute.
            req_state.output_token_ids.extend([0])
            if logits_indices_selector is None:
                placeholder_req_id_to_index[req_state.req_id] = req_idx
            else:
                placeholder_req_id_to_index[
                    req_state.req_id] = logits_indices_selector[req_idx]
        return placeholder_req_id_to_index

    def _execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> tuple[AttentionMetadata, JaxIntermediateTensors | ModelRunnerOutput
               | None]:
        self.persistent_batch_manager.update_states(
            scheduler_output, self.get_mrope_input_positions_fn)
        if not scheduler_output.total_num_scheduled_tokens:
            if has_kv_transfer_group():
                return DUMMY_METADATA, self.kv_connector_no_forward(
                    scheduler_output, self.vllm_config)

            # Return empty ModelRunnerOutput if there's no work to do.
            # TODO(fhzhang): We rely on empty cycles to remove requests in input batch. Fix it to reduce overhead.
            logger.debug(f"Nothing scheduled: {scheduler_output}!")
            # NOTE(pooyam): There is no guarantee that scheduler is not sending empty output: https://github.com/vllm-project/vllm/blob/7cfea0df390c154c1026f77d3682e2733ca4aca8/vllm/v1/engine/core.py#L275
            # Why they are not preventing that is not clear to me.
            if len(scheduler_output.finished_req_ids) == 0:
                logger.warning(
                    "Should not schedule a request that does nothing!")
                # raise Exception(
                #     "Should not schedule a request that does nothing!")
            return DUMMY_METADATA, EMPTY_MODEL_RUNNER_OUTPUT

        # TODO(pooyam): I guess we can remove returning sampling_metadata in `_prepare_inputs` after https://github.com/njhill/vllm/commit/b7433ca1a47732394b1bdea4099d98389515954b
        (
            input_ids,
            input_positions,
            attn_metadata,
            _,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
        ) = self._prepare_inputs(scheduler_output)

        is_llama_guard_4 = (
            hasattr(
                self.model_config.hf_config, "architectures"
            )  #TODO: Remove Llama Guard 4 specific condition once the LG4 Vision portion is implemented
            and len(self.model_config.hf_config.architectures) >= 1
            and self.model_config.hf_config.architectures[0]
            == "Llama4ForConditionalGeneration")

        # multi-modal support
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            # We have the modality embeds at this time.
            self.mm_manager.execute_mm_encoder(scheduler_output)
            mm_embeds = self.mm_manager.gather_mm_embeddings(
                scheduler_output, input_ids.shape[0])
        #TODO: Remove the follow elif statement once Llama Guard 4 Vision portion has been implemented
        elif is_llama_guard_4 and any(
                self.mm_manager.runner.requests[req_id].mm_features
                for req_id in self.mm_manager.runner.input_batch.req_ids):
            raise NotImplementedError(
                "Llama Guard 4 (JAX) currently supports only text inputs. "
                "Multimodal processing not yet implemented.")
        else:
            mm_embeds = []

        # NOTE(Wenlong): For multi-modal model,
        # it will embed the text tokens and merge with the existing modality embeds
        # Later, the multi-modality model will take the embedding as the input.
        # For text-only model, this does nothing. It will input the input_ids and
        # leave the mebedding job inside the forward pass
        input_ids, inputs_embeds = self._get_input_ids_embeds(
            input_ids, mm_embeds)

        lora_metadata = self.lora_utils.extract_lora_metadata()
        # TODO: make _get_input_ids_embeds within this context
        # NOTE: right now, mm model will use embeddings as the input,
        # but text-only model will use input_ids
        with self.maybe_forbid_compile:

            with set_forward_context(
                    None,
                    self.vllm_config,
            ), self.maybe_get_kv_connector_output(
                    scheduler_output) as kv_connector_output:
                # NOTE(Wenlong): It takes both `input_ids` and `inputs_embeds`,
                # but one of them would be `None`
                (self.kv_caches, hidden_states,
                 aux_hidden_states) = self.model_fn(
                     self.state,
                     self.kv_caches,
                     input_ids,
                     attn_metadata,
                     inputs_embeds,
                     input_positions,
                     tuple(self.layer_name_to_kvcache_index.items()),
                     lora_metadata,
                     intermediate_tensors,
                     self.is_first_rank,
                     self.is_last_rank,
                 )
            if not get_pp_group().is_last_rank:
                assert isinstance(hidden_states, JaxIntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                return attn_metadata, hidden_states
            hidden_states = self._select_from_array_fn(hidden_states,
                                                       logits_indices)
            logits = self.compute_logits_fn(
                self.state,
                hidden_states,
                lora_metadata,
            )

        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits,
            aux_hidden_states=aux_hidden_states,
            spec_decode_metadata=spec_decode_metadata,
            kv_connector_output=kv_connector_output,
            logits_indices_selector=logits_indices_selector,
            padded_num_reqs=padded_num_reqs)
        return attn_metadata, None

    def _sample_from_logits(
        self,
        scheduler_output: "VllmSchedulerOutput",
        attn_metadata: AttentionMetadata,
        input_ids: Optional[jax.Array],
        hidden_states: jax.Array,
        logits: jax.Array,
        aux_hidden_states: Optional[jax.Array],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        kv_connector_output: Optional[KVConnectorOutput],
        logits_indices_selector: Optional[List[int]] = None,
        padded_num_reqs: Optional[int] = None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if padded_num_reqs is None:
            padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
                self.input_batch.num_reqs, self.max_num_reqs)

        sharding = None
        if self.dp_size > 1:
            sharding = NamedSharding(self.mesh,
                                     PartitionSpec(ShardingAxisName.MLP_DATA))

        tpu_sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh, self.input_batch, padded_num_reqs, sharding=sharding)

        # TODO(pooyam): Should we move this to `_prepare_inputs`?
        if tpu_sampling_metadata.do_sampling:
            self.rng_params_for_sampling, step_rng = jax.random.split(
                self.rng_params_for_sampling)
        else:
            step_rng = self.rng_params_for_sampling

        if spec_decode_metadata is None:
            next_tokens = sample(
                step_rng,
                self.mesh,
                logits,
                tpu_sampling_metadata,
            )
        else:
            if tpu_sampling_metadata.do_sampling:
                bonus_rng, rejection_rng = jax.random.split(step_rng)
            else:
                bonus_rng = step_rng
                rejection_rng = step_rng
            bonus_logits = self._select_from_array_fn(
                logits, spec_decode_metadata.bonus_logits_indices)
            bonus_token_ids = sample(
                bonus_rng,
                self.mesh,
                bonus_logits,
                tpu_sampling_metadata,
            )
            target_logits = self._select_from_array_fn(
                logits, spec_decode_metadata.target_logits_indices)
            next_tokens = self.rejection_sampler(
                draft_token_ids=spec_decode_metadata.draft_token_ids,
                num_draft_tokens=spec_decode_metadata.draft_lengths,
                draft_probs=None,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=tpu_sampling_metadata,
                key=rejection_rng,
            )

        if tpu_sampling_metadata.logprobs:
            logprobs = self._compute_and_gather_logprobs(
                logits, next_tokens, self.model_config.max_logprobs)
        else:
            logprobs = None

        num_reqs = self.input_batch.num_reqs

        # Update the cache state concurrently. Code above will not block until
        # We use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        # If async scheduler enabled
        if self.scheduler_config.async_scheduling:
            # Get previous results from TPU and replace the placeholder.
            if self._pre_async_results is not None:
                assert not self.speculative_config and spec_decode_metadata is None, "Async scheduler does not support speculative decoding yet."
                self._modify_prev_results()

            # Set placeholder for next tokens that is not yet generated
            placeholder_req_id_to_index: dict[
                str, int] = self._update_placeholder(
                    discard_sampled_tokens_req_indices, request_seq_lens,
                    logits_indices_selector)

            if logprobs is not None:
                # Map logprobs back to the pre-dp shuffling order
                logprobs_lists = _jax_logprobs_to_lists(
                    logprobs, logits_indices_selector)

            else:
                logprobs_lists = None

            # Save the previous results
            next_tokens = jax.copy_to_host_async(next_tokens)
            self._pre_async_results = AsyncPreResults(
                req_ids=req_ids,
                next_tokens=next_tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=
                discard_sampled_tokens_req_indices,
                placeholder_req_id_to_index=placeholder_req_id_to_index,
                logits_indices_selector=logits_indices_selector)

            # Return Model output to executor
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=copy.deepcopy(
                    self.input_batch.req_id_to_index),
                sampled_token_ids=[],  # Fill in async get
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
            )
            # Return async_model_runner_output
            async_model_runner_output = AsyncTPUModelRunnerOutput(
                model_runner_output, next_tokens, num_reqs,
                discard_sampled_tokens_req_indices, logits_indices_selector)
            return async_model_runner_output

        if spec_decode_metadata is None:
            next_tokens = np.asarray(jax.device_get(next_tokens))
            # Map tokens back to the pre-dp shuffling order
            if logits_indices_selector is not None:
                next_tokens = next_tokens[logits_indices_selector]
            selected_token_ids = np.expand_dims(next_tokens[:num_reqs], 1)
            valid_sampled_token_ids = selected_token_ids.tolist()
        else:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                next_tokens, self.input_batch.vocab_size,
                spec_decode_metadata.draft_lengths_cpu, num_reqs,
                spec_decode_metadata.draft_token_ids.shape[0])

        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        # Append sampled tokens
        for req_idx, req_state, _ in request_seq_lens:
            sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_state.output_token_ids.extend(sampled_ids)

        if logprobs is not None:
            # Map logprobs back to the pre-dp shuffling order
            logprobs_lists = _jax_logprobs_to_lists(logprobs,
                                                    logits_indices_selector)
        else:
            logprobs_lists = None

        if self.speculative_config:
            with self.maybe_forbid_compile:
                self.speculative_decoding_manager.propose_draft_token_ids(
                    valid_sampled_token_ids,
                    aux_hidden_states,
                    attn_metadata,
                    spec_decode_metadata,
                    scheduler_output,
                    input_ids,
                )

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )
        return model_runner_output

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _select_from_array_fn(self, array, indices_to_select):

        def select_local_fn(local_array, local_indices):
            return local_array[local_indices]

        ret = jax.shard_map(
            select_local_fn,
            mesh=self.mesh,
            in_specs=(PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA)),
            out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA))(
                array, indices_to_select)

        return ret

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_logprobs", ))
    def _compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
        logprobs = compute_logprobs(logits)
        return gather_logprobs(logprobs, next_tokens, max_logprobs)

    def _prepare_dp_input_metadata(self,
                                   scheduler_output: "VllmSchedulerOutput"):

        dp_size = self.dp_size
        num_reqs = self.input_batch.num_reqs
        max_num_reqs_per_dp_rank = self.max_num_reqs // dp_size
        req_ids_dp = {dp_rank: [] for dp_rank in range(dp_size)}
        req_indices_dp = {dp_rank: [] for dp_rank in range(dp_size)}
        num_scheduled_tokens_per_dp_rank = {
            dp_rank: 0
            for dp_rank in range(dp_size)
        }
        scheduled_tokens_per_dp_rank = {
            dp_rank: []
            for dp_rank in range(dp_size)
        }
        num_req_per_dp_rank = {dp_rank: 0 for dp_rank in range(dp_size)}

        for req_id in self.input_batch.req_ids[:num_reqs]:
            dp_rank = scheduler_output.assigned_dp_rank[req_id]
            req_ids_dp[dp_rank].append(req_id)
            req_indices_dp[dp_rank].append(
                self.input_batch.req_id_to_index[req_id])
            num_scheduled_tokens_per_dp_rank[
                dp_rank] += scheduler_output.num_scheduled_tokens[req_id]
            scheduled_tokens_per_dp_rank[dp_rank].append(
                scheduler_output.num_scheduled_tokens[req_id])
            num_req_per_dp_rank[dp_rank] += 1

        # Find maximum number of scheduled tokens across DP ranks
        max_num_scheduled_tokens_across_dp = max(
            num_scheduled_tokens_per_dp_rank.values())

        padded_num_scheduled_tokens_per_dp_rank = runner_utils.get_padded_token_len(
            self.num_tokens_paddings_per_dp,
            max_num_scheduled_tokens_across_dp)

        padded_total_num_scheduled_tokens = (
            padded_num_scheduled_tokens_per_dp_rank * dp_size)

        assert max_num_scheduled_tokens_across_dp > 0

        # Find maximum number of requests across DP ranks
        max_num_reqs_across_dp = max(
            len(req_ids) for req_ids in req_ids_dp.values())
        padded_num_reqs_per_dp_rank = runner_utils.get_padded_token_len(
            self.num_reqs_paddings_per_dp, max_num_reqs_across_dp)
        padded_num_reqs = padded_num_reqs_per_dp_rank * dp_size

        all_req_indices = np.concatenate(
            [req_indices_dp[dp_rank] for dp_rank in range(dp_size)])
        all_positions = np.concatenate([
            np.arange(len(req_indices_dp[dp_rank])) +
            padded_num_reqs_per_dp_rank * dp_rank for dp_rank in range(dp_size)
        ])

        # Sort positions by request indices
        sorted_indices = np.argsort(all_req_indices)
        logits_indices_selector = all_positions[sorted_indices]

        return (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
                scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
                padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
                padded_total_num_scheduled_tokens, padded_num_reqs_per_dp_rank,
                logits_indices_selector, max_num_reqs_per_dp_rank)

    def _prepare_async_token_substitution_indices_dp(
            self, req_ids_dp, scheduled_tokens_per_dp_rank,
            padded_num_scheduled_tokens_per_dp_rank, dp_size):
        """Prepare token substitution indices for async scheduling in DP mode."""
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}

        for dp_rank in range(dp_size):
            token_in_tpu_cur_input_indices_dp[dp_rank] = []
            token_in_tpu_pre_next_tokens_indices_dp[dp_rank] = []

            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            acc_cur_len = token_offset

            for i, req_id in enumerate(req_ids_dp[dp_rank]):
                acc_cur_len += scheduled_tokens_per_dp_rank[dp_rank][i]
                if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                    continue

                token_in_tpu_cur_input_indices_dp[dp_rank].append(acc_cur_len -
                                                                  1)
                token_in_tpu_pre_next_tokens_indices_dp[dp_rank].append(
                    self._pre_async_results.placeholder_req_id_to_index[req_id]
                )

        return token_in_tpu_cur_input_indices_dp, token_in_tpu_pre_next_tokens_indices_dp

    def _prepare_async_token_substitution_indices_non_dp(
            self, num_reqs, num_scheduled_tokens_per_req):
        """Prepare token substitution indices for async scheduling in non-DP mode."""
        token_in_tpu_cur_input_indices_list = []
        token_in_tpu_pre_next_tokens_indices_list = []
        acc_cur_len = 0

        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            acc_cur_len += num_scheduled_tokens_per_req[i]
            assert req_id is not None
            if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                continue

            token_in_tpu_cur_input_indices_list.append(acc_cur_len - 1)
            token_in_tpu_pre_next_tokens_indices_list.append(
                self._pre_async_results.placeholder_req_id_to_index[req_id])

        if len(token_in_tpu_cur_input_indices_list) > 0:
            return (np.array(token_in_tpu_cur_input_indices_list),
                    np.array(token_in_tpu_pre_next_tokens_indices_list))
        else:
            return np.array([]), np.array([])

    def _apply_async_token_substitution(self, input_ids,
                                        token_in_tpu_cur_input_indices,
                                        token_in_tpu_pre_next_tokens_indices):
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input_ids

        idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside self._substitute_placeholder_token_fn
        full_range = np.arange(0, len(input_ids))
        missing_values = np.setdiff1d(full_range,
                                      token_in_tpu_cur_input_indices)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1)

        (padded_token_in_tpu_cur_input_indices,
         padded_token_in_tpu_pre_next_tokens_indices) = device_array(
             self.mesh, (padded_token_in_tpu_cur_input_indices,
                         padded_token_in_tpu_pre_next_tokens_indices))

        with self.maybe_forbid_compile:
            input_ids = self._substitute_placeholder_token_fn(
                input_ids, padded_token_in_tpu_cur_input_indices,
                padded_token_in_tpu_pre_next_tokens_indices,
                self._pre_async_results.next_tokens,
                len(token_in_tpu_cur_input_indices))

        return input_ids

    def _prepare_inputs(self, scheduler_output: "VllmSchedulerOutput"):
        if self.dp_size > 1:
            return self._prepare_inputs_dp(scheduler_output)
        else:
            return self._prepare_inputs_non_dp(scheduler_output)

    def _prepare_inputs_dp(self, scheduler_output: "VllmSchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        dp_size = self.dp_size
        data_parallel_attn_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))

        (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
         scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
         padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
         padded_total_num_scheduled_tokens, padded_num_reqs_per_dp_rank,
         logits_indices_selector, max_num_reqs_per_dp_rank
         ) = self._prepare_dp_input_metadata(scheduler_output)
        # Multi-modal support
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self.mm_manager.calc_mrope_positions(scheduler_output)

        # Async scheduling: prepare token substitution indices for DP
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # If async previous results exists, we will prepare for the token substitution here
            # The actual substitution will be performed in tpu during later parts of this function.
            (token_in_tpu_cur_input_indices_dp,
             token_in_tpu_pre_next_tokens_indices_dp
             ) = self._prepare_async_token_substitution_indices_dp(
                 req_ids_dp, scheduled_tokens_per_dp_rank,
                 padded_num_scheduled_tokens_per_dp_rank, dp_size)

        # Populates input_ids and positions
        for dp_rank in range(dp_size):
            if num_req_per_dp_rank[dp_rank] == 0:
                continue
            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]
            total_num_scheduled_tokens = num_scheduled_tokens_per_dp_rank[
                dp_rank]
            input_ids_cpu = self.input_ids_cpu[
                token_offset:token_offset +
                padded_num_scheduled_tokens_per_dp_rank]
            positions_cpu = self.positions_cpu[
                token_offset:token_offset +
                padded_num_scheduled_tokens_per_dp_rank]
            # Get request indices.
            # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
            # For each scheduled token, what are the corresponding req index.
            req_indices = np.repeat(req_indices_dp[dp_rank],
                                    num_scheduled_tokens_per_req)
            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # For each scheduled token, what is its position in corresponding req.
            arange = np.concatenate(
                [self.arange_cpu[:n] for n in num_scheduled_tokens_per_req])
            # Get positions.
            positions_np = positions_cpu[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )
            # Get token indices.
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
            # where M is the max_model_len.
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])
            # NOTE(woosuk): We use torch.index_select instead of np.take here
            # because torch.index_select is much faster than np.take for large
            # tensors.
            np.take(
                self.input_batch.token_ids_cpu.ravel(),
                token_indices,
                out=input_ids_cpu[:total_num_scheduled_tokens],
            )

            input_ids_cpu[total_num_scheduled_tokens:] = 0

        # Prepare the attention metadata (query_start_loc_cpu, seq_lens_cpu)
        for dp_rank in range(dp_size):
            req_offset = dp_rank * max_num_reqs_per_dp_rank
            query_start_loc_cpu = self.query_start_loc_cpu[
                req_offset + dp_rank:req_offset + max_num_reqs_per_dp_rank +
                dp_rank + 1]
            seq_lens_cpu = self.seq_lens_cpu[req_offset:req_offset +
                                             max_num_reqs_per_dp_rank]
            _num_reqs = num_req_per_dp_rank[dp_rank]
            req_indices = req_indices_dp[dp_rank]
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]

            if _num_reqs == 0:
                query_start_loc_cpu[:] = 0
                seq_lens_cpu[:] = 0
                continue

            np.cumsum(
                num_scheduled_tokens_per_req,
                out=query_start_loc_cpu[1:_num_reqs + 1],
            )
            query_start_loc_cpu[_num_reqs + 1:] = 1

            seq_lens_cpu[:_num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[req_indices] +
                num_scheduled_tokens_per_req)
            seq_lens_cpu[_num_reqs:] = 0

        # populate logits_indices
        for dp_rank in range(dp_size):
            req_offset = dp_rank * padded_num_reqs_per_dp_rank
            query_loc_req_offset = dp_rank * (max_num_reqs_per_dp_rank + 1)
            _num_reqs = num_req_per_dp_rank[dp_rank]

            logits_indices_cpu = self.logits_indices_cpu[
                req_offset:req_offset + padded_num_reqs_per_dp_rank]
            logits_indices_cpu[:_num_reqs] = (
                self.query_start_loc_cpu[query_loc_req_offset +
                                         1:query_loc_req_offset + _num_reqs +
                                         1] - 1)
            logits_indices_cpu[_num_reqs:] = -1

        logits_indices = self.logits_indices_cpu[:padded_num_reqs]

        # Please see runner_utils.PhasedBasedProfiler for details
        if self.phase_based_profiler:
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                self.input_batch, total_num_scheduled_tokens, num_reqs,
                padded_total_num_scheduled_tokens, scheduler_output)

            self.phase_based_profiler.step(batch_composition_stats)

        # Inputs
        input_ids = self.input_ids_cpu[:padded_total_num_scheduled_tokens]
        positions = self.positions_cpu[:padded_total_num_scheduled_tokens]
        mrope_positions = self.mrope_positions_cpu[:, :
                                                   padded_total_num_scheduled_tokens]

        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs +
                                                   dp_size]
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs]

        _request_distribution = []
        for dp_rank in range(dp_size):
            _num_reqs = num_req_per_dp_rank[dp_rank]
            # The batch has been reordered by _reorder_batch so decode requests come first
            # Count decode requests (those with num_scheduled_tokens == 1) in this DP rank
            num_decode_in_dp_rank = 0
            for req_id in req_ids_dp[dp_rank]:
                if scheduler_output.num_scheduled_tokens[req_id] == 1:
                    num_decode_in_dp_rank += 1
            _request_distribution.append(
                [num_decode_in_dp_rank, num_decode_in_dp_rank, _num_reqs])
        request_distribution = np.array(_request_distribution).ravel()

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            spec_decode_metadata = None
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for (
                    req_id,
                    draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = (
                self.speculative_decoding_manager.get_spec_decode_metadata(
                    num_draft_tokens,
                    self.query_start_loc_cpu[1:num_reqs + 1],
                    padded_num_reqs,
                ))
            logits_indices = spec_decode_metadata.final_logits_indices

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh,
            self.input_batch,
            padded_num_reqs,
            sharding=NamedSharding(self.mesh,
                                   PartitionSpec(ShardingAxisName.MLP_DATA)),
        )
        if self.uses_mrope:
            positions = mrope_positions

        query_start_loc_cpu = query_start_loc
        logits_indices_cpu = logits_indices
        seq_lens_cpu = seq_lens

        (input_ids, positions, query_start_loc, seq_lens, logits_indices,
         request_distribution) = device_array(
             self.mesh,
             (input_ids, positions, query_start_loc, seq_lens, logits_indices,
              request_distribution),
             sharding=data_parallel_attn_sharding,
         )

        attention_metadata_per_layer: Dict[str, AttentionMetadata] = {}
        uniform_attention_metadata: AttentionMetadata = None
        for kv_cache_gid, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_tables = self.block_tables_cpu[kv_cache_gid][:self.
                                                               max_num_reqs]
            for dp_rank in range(dp_size):
                req_offset = dp_rank * max_num_reqs_per_dp_rank
                _num_reqs = num_req_per_dp_rank[dp_rank]

                block_tables[
                    req_offset:req_offset + _num_reqs, :self.
                    max_num_blocks_per_req] = self.input_batch.block_table[
                        kv_cache_gid].get_cpu_tensor()[req_indices_dp[dp_rank]]
            # Convert block_tables to 1D on cpu.
            block_tables = block_tables.reshape(-1)
            block_tables = device_array(
                self.mesh,
                (block_tables),
                sharding=data_parallel_attn_sharding,
            )

            attention_metadata_gid = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )

            # This is for making these cpu buffers hidden during tracing
            attention_metadata_gid.query_start_loc_cpu = query_start_loc_cpu
            attention_metadata_gid.seq_lens_cpu = seq_lens_cpu

            if not self.use_hybrid_kvcache:
                uniform_attention_metadata = attention_metadata_gid
            else:
                for layer_name in kv_cache_group.layer_names:
                    attention_metadata_per_layer[
                        layer_name] = attention_metadata_gid

        # Async scheduling: substitute placeholder tokens for DP
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # Collect all token indices that need substitution across all DP ranks
            all_token_indices_to_substitute = []
            all_pre_next_tokens_indices = []

            for dp_rank in range(dp_size):
                cur_indices = token_in_tpu_cur_input_indices_dp[dp_rank]
                pre_indices = token_in_tpu_pre_next_tokens_indices_dp[dp_rank]
                all_token_indices_to_substitute.extend(cur_indices)
                all_pre_next_tokens_indices.extend(pre_indices)

            if len(all_token_indices_to_substitute) > 0:
                token_in_tpu_cur_input_indices = np.array(
                    all_token_indices_to_substitute)
                token_in_tpu_pre_next_tokens_indices = np.array(
                    all_pre_next_tokens_indices)
                input_ids = self._apply_async_token_substitution(
                    input_ids, token_in_tpu_cur_input_indices,
                    token_in_tpu_pre_next_tokens_indices)

        if self.lora_config is not None:
            self.lora_utils.set_active_loras(
                num_scheduled_tokens_per_req,
                total_num_scheduled_tokens,
                padded_total_num_scheduled_tokens,
            )

        if self.use_hybrid_kvcache:
            attention_metadata = attention_metadata_per_layer
        else:
            attention_metadata = uniform_attention_metadata
        return (
            input_ids,
            positions,
            attention_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
        )

    def _prepare_inputs_non_dp(self, scheduler_output: "VllmSchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(
                max_num_scheduled_tokens_all_reqs, num_tokens)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        assert max_num_scheduled_tokens_all_reqs > 0
        padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_cpu[:num_reqs],
                                num_scheduled_tokens_per_req)
        token_in_tpu_cur_input_indices = np.array([])
        token_in_tpu_pre_next_tokens_indices = np.array([])
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # If async previous results exists, we will prepare for the token substitution here
            # The actual substitution will be performed in tpu during later parts of this function.
            (token_in_tpu_cur_input_indices,
             token_in_tpu_pre_next_tokens_indices
             ) = self._prepare_async_token_substitution_indices_non_dp(
                 num_reqs, num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_cpu[:n] for n in num_scheduled_tokens_per_req])

        # Get positions.
        positions_np = self.positions_cpu[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Multi-modal support
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self.mm_manager.calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        np.take(self.input_batch.token_ids_cpu.ravel(),
                token_indices,
                out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_cpu[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_cpu[1:num_reqs + 1])
        self.query_start_loc_cpu[num_reqs + 1:] = 1

        self.seq_lens_cpu[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = runner_utils.get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0

        # Please see runner_utils.PhasedBasedProfiler for details
        if self.phase_based_profiler:
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                self.input_batch, total_num_scheduled_tokens, num_reqs,
                padded_total_num_scheduled_tokens, scheduler_output)

            self.phase_based_profiler.step(batch_composition_stats)

        # Inputs
        input_ids = self.input_ids_cpu[:padded_total_num_scheduled_tokens]
        positions = self.positions_cpu[:padded_total_num_scheduled_tokens]
        mrope_positions = self.mrope_positions_cpu[:, :
                                                   padded_total_num_scheduled_tokens]

        # TODO(pooyam): Some paddings are up to `num_reqs_paddings` (spec decoding, select hidden states, etc) and some other are to `max_num_reqs` (block table, seq_lens). We should stick to one of them maybe?
        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs + 1]
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs]
        request_distribution = np.array(self.input_batch.request_distribution)
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            logits_indices = self.query_start_loc_cpu[1:padded_num_reqs +
                                                      1] - 1
            spec_decode_metadata = None
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self.speculative_decoding_manager.get_spec_decode_metadata(
                num_draft_tokens, self.query_start_loc_cpu[1:num_reqs + 1],
                padded_num_reqs)
            logits_indices = spec_decode_metadata.final_logits_indices

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh, self.input_batch, padded_num_reqs)
        if self.uses_mrope:
            positions = mrope_positions
        query_start_loc_cpu = query_start_loc
        seq_lens_cpu = seq_lens

        (input_ids, positions, query_start_loc, seq_lens,
         logits_indices, request_distribution) = device_array(
             self.mesh, (input_ids, positions, query_start_loc, seq_lens,
                         logits_indices, request_distribution))

        attention_metadata_per_layer: Dict[str, AttentionMetadata] = {}
        uniform_attention_metadata: AttentionMetadata = None
        for kv_cache_gid, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_tables = self.block_tables_cpu[kv_cache_gid][:self.
                                                               max_num_reqs]
            block_tables[:num_reqs] = (
                self.input_batch.block_table[kv_cache_gid].get_cpu_tensor()
                [:num_reqs])
            # Convert block_tables to 1D on cpu.
            block_tables = block_tables.reshape(-1)
            block_tables = device_array(self.mesh, (block_tables))

            attention_metadata_gid = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution)
            # This is for making these cpu buffers hidden during tracing
            attention_metadata_gid.query_start_loc_cpu = query_start_loc_cpu
            attention_metadata_gid.seq_lens_cpu = seq_lens_cpu

            if not self.use_hybrid_kvcache:
                # all layers share the same attention metadata
                uniform_attention_metadata = attention_metadata_gid
            else:
                for layer_name in kv_cache_group.layer_names:
                    attention_metadata_per_layer[
                        layer_name] = attention_metadata_gid

        if self.scheduler_config.async_scheduling and len(
                token_in_tpu_cur_input_indices) > 0:
            assert self._pre_async_results is not None
            input_ids = self._apply_async_token_substitution(
                input_ids, token_in_tpu_cur_input_indices,
                token_in_tpu_pre_next_tokens_indices)

        if self.lora_config is not None:
            self.lora_utils.set_active_loras(
                num_scheduled_tokens_per_req, total_num_scheduled_tokens,
                padded_total_num_scheduled_tokens)
        logits_indices_selector = None

        if self.use_hybrid_kvcache:
            attention_metadata = attention_metadata_per_layer
        else:
            attention_metadata = uniform_attention_metadata
        return (input_ids, positions, attention_metadata, sampling_metadata,
                logits_indices, spec_decode_metadata, logits_indices_selector,
                padded_num_reqs)

    def _get_input_ids_embeds(self, input_ids: jax.Array,
                              mm_embeds: list[jax.Array]):
        if self.is_multimodal_model:
            inputs_embeds = self.embed_input_ids_fn(
                self.state,
                input_ids,
                mm_embeds,
            )
            return None, inputs_embeds
        else:
            return input_ids, None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.speculative_decoding_manager.take_draft_token_ids()

    ###### Local disagg utilities ######

    def get_kv_cache_for_block_ids(
        self,
        block_ids: List[int],
    ) -> List[jax.Array]:
        return self.kv_cache_manager.get_kv_cache_for_block_ids(block_ids)

    def transfer_kv_cache(self,
                          kv_cache_slices: List[jax.Array]) -> List[jax.Array]:
        return self.kv_cache_manager.transfer_kv_cache(kv_cache_slices)

    def insert_request_with_kv_cache(
        self,
        request: "Request",
        kv_cache_slices: List[jax.Array],
        block_ids: List[List[int]],
    ):
        return self.kv_cache_manager.insert_request_with_kv_cache(
            request, kv_cache_slices, block_ids)

    ###### RL framework integration ######

    def _sync_weights(
        self,
        updated_weights: jaxtyping.PyTree,
        mappings: Dict[str, Tuple[str, Tuple[str]]],
        transpose_keys: Dict[str, Tuple[int]],
        reshard_fn: Callable[[jaxtyping.PyTree, jaxtyping.PyTree],
                             jaxtyping.PyTree] = None
    ) -> None:
        """For RL framework integration."""
        if reshard_fn is not None:
            updated_weights = reshard_fn(updated_weights, self.state)
            shard = None
        else:
            shard = functools.partial(shard_put, mesh=self.mesh)
        self.state = transfer_state_with_mappings(
            src_state=updated_weights,
            tgt_state=self.state,
            mappings=mappings,
            transpose_keys=transpose_keys,
            shard=shard)

    def get_intermediate_tensor_spec(self, num_tokens: int):
        jax_dtype = to_jax_dtype(self.dtype)
        num_padded_tokens = runner_utils.get_padded_token_len(
            self.num_tokens_paddings, num_tokens)
        sharding = NamedSharding(self.mesh, PartitionSpec())
        hidden_size = self.model_config.get_hidden_size()
        spec = jax.ShapeDtypeStruct(shape=(num_padded_tokens, hidden_size),
                                    dtype=jax_dtype,
                                    sharding=sharding)
        tensor_spec = {"hidden_states": spec, "residual": spec}
        return tensor_spec

    def get_uuid_for_jax_transfer(self,
                                  scheduler_output: "VllmSchedulerOutput",
                                  rank: int, step: int) -> int:
        '''
        Get a uuid for jax.transfer, here we use the hash of
        scheduler_output + counter_step + sender's rank
        '''
        scheduler_output_str = ""
        if not scheduler_output.num_scheduled_tokens:
            scheduler_output_str = "empty_batch"
        else:
            scheduler_output_str = str(
                sorted(scheduler_output.num_scheduled_tokens.items()))
        unique_str = f'{scheduler_output_str} {step} {rank}'
        import hashlib
        hasher = hashlib.sha1()
        hasher.update(unique_str.encode('utf-8'))
        return int.from_bytes(hasher.digest()[:8], 'big')
