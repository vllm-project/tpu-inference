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

import functools
import logging
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import torch
import vllm.envs as vllm_envs
import vllm.lora.utils as lora_utils_mod
from flax import nnx
from jax._src.pallas.utils import next_power_of_2
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
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
                             LogprobsTensors, ModelRunnerOutput,
                             RoutedExpertsLists)
from vllm.v1.request import Request
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import \
    KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

import tpu_inference.envs as envs
from tpu_inference import utils as common_utils
from tpu_inference.core.sched.utils import DEFAULT_MAX_DECODE_STEPS
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D,
                                                  ShardingAxisName,
                                                  ShardingConfigManager)
from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.layers.jax.sample.sampling import (
    PromptLogprobsAsyncData, PromptLogprobsReqSnap,
    _jax_logprobs_copy_to_host_async, compute_and_gather_logprobs,
    compute_prompt_logprobs, sample)
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
from tpu_inference.runner.decode_loop import TpuSamplingState, continue_decode
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
from tpu_inference.spec_decode.jax.dflash import DFlashProposer
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.spec_decode.jax.utils import (
    concat_last_sampled_tokens_and_draft_tokens, extend_logits_simple,
    extract_last_sampled_tokens, filter_speculative_logprobs,
    process_and_extend_logits)
from tpu_inference.utils import (device_array, make_optimized_mesh,
                                 time_function, to_jax_dtype, to_torch_dtype)

# Patch to classify specific non-LoRA keys as base weights. Without this, vLLM's `add_lora`
# crashes with "unsupported LoRA weight" on base model parameters like `lm_head.weight`
# because upstream's `is_base_embedding_weights` uses a strict layer-name whitelist.
# Compare upstream: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/lora/utils.py#L97-L99
_orig_is_base_embedding_weights = getattr(lora_utils_mod,
                                          "is_base_embedding_weights", None)


def _patched_is_base_embedding_weights(name: str) -> bool:
    if name.endswith(("lm_head.weight", "embed_tokens.weight")):
        return True
    if _orig_is_base_embedding_weights is not None:
        return _orig_is_base_embedding_weights(name)
    return False


lora_utils_mod.is_base_embedding_weights = _patched_is_base_embedding_weights

# Target module updates in loaded modules to avoid broad sys.modules loops
for m in ("vllm.lora.utils", "vllm.lora.lora_model"):
    if m in sys.modules:
        mod = sys.modules[m]
        if hasattr(mod, "is_base_embedding_weights"):
            setattr(mod, "is_base_embedding_weights",
                    _patched_is_base_embedding_weights)

logger = init_logger(__name__)

logging.getLogger("torchax.tensor").setLevel(logging.ERROR)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8


@functools.partial(jax.jit, static_argnames=["dp_size", "tokens_per_dp"])
def _compute_active_mask(
    logits_indices: jax.Array,
    dp_size: int,
    tokens_per_dp: int,
) -> jax.Array:
    reqs_per_dp = logits_indices.shape[0] // dp_size
    active_reqs_mask = (logits_indices >= 0).reshape(dp_size, reqs_per_dp)
    if tokens_per_dp == reqs_per_dp:
        active_mask = logits_indices >= 0
    else:
        num_active_per_dp = active_reqs_mask.sum(axis=1, keepdims=True)
        active_mask = (jnp.arange(tokens_per_dp) < num_active_per_dp).ravel()
    return active_mask


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner.

    This class acts as a wrapper around the standard ModelRunnerOutput. Its
    primary purpose is to hold references to data still on the TPU device
    (like the `next_tokens` JAX array) without blocking the main thread.

    The `get_output()` method is called to resolve these async results,
    triggering the JAX device-to-host (CPU) data transfer and populating
    the final `ModelRunnerOutput` object.
    """

    def __init__(self,
                 model_runner_output: ModelRunnerOutput,
                 next_tokens: jax.Array,
                 num_reqs: int,
                 discard_sampled_tokens_req_indices: list[int],
                 logits_indices_selector: Optional[List[int]] = None,
                 logprobs_tensors: Optional[LogprobsTensors] = None,
                 prompt_logprobs_async_data: Optional[
                     "PromptLogprobsAsyncData"] = None,
                 expert_indices: Optional[jax.Array] = None,
                 total_num_scheduled_tokens: int = 0,
                 spec_decode_metadata: Optional[SpecDecodeMetadata] = None,
                 scheduler_output: Optional["VllmSchedulerOutput"] = None,
                 req_ids_dp: Optional[Dict] = None,
                 padded_num_scheduled_tokens_per_dp_rank: int = 0,
                 runner=None):
        self._model_runner_output = model_runner_output
        self._next_tokens = next_tokens
        self._num_reqs = num_reqs
        self._discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices
        self.logits_indices_selector: list[int] = logits_indices_selector
        self._logprobs_tensors = logprobs_tensors
        self._prompt_logprobs_async_data = prompt_logprobs_async_data
        self._expert_indices = expert_indices
        self._total_num_scheduled_tokens = total_num_scheduled_tokens
        self._spec_decode_metadata = spec_decode_metadata
        self._scheduler_output = scheduler_output
        self._req_ids_dp = req_ids_dp
        self._padded_num_scheduled_tokens_per_dp_rank = padded_num_scheduled_tokens_per_dp_rank
        self._runner = runner

    def get_output(self) -> ModelRunnerOutput:
        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self._runner, self._spec_decode_metadata, self._next_tokens,
            self.logits_indices_selector,
            self._discard_sampled_tokens_req_indices, self._num_reqs)

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids

        if self._logprobs_tensors is not None:
            # Use materialize to ensure logprobs are ready on host when we return async results
            self._model_runner_output.logprobs = _jax_logprobs_materialize(
                self._logprobs_tensors,
                self.logits_indices_selector,
                spec_decode_metadata=self._spec_decode_metadata,
                runner=self._runner,
                num_reqs=self._num_reqs)

        if self._prompt_logprobs_async_data is not None:
            self._model_runner_output.prompt_logprobs_dict = (
                self._runner._get_prompt_logprobs_dict(
                    self._prompt_logprobs_async_data))

        if self._runner.model_config.enable_return_routed_experts and self._expert_indices is not None:
            expert_indices_cpu = np.asarray(
                jax.device_get(self._expert_indices))

            if self._scheduler_output is not None:
                routed_experts = _reconstruct_routed_experts(
                    runner=self._runner,
                    scheduler_output=self._scheduler_output,
                    expert_indices_cpu=expert_indices_cpu,
                    req_ids=self._model_runner_output.req_ids,
                    req_ids_dp=self._req_ids_dp,
                    padded_num_scheduled_tokens_per_dp_rank=self.
                    _padded_num_scheduled_tokens_per_dp_rank,
                )
                self._model_runner_output.routed_experts = routed_experts

        return self._model_runner_output


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens: jax.Array
    request_seq_lens: list[tuple[int, CachedRequestState, int]]
    discard_sampled_tokens_req_indices: list[int]
    placeholder_req_id_to_index: dict[str, int]
    logits_indices_selector: Optional[List[int]] = None
    scheduler_output: "VllmSchedulerOutput" = None

    # Only when spec decoding is enabled, the follow variables
    # are populated.
    spec_decode_next_tokens: Optional[
        jax.Array] = None  # [max_num_reqs * (gamma + 1)]
    spec_decode_num_rejected_tokens: Optional[
        jax.Array] = None  # [max_num_reqs]
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None


@dataclass
class ExecuteModelState:
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "VllmSchedulerOutput"
    attn_metadata: AttentionMetadata
    shared_attn_metadata: SharedAttentionMetadata
    sampling_metadata: TPUSupportedSamplingMetadata
    input_ids: Optional[jax.Array]
    hidden_states: jax.Array
    logits: jax.Array
    aux_hidden_states: Optional[jax.Array]
    spec_decode_metadata: Optional[SpecDecodeMetadata]
    kv_connector_output: Optional[KVConnectorOutput]
    logits_indices_selector: Optional[List[int]] = None
    padded_num_reqs: Optional[int] = None
    expert_indices: Optional[jax.Array] = None
    full_hidden_states: Optional[jax.Array] = None
    # Prompt logprobs fields: populated only when any request has prompt_logprobs set.
    full_logits: Optional[jax.Array] = None
    req_ids_dp: Optional[Dict] = None
    padded_num_scheduled_tokens_per_dp_rank: int = 0


@jax.jit(donate_argnums=(0, 1, 2))
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


@jax.jit(donate_argnums=(0, 1))
def _subtract_num_rejected_tokens_fn(seq_lens: jax.Array, positions: jax.Array,
                                     num_rejected_tokens: jax.Array,
                                     seq_lens_subtract_indices: jax.Array,
                                     positions_subtract_indices: jax.Array):
    """Subtract the previous step's rejected-token counts from `seq_lens` and
    `positions`.
    """
    seq_valid = seq_lens_subtract_indices >= 0
    seq_subtract = jnp.where(seq_valid,
                             num_rejected_tokens[seq_lens_subtract_indices], 0)
    seq_lens = seq_lens - seq_subtract

    pos_valid = positions_subtract_indices >= 0
    pos_subtract = jnp.where(pos_valid,
                             num_rejected_tokens[positions_subtract_indices],
                             0)
    if positions.ndim == 2:
        pos_subtract = jnp.expand_dims(pos_subtract, axis=0)
    positions = positions - pos_subtract
    return seq_lens, positions


def _jax_logprobs_materialize(
        logprobs_tensors: LogprobsTensors,
        logits_indices_selector: Optional[List[int]] = None,
        cu_num_generated_tokens: Optional[Any] = None,
        spec_decode_metadata: Optional[SpecDecodeMetadata] = None,
        runner: Optional[Any] = None,
        num_reqs: Optional[int] = None) -> LogprobsLists:
    """Materializes logprobs from JAX arrays into NumPy-backed LogprobsLists."""
    log_token_ids = np.asarray(
        jax.device_get(logprobs_tensors.logprob_token_ids))
    logprobs_arr = np.asarray(jax.device_get(logprobs_tensors.logprobs))
    selected_token_ranks = np.asarray(
        jax.device_get(logprobs_tensors.selected_token_ranks))

    # For speculative decoding, we need to filter and reorganize the materialized
    # logprobs. The raw logprobs contain info for all proposed draft tokens (including
    # rejected ones) and bonus tokens. We filter them to only keep the accepted
    # draft tokens and the actual bonus token for each request, and flatten them
    # back to match the output format.
    if (spec_decode_metadata is not None
            and np.sum(spec_decode_metadata.draft_lengths_cpu) > 0):
        assert runner is not None
        vocab_size = runner.input_batch.vocab_size
        dp_size = runner.dp_size
        num_reqs = runner.input_batch.num_reqs if num_reqs is None else num_reqs

        (
            log_token_ids,
            logprobs_arr,
            selected_token_ranks,
            cu_num_generated_tokens,
        ) = filter_speculative_logprobs(
            log_token_ids,
            logprobs_arr,
            selected_token_ranks,
            spec_decode_metadata,
            vocab_size,
            dp_size,
            num_reqs,
        )

    else:
        if logits_indices_selector is not None:
            log_token_ids = log_token_ids[logits_indices_selector]
            logprobs_arr = logprobs_arr[logits_indices_selector]
            selected_token_ranks = selected_token_ranks[
                logits_indices_selector]

        if cu_num_generated_tokens is None and runner is not None:
            num_reqs = runner.input_batch.num_reqs if num_reqs is None else num_reqs
            cu_num_generated_tokens = list(range(num_reqs + 1))

    return LogprobsLists(
        logprob_token_ids=np.array(log_token_ids.tolist()),
        logprobs=np.array(logprobs_arr.tolist()),
        sampled_token_ranks=np.array(selected_token_ranks.tolist()),
        cu_num_generated_tokens=cu_num_generated_tokens,
    )


def _reconstruct_slots_for_request(
    req_state: CachedRequestState,
    num_tokens: int,
    block_size: int,
    start_pos: int,
) -> np.ndarray:
    """Reconstructs physical KV-cache slots for ``num_tokens`` tokens of a
    request using vectorized NumPy.

    ``start_pos`` is the absolute sequence position of the first of these
    tokens and is supplied by the caller, because the two call sites have
    different token contracts: the routed-experts reconstruction passes a
    single scheduler chunk (whose tokens end at ``num_computed_tokens``, so it
    passes ``num_computed_tokens - num_tokens``), while the continue_decode
    path passes an on-device decode burst (not reflected in
    ``num_computed_tokens``, so it passes ``num_computed_tokens``). The slots
    must match the scheduler-side read (``RoutedExpertsManager.get``, which is
    block-relative from position 0).
    """
    if num_tokens <= 0:
        return np.array([], dtype=np.int32)

    assert start_pos >= 0, (
        f"[routed-experts] start_pos must be non-negative, got {start_pos} "
        f"(num_tokens={num_tokens}); slots would be wrong via numpy negative "
        f"indexing")
    block_ids = req_state.block_ids[0] if req_state.block_ids else []

    pos = np.arange(start_pos, start_pos + num_tokens, dtype=np.int32)
    block_idx = pos // block_size

    if block_ids:
        # Pad block_ids with 0s up to the max required index to avoid IndexError
        block_ids_arr = np.zeros(max(len(block_ids),
                                     int(block_idx[-1]) + 1),
                                 dtype=np.int32)
        block_ids_arr[:len(block_ids)] = block_ids
        block_id = block_ids_arr[block_idx]
        slots_arr = block_id * block_size + (pos % block_size)
    else:
        slots_arr = np.zeros_like(pos, dtype=np.int32)

    return slots_arr


def _reconstruct_routed_experts(
    runner,
    scheduler_output: "VllmSchedulerOutput",
    expert_indices_cpu: np.ndarray,
    req_ids: List[str],
    req_ids_dp: Dict,
    padded_num_scheduled_tokens_per_dp_rank: int,
) -> RoutedExpertsLists:
    """Reconstructs physical slot mappings and performs DP-rank reordering for MoE routed expert indices."""
    num_layers, _, top_k = expert_indices_cpu.shape
    block_size = runner.block_size
    total_active_tokens = scheduler_output.total_num_scheduled_tokens
    dp_size = runner.dp_size

    # Absolute start position of each request's chunk = its PRE-step
    # computed-token count, sourced from scheduler_output (not from
    # req_state.num_computed_tokens). req_state.num_computed_tokens is advanced
    # at different times across the two output paths that reach this function:
    # it is already advanced past this step on the async get_output path, but is
    # still the pre-step value on the sync _sample_from_logits path
    # (async_scheduling=False, required by continue_decode). scheduler_output
    # always carries the pre-step value, so it is correct for both; deriving
    # start_pos as `num_computed_tokens - n` underflows to a negative value on a
    # fresh prefill in the sync path.
    chunk_start = {}
    for new_req in scheduler_output.scheduled_new_reqs:
        chunk_start[new_req.req_id] = new_req.num_computed_tokens
    cached_reqs = scheduler_output.scheduled_cached_reqs
    for i, rid in enumerate(cached_reqs.req_ids):
        chunk_start[rid] = cached_reqs.num_computed_tokens[i]

    # 1. Compute global start offsets of every request in input batch order
    global_start_offsets = {}
    current_global_offset = 0
    for req_id in req_ids:
        global_start_offsets[req_id] = current_global_offset
        current_global_offset += scheduler_output.num_scheduled_tokens[req_id]

    # 2. Map DP-sharded indices to global contiguous input batch order,
    # and reconstruct physical slots in a single pre-allocated array.
    indices_map = np.zeros(total_active_tokens, dtype=np.int32)
    global_slots = np.zeros(total_active_tokens, dtype=np.int32)

    for dp_rank in range(dp_size):
        if dp_rank not in req_ids_dp:
            continue

        token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
        current_dp_offset = token_offset

        for req_id in req_ids_dp[dp_rank]:
            n = scheduler_output.num_scheduled_tokens[req_id]
            dp_start = current_dp_offset
            dp_end = dp_start + n
            current_dp_offset = dp_end

            global_start = global_start_offsets[req_id]
            global_end = global_start + n

            # Map sharded JAX positions to global contiguous positions
            indices_map[global_start:global_end] = np.arange(dp_start,
                                                             dp_end,
                                                             dtype=np.int32)

            # Reconstruct slots for this request using vectorized NumPy. Use the
            # pre-step chunk start from scheduler_output (see chunk_start above);
            # the slots must match the scheduler-side block-relative read
            # (RoutedExpertsManager.get, from position 0).
            req_state = runner.requests[req_id]
            if n > 0:
                global_slots[
                    global_start:global_end] = _reconstruct_slots_for_request(
                        req_state,
                        n,
                        block_size,
                        start_pos=chunk_start[req_id])

    # 3. Perform global rank reordering and transpose in a single fancy indexing sweep!
    expert_indices_reordered = expert_indices_cpu[:, indices_map, :].transpose(
        1, 0, 2)

    routed_experts = RoutedExpertsLists(routing_data=expert_indices_reordered,
                                        slot_mapping=global_slots)

    return routed_experts


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
        self.parallel_config: ParallelConfig = vllm_config.parallel_config
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
        self._init_aggregated_stats_logging()
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
        self._continue_decode_output = None
        self.batch_counter = 0

        self.kv_caches: list[jax.Array] = []
        self.layer_name_to_kvcache_index: dict[str, int] = {}

        self.is_pooling_model: bool = self.model_config.runner_type == "pooling"
        """Generative model or pooling model select different computations."""
        self.enable_continue_decode = self.vllm_config.additional_config.get(
            "enable_continue_decode", False)
        self.static_max_decode_steps = self.vllm_config.additional_config.get(
            "max_decode_steps", DEFAULT_MAX_DECODE_STEPS)
        self.eos_token_id = runner_utils.get_eos_token_id(self.model_config)
        self.pad_token_id = runner_utils.get_pad_token_id(self.model_config)

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
        sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
        mesh_shape = (
            sharding_config.model_dp_size,
            sharding_config.attn_dp_size,
            sharding_config.attn_dp_expert_size,
            sharding_config.expert_size,
            sharding_config.tp_size,
            sharding_config.decode_cp_size,
        )

        # Attempt to create a physically optimized mesh. Fall back to a simple
        # logical reshape for non-power-of-two device counts (e.g., DP=6) to
        # bypass strict physical topology constraints.
        try:
            return mesh_utils.create_device_mesh(
                mesh_shape,
                self.devices,
                allow_split_physical_axes=True,
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning(
                "Physical mesh creation failed (shape=%s, devices=%d). "
                "Falling back to logical reshape. Error: %s", mesh_shape,
                len(self.devices), e)
            return np.array(self.devices).reshape(mesh_shape)

    def _create_multi_slice_mesh(self, num_slices: int) -> jax.Array:
        sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
        dp_inner = sharding_config.model_dp_size // num_slices

        # Splits data parallelism across multiple slices.
        ici_mesh_shape = (
            dp_inner,
            sharding_config.attn_dp_size,
            sharding_config.attn_dp_expert_size,
            sharding_config.expert_size,
            sharding_config.tp_size,
            sharding_config.decode_cp_size,
        )
        dcn_mesh_shape = (num_slices, 1, 1, 1, 1, 1)

        # Attempt to create a physically optimized hybrid mesh (ICI + DCN).
        # Fall back to a logical reshape for non-power-of-two device counts
        # to bypass strict hardware topology constraints across slices.
        try:
            return mesh_utils.create_hybrid_device_mesh(
                mesh_shape=ici_mesh_shape,
                dcn_mesh_shape=dcn_mesh_shape,
                devices=self.devices,
                allow_split_physical_axes=True,
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning(
                "Hybrid physical mesh creation failed. Falling back to logical reshape. "
                "ICI shape: %s, DCN shape: %s, Error: %s", ici_mesh_shape,
                dcn_mesh_shape, e)
            return np.array(self.devices).reshape(
                tuple(i * d for i, d in zip(ici_mesh_shape, dcn_mesh_shape)))

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
            # Under MPMD each DP rank runs its own profiler on the same host
            # and produces an identically-named xplane.pb (same hostname,
            # same JAX worker id). Without a per-rank segment they collide on
            # the second-resolution timestamp dir and one rank's capture
            # overwrites another's.
            dp_rank = self.parallel_config.data_parallel_index if envs.TPU_MULTIPROCESS_DP else 0
            self.phase_based_profiler = runner_utils.PhasedBasedProfiler(
                self.phased_profiling_dir, worker_rank=dp_rank)

    def _init_aggregated_stats_logging(self) -> None:
        self.aggregated_stats_dir = envs.AGGREGATED_STATS_DIR
        self.aggregated_stats_logger = None
        # Only enable stats aggregation on one worker to avoid duplicate records
        if self.aggregated_stats_dir and self.rank == 0:
            self.aggregated_stats_logger = runner_utils.AggregatedStatsLogger(
                self.aggregated_stats_dir)

    def _init_mm(self) -> None:
        self.is_multimodal_model = None
        self.uses_mrope = self.model_config.uses_mrope
        self.supports_mm_inputs = True

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        if self.speculative_config:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "dflash":
                self.drafter = DFlashProposer(self.vllm_config, self)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self.vllm_config, self)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler(self.mesh)

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

        additional_sizes = self.vllm_config.additional_config.get(
            "compilation_sizes", [])
        # [16, 32, 64, 128, 256, 512, 1024, 2048]
        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype == "auto":
            cache_dtype = self.dtype
        kv_cache_dtype = to_jax_dtype(cache_dtype)
        kv_packing = common_utils.get_dtype_packing(kv_cache_dtype)
        self.num_tokens_paddings = runner_utils.get_token_paddings(
            min_token_size=max(envs.MIN_TOKEN_BUCKET,
                               next_power_of_2(self.dp_size * kv_packing)),
            max_token_size=scheduler_config.max_num_batched_tokens *
            self.dp_size,
            padding_gap=vllm_envs.VLLM_TPU_BUCKET_PADDING_GAP)
        self.num_tokens_paddings = sorted(self.num_tokens_paddings +
                                          additional_sizes)
        self.num_tokens_paddings_per_dp = [
            padding // self.dp_size for padding in self.num_tokens_paddings
        ]
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        self.requests: dict[str, CachedRequestState] = {}
        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, jax.Array] = {}

        # Use parallel_config as the primary source of truth during initialization
        # to avoid AttributeError when self.mesh is not yet assigned (common in unit tests).
        tp_size = 1
        if self.vllm_config.parallel_config is not None:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        elif hasattr(self, 'mesh') and self.mesh is not None:
            tp_size = self.mesh.shape.get(ShardingAxisName.MODEL, 1)

        self.vocab_size = common_utils.align_to(model_config.get_vocab_size(),
                                                tp_size)
        num_speculative_tokens = 0
        if self.vllm_config.speculative_config:
            num_speculative_tokens = self.vllm_config.speculative_config.num_speculative_tokens

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            pin_memory=False,
            vocab_size=self.vocab_size,
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            num_speculative_tokens=num_speculative_tokens,
            dp_size=self.dp_size,
        )

        self.positions_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)
        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_cpu = np.arange(self.max_num_tokens, dtype=np.int64)
        min_num_reqs = max(MIN_NUM_SEQS, next_power_of_2(self.dp_size))
        self.num_reqs_paddings = runner_utils.get_req_paddings(
            min_req_size=min_num_reqs, max_req_size=self.max_num_reqs)

        # The num_reqs paddings for attention only, by default it is
        # the max_reqs. If ATTN_BUCKETIZE_NUM_REQS=true, it is the
        # power-of-two between min and max reqs.
        # User can set ATTN_CUSTOM_NUM_REQS_BUCKETS to provide custom buckets.
        self.attn_num_reqs_paddings_per_dp = runner_utils.get_attn_req_paddings(
            min_req_size=MIN_NUM_SEQS,
            max_req_size=scheduler_config.max_num_seqs)
        self.attn_num_reqs_paddings = [
            padding * self.dp_size
            for padding in self.attn_num_reqs_paddings_per_dp
        ]

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

        # tensors for structured decoding
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

        # Contiguous buffer for metadata array. By using a single buffer, we are
        # able to avoid the overhead of multiple device_put operations.
        # Initialize to a constant size, and resize later after kv cache size is
        # known.
        self.device_buffer = common_utils.DeviceBuffer(initial_capacity=1024)
        # Cache a zero scalar JAX array to avoid eager allocation overhead during continue_decode cycles.
        self.zero_array = jnp.array(0, dtype=jnp.int32)

    def load_model(self):
        with set_current_vllm_config(self.vllm_config):
            model = get_model(
                self.vllm_config,
                self.rng_key,
                self.mesh,
            )

            # Only pre-warm torchax for pooling tasks to prevent JIT during inference
            # while avoiding unnecessary overhead for standard generative models.
            if self.is_pooling_model:
                import torchax
                from torchax.interop import torch_view

                h_size = self.model_config.get_hidden_size()
                # Use the actual model mesh and ATTN_DATA axis for sharding alignment
                pooling_sharding = NamedSharding(
                    self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))

                logger.info(
                    f"Pre-warming StepPooler for shapes: {self.num_tokens_paddings}"
                )
                with torchax.default_env():
                    for max_tokens in self.num_tokens_paddings:
                        # Create a dummy JAX array with exact shape and sharding metadata
                        dummy_jax = jnp.zeros((max_tokens, h_size),
                                              dtype=jnp.bfloat16)
                        # Shard the array to match real-time hidden_states
                        dummy_jax = jax.device_put(dummy_jax, pooling_sharding)

                        # Trigger the casting and host-transfer kernels
                        # Using non_blocking=False to ensure AOT compilation completes during load
                        _ = torch_view(dummy_jax).to('cpu', non_blocking=False)
                logger.debug("Universal StepPooler pre-warming successful.")

            self.state = model.state
            self.model = model.model
            self.state_leaves = model.state_leaves

            if self.drafter is not None:
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.state)

        self.model_fn = model.model_fn
        self.compute_logits_fn = model.compute_logits_fn
        self.pooler_fn = model.pooler_fn
        self.combine_hidden_states_fn = model.combine_hidden_states_fn
        # For the flax_nnx path, `model_fn` (== `run_model`) accepts a flat
        # tuple of array leaves and reconstructs the nnx.State inside the
        # jit. Pre-flatten here so subsequent dispatches skip the per-call
        # walk of `nnx.Variable` wrappers
        self.lora_manager = model.lora_manager

        self.precompile_vision_encoder_fn = model.multimodal_fns.precompile_vision_encoder_fn
        self.embed_multimodal_fn = model.multimodal_fns.embed_multimodal_fn
        self.embed_input_ids_fn = model.multimodal_fns.embed_input_ids_fn
        self.get_mrope_input_positions_fn = model.multimodal_fns.get_mrope_input_positions_fn

        rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()
        self.rng_params_for_sampling = device_array(self.mesh,
                                                    rng_key,
                                                    sharding=NamedSharding(
                                                        self.mesh,
                                                        PartitionSpec()))
        # This allows a multi-modal model to be used as text-only, assuming the user
        # passes the following to vLLM (on the CLI):
        # --limit-mm-per-prompt '{"image": 0, "video": 0}'
        disable_mm_from_limits = False
        if self.model_config.is_multimodal_model:
            mm_limits = self.model_config.multimodal_config.limit_per_prompt
            # According to https://github.com/vllm-project/vllm/blob/21d2b53f88d99f9ab369444f6d53ed2b9c260e4f/vllm/config/multimodal.py#L79-L95
            # if a modality limit is missing, we should treat count as 999. So here we disable multi-modality only when all limits are set to 0.
            if mm_limits and all(limit.count == 0
                                 for limit in mm_limits.values()):
                disable_mm_from_limits = True

            if disable_mm_from_limits:
                logger.warning(
                    f"Disabling multi-modality for model because limits are set to 0. {mm_limits=}"
                )

        self.is_multimodal_model = (self.model_config.is_multimodal_model
                                    and self.embed_multimodal_fn is not None
                                    and hasattr(self.model_config.hf_config,
                                                "architectures")
                                    and not disable_mm_from_limits)

        # Clear JIT compilation caches from weight loading to free XLA
        # program reservations (bytes_reserved) on TPU HBM.
        jax.clear_caches()
        logger.info("Cleared JIT caches after weight loading")

        logger.info(f"Init model | "
                    f"hbm={common_utils.hbm_usage_gb(self.devices)}GiB")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        runner_type = self.model_config.runner_type
        if runner_type == "generate":
            return ("generate", )
        if runner_type == "pooling":
            return (self.model_config.convert_type, )
        assert False, f"unsupported runner type: {runner_type}"

    def get_kv_cache_spec(self):
        return self.kv_cache_manager.get_kv_cache_spec()

    def get_kv_cache_layout(self):
        return self.kv_cache_manager.get_kv_cache_layout()

    def initialize_kv_cache(self,
                            kv_cache_config: KVCacheConfig,
                            topology_order_id: int = 0) -> None:
        self.topology_order_id = topology_order_id
        self.kv_cache_config = kv_cache_config
        self.use_hybrid_kvcache = len(kv_cache_config.kv_cache_groups) > 1
        self.kv_cache_manager.initialize_kv_cache(kv_cache_config)
        self.input_batch.has_mamba_layers = kv_cache_config.has_mamba_layers

        if self.kv_cache_manager.actual_mamba_num_blocks is not None:
            self.input_batch.init_mamba_pools(
                self.kv_cache_manager.actual_mamba_num_blocks)

        # This buffer grows dynamically to accommodate metadata and block tables.
        # We re-initialize with a precise capacity now that kv_cache_config is known.
        num_kv_groups = len(kv_cache_config.kv_cache_groups)
        sampling_params_size = 3 * self.max_num_reqs
        input_ids_size = self.max_num_tokens
        query_start_loc_size = self.max_num_reqs + self.dp_size
        seq_lens_size = self.max_num_reqs
        logits_indices_size = self.max_num_reqs
        # Block tables for each KV cache group.
        block_tables_size = num_kv_groups * self.max_num_reqs * self.max_num_blocks_per_req

        initial_capacity = (sampling_params_size + input_ids_size +
                            query_start_loc_size + seq_lens_size +
                            logits_indices_size + block_tables_size)
        self.device_buffer = common_utils.DeviceBuffer(
            initial_capacity=initial_capacity)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self)

    def delete_kv_cache(self) -> None:
        self.kv_cache_manager.delete_kv_cache()

    def reinitialize_kv_cache(self) -> None:
        self.kv_cache_manager.reinitialize_kv_cache()

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
        reqs = self.input_batch.num_reqs
        toks = scheduler_output.total_num_scheduled_tokens

        req_id_kwargs = {}
        if jax.profiler.TraceAnnotation.is_enabled():
            req_id_kwargs = runner_utils.extract_request_ids_for_tracing(
                self.input_batch, scheduler_output)

        with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation(
                f"execute_model: {reqs} reqs, {toks} toks", **req_id_kwargs):
            output = self._execute_model(scheduler_output,
                                         intermediate_tensors)
        return output

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self._continue_decode_output is not None:
            output = self._continue_decode_output
            self._continue_decode_output = None
            return output

        if self.execute_model_state is None:
            # This can happen in pipeline parallel case.
            return EMPTY_MODEL_RUNNER_OUTPUT

        (scheduler_output, attn_metadata, sampling_metadata, input_ids,
         hidden_states, logits, aux_hidden_states, spec_decode_metadata,
         kv_connector_output, logits_indices_selector, padded_num_reqs,
         expert_indices, full_hidden_states, full_logits, req_ids_dp,
         padded_num_scheduled_tokens_per_dp_rank) = (
             self.execute_model_state.scheduler_output,
             self.execute_model_state.attn_metadata,
             self.execute_model_state.sampling_metadata,
             self.execute_model_state.input_ids,
             self.execute_model_state.hidden_states,
             self.execute_model_state.logits,
             self.execute_model_state.aux_hidden_states,
             self.execute_model_state.spec_decode_metadata,
             self.execute_model_state.kv_connector_output,
             self.execute_model_state.logits_indices_selector,
             self.execute_model_state.padded_num_reqs,
             self.execute_model_state.expert_indices,
             self.execute_model_state.full_hidden_states,
             self.execute_model_state.full_logits,
             self.execute_model_state.req_ids_dp,
             self.execute_model_state.padded_num_scheduled_tokens_per_dp_rank,
         )
        self.execute_model_state = None

        with jax.set_mesh(self.mesh):
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
                scheduler_output, attn_metadata, sampling_metadata, input_ids,
                hidden_states, logits, aux_hidden_states, spec_decode_metadata,
                kv_connector_output, logits_indices_selector, padded_num_reqs,
                expert_indices, full_hidden_states, full_logits, req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank)

    def _modify_prev_results(self):
        # If copy to host has not been done, we just wait.
        # device_get should return immediately as we have scheduled it in previous function call.
        assert self._pre_async_results is not None, "When we call _modify_prev_results(), self._pre_async_results should already exist"
        pre_req_ids = self._pre_async_results.req_ids
        pre_num_reqs = len(pre_req_ids)
        pre_next_tokens = self._pre_async_results.next_tokens
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices
        pre_logits_indices_selector = self._pre_async_results.logits_indices_selector
        pre_spec_decode_metadata = self._pre_async_results.spec_decode_metadata
        pre_scheduler_output = self._pre_async_results.scheduler_output

        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self, pre_spec_decode_metadata, pre_next_tokens,
            pre_logits_indices_selector,
            pre_discard_sampled_tokens_req_indices, pre_num_reqs)

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
            pre_num_placeholder_tokens = 1
            assert pre_scheduler_output is not None
            pre_num_placeholder_tokens += len(
                pre_scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, []))

            end_idx = self.input_batch.num_tokens_no_spec[req_idx]
            num_sampled_tokens = len(sampled_ids)
            assert num_sampled_tokens <= pre_num_placeholder_tokens
            start_idx = end_idx - pre_num_placeholder_tokens
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx, start_idx:start_idx +
                                           num_sampled_tokens] = sampled_ids
            self.input_batch.num_tokens_no_spec[
                req_idx] = start_idx + num_sampled_tokens
            self.input_batch.num_tokens[
                req_idx] = start_idx + num_sampled_tokens
            # Replace previous placeholder
            for j in range(pre_num_placeholder_tokens):
                req_state.output_token_ids.pop()
            req_state.output_token_ids.extend(sampled_ids)

    def _update_placeholder(
            self,
            discard_sampled_tokens_req_indices,
            request_seq_lens,
            scheduler_output: "VllmSchedulerOutput",
            logits_indices_selector=None,
            spec_decode_metadata: Optional[SpecDecodeMetadata] = None):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_sampled_tokens_req_indices_set = set(
            discard_sampled_tokens_req_indices)
        for req_idx, req_state, _ in request_seq_lens:
            if req_idx in discard_sampled_tokens_req_indices_set:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + 1
            if req_state.req_id in scheduler_output.scheduled_spec_decode_tokens:
                end_idx += len(scheduler_output.scheduled_spec_decode_tokens[
                    req_state.req_id])
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            # Update cpu tokens at next execute and prepare input from tpu
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            # For placeholder, should be update on next execute.
            req_state.output_token_ids.extend([0] * (end_idx - start_idx))
            if logits_indices_selector is None:
                placeholder_req_id_to_index[req_state.req_id] = req_idx
            else:
                placeholder_req_id_to_index[
                    req_state.req_id] = logits_indices_selector[req_idx]

        if spec_decode_metadata is not None:
            placeholder_req_id_to_index.clear()
            for rank in range(self.dp_size):
                for i, req_id in enumerate(
                        spec_decode_metadata.req_ids_dp[rank]):
                    placeholder_req_id_to_index[req_id] = i + rank * (
                        self.max_num_reqs // self.dp_size)

        return placeholder_req_id_to_index

    def _execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> JaxIntermediateTensors | ModelRunnerOutput | None:
        self.persistent_batch_manager.update_states(
            scheduler_output, self.get_mrope_input_positions_fn)
        if not scheduler_output.total_num_scheduled_tokens:
            if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
                self._modify_prev_results()
                self._pre_async_results = None

            if has_kv_transfer_group():
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

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
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Check if the entire batch is in the decode phase.
        # request_distribution[0] tracks the number of decode requests.
        is_decode_only = self.input_batch.request_distribution[
            0] == self.input_batch.num_reqs
        if is_decode_only and self.enable_continue_decode:
            return self._execute_continue_decode(scheduler_output)

        # TODO(pooyam): I guess we can remove returning sampling_metadata in `_prepare_inputs` after https://github.com/njhill/vllm/commit/b7433ca1a47732394b1bdea4099d98389515954b
        (
            input_ids,
            input_positions,
            attn_metadata,
            shared_attn_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
            req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank,
            _,
        ) = self._prepare_inputs(scheduler_output)

        # multi-modal support
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            # We have the modality embeds at this time.
            self.mm_manager.execute_mm_encoder(scheduler_output)
            mm_embeds, is_mm_embed = self.mm_manager.gather_mm_embeddings(
                scheduler_output, input_ids.shape[0], req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank)
        else:
            mm_embeds, is_mm_embed = None, None

        if self.is_multimodal_model and self.input_batch.num_prompt_logprobs:
            raise ValueError(
                "prompt_logprobs is not supported for multimodal models.")

        if self.speculative_config and self.input_batch.num_prompt_logprobs:
            raise ValueError(
                "prompt_logprobs is not supported with speculative decoding.")

        # NOTE(Wenlong): For multi-modal model,
        # it will embed the text tokens and merge with the existing modality embeds
        # Later, the multi-modality model will take the embedding as the input.
        # For text-only model, this does nothing. It will input the input_ids and
        # leave the embedding job inside the forward pass
        input_ids, inputs_embeds = self._get_input_ids_embeds(
            input_ids, mm_embeds, is_mm_embed)

        lora_metadata = self.lora_utils.extract_lora_metadata()
        # TODO: make _get_input_ids_embeds within this context
        # NOTE: right now, mm model will use embeddings as the input,
        # but text-only model will use input_ids
        with self.maybe_forbid_compile:
            # attn_metadata=None (and no num_tokens) is intentional: vLLM's
            # set_forward_context only builds DP metadata when
            # `attn_metadata is not None or num_tokens is not None`, so the
            # num_tokens-requiring MoE sequence-parallel path is skipped here.
            # Do NOT add num_tokens without also handling that path -- unlike the
            # VllmModelWrapper forward, this native path does not need it.
            with set_forward_context(
                    None,
                    self.vllm_config,
            ), self.maybe_get_kv_connector_output(
                    scheduler_output) as kv_connector_output:
                # NOTE(Wenlong): It takes both `input_ids` and `inputs_embeds`,
                # but one of them would be `None`
                (self.kv_caches, hidden_states, aux_hidden_states,
                 expert_indices) = self.model_fn(
                     self.state_leaves,
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
                     shared_attention_metadata=shared_attn_metadata,
                 )
            if not self.is_last_rank:
                assert isinstance(hidden_states, JaxIntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                hidden_states.expert_indices = expert_indices
                return hidden_states

        if self.is_pooling_model:
            num_reqs = self.input_batch.num_reqs

            # Retrieve sequence lengths
            seq_lens_view = self.device_buffer.get_view((self.max_num_reqs, ),
                                                        key="seq_lens")
            seq_lens = seq_lens_view[:num_reqs]

            pooling_metadata = self.input_batch.get_pooling_metadata()

            # Extract scheduled token counts for the current chunk
            num_scheduled_tokens = np.array([
                scheduler_output.num_scheduled_tokens[req_id]
                for req_id in self.input_batch.req_ids[:num_reqs]
            ],
                                            dtype=np.int32)

            # Call the pooler with the decoupled interface
            pooler_output = self.pooler_fn(
                hidden_states,
                pooling_metadata,
                seq_lens,
                num_scheduled_tokens,
            )

            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=pooler_output,
            )

        full_hidden_states = hidden_states

        if self.input_batch.num_prompt_logprobs:
            # Compute logits for ALL token positions once.
            full_logits = self.compute_logits_fn(
                self.state_leaves,
                full_hidden_states,
                lora_metadata,
            )
            logits = self._select_from_array_fn(full_logits, logits_indices,
                                                self.mesh)
        else:
            full_logits = None
            hidden_states = self._select_from_array_fn(hidden_states,
                                                       logits_indices,
                                                       self.mesh)
            logits = self.compute_logits_fn(
                self.state_leaves,
                hidden_states,
                lora_metadata,
            )

        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            attn_metadata=attn_metadata,
            shared_attn_metadata=shared_attn_metadata,
            sampling_metadata=sampling_metadata,
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits,
            aux_hidden_states=aux_hidden_states,
            spec_decode_metadata=spec_decode_metadata,
            kv_connector_output=kv_connector_output,
            logits_indices_selector=logits_indices_selector,
            padded_num_reqs=padded_num_reqs,
            expert_indices=expert_indices,
            full_hidden_states=full_hidden_states,
            full_logits=full_logits,
            req_ids_dp=req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank=
            padded_num_scheduled_tokens_per_dp_rank,
        )
        return None

    def _get_min_remaining_slots(self) -> int:
        # Conservatively calculate the minimum remaining token capacity based on max_model_len.
        num_tokens = self.input_batch.num_tokens[:self.input_batch.num_reqs]
        remaining_slots = self.max_model_len - num_tokens
        if len(remaining_slots) == 0:
            return 0
        return int(np.min(remaining_slots))

    def _execute_continue_decode(
        self,
        scheduler_output: "VllmSchedulerOutput",
    ) -> ModelRunnerOutput:
        (
            input_ids,
            input_positions,
            attn_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
            _,
            _,
            tokens_indices_selector,
        ) = self._prepare_inputs(scheduler_output)

        init_tokens = input_ids
        # Map active rows correctly across DP buckets by checking valid query locations.
        # Pad active_mask to match the full padded_total_num_scheduled_tokens length of init_tokens.
        tokens_per_dp = init_tokens.shape[0] // self.dp_size
        active_mask = _compute_active_mask(logits_indices, self.dp_size,
                                           tokens_per_dp)

        init_state = TpuSamplingState(
            current_tokens=init_tokens,
            active_mask=active_mask,
            attn_metadata=attn_metadata,
            step_counter=self.zero_array,
        )

        from tpu_inference.layers.jax.sample.sampling import sample

        min_remaining = self._get_min_remaining_slots()

        # Limit max_decode_steps to not cross block boundaries
        max_decode_steps = min(self.static_max_decode_steps, min_remaining)
        if max_decode_steps <= 0:
            max_decode_steps = 1
        max_decode_steps_arr = jnp.array(max_decode_steps, dtype=jnp.int32)

        lora_metadata = self.lora_utils.extract_lora_metadata()

        # Run continue-decode as a single JIT'd on-device loop (JAX while_loop with
        # donated KV cache) to avoid host syncs. EOS early-exit happens on-device.
        # Mirror standard path wrappers to preserve forward context and KV hooks.
        # attn_metadata=None (and no num_tokens) is intentional -- see the note in
        # the standard execute path: it skips vLLM's num_tokens-requiring MoE
        # sequence-parallel DP-metadata path, which this native runner does not use.
        with self.maybe_forbid_compile, \
             set_forward_context(None, self.vllm_config), \
             self.maybe_get_kv_connector_output(
                 scheduler_output) as kv_connector_output:
            (generated_tokens, final_kv_caches, final_state, final_rng,
             all_expert_indices, logprobs_tensors) = continue_decode(
                 state=self.state_leaves,
                 model_fn=getattr(self.model, "step_fn_no_options",
                                  self.model_fn),
                 compute_logits_fn=self.compute_logits_fn,
                 sample_fn=sample,
                 mesh=self.mesh,
                 sampling_metadata=sampling_metadata,
                 init_state=init_state,
                 kv_caches=self.kv_caches,
                 max_decode_steps=max_decode_steps_arr,
                 static_max_decode_steps=self.static_max_decode_steps,
                 eos_token_id=self.eos_token_id,
                 padding_token_id=self.pad_token_id,
                 rng=self.rng_params_for_sampling,
                 inputs_embeds=None,
                 layer_name_to_kvcache_index=tuple(
                     self.layer_name_to_kvcache_index.items()),
                 lora_metadata=lora_metadata,
                 intermediate_tensors=None,
                 is_first_rank=self.is_first_rank,
                 is_last_rank=self.is_last_rank,
                 dp_size=self.dp_size,
                 collect_expert_indices=getattr(
                     self.vllm_config.model_config,
                     "enable_return_routed_experts", False),
                 max_logprobs=self.model_config.max_logprobs,
                 logprobs_mode=self.model_config.logprobs_mode,
             )
        self.rng_params_for_sampling = final_rng

        self.kv_caches = final_kv_caches

        # continue_decode now returns fixed-size stacked buffers plus the
        # number of steps actually executed (early EOS exit can stop before
        # the cap). This is the single, end-of-run host transfer.
        if logprobs_tensors is not None:
            (
                generated_tokens_cpu,
                all_expert_indices_cpu,
                actual_steps,
                lp_token_ids_cpu,
                lp_vals_cpu,
                lp_ranks_cpu,
            ) = jax.device_get((
                generated_tokens,
                all_expert_indices,
                final_state.step_counter,
                logprobs_tensors.logprob_token_ids,
                logprobs_tensors.logprobs,
                logprobs_tensors.selected_token_ranks,
            ))
        else:
            generated_tokens_cpu, all_expert_indices_cpu, actual_steps = jax.device_get(
                (generated_tokens, all_expert_indices,
                 final_state.step_counter))
            lp_token_ids_cpu = None
            lp_vals_cpu = None
            lp_ranks_cpu = None

        actual_steps = int(actual_steps)
        generated_tokens_cpu = np.asarray(generated_tokens_cpu)[:actual_steps]
        if all_expert_indices_cpu is not None:
            all_expert_indices_cpu = np.asarray(
                all_expert_indices_cpu)[:actual_steps]

        if lp_token_ids_cpu is not None:
            lp_token_ids_cpu = np.asarray(lp_token_ids_cpu)[:actual_steps]
            lp_vals_cpu = np.asarray(lp_vals_cpu)[:actual_steps]
            lp_ranks_cpu = np.asarray(lp_ranks_cpu)[:actual_steps]

        # Expose request dimension as axis 0 after transpose: shape (batch_size, actual_steps)
        generated_tokens_cpu = generated_tokens_cpu.T
        if tokens_indices_selector is not None:
            # Realign physical rows back to logical input batch order upfront
            generated_tokens_cpu = generated_tokens_cpu[
                tokens_indices_selector]

            if all_expert_indices_cpu is not None:
                # Shape: (steps, layers, batch, top_k) -> realign batch dimension
                all_expert_indices_cpu = all_expert_indices_cpu[:, :,
                                                                tokens_indices_selector, :]

            if lp_token_ids_cpu is not None:
                lp_token_ids_cpu = lp_token_ids_cpu[:,
                                                    tokens_indices_selector, :]
                lp_vals_cpu = lp_vals_cpu[:, tokens_indices_selector, :]
                lp_ranks_cpu = lp_ranks_cpu[:, tokens_indices_selector]

        num_reqs = self.input_batch.num_reqs
        sampled_token_ids = []
        expert_indices_list = []
        expert_slots_list = []
        lp_token_ids_list = []
        lp_vals_list = []
        lp_ranks_list = []
        num_eos_hits = 0

        for req_idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests.get(req_id)
            tokens = generated_tokens_cpu[req_idx]

            eos_arr = np.atleast_1d(self.eos_token_id)
            is_eos = np.any(tokens[:, None] == eos_arr[None, :], axis=-1)
            eos_indices = np.where(is_eos)[0]
            if len(eos_indices) > 0:
                num_eos_hits += 1
                first_eos_idx = eos_indices[0]
                valid_tokens = tokens[:first_eos_idx + 1].tolist()
            else:
                valid_tokens = tokens.tolist()

            actual_len = len(valid_tokens)
            sampled_token_ids.append(valid_tokens)

            # 2. Update scheduler_output directly to the true trimmed length
            scheduler_output.num_scheduled_tokens[req_id] = actual_len

            # 3. Extract exact valid expert indices for this request
            if all_expert_indices_cpu is not None:
                # Slice to (actual_len, layers, top_k)
                req_experts = all_expert_indices_cpu[:actual_len, :,
                                                     req_idx, :]
                # Transpose to (layers, actual_len, top_k) and collect
                expert_indices_list.append(req_experts.transpose(1, 0, 2))

                # Reconstruct slots for this request. The on-device decode
                # burst length (actual_len) is decided on-device via EOS
                # early-exit and is not reflected in num_computed_tokens, so the
                # burst's tokens begin at num_computed_tokens.
                if req_state is not None and actual_len > 0:
                    slots_arr = _reconstruct_slots_for_request(
                        req_state,
                        actual_len,
                        self.block_size,
                        start_pos=req_state.num_computed_tokens)
                    expert_slots_list.append(slots_arr)

            if lp_token_ids_cpu is not None:
                # Shape: [actual_len, max_logprobs + 1]
                req_lp_token_ids = lp_token_ids_cpu[:actual_len, req_idx]
                req_lp_vals = lp_vals_cpu[:actual_len, req_idx]
                # Shape: [actual_len]
                req_lp_ranks = lp_ranks_cpu[:actual_len, req_idx]

                lp_token_ids_list.append(req_lp_token_ids)
                lp_vals_list.append(req_lp_vals)
                lp_ranks_list.append(req_lp_ranks)

            if req_state is not None:
                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(valid_tokens)
                if req_idx < self.max_num_reqs and end_idx <= self.max_model_len:
                    self.input_batch.token_ids_cpu[
                        req_idx, start_idx:end_idx] = valid_tokens
                    self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                    self.input_batch.num_tokens[req_idx] = end_idx
                    req_state.output_token_ids.extend(valid_tokens)

            if hasattr(
                    attn_metadata,
                    "seq_lens_cpu") and attn_metadata.seq_lens_cpu is not None:
                attn_metadata.seq_lens_cpu[req_idx] += len(valid_tokens)

        # 4. Concatenate along the token dimension (axis 1)
        expert_indices_cpu = None
        if expert_indices_list:
            expert_indices_cpu = np.concatenate(expert_indices_list, axis=1)

        routed_experts = None
        if expert_indices_cpu is not None and expert_slots_list:
            routing_data = expert_indices_cpu.transpose(1, 0, 2)
            slot_mapping = np.concatenate(expert_slots_list, axis=0)
            routed_experts = RoutedExpertsLists(
                routing_data=routing_data,
                slot_mapping=slot_mapping,
            )

        logprobs_lists = None
        if lp_token_ids_list:
            logprob_token_ids = np.concatenate(lp_token_ids_list, axis=0)
            logprobs_arr = np.concatenate(lp_vals_list, axis=0)
            sampled_token_ranks = np.concatenate(lp_ranks_list, axis=0)
            cu_num_generated_tokens = [0] + list(
                np.cumsum([len(x) for x in sampled_token_ids]))

            from vllm.v1.outputs import LogprobsLists
            logprobs_lists = LogprobsLists(
                logprob_token_ids=logprob_token_ids,
                logprobs=logprobs_arr,
                sampled_token_ranks=sampled_token_ranks,
                cu_num_generated_tokens=cu_num_generated_tokens,
            )

        termination_reason = "unknown"
        if actual_steps < max_decode_steps:
            termination_reason = "eos_hit"
        elif actual_steps == max_decode_steps:
            reasons = []
            if max_decode_steps == self.static_max_decode_steps:
                reasons.append("max_decode_steps_limit")
            if max_decode_steps == min_remaining or min_remaining <= 0:
                reasons.append("kv_cache_limit")
            termination_reason = "_".join(reasons) if reasons else "max_steps"

        logger.debug(
            "continue_decode finished: actual steps: %d, reason: %s, initial_active_reqs: %d (max_steps: %d, EOS hits: %d)",
            actual_steps, termination_reason, num_reqs, max_decode_steps,
            num_eos_hits)

        output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        if routed_experts is not None:
            output.routed_experts = routed_experts

        self._continue_decode_output = output
        return None

    def _sample_from_logits(
        self,
        scheduler_output: "VllmSchedulerOutput",
        attn_metadata: AttentionMetadata,
        tpu_sampling_metadata: TPUSupportedSamplingMetadata,
        input_ids: Optional[jax.Array],
        hidden_states: jax.Array,
        logits: jax.Array,
        aux_hidden_states: Optional[jax.Array],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        kv_connector_output: Optional[KVConnectorOutput],
        logits_indices_selector: Optional[List[int]] = None,
        padded_num_reqs: Optional[int] = None,
        expert_indices: Optional[jax.Array] = None,
        full_hidden_states: Optional[jax.Array] = None,
        full_logits: Optional[jax.Array] = None,
        req_ids_dp: Optional[Dict] = None,
        padded_num_scheduled_tokens_per_dp_rank: int = 0,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if padded_num_reqs is None:
            padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
                self.input_batch.num_reqs, self.max_num_reqs)

        if tpu_sampling_metadata.do_sampling:
            self.rng_params_for_sampling, step_rng = jax.random.split(
                self.rng_params_for_sampling)
        else:
            step_rng = self.rng_params_for_sampling

        processed_bonus_logits = None
        if spec_decode_metadata is None:
            logits = logits.astype(jnp.float32)
            with self.maybe_forbid_compile:
                next_tokens, processed_logits = sample(
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
                logits, spec_decode_metadata.bonus_logits_indices, self.mesh)
            bonus_token_ids, processed_bonus_logits = sample(
                bonus_rng,
                self.mesh,
                bonus_logits,
                tpu_sampling_metadata,
            )
            target_logits = self._select_from_array_fn(
                logits, spec_decode_metadata.target_logits_indices, self.mesh)
            assert input_ids is not None
            draft_token_ids = self._extract_draft_token_ids(
                input_ids, spec_decode_metadata.final_logits_indices,
                spec_decode_metadata.target_logits_indices)
            next_tokens = self.rejection_sampler(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=spec_decode_metadata.draft_lengths,
                draft_probs=None,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=tpu_sampling_metadata,
                key=rejection_rng,
            )

        logits = logits.astype(jnp.float32)
        if full_logits is not None:
            full_logits = full_logits.astype(jnp.float32)
        with self.maybe_forbid_compile:
            if tpu_sampling_metadata.logprobs:
                if spec_decode_metadata is not None:
                    with jax.set_mesh(self.mesh):
                        if (self.model_config.logprobs_mode
                                == "processed_logprobs"
                                and tpu_sampling_metadata.do_sampling):
                            extended_logits = process_and_extend_logits(
                                self.mesh, target_logits,
                                processed_bonus_logits, spec_decode_metadata,
                                tpu_sampling_metadata)
                        else:
                            extended_logits = extend_logits_simple(
                                target_logits, bonus_logits, self.mesh)

                        logprobs_logits = extended_logits
                else:
                    logprobs_logits = (processed_logits
                                       if self.model_config.logprobs_mode
                                       == "processed_logprobs" else logits)
                logprobs = compute_and_gather_logprobs(
                    logprobs_logits, next_tokens,
                    self.model_config.max_logprobs)
                logprobs = _jax_logprobs_copy_to_host_async(logprobs)
            else:
                logprobs = None

            prompt_logprobs_async = compute_prompt_logprobs(
                full_logits,
                input_ids,
                self.input_batch.num_prompt_logprobs,
                self.requests,
                scheduler_output,
                req_ids_dp,
                self.dp_size,
                max_logprobs=self.model_config.max_logprobs,
            )

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

        spec_decode_last_sampled_token_id = None
        spec_decode_num_rejected_tokens = None
        if self.speculative_config:
            with self.maybe_forbid_compile, jax.set_mesh(self.mesh):
                last_sampled_token_id, num_rejected_tokens = extract_last_sampled_tokens(
                    spec_decode_metadata, next_tokens,
                    self.speculative_config.num_speculative_tokens,
                    self.input_batch.vocab_size,
                    self.max_num_reqs // self.dp_size, self.mesh)
                self.speculative_decoding_manager.propose_draft_token_ids(
                    next_tokens,
                    logits_indices_selector,
                    last_sampled_token_id,
                    num_rejected_tokens,
                    discard_sampled_tokens_req_indices,
                    aux_hidden_states,
                    attn_metadata,
                    bool(self.scheduler_config.async_scheduling),
                    spec_decode_metadata,
                    scheduler_output,
                    input_ids,
                    full_hidden_states,
                )
                spec_decode_last_sampled_token_id = last_sampled_token_id
                spec_decode_num_rejected_tokens = num_rejected_tokens

        spec_decode_next_tokens = None
        if self.speculative_config and self.scheduler_config.async_scheduling:
            assert spec_decode_last_sampled_token_id is not None
            with self.maybe_forbid_compile, jax.set_mesh(self.mesh):
                spec_decode_next_tokens = concat_last_sampled_tokens_and_draft_tokens(
                    spec_decode_last_sampled_token_id,
                    self.speculative_decoding_manager._draft_token_ids)

        # If async scheduler enabled
        if self.scheduler_config.async_scheduling:
            # Get previous results from TPU and replace the placeholder.
            if self._pre_async_results is not None:
                self._modify_prev_results()

            # Set placeholder for next tokens that is not yet generated
            placeholder_req_id_to_index: dict[
                str, int] = self._update_placeholder(
                    discard_sampled_tokens_req_indices, request_seq_lens,
                    scheduler_output, logits_indices_selector,
                    spec_decode_metadata)

            # Save the previous results
            next_tokens = jax.copy_to_host_async(next_tokens)
            self._pre_async_results = AsyncPreResults(
                req_ids=req_ids,
                next_tokens=next_tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=
                discard_sampled_tokens_req_indices,
                placeholder_req_id_to_index=placeholder_req_id_to_index,
                logits_indices_selector=logits_indices_selector,
                scheduler_output=scheduler_output,
                spec_decode_next_tokens=spec_decode_next_tokens,
                spec_decode_num_rejected_tokens=spec_decode_num_rejected_tokens,
                spec_decode_metadata=spec_decode_metadata,
            )

            # Return Model output to executor
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=self.input_batch.req_id_to_index.copy(),
                sampled_token_ids=[],  # Fill in async get
                logprobs=None,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
            )
            # Return async_model_runner_output
            async_model_runner_output = AsyncTPUModelRunnerOutput(
                model_runner_output,
                next_tokens,
                num_reqs,
                discard_sampled_tokens_req_indices,
                logits_indices_selector,
                logprobs_tensors=logprobs,
                prompt_logprobs_async_data=prompt_logprobs_async,
                expert_indices=expert_indices,
                total_num_scheduled_tokens=scheduler_output.
                total_num_scheduled_tokens,
                spec_decode_metadata=spec_decode_metadata,
                scheduler_output=scheduler_output,
                req_ids_dp=req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank=
                padded_num_scheduled_tokens_per_dp_rank,
                runner=self)
            return async_model_runner_output

        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self, spec_decode_metadata, next_tokens, logits_indices_selector,
            discard_sampled_tokens_req_indices, num_reqs)

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
            # Use materialize to ensure logprobs are ready on host when we return async results
            logprobs_lists = _jax_logprobs_materialize(
                logprobs,
                logits_indices_selector,
                spec_decode_metadata=spec_decode_metadata,
                runner=self,
                num_reqs=num_reqs)
        else:
            logprobs_lists = None

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=(
                self._get_prompt_logprobs_dict(prompt_logprobs_async)
                if prompt_logprobs_async else {}),
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        if self.model_config.enable_return_routed_experts and expert_indices is not None:
            expert_indices_cpu = np.asarray(jax.device_get(expert_indices))

            routed_experts = _reconstruct_routed_experts(
                runner=self,
                scheduler_output=scheduler_output,
                expert_indices_cpu=expert_indices_cpu,
                req_ids=self.input_batch.req_ids[:num_reqs],
                req_ids_dp=req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank=
                padded_num_scheduled_tokens_per_dp_rank,
            )
            model_runner_output.routed_experts = routed_experts

        return model_runner_output

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(2, ))
    def _select_from_array_fn(array, indices_to_select, mesh):

        def select_local_fn(local_array, local_indices):
            # XLA keeps the whole row-gather output in scoped VMEM, so one
            # gather of [n, vocab_shard] logits rows exceeds the 32M scoped
            # limit once n grows (e.g. spec decode with max_num_seqs=16:
            # [128, 62080] bf16 = 37.9M -> CompileTimeScopedVmemOom).
            # Gather in fixed-size row chunks so each gather stays small.
            chunk = 32
            n = local_indices.shape[0]
            if n <= chunk or local_array.ndim < 2:
                return local_array[local_indices]
            num_pads = (-n) % chunk
            idx = jnp.pad(local_indices, (0, num_pads)).reshape(-1, chunk)
            out = jax.lax.map(lambda ix: local_array[ix], idx)
            return out.reshape(-1, *local_array.shape[1:])[:n]

        ret = jax.shard_map(
            select_local_fn,
            mesh=mesh,
            in_specs=(PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA)),
            out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA))(
                array, indices_to_select)

        return ret

    @jax.jit(static_argnums=(0, ))
    def _extract_draft_token_ids(self, input_ids, logits_indices,
                                 target_logits_indices):

        def _fn(local_input_ids, local_logits_indices,
                local_target_logits_indices):
            draft_token_ids = local_input_ids[local_logits_indices]
            return draft_token_ids[local_target_logits_indices + 1]

        ret = jax.shard_map(
            _fn,
            mesh=self.mesh,
            in_specs=(PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA)),
            out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA))(
                input_ids, logits_indices, target_logits_indices)

        return ret

    def _get_prompt_logprobs_dict(
        self,
        data: "PromptLogprobsAsyncData",
    ) -> Dict[str, Any]:
        """Materializes TPU arrays on CPU and slices per-request prompt logprobs.
        Uses snapshotted request metadata (start_idx, num_logits) to safely slice
        async-copied tensors, which overlap with the execution of the next step.
        """
        token_ids_np = np.asarray(
            jax.device_get(data.tensors.logprob_token_ids))
        logprobs_np = np.asarray(jax.device_get(data.tensors.logprobs))
        ranks_np = np.asarray(jax.device_get(
            data.tensors.selected_token_ranks))

        prompt_logprobs_dict: Dict[str, Any] = {}
        completed_snaps: List[PromptLogprobsReqSnap] = []

        for snap in data.req_snaps:
            req_state = snap.req_state
            if snap.num_logits > 0:
                ids_buf, lp_buf, ranks_buf = (
                    req_state.in_progress_prompt_logprobs_cpu)
                s, n, k = snap.start_idx, snap.num_logits, snap.num_k
                o = snap.req_offset
                ids_buf[s:s + n] = token_ids_np[o:o + n, :k + 1]
                lp_buf[s:s + n] = logprobs_np[o:o + n, :k + 1]
                ranks_buf[s:s + n] = ranks_np[o:o + n]

            if snap.is_last_chunk:
                if req_state.in_progress_prompt_logprobs_cpu is not None:
                    ids_buf, lp_buf, ranks_buf = (
                        req_state.in_progress_prompt_logprobs_cpu)
                    # LogprobsTensors fields must be torch.Tensors: EngineCoreOutputs
                    # is msgpack-serialized between the EngineCore and client
                    # processes, and the type-driven decoder reconstructs these
                    # fields as torch.Tensor. numpy arrays serialize with a numpy
                    # dtype string (e.g. '<i4') that the tensor decoder rejects.
                    prompt_logprobs_dict[snap.req_id] = LogprobsTensors(
                        logprob_token_ids=torch.from_numpy(ids_buf.copy()),
                        logprobs=torch.from_numpy(lp_buf.copy()),
                        selected_token_ranks=torch.from_numpy(
                            ranks_buf.copy()),
                    )
                completed_snaps.append(snap)

        for snap in completed_snaps:
            self.input_batch.num_prompt_logprobs.pop(snap.req_id, None)
            snap.req_state.in_progress_prompt_logprobs_cpu = None

        return prompt_logprobs_dict

    def _prepare_input_metadata(self, scheduler_output: "VllmSchedulerOutput"):

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
            dp_rank = (scheduler_output.assigned_dp_rank[req_id]
                       if dp_size > 1 else 0)
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

        # Find maximum number of requests across DP ranks
        max_num_reqs_across_dp = max(
            len(req_ids) for req_ids in req_ids_dp.values())

        is_decode_only = (self.input_batch.request_distribution[0] ==
                          self.input_batch.num_reqs)

        padded_num_reqs_per_dp_rank = runner_utils.get_padded_token_len(
            self.num_reqs_paddings_per_dp, max_num_reqs_across_dp)
        if is_decode_only and self.enable_continue_decode:
            padded_num_scheduled_tokens_per_dp_rank = padded_num_reqs_per_dp_rank
        else:
            padded_num_scheduled_tokens_per_dp_rank = runner_utils.get_padded_token_len(
                self.num_tokens_paddings_per_dp,
                max_num_scheduled_tokens_across_dp)

        padded_total_num_scheduled_tokens = (
            padded_num_scheduled_tokens_per_dp_rank * dp_size)

        assert max_num_scheduled_tokens_across_dp > 0

        padded_num_reqs = padded_num_reqs_per_dp_rank * dp_size
        attn_padded_num_reqs = runner_utils.get_padded_token_len(
            self.attn_num_reqs_paddings_per_dp,
            max_num_reqs_across_dp) * dp_size

        # logits_indices_selector reorders per-rank outputs back to the
        # original batch ordering; with a single rank the ordering is already
        # the input-batch ordering, so no selector is needed.
        if dp_size > 1:
            all_req_indices = np.concatenate(
                [req_indices_dp[dp_rank] for dp_rank in range(dp_size)])
            all_positions = np.concatenate([
                np.arange(len(req_indices_dp[dp_rank])) +
                padded_num_reqs_per_dp_rank * dp_rank
                for dp_rank in range(dp_size)
            ])
            sorted_indices = np.argsort(all_req_indices)
            logits_indices_selector = all_positions[sorted_indices]

            tokens_indices_selector = None
            if self.enable_continue_decode:
                all_token_positions = np.concatenate([
                    np.arange(len(req_indices_dp[dp_rank])) +
                    padded_num_scheduled_tokens_per_dp_rank * dp_rank
                    for dp_rank in range(dp_size)
                ])
                tokens_indices_selector = all_token_positions[sorted_indices]
        else:
            logits_indices_selector = None
            tokens_indices_selector = None

        return (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
                scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
                padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
                attn_padded_num_reqs, padded_total_num_scheduled_tokens,
                padded_num_reqs_per_dp_rank, logits_indices_selector,
                tokens_indices_selector, max_num_reqs_per_dp_rank)

    def _prepare_async_token_substitution_indices(
            self, req_ids_dp, scheduled_tokens_per_dp_rank,
            padded_num_scheduled_tokens_per_dp_rank, dp_size):
        """Prepare token substitution indices for async scheduling."""
        # For input_ids substitution.
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}
        spec_decode_enabled = (self.speculative_config is not None)

        for dp_rank in range(dp_size):
            token_in_tpu_cur_input_indices_dp[dp_rank] = []
            token_in_tpu_pre_next_tokens_indices_dp[dp_rank] = []

            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]
            token_in_tpu_cur_input_indices_list = token_in_tpu_cur_input_indices_dp[
                dp_rank]
            token_in_tpu_pre_next_tokens_indices_list = token_in_tpu_pre_next_tokens_indices_dp[
                dp_rank]

            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            acc_cur_len = token_offset

            for i, req_id in enumerate(req_ids_dp[dp_rank]):
                acc_cur_len += num_scheduled_tokens_per_req[i]

                req_idx = self.input_batch.req_id_to_index[req_id]
                is_prefill = self.input_batch.num_computed_tokens_cpu[
                    req_idx] < self.input_batch.num_prompt_tokens[req_idx]

                # We need an explicit `is_prefill` check here because of preemption.
                # If a request is preempted and immediately resumed, it goes back to
                # the prefill stage. However, its `req_id` might still be in
                # `_pre_async_results.placeholder_req_id_to_index` from the previous step.
                # Without this check, we would incorrectly perform token substitution
                # for a resumed prefill request.
                if is_prefill:
                    # Skip substitution for prefill requests (including chunked prefill)
                    continue

                if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                    continue

                if not spec_decode_enabled:
                    # Treat as normal decode: substitute 1 token
                    assert num_scheduled_tokens_per_req[i] == 1, (
                        f"Expected 1 token for normal decode request {req_id}, "
                        f"but got {num_scheduled_tokens_per_req[i]}")
                    token_in_tpu_cur_input_indices_list.append(acc_cur_len - 1)
                    token_in_tpu_pre_next_tokens_indices_list.append(
                        self._pre_async_results.
                        placeholder_req_id_to_index[req_id])
                else:
                    max_num_spec_tokens = self.speculative_config.num_speculative_tokens
                    assert num_scheduled_tokens_per_req[
                        i] <= max_num_spec_tokens + 1
                    idx = self._pre_async_results.placeholder_req_id_to_index[
                        req_id]

                    base_offset = acc_cur_len - num_scheduled_tokens_per_req[i]
                    for j in range(num_scheduled_tokens_per_req[i]):
                        token_in_tpu_cur_input_indices_list.append(
                            base_offset + j)
                        token_in_tpu_pre_next_tokens_indices_list.append(
                            idx * (max_num_spec_tokens + 1) + j)

        return token_in_tpu_cur_input_indices_dp, token_in_tpu_pre_next_tokens_indices_dp

    def _apply_async_token_substitution(self, input, next_tokens_in_tpu,
                                        token_in_tpu_cur_input_indices,
                                        token_in_tpu_pre_next_tokens_indices):
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input

        idx_pad_len = len(input) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside self._substitute_placeholder_token_fn
        full_range = np.arange(0, len(input), dtype=np.int32)
        missing_values = np.setdiff1d(full_range,
                                      token_in_tpu_cur_input_indices)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1).astype(np.int32)
        placeholder_num = np.array([len(token_in_tpu_cur_input_indices)
                                    ]).astype(np.int32)

        (padded_token_in_tpu_cur_input_indices,
         padded_token_in_tpu_pre_next_tokens_indices,
         placeholder_num) = device_array(
             self.mesh,
             (padded_token_in_tpu_cur_input_indices,
              padded_token_in_tpu_pre_next_tokens_indices, placeholder_num))

        with self.maybe_forbid_compile:
            input = self._substitute_placeholder_token_fn(
                input, padded_token_in_tpu_cur_input_indices,
                padded_token_in_tpu_pre_next_tokens_indices,
                next_tokens_in_tpu, placeholder_num)
        return input

    def _subtract_num_rejected_tokens(self, seq_lens, positions, req_ids_dp,
                                      scheduled_tokens_per_dp_rank):
        """Apply rejection-count subtraction to seq_lens and positions if needed.

        `num_computed_tokens_cpu` was advanced on the host assuming every
        speculatively proposed token from the previous step was accepted. Here
        we subtract the actual rejection counts on TPU for the requests that
        ran spec decoding in the previous step.
        """
        assert self._pre_async_results is not None
        assert self._pre_async_results.spec_decode_num_rejected_tokens is not None
        seq_lens_subtract_indices = np.full(self.max_num_reqs,
                                            -1,
                                            dtype=np.int32)
        positions_subtract_indices = np.full(positions.shape[-1],
                                             -1,
                                             dtype=np.int32)

        for rank in range(self.dp_size):
            acc_cur_len = 0
            scheduled_tokens_cur_rank = scheduled_tokens_per_dp_rank[rank]
            for i, req_id in enumerate(req_ids_dp[rank]):
                acc_cur_len += scheduled_tokens_cur_rank[i]

                if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                    continue

                idx = self._pre_async_results.placeholder_req_id_to_index[
                    req_id]
                seq_lens_subtract_indices[
                    i + rank * (self.max_num_reqs // self.dp_size)] = idx
                base_offset = acc_cur_len - scheduled_tokens_cur_rank[i]
                for j in range(scheduled_tokens_cur_rank[i]):
                    positions_subtract_indices[
                        base_offset + j + rank *
                        (positions.shape[-1] // self.dp_size)] = idx

        seq_lens_subtract_indices, positions_subtract_indices = device_array(
            self.mesh, (seq_lens_subtract_indices, positions_subtract_indices))

        with self.maybe_forbid_compile:
            seq_lens, positions = _subtract_num_rejected_tokens_fn(
                seq_lens, positions,
                self._pre_async_results.spec_decode_num_rejected_tokens,
                seq_lens_subtract_indices, positions_subtract_indices)
        return seq_lens, positions

    def _prepare_inputs(self, scheduler_output: "VllmSchedulerOutput"):
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
         attn_padded_num_reqs, padded_total_num_scheduled_tokens,
         padded_num_reqs_per_dp_rank, logits_indices_selector,
         tokens_indices_selector, max_num_reqs_per_dp_rank
         ) = self._prepare_input_metadata(scheduler_output)
        # Multi-modal support
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self.mm_manager.calc_mrope_positions(
                scheduler_output, req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank)

        # Async scheduling: prepare token substitution indices for DP
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for (req_id, draft_token_ids
             ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # If async previous results exists, we will prepare for the token substitution here
            # The actual substitution will be performed in tpu during later parts of this function.
            (
                token_in_tpu_cur_input_indices_dp,
                token_in_tpu_pre_next_tokens_indices_dp,
            ) = self._prepare_async_token_substitution_indices(
                req_ids_dp, scheduled_tokens_per_dp_rank,
                padded_num_scheduled_tokens_per_dp_rank, dp_size)

        self.device_buffer.reset()

        input_ids_view = self.device_buffer.get_view(
            (padded_total_num_scheduled_tokens, ), key="input_ids")
        query_start_loc_view = self.device_buffer.get_view(
            (self.max_num_reqs + dp_size, ), key="query_start_loc")
        seq_lens_view = self.device_buffer.get_view((self.max_num_reqs, ),
                                                    key="seq_lens")

        if self.speculative_config:
            padded_logits_length = None
            for dp_rank in range(dp_size):
                cur_rank_req_idxs = req_indices_dp[dp_rank]
                cur_rank_num_draft_tokens = num_draft_tokens[cur_rank_req_idxs]

                num_sampled_tokens = cur_rank_num_draft_tokens + 1
                total_sampled_tokens = np.sum(num_sampled_tokens)
                if padded_logits_length is None:
                    padded_logits_length = runner_utils.get_padded_token_len(
                        self.num_logits_paddings, total_sampled_tokens)
                else:
                    padded_logits_length = max(
                        padded_logits_length,
                        runner_utils.get_padded_token_len(
                            self.num_logits_paddings, total_sampled_tokens))

            assert padded_logits_length is not None
            logits_indices_shape = (padded_logits_length * dp_size, )
        else:
            logits_indices_shape = (padded_num_reqs, )

        logits_indices_view = self.device_buffer.get_view(logits_indices_shape,
                                                          key="logits_indices")

        # Populates input_ids and positions
        for dp_rank in range(dp_size):
            if num_req_per_dp_rank[dp_rank] == 0:
                continue
            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]
            total_num_scheduled_tokens = num_scheduled_tokens_per_dp_rank[
                dp_rank]
            input_ids_cpu = input_ids_view[
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
            np.take(
                self.input_batch.token_ids_cpu.ravel(),
                token_indices,
                out=input_ids_cpu[:total_num_scheduled_tokens],
            )

            input_ids_cpu[total_num_scheduled_tokens:] = 0

        # Prepare the attention metadata (query_start_loc_cpu, seq_lens_cpu)
        for dp_rank in range(dp_size):
            req_offset = dp_rank * max_num_reqs_per_dp_rank
            query_start_loc_cpu = query_start_loc_view[
                req_offset + dp_rank:req_offset + max_num_reqs_per_dp_rank +
                dp_rank + 1]
            seq_lens_cpu = seq_lens_view[req_offset:req_offset +
                                         max_num_reqs_per_dp_rank]
            _num_reqs = num_req_per_dp_rank[dp_rank]
            req_indices = req_indices_dp[dp_rank]
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]

            if _num_reqs == 0:
                query_start_loc_cpu[:] = 0
                seq_lens_cpu[:] = 0
                continue

            # After buffer.reset(), the buffer is still dirty, so we need to zero
            # Out the starting index.
            query_start_loc_cpu[0] = 0
            np.cumsum(
                num_scheduled_tokens_per_req,
                out=query_start_loc_cpu[1:_num_reqs + 1],
            )
            query_start_loc_cpu[_num_reqs +
                                1:] = query_start_loc_cpu[_num_reqs]

            seq_lens_cpu[:_num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[req_indices] +
                num_scheduled_tokens_per_req)
            seq_lens_cpu[_num_reqs:] = 0

        # populate logits_indices
        for dp_rank in range(dp_size):
            req_offset = dp_rank * padded_num_reqs_per_dp_rank
            query_loc_req_offset = dp_rank * (max_num_reqs_per_dp_rank + 1)
            _num_reqs = num_req_per_dp_rank[dp_rank]

            logits_indices_cpu = logits_indices_view[
                req_offset:req_offset + padded_num_reqs_per_dp_rank]
            logits_indices_cpu[:_num_reqs] = (
                query_start_loc_view[query_loc_req_offset +
                                     1:query_loc_req_offset + _num_reqs + 1] -
                1)
            logits_indices_cpu[_num_reqs:] = -1

            # Calculate batch composition statistics for active hardware profilers
            # and/or continuous batch logging.
        if self.phase_based_profiler or self.aggregated_stats_logger:
            self.batch_counter += 1
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                self.batch_counter, self.input_batch,
                total_num_scheduled_tokens, num_reqs,
                padded_total_num_scheduled_tokens, scheduler_output)

            if self.phase_based_profiler:
                self.phase_based_profiler.step(batch_composition_stats)
            if self.aggregated_stats_logger:
                self.aggregated_stats_logger.log(batch_composition_stats)

        positions = self.positions_cpu[:padded_total_num_scheduled_tokens]
        mrope_positions = self.mrope_positions_cpu[:, :
                                                   padded_total_num_scheduled_tokens]
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
        request_distribution = np.array(_request_distribution,
                                        dtype=np.int32).ravel()

        spec_decode_metadata = None
        if self.speculative_config:
            spec_decode_metadata = (
                self.speculative_decoding_manager.get_spec_decode_metadata(
                    num_draft_tokens_dp=num_draft_tokens,
                    dp_size=dp_size,
                    req_indices_dp=req_indices_dp,
                    req_ids_dp=req_ids_dp,
                    query_start_loc=query_start_loc_view,
                    padded_num_reqs_per_dp_rank=padded_num_reqs_per_dp_rank,
                    padded_logits_length_dp_rank=(logits_indices_shape[0] //
                                                  dp_size),
                    max_num_reqs_per_dp_rank=max_num_reqs_per_dp_rank,
                ))
            logits_indices_view[:] = spec_decode_metadata.final_logits_indices

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh,
            self.input_batch,
            padded_num_reqs,
            sharding=data_parallel_attn_sharding,
            req_indices_dp=req_indices_dp,
        )

        if self.uses_mrope:
            # M-RoPE positions are of the shape (3, max_num_tokens).
            # https://github.com/vllm-project/tpu-inference/blob/efc9608acd925bb3b64db6fda509514f799ab7be/tpu_inference/runner/tpu_runner.py#L555
            # Shard the positions accordingly.
            mrope_sharding = NamedSharding(
                self.mesh, PartitionSpec(None, ShardingAxisName.ATTN_DATA))
            positions = device_array(self.mesh,
                                     mrope_positions,
                                     sharding=mrope_sharding)
        else:
            positions = device_array(self.mesh,
                                     positions,
                                     sharding=data_parallel_attn_sharding)

        # Collect block tables host arrays loops zone presence zones legality
        def build_block_table_host(kv_cache_gid: int) -> None:

            block_table_obj = self.input_batch.block_table[kv_cache_gid]
            block_tables_view = self.device_buffer.get_view(
                (self.max_num_reqs, block_table_obj.max_num_blocks_per_req),
                key=f"block_tables_gid_{kv_cache_gid}")

            # Zero out the view once for correct padding
            block_tables_view.fill(0)

            cpu_tensor = block_table_obj.get_cpu_tensor()
            for dp_rank in range(dp_size):
                _num_reqs = num_req_per_dp_rank[dp_rank]
                if _num_reqs == 0:
                    continue

                req_offset = dp_rank * max_num_reqs_per_dp_rank
                # Use np.take with out= to avoid intermediate copies from advanced indexing
                np.take(cpu_tensor,
                        req_indices_dp[dp_rank],
                        axis=0,
                        out=block_tables_view[req_offset:req_offset +
                                              _num_reqs])

        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            no_kv_cache = len(self.kv_cache_config.kv_cache_groups) == 0
            if not no_kv_cache:
                build_block_table_host(0)
        else:
            for gid, kv_cache_group in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                build_block_table_host(gid)

        metadata_blob, metadata_layout = self.device_buffer.build()

        # Mamba slot ids are only consumed by hybrid attn+mamba models; for
        # pure-attention models, leaving the field None keeps AttentionMetadata
        # byte-identical to the pre-compact-mamba layout (so the model_fn
        # signature on those models is unchanged).
        if self.kv_cache_config.has_mamba_layers:
            # Reorder mamba_state_indices per DP rank (like block_tables)
            # and convert global slot ids to rank-local indices so they
            # index correctly into the per-rank shard of the mamba state.
            local_slots = self.input_batch._mamba_local_slots
            mamba_state_indices_cpu = np.zeros(self.max_num_reqs,
                                               dtype=np.int32)
            for dp_rank in range(dp_size):
                _num_reqs = num_req_per_dp_rank[dp_rank]
                if _num_reqs == 0:
                    continue
                req_offset = dp_rank * max_num_reqs_per_dp_rank
                global_slots = self.input_batch.mamba_state_indices_cpu[
                    req_indices_dp[dp_rank]]
                mamba_state_indices_cpu[req_offset:req_offset +
                                        _num_reqs] = (global_slots %
                                                      local_slots)
            (request_distribution, mamba_state_indices,
             dev_arrays_payload) = device_array(
                 self.mesh, (request_distribution, mamba_state_indices_cpu,
                             metadata_blob),
                 sharding=data_parallel_attn_sharding)
        else:
            mamba_state_indices = None
            (request_distribution, dev_arrays_payload) = device_array(
                self.mesh, (request_distribution, metadata_blob),
                sharding=data_parallel_attn_sharding)

        metadata = common_utils.DeviceBuffer.unpack_arrays(
            dev_arrays_payload, metadata_layout)
        input_ids = metadata["input_ids"]
        query_start_loc = metadata["query_start_loc"]
        seq_lens = metadata["seq_lens"]
        logits_indices = metadata["logits_indices"]

        # The host-side `num_computed_tokens_cpu` assumes all speculatively
        # proposed tokens from the previous step were accepted. Subtract the
        # actual rejection counts from `seq_lens` and `positions` on TPU.
        if self.speculative_config and self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            seq_lens, positions = self._subtract_num_rejected_tokens(
                seq_lens, positions, req_ids_dp, scheduled_tokens_per_dp_rank)

        def build_attn(block_tables: jax.Array | None) -> AttentionMetadata:
            attention_metadata_gid = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                padded_num_reqs=attn_padded_num_reqs,
            )

            return attention_metadata_gid

        def build_shared_attn() -> SharedAttentionMetadata:
            return SharedAttentionMetadata(
                input_positions=positions,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                padded_num_reqs=attn_padded_num_reqs,
            )

        attention_metadata: AttentionMetadata | dict[str, AttentionMetadata]
        shared_attention_metadata: SharedAttentionMetadata
        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            # Pooling model will not using kv cache
            no_kv_cache = len(self.kv_cache_config.kv_cache_groups) == 0
            block_tables = metadata.get(
                "block_tables_gid_0") if not no_kv_cache else None
            attention_metadata = build_attn(block_tables)
            shared_attention_metadata = build_shared_attn()
        else:
            attention_metadata = {
                name: build_attn(metadata[f"block_tables_gid_{gid}"])
                for gid, kv_cache_group in enumerate(
                    self.kv_cache_config.kv_cache_groups)
                for name in kv_cache_group.layer_names
            }
            shared_attention_metadata = build_shared_attn()

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

            if self.scheduler_config.async_scheduling and self._pre_async_results:
                if self.speculative_config:
                    next_tokens = self._pre_async_results.spec_decode_next_tokens
                else:
                    next_tokens = self._pre_async_results.next_tokens
                token_in_tpu_cur_input_indices = np.array(
                    all_token_indices_to_substitute)
                token_in_tpu_pre_next_tokens_indices = np.array(
                    all_pre_next_tokens_indices)

                input_ids = self._apply_async_token_substitution(
                    input_ids, next_tokens, token_in_tpu_cur_input_indices,
                    token_in_tpu_pre_next_tokens_indices)

        num_scheduled_tokens_per_req = np.concatenate([
            np.array(scheduled_tokens_per_dp_rank[dp_rank], dtype=np.int32)
            for dp_rank in range(dp_size)
        ])
        if self.lora_config is not None:
            self.lora_utils.set_active_loras(
                num_scheduled_tokens_per_req,
                total_num_scheduled_tokens,
                padded_total_num_scheduled_tokens,
            )

        return (
            input_ids,
            positions,
            attention_metadata,
            shared_attention_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
            req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank,
            tokens_indices_selector,
        )

    def _get_input_ids_embeds(self, input_ids: jax.Array,
                              mm_embeds: list[jax.Array] | None,
                              is_mm_embed: jax.Array | None):
        # Prevent the cost of calling additional function.
        if self.is_multimodal_model and mm_embeds is not None:
            assert self.embed_input_ids_fn is not None
            inputs_embeds = self.embed_input_ids_fn(
                self.state_leaves,
                input_ids,
                mm_embeds,
                is_multimodal=is_mm_embed,
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
        # Keep the dispatch-side view in sync with the updated state so
        # subsequent jit dispatches see the new weights.
        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state

    def _get_padded_total_tokens(
            self, scheduler_output: "VllmSchedulerOutput") -> int:
        num_tokens = scheduler_output.total_num_scheduled_tokens

        # Determine the capacity per rank (max tokens assigned to any single device)
        max_tokens_per_rank = getattr(
            scheduler_output, "max_num_scheduled_tokens_per_dp_rank",
            (num_tokens + self.dp_size - 1) // self.dp_size)

        # Map to the next local bucket and multiply by world size to get global shape
        padded_per_rank = runner_utils.get_padded_token_len(
            self.num_tokens_paddings_per_dp, max_tokens_per_rank)

        return padded_per_rank * self.dp_size

    def get_intermediate_tensor_spec(self,
                                     scheduler_output: "VllmSchedulerOutput"):
        jax_dtype = to_jax_dtype(self.dtype)
        num_padded_tokens = self._get_padded_total_tokens(scheduler_output)

        if self.dp_size > 1:
            sharding = NamedSharding(
                self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        else:
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
