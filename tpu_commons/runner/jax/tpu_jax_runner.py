import functools
import os
import random
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import vllm.envs as envs
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.v1.request import Request
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import \
    KVConnectorModelRunnerMixin
from vllm.v1.worker.utils import (gather_mm_placeholders,
                                  scatter_mm_placeholders)

from tpu_commons import utils as common_utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.sharding import build_mesh
from tpu_commons.models.jax.layers.sample.rejection_sampler import \
    RejectionSampler
from tpu_commons.models.jax.layers.sample.sampling import (compute_logprobs,
                                                           gather_logprobs,
                                                           sample)
from tpu_commons.models.jax.layers.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_commons.models.jax.model_loader import get_model
from tpu_commons.models.jax.utils.multi_modal_utils import \
    sanity_check_mm_encoder_outputs
from tpu_commons.models.jax.utils.weight_utils import (
    shard_put, transfer_state_with_mappings)
from tpu_commons.runner import utils as runner_utils
from tpu_commons.runner.jax.compilation_manager import CompilationManager
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.jax.kv_cache_manager import KVCacheManager
from tpu_commons.runner.jax.metadata import SpecDecodeMetadata
from tpu_commons.runner.jax.persistent_batch_manager import \
    PersistentBatchManager
from tpu_commons.utils import make_optimized_mesh

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8

DUMMY_METADATA = AttentionMetadata(
    input_positions=[],
    block_tables=[],
    request_distribution=[0, 0, 0],
)


class TPUModelRunner(KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[Any],
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
        self._verify_chunked_prefill_config()

        self.devices = devices
        self.dtype = self.model_config.dtype

        self.phased_profiling_dir = os.getenv("PHASED_PROFILING_DIR", "")
        self.phase_based_profiler = None
        if self.phased_profiling_dir:
            self.phase_based_profiler = runner_utils.PhasedBasedProfiler(
                self.phased_profiling_dir)

        # multi-modal related
        self.is_multimodal_model = None  # Will get updated once the model is loaded.
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = self.model_config.uses_mrope

        # Set up speculative decoding.
        if self.speculative_config:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

        self._init_random()
        self._init_mesh()
        self._init_inputs()

        self.maybe_forbid_compile = runner_utils.ForbidCompile(
        ) if envs.VLLM_XLA_CHECK_RECOMPILATION else nullcontext()

        # Cached draft tokens.
        self._draft_token_ids: Optional[list[list[int]]] = None

        self.compilation_manager = CompilationManager(self)
        self.kv_cache_manager = KVCacheManager(self)
        self.persistent_batch_manager = PersistentBatchManager(
            self.requests, self.input_batch, self.encoder_cache,
            self.uses_mrope, self.model_config)

    def _verify_chunked_prefill_config(self):
        if (self.scheduler_config.max_num_batched_tokens
                < self.scheduler_config.max_num_seqs):
            raise ValueError(
                "max_num_batched_tokens needs to be larger than or equal to max_num_seqs."
            )

    def _init_random(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        self.rng_key = jax.random.key(self.model_config.seed)

    def _init_mesh(self) -> None:
        try:
            # TODO: Update override steps.
            sharding_strategy = \
                self.vllm_config.additional_config["sharding"]["sharding_strategy"]
        except KeyError:
            sharding_strategy = {"tensor_parallelism": len(self.devices)}

        if os.getenv("NEW_MODEL_DESIGN", False):
            self.mesh = build_mesh(self.devices, sharding_strategy)
        else:
            try:
                dp = sharding_strategy["data_parallelism"]
            except KeyError:
                dp = 1
            try:
                tp = sharding_strategy["tensor_parallelism"]
            except KeyError:
                tp = len(self.devices)

            axis_names = ("data", "model")
            mesh_shape = (dp, tp)

            self.mesh = make_optimized_mesh(mesh_shape,
                                            axis_names,
                                            devices=self.devices)
        logger.info(f"Init mesh | mesh={self.mesh}")

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
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        # [16, 32, 64, 128, 256, 512, 1024, 2048]
        self.num_tokens_paddings = runner_utils.get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
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
        self.block_table_cpu = np.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req), dtype=np.int32)
        self.query_start_loc_cpu = np.zeros(self.max_num_tokens + 1,
                                            dtype=np.int32)
        self.seq_lens_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)
        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_cpu = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = runner_utils.get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

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

    @functools.partial(jax.jit, static_argnums=(0, ))
    def select_from_array_fn(self, array, indices_to_select):
        return array[indices_to_select]

    def load_model(self):
        self.model_fn, self.compute_logits_fn, self.get_multimodal_embeddings_fn, self.get_input_embeddings_fn, self.state = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )
        self.rng_params_for_sampling = nnx.Rngs(
            jax.random.key(self.model_config.seed)).params()
        self.is_multimodal_model = (self.model_config.is_multimodal_model
                                    and self.get_multimodal_embeddings_fn
                                    is not None)

        logger.info(f"Init model | "
                    f"hbm={common_utils.hbm_usage_gb(self.devices)}Gb")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate", )

    def get_kv_cache_spec(self):
        return self.kv_cache_manager.get_kv_cache_spec()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_manager.initialize_kv_cache(kv_cache_config)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_logprobs", ))
    def _compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
        logprobs = compute_logprobs(logits)
        return gather_logprobs(logprobs, next_tokens, max_logprobs)

    def capture_model(self) -> None:
        self.compilation_manager.capture_model()

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

    def _calc_mrope_positions(self, scheduler_output: "VllmSchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions_cpu,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _execute_mm_encoder(self, scheduler_output: "VllmSchedulerOutput"):
        import torch
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_hash = req_state.mm_hashes[mm_input_id]
                mm_kwargs.append(req_state.mm_kwargs[mm_input_id])
                mm_hashes_pos.append(
                    (mm_hash, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs):
            batched_mm_inputs = mm_kwargs_group
            # Convert torch tensors to numpy arrays that JAX can handle.
            if "pixel_values" in batched_mm_inputs and isinstance(
                    batched_mm_inputs["pixel_values"], list):
                batched_mm_inputs["pixel_values"] = torch.cat(
                    batched_mm_inputs["pixel_values"], dim=0)

            image_grid_thw = ()
            for key, value in batched_mm_inputs.items():
                if isinstance(value, torch.Tensor):
                    if key == 'image_grid_thw':
                        # change it to tuple of tuples to make it hashable for JIT

                        # Shape: (B, N, 3) -> (B*N, 3) -> tuple of tuples
                        grid_thw_tensor = batched_mm_inputs[key]
                        grid_thw_reshaped = grid_thw_tensor.reshape(-1, 3)
                        image_grid_thw = tuple(
                            tuple(row) for row in grid_thw_reshaped.tolist())

                        continue

                    if value.dtype == torch.bfloat16:
                        batched_mm_inputs[key] = value.to(
                            torch.float32).numpy().astype(jnp.bfloat16)
                    else:
                        batched_mm_inputs[key] = value.numpy()
            batched_mm_inputs.pop('image_grid_thw')

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.get_multimodal_embeddings_fn(
                self.state, image_grid_thw, **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (mm_hash, pos_info), output in zip(
                mm_hashes_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "VllmSchedulerOutput",
    ) -> list[jax.Array]:
        mm_embeds: list[jax.Array] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            mm_hashes = req_state.mm_hashes
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                mm_hash = mm_hashes[i]
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                      f"Encoder cache miss for {mm_hash}."
                encoder_output = self.encoder_cache[mm_hash]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def _get_input_ids_embeds(self, input_ids: jax.Array,
                              mm_embeds: list[jax.Array]):
        if self.is_multimodal_model:
            inputs_embeds = self.get_input_embeddings_fn(
                self.state,
                input_ids=input_ids,
                multimodal_embeddings=mm_embeds,
            )
            return None, inputs_embeds
        else:
            return input_ids, None

    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        return self._execute_model(scheduler_output)[1]

    def _execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
    ) -> tuple[AttentionMetadata, ModelRunnerOutput]:
        self.persistent_batch_manager.update_states(scheduler_output)
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
            return DUMMY_METADATA, EMPTY_MODEL_RUNNER_OUTPUT,

        (input_ids, attn_metadata, sampling_metadata, logits_indices,
         spec_decode_metadata) = self._prepare_inputs(scheduler_output)

        # multi-modal support
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            # We have the modality embeds at this time.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # NOTE(Wenlong): For multi-modal model,
        # it will embed the text tokens and merge with the existing modality embeds
        # Later, the multi-modality model will take the embedding as the input.
        # For text-only model, this does nothing. It will input the input_ids and
        # leave the mebedding job inside the forward pass
        input_ids, inputs_embeds = self._get_input_ids_embeds(
            input_ids, mm_embeds)

        # TODO: Disable this for now
        if self.is_multimodal_model:
            self.maybe_forbid_compile = nullcontext()

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
                self.kv_caches, hidden_states = self.model_fn(
                    self.state,
                    self.kv_caches,
                    input_ids,
                    attn_metadata,
                    inputs_embeds,
                )

            hidden_states = self.select_from_array_fn(hidden_states,
                                                      logits_indices)
            logits = self.compute_logits_fn(self.state, hidden_states)
            if scheduler_output.grammar_bitmask is not None:
                require_struct_decoding, grammar_bitmask_padded, arange = \
                    self.prepare_structured_decoding_input(
                        logits, scheduler_output
                    )
                logits = self.structured_decode_fn(
                    require_struct_decoding,
                    grammar_bitmask_padded,
                    logits,
                    arange,
                )
            tpu_sampling_metadata = sampling_metadata
            if spec_decode_metadata is None:
                next_tokens = sample(
                    self.rng_params_for_sampling,
                    self.mesh,
                    logits,
                    tpu_sampling_metadata,
                )
            else:
                bonus_logits = self.select_from_array_fn(
                    logits, spec_decode_metadata.bonus_logits_indices)
                bonus_token_ids = sample(
                    self.rng_params_for_sampling,
                    self.mesh,
                    bonus_logits,
                    tpu_sampling_metadata,
                )
                target_logits = self.select_from_array_fn(
                    logits, spec_decode_metadata.target_logits_indices)
                next_tokens = self.rejection_sampler(
                    draft_token_ids=spec_decode_metadata.draft_token_ids,
                    num_draft_tokens=spec_decode_metadata.draft_lengths,
                    max_spec_len=spec_decode_metadata.max_spec_len,
                    draft_probs=None,
                    target_probs=target_logits,
                    bonus_token_ids=bonus_token_ids,
                    sampling_metadata=tpu_sampling_metadata,
                )

            if tpu_sampling_metadata.logprobs:
                logprobs = self._compute_and_gather_logprobs(
                    logits, next_tokens, self.model_config.max_logprobs)
            else:
                logprobs = None

        num_reqs = self.input_batch.num_reqs

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
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

        if spec_decode_metadata is None:
            next_tokens = np.asarray(jax.device_get(next_tokens))
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
            logprobs_lists = logprobs.tolists()
        else:
            logprobs_lists = None

        if self.speculative_config:
            self._draft_token_ids = self.propose_draft_token_ids(
                valid_sampled_token_ids)

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )
        return attn_metadata, model_runner_output

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        spec_token_ids = self.propose_ngram_draft_token_ids(sampled_token_ids)
        return spec_token_ids

    def propose_ngram_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        assert isinstance(self.drafter, NgramProposer)
        draft_token_ids: list[list[int]] = []
        num_reqs = self.input_batch.num_reqs
        for i, sampled_ids in enumerate(sampled_token_ids[:num_reqs]):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = self.input_batch.req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = self.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :num_tokens])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())

        return draft_token_ids

    @functools.partial(jax.jit, static_argnums=(0, ))
    def structured_decode_fn(self, require_struct_decoding: jax.Array,
                             grammar_bitmask: jax.Array, logits: jax.Array,
                             arange: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.any(require_struct_decoding),
            lambda: self._apply_grammar_bitmask_kernel(
                logits, grammar_bitmask, require_struct_decoding, arange),
            lambda: logits)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _apply_grammar_bitmask_kernel(self, logits: jax.Array,
                                      grammar_bitmask: jax.Array,
                                      require_struct_decoding: jax.Array,
                                      arange: jax.Array) -> jax.Array:

        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=num_reqs, N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = jnp.right_shift(grammar_bitmask[:, :, None],
                                           arange[None, None, :]) & 1 == 0

        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(logits.shape[0],
                                                    -1)[:, :self.vocab_size]

        masked_logits = jnp.where(unpacked_bitmask, -jnp.inf, logits)

        return jnp.where(require_struct_decoding, masked_logits, logits)

    def prepare_structured_decoding_input(
        self, logits: jax.Array, scheduler_output: "VllmSchedulerOutput"
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.fill(0)
        self.require_structured_out_cpu.fill(0)

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = grammar_bitmask[
            mask_indices]
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        self.require_structured_out_cpu[struct_out_indices] = True

        require_structured_out_cpu, grammar_bitmask_cpu, structured_decode_arange = self._device_array(
            (self.require_structured_out_cpu[:num_reqs],
             self.grammar_bitmask_cpu[:num_reqs],
             self.structured_decode_arange))

        return require_structured_out_cpu, grammar_bitmask_cpu, structured_decode_arange

    def _get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        max_spec_len = np.max(num_draft_tokens)

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)
        arange = np.concatenate(
            [self.arange_cpu[:n] for n in num_sampled_tokens])
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange
        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # arange: [0, 1, 2, 0, 1, 0]
        arange = np.concatenate(
            [self.arange_cpu[:n] for n in num_draft_tokens])
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids_cpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        padded_logits_length = runner_utils.get_padded_token_len(
            self.num_logits_paddings, logits_indices.shape[0])
        padded_logits_indices = np.concatenate([
            logits_indices,
            np.zeros(padded_logits_length - logits_indices.shape[0],
                     dtype=np.int32)
        ])

        assert bonus_logits_indices.shape[0] <= padded_num_reqs, (
            f"bonus_logits_indices.shape[0]={bonus_logits_indices.shape[0]} "
            f"padded_num_reqs={padded_num_reqs}")

        padded_bonus_logits_indices = np.concatenate([
            bonus_logits_indices,
            np.zeros(padded_num_reqs - bonus_logits_indices.shape[0],
                     dtype=np.int32)
        ])
        padded_num_draft_tokens = np.concatenate([
            num_draft_tokens,
            np.zeros(padded_num_reqs - num_draft_tokens.shape[0],
                     dtype=np.int32)
        ])
        padded_draft_token_ids = np.concatenate([
            draft_token_ids,
            np.zeros(padded_logits_length - draft_token_ids.shape[0],
                     dtype=np.int32)
        ])
        padded_target_logits_indices = np.concatenate([
            target_logits_indices,
            np.zeros(padded_logits_length - target_logits_indices.shape[0],
                     dtype=np.int32)
        ])

        padded_num_draft_tokens_cpu = padded_num_draft_tokens
        # CPU -> TPU copy.
        (padded_num_draft_tokens, padded_draft_token_ids,
         padded_logits_indices, padded_target_logits_indices,
         padded_bonus_logits_indices) = self._device_array(
             (padded_num_draft_tokens, padded_draft_token_ids,
              padded_logits_indices, padded_target_logits_indices,
              padded_bonus_logits_indices))

        metadata = SpecDecodeMetadata(
            draft_token_ids=padded_draft_token_ids,
            draft_lengths=padded_num_draft_tokens,
            draft_lengths_cpu=padded_num_draft_tokens_cpu,
            target_logits_indices=padded_target_logits_indices,
            bonus_logits_indices=padded_bonus_logits_indices,
            final_logits_indices=padded_logits_indices,
            max_spec_len=max_spec_len,
        )
        return metadata

    def _prepare_inputs(self, scheduler_output: "VllmSchedulerOutput"):
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

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_cpu[:num_reqs],
                                num_scheduled_tokens_per_req)

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
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        np.take(self.input_batch.token_ids_cpu.flatten(),
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
        block_tables = self.block_table_cpu[:self.max_num_reqs]
        block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs])
        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs + 1]
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs]
        request_distribution = np.array(self.input_batch.request_distribution)
        padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
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

            spec_decode_metadata = self._get_spec_decode_metadata(
                num_draft_tokens, self.query_start_loc_cpu[1:num_reqs + 1],
                padded_num_reqs)
            logits_indices = spec_decode_metadata.final_logits_indices

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh, self.input_batch, padded_num_reqs)
        if self.uses_mrope:
            positions = mrope_positions

        # Convert block_tables to 1D on cpu.
        block_tables = block_tables.reshape(-1)

        (input_ids, positions, block_tables, query_start_loc, seq_lens,
         logits_indices, request_distribution) = self._device_array(
             (input_ids, positions, block_tables, query_start_loc, seq_lens,
              logits_indices, request_distribution))

        return (
            input_ids,
            AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            ),
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
        )

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

    def _sync_weights(
        self,
        updated_weights: jaxtyping.PyTree,
        mappings: Dict[str, Tuple[str, Tuple[str]]],
        transpose_keys: Dict[str, Tuple[int]],
        reshard_fn: Callable[[jaxtyping.PyTree, jaxtyping.PyTree],
                             jaxtyping.PyTree] = None
    ) -> None:
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
