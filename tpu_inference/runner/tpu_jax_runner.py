import functools
import os
import random
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import json
import numpy as np
import torch
import vllm.envs as envs
from flax import nnx
from torchax.ops.mappings import j2t_dtype
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import set_forward_context
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
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from tpu_inference import utils as common_utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.layers.jax.sample.sampling import (compute_logprobs,
                                                      gather_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.layers.jax.sharding import build_mesh
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.models.jax.utils.weight_utils import (
    shard_put, transfer_state_with_mappings)
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.compilation_manager import CompilationManager
from tpu_inference.runner.input_batch_jax import CachedRequestState, InputBatch
from tpu_inference.runner.kv_cache_manager import KVCacheManager
from tpu_inference.runner.lora_utils import LoraUtils
from tpu_inference.runner.multimodal_manager import MultiModalManager
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager
from tpu_inference.runner.speculative_decoding_manager import \
    SpeculativeDecodingManager
from tpu_inference.runner.structured_decoding_manager import \
    StructuredDecodingManager
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.utils import device_array, make_optimized_mesh

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8

DUMMY_METADATA = AttentionMetadata(
    input_positions=[],
    block_tables=[],
    request_distribution=[0, 0, 0],
)

TPU_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

def to_python_types(pytree):
    """Converts a PyTree of JAX/NumPy arrays to Python lists/floats."""
    return jax.tree_util.tree_map(
        lambda x: x.tolist() if isinstance(x, (np.ndarray, jax.Array)) else x,
        pytree
    )


class TPUModelRunner(KVConnectorModelRunnerMixin, LoRAModelRunnerMixin):

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

        self.devices = devices
        self.dtype = self.model_config.dtype
        self.maybe_forbid_compile = runner_utils.ForbidCompile(
        ) if envs.VLLM_XLA_CHECK_RECOMPILATION else nullcontext()

        self._init_random()
        self._init_mesh()
        self._init_phased_profiling()
        self._init_mm()
        self._init_inputs()
        self._init_speculative_decoding()

        # Delegate functions to specific manager classes.
        self.compilation_manager = CompilationManager(self)
        self.speculative_decoding_manager = SpeculativeDecodingManager(self)
        self.structured_decoding_manager = StructuredDecodingManager(self)
        self.kv_cache_manager = KVCacheManager(self)
        self.mm_manager = MultiModalManager(self)
        self.persistent_batch_manager = PersistentBatchManager(
            self.requests, self.input_batch, self.encoder_cache,
            self.uses_mrope, self.model_config)
        self.lora_utils = LoraUtils(self)

        cache_config = self.cache_config
        if cache_config.cache_dtype == "auto":
            model_dtype = self.dtype
            if isinstance(model_dtype, str):
                self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(getattr(model_dtype, 'dtype', None), jnp.dtype):
                self.kv_cache_dtype = j2t_dtype(model_dtype.dtype)
            elif isinstance(model_dtype, torch.dtype):
                self.kv_cache_dtype = model_dtype
            else:
                raise ValueError(
                    "KV cache is unsupported for model_dtype of %s",
                    model_dtype)
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

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

    def _init_phased_profiling(self) -> None:
        self.phased_profiling_dir = os.getenv("PHASED_PROFILING_DIR", "")
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
        if self.lora_config is not None:
            # lora_config.lora_extra_vocab_size is the "Maximum size of extra vocabulary that can be present in a LoRA adapter" per https://github.com/vanbasten23/vllm/blob/7f4a8b6705622fde952a2e633e86716f902d6e1b/vllm/config.py#L3040
            self.vocab_size += self.lora_config.lora_extra_vocab_size
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
        self.model_fn, self.compute_logits_fn, self.combine_hidden_states_fn, self.get_multimodal_embeddings_fn, self.get_input_embeddings_fn, self.get_mrope_input_positions_fn, self.state, self.lora_manager, self.model = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )

        if self.drafter is not None:
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.state)

        self.rng_params_for_sampling = nnx.Rngs(
            jax.random.key(self.model_config.seed)).params()
        self.is_multimodal_model = (self.model_config.is_multimodal_model
                                    and self.get_multimodal_embeddings_fn
                                    is not None)

        logger.info(f"Init model | "
                    f"hbm={common_utils.hbm_usage_gb(self.devices)}GiB")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate", )

    def get_kv_cache_spec(self):
        return self.kv_cache_manager.get_kv_cache_spec()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_config = kv_cache_config
        self.kv_caches = []
        self.kv_cache_manager.initialize_kv_cache(kv_cache_config)
        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self)

    def capture_model(self) -> None:
        self.compilation_manager.capture_model()

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
            return DUMMY_METADATA, EMPTY_MODEL_RUNNER_OUTPUT,

        (input_ids, attn_metadata, sampling_metadata, logits_indices,
         spec_decode_metadata) = self._prepare_inputs(scheduler_output)

        # multi-modal support
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            # We have the modality embeds at this time.
            self.mm_manager.execute_mm_encoder(scheduler_output)
            mm_embeds = self.mm_manager.gather_mm_embeddings(scheduler_output)
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
                 aux_hidden_states, metrics) = self.model_fn(
                     self.state,
                     self.kv_caches,
                     input_ids,
                     attn_metadata,
                     inputs_embeds,
                     tuple(self.layer_name_to_kvcache_index.items()),
                     lora_metadata,
                 )
            if True:
                metrics_cpu = to_python_types(jax.device_get(metrics))
                metrics_path = self.vllm_config.additional_config["metrics_path"]
                with open(metrics_path, 'a') as f:
                    json.dump(metrics_cpu.to_pure_dict(), f)
                    f.write('\n')
            hidden_states = self._select_from_array_fn(hidden_states,
                                                       logits_indices)
            logits = self.compute_logits_fn(
                self.state,
                hidden_states,
                lora_metadata,
            )
            if scheduler_output.grammar_bitmask is not None:
                (
                    require_struct_decoding, grammar_bitmask_padded, arange
                ) = self.structured_decoding_manager.prepare_structured_decoding_input(
                    logits, scheduler_output)
                logits = self.structured_decoding_manager.structured_decode_fn(
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
                bonus_logits = self._select_from_array_fn(
                    logits, spec_decode_metadata.bonus_logits_indices)
                bonus_token_ids = sample(
                    self.rng_params_for_sampling,
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
                    key=self.rng_params_for_sampling,
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
        return attn_metadata, model_runner_output

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _select_from_array_fn(self, array, indices_to_select):
        return array[indices_to_select]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_logprobs", ))
    def _compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
        logprobs = compute_logprobs(logits)
        return gather_logprobs(logprobs, next_tokens, max_logprobs)

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
        padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)

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

        # Convert block_tables to 1D on cpu.
        block_tables = block_tables.reshape(-1)

        query_start_loc_cpu = query_start_loc
        seq_lens_cpu = seq_lens
        (input_ids, positions, block_tables, query_start_loc, seq_lens,
         logits_indices, request_distribution) = device_array(
             self.mesh, (input_ids, positions, block_tables, query_start_loc,
                         seq_lens, logits_indices, request_distribution))

        if self.lora_config is not None:
            self.lora_utils.set_active_loras(
                num_scheduled_tokens_per_req, total_num_scheduled_tokens,
                padded_total_num_scheduled_tokens)

        attention_metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution)

        # This is for making these cpu buffers hidden during tracing
        attention_metadata.query_start_loc_cpu = query_start_loc_cpu
        attention_metadata.seq_lens_cpu = seq_lens_cpu

        return (
            input_ids,
            attention_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
        )

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
