import functools
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import vllm.envs as envs
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
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
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.jax.metadata import SpecDecodeMetadata
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
        self.encoder_cache: dict[str, dict[int, jax.Array]] = {}
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
    def select_hidden_states_fn(self, hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

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
        import torch

        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config

        # Pad num_kv_heads to multiple of TP size.
        num_kv_heads = common_utils.get_padded_num_heads(
            model_config.get_total_num_kv_heads(), self.mesh.shape["model"])

        # Pad head_dim to multiple of 128.
        head_size = model_config.get_head_size()
        head_size = common_utils.get_padded_head_dim(head_size)

        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_caches: List[jax.Array] = []

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        layer_names = kv_cache_groups[0].layer_names

        # NOTE: we'll multiply the num_kv_heads by 2 in the function
        self.kv_caches = runner_utils.create_kv_caches(
            num_blocks=kv_cache_config.num_blocks,
            block_size=kv_cache_spec.block_size,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            head_size=kv_cache_spec.head_size,
            mesh=self.mesh,
            layer_names=layer_names,
            devices=self.devices,
        )

        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self)

    def _precompile_backbone(self) -> None:
        for num_tokens in self.num_tokens_paddings:
            input_ids = np.ones((num_tokens, ), dtype=np.int32)
            positions = np.ones((num_tokens, ), dtype=np.int32)
            block_tables = self.block_table_cpu[:self.max_num_reqs]
            seq_lens = np.ones((self.max_num_reqs, ), dtype=np.int32)
            query_start_loc = np.ones((self.max_num_reqs + 1, ),
                                      dtype=np.int32)
            request_distribution = np.array([0, 0, 0], dtype=np.int32)

            # Convert block_tables to 1D on cpu.
            block_tables = block_tables.reshape(-1)

            (input_ids, positions, block_tables, query_start_loc, seq_lens,
             request_distribution) = self._device_array(
                 (input_ids, positions, block_tables, query_start_loc,
                  seq_lens, request_distribution))
            logger.info(f"Precompile backbone --> num_tokens={num_tokens}")

            attention_metadata = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )

            # TODO:
            # Use a None inputs_embeds here, assuming it is text-only model
            # For mm model, we will use another precompile function
            inputs_embeds = None
            start = time.perf_counter()
            self.kv_caches, hidden_states = self.model_fn(
                self.state, self.kv_caches, input_ids, attention_metadata,
                inputs_embeds)
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)
            hidden_states.block_until_ready()

    def _precompile_select_hidden_states(self) -> None:
        logger.info(
            "Compiling select_hidden_states with different input shapes.")
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            for num_reqs in self.num_reqs_paddings:
                if num_reqs > num_tokens:
                    continue
                hidden_states = jnp.ones((num_tokens, hsize),
                                         dtype=jnp.bfloat16)
                indices_do_sample = jnp.ones((num_reqs, ), dtype=jnp.int32)
                hidden_states, indices_do_sample = self._device_array(
                    (hidden_states, indices_do_sample))
                start = time.perf_counter()
                logger.info(
                    f"Precompile select_hidden_states --> num_tokens={num_tokens} | "
                    f"num_reqs={num_reqs}")
                result = self.select_hidden_states_fn(hidden_states,
                                                      indices_do_sample)
                result.block_until_ready()
                end = time.perf_counter()
                logger.info("Compilation finished in %.2f [secs].",
                            end - start)

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        hsize = self.model_config.get_hidden_size()
        for num_reqs in self.num_reqs_paddings:
            hidden_states = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            hidden_states = self._device_array(hidden_states)
            logger.info(f"Precompile compute_logits --> num_reqs={num_reqs}")
            start = time.perf_counter()
            result = self.compute_logits_fn(self.state, hidden_states)
            result.block_until_ready()
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_sampling(self) -> None:
        logger.info("Compiling sampling with different input shapes.")
        hsize = self.model_config.get_vocab_size()
        for num_reqs in self.num_reqs_paddings:
            logits = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            # logits is expected to be sharded along hsize dim.
            sharding = NamedSharding(self.mesh, PartitionSpec(None, "model"))
            logits = self._device_array(logits, sharding=sharding)
            logger.info(f"Precompile sampling --> num_reqs={num_reqs}")
            start = time.perf_counter()
            for do_sampling in (True, False):
                if do_sampling:
                    temperature = np.full((num_reqs, ), 0.7, dtype=np.float32)
                    top_k = np.full((num_reqs, ), 20, dtype=np.int32)
                    top_p = np.full((num_reqs, ), 0.8, dtype=np.float32)
                    (temperature, top_k, top_p) = self._device_array(
                        (temperature, top_k, top_p))
                else:
                    temperature = None
                    top_k = None
                    top_p = None

                sampling_metadata = TPUSupportedSamplingMetadata(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sampling=do_sampling,
                )
                result = sample(self.rng_params_for_sampling, self.mesh,
                                logits, sampling_metadata)
                result.block_until_ready()
                end = time.perf_counter()
                logger.info("Compilation finished in %.2f [secs].",
                            end - start)

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        hsize = self.model_config.get_vocab_size()
        for num_reqs in self.num_reqs_paddings:
            logits = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            logits, = self._device_array((logits, ))
            logger.info(f"Precompile gather_logprobs --> num_reqs={num_reqs}")
            start = time.perf_counter()
            token_ids = jnp.ones((num_reqs, ), dtype=jnp.int32)
            token_ids, = self._device_array((token_ids, ))
            result = self._compute_and_gather_logprobs(
                logits, token_ids, self.model_config.max_logprobs)
            result.logprob_token_ids.block_until_ready()
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_logprobs", ))
    def _compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
        logprobs = compute_logprobs(logits)
        return gather_logprobs(logprobs, next_tokens, max_logprobs)

    def capture_model(self) -> None:
        if os.getenv("SKIP_JAX_PRECOMPILE", False):
            return
        logger.info("Precompile all the subgraphs with possible input shapes.")

        # TODO: wenlong: skip precompiling for mm models
        # because the model requires input_embeds
        if self.is_multimodal_model:
            logger.info("[TEMP] skip precompiling for multi-modal models")
            return

        self._precompile_backbone()
        self._precompile_select_hidden_states()
        self._precompile_compute_logits()
        self._precompile_sampling()
        self._precompile_gather_logprobs()
        self._precompile_structured_decoding()

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = jnp.zeros((num_reqs, self.vocab_size),
                                     dtype=jnp.bfloat16)
            dummy_require_struct_decoding = self.require_structured_out_cpu[:
                                                                            num_reqs]
            dummy_grammar_bitmask = self.grammar_bitmask_cpu[:num_reqs]

            (dummy_logits, dummy_require_struct_decoding,
             dummy_grammar_bitmask, arange) = self._device_array(
                 (dummy_logits, dummy_require_struct_decoding,
                  dummy_grammar_bitmask, self.structured_decode_arange))

            self.structured_decode_fn(dummy_require_struct_decoding,
                                      dummy_grammar_bitmask, dummy_logits,
                                      arange)
            logger.info("  -- num_seqs: %d", num_reqs)

        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    @staticmethod
    @functools.partial(jax.jit)
    def _jitted_gather_kv_cache(
            kv_caches: List[jax.Array],
            indices_to_gather: jax.Array) -> List[jax.Array]:
        """
        JIT-compiled function to gather KV cache slices for all layers at once.
        This fuses the reshape and take operations across all layers into a
        single efficient kernel.
        """
        batched_kv_cache_per_layer = []
        for layer_kv_cache in kv_caches:
            flat_layer_cache = layer_kv_cache.reshape(
                -1, *layer_kv_cache.shape[2:])
            all_gathered_slices = flat_layer_cache.take(indices_to_gather,
                                                        axis=0)
            batched_kv_cache_per_layer.append(all_gathered_slices)
        return batched_kv_cache_per_layer

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("block_size"),
        donate_argnames=("kv_caches"),
    )
    def _jitted_insert_kv_cache(
        block_size,
        kv_caches: List[jax.Array],
        kv_cache_slices: List[jax.Array],
        block_numbers: jax.Array,
    ) -> List[jax.Array]:
        """
        JIT-compiled function to insert KV cache slices into the physical
        cache for all layers at once. This fuses the pad, reshape, and scatter
        operations into a single efficient kernel.
        """
        new_kv_caches = []
        # Assuming block numbers are non-negative and sorted.
        for i, layer_kv_cache_slices in enumerate(kv_cache_slices):
            padded_seq_len, packing_div, packing, head_dim = layer_kv_cache_slices.shape
            padding_config = ((0, block_numbers.shape[0] * block_size -
                               padded_seq_len), (0, 0), (0, 0), (0, 0))
            layer_kv_cache_slices = jnp.pad(layer_kv_cache_slices,
                                            pad_width=padding_config)
            layer_kv_cache_slices = layer_kv_cache_slices.reshape(
                -1, block_size, packing_div, packing, head_dim)
            updated_cache = kv_caches[i].at[block_numbers].set(
                layer_kv_cache_slices)
            new_kv_caches.append(updated_cache)
        return new_kv_caches

    def get_kv_cache_for_block_ids(
        self,
        block_ids: List[int],
    ) -> List[jax.Array]:
        """
        Extracts the KV cache slices for a given list of block IDs.
        This assumes all provided blocks are full.

        Args:
            block_ids: A list of block IDs to extract KV cache for.

        Returns:
            A list of JAX arrays, with each array representing the KV cache
            slices for a layer, concatenated for all blocks.
        """
        all_indices_to_gather = []
        for block_id in block_ids:
            start_index = block_id * self.block_size
            all_indices_to_gather.extend(
                range(start_index, start_index + self.block_size))

        indices_to_gather_jnp = jnp.array(all_indices_to_gather,
                                          dtype=jnp.int32)
        with runner_utils.LatencyTracker("BatchedGatherKVSlices-for-blocks"):
            batched_kv_cache_per_layer = self._jitted_gather_kv_cache(
                self.kv_caches, indices_to_gather_jnp)

        return batched_kv_cache_per_layer

    def transfer_kv_cache(self,
                          kv_cache_slices: List[jax.Array]) -> List[jax.Array]:
        """
        Transfers KV cache slices to the runner's mesh.

        This is used when a KV cache generated on one runner (e.g., a prefill
        runner) needs to be used on another runner (e.g., a decode runner)
        with a different device mesh. The transfer is asynchronous.

        Args:
            kv_cache_slices: A list of JAX arrays, where each array contains
                the KV cache slices for a specific layer. The shape of each
                slice is expected to be (num_tokens, num_kv_heads * 2, head_size).

        Returns:
            A new list of JAX arrays representing the KV cache slices, sharded
            across the runner's device mesh.
        """
        # The KV cache slices have a shape of (num_tokens, num_kv_heads * 2, head_size).
        # We shard along the num_kv_heads dimension (axis=1), which corresponds
        # to the "model" axis of the mesh for tensor parallelism.
        sharding = NamedSharding(self.mesh, PartitionSpec())
        transferred_kv_cache = jax.device_put(kv_cache_slices, sharding)
        for cache in transferred_kv_cache:
            cache.block_until_ready()
        return transferred_kv_cache

    def insert_request_with_kv_cache(
        self,
        request: "Request",
        kv_cache_slices: List[jax.Array],
        block_ids: List[List[int]],
    ):
        """
        Inserts a request and its KV cache into the runner. This is used to
        transfer a request from a prefill runner to a decode runner.

        The provided KV cache slices are copied into the physical blocks
        allocated for the request. The runner's internal state is then updated
        to include the request.

        Args:
            request: The vLLM request object, containing the state after prefill.
            kv_cache_slices: The KV cache for the request, already transferred
                to this runner's mesh. This is a list of JAX arrays, one per layer.
            block_ids: The physical block numbers allocated for this request on
                this runner. This is a list of lists, for each KV cache group.
        """
        # Assume one KV cache group for now, which is consistent with current setup.
        if len(block_ids) > 1:
            raise NotImplementedError(
                "Inserting KV cache for models with multiple KV cache groups "
                "is not supported yet.")
        block_numbers = block_ids[0]
        num_blocks = len(block_numbers)

        # Pad the number of blocks to static bucket sizes to avoid recompilation.
        block_buckets = [1, 2, 4, 8, 16, 32, 64, self.max_num_blocks_per_req]
        import bisect
        bucket_index = bisect.bisect_left(block_buckets, num_blocks)
        padded_num_blocks = block_buckets[bucket_index]
        padding_size = padded_num_blocks - num_blocks

        # Pad target_block_numbers. Pad with the max_num_blocks + 1 to avoid writing
        # to unintended blocks. JAX would ignore as it's beyond the number of blocks
        # in the cache.
        max_num_blocks = self.vllm_config.cache_config.num_gpu_blocks
        assert max_num_blocks is not None and max_num_blocks != 0
        block_numbers.extend([0] * padding_size)

        padded_block_numbers = jnp.array(block_numbers, dtype=jnp.int32)

        # Call the JIT-compiled function to perform the scatter operation
        # for all layers in a single, fused kernel.
        with runner_utils.LatencyTracker(
                f"JittedInsertKVCache-b{padded_num_blocks}"):
            self.kv_caches = self._jitted_insert_kv_cache(
                self.block_size,
                self.kv_caches,
                kv_cache_slices,
                padded_block_numbers,
            )

        logger.debug(f"Updated kv cache entries cnt={len(self.kv_caches)}")

        # Update runner's internal state to track the new request.
        req_id = request.request_id
        if req_id in self.requests:
            logger.warning(
                f"Request {req_id} already exists in the runner. Overwriting.")

        # Create a CachedRequestState object to add to the input batch.
        req_state = CachedRequestState(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=[request.all_token_ids[-1]],
            sampling_params=request.sampling_params,
            block_ids=tuple(block_ids),
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            mm_kwargs=getattr(request, "mm_kwargs", []),
            mm_hashes=[],
            mm_positions=getattr(request, "mm_positions", []),
            pooling_params=getattr(request, "pooling_params", None),
            generator=None,
        )

        self.requests[req_id] = req_state
        self.input_batch.add_request(req_state)

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
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_kwargs.append(req_state.mm_kwargs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

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
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(
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
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]

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
        self._update_states(scheduler_output)
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

            hidden_states = self.select_hidden_states_fn(
                hidden_states, logits_indices)
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
                bonus_logits = self.select_hidden_states_fn(
                    logits, spec_decode_metadata.bonus_logits_indices)
                bonus_token_ids = sample(
                    self.rng_params_for_sampling,
                    self.mesh,
                    bonus_logits,
                    tpu_sampling_metadata,
                )
                target_logits = self.select_hidden_states_fn(
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
            self.mesh, self.input_batch, logits_indices.shape[0])
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

    def _update_states(self, scheduler_output: "VllmSchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input TPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the TPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id

            data_items = asdict(new_req_data)
            data_items["mm_hashes"] = []

            self.requests[req_id] = CachedRequestState(**data_items,
                                                       output_token_ids=[])

            # NOTE(wenlong): We need to explicitly set this
            # because asdict will convert List[PlaceholderRange] to list[dict]
            self.requests[req_id].mm_positions = new_req_data.mm_positions

            # multi-modal related
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for item in self.requests[req_id].mm_kwargs:
                    mm_input = item.get_data()
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.append(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.append(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.append(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.append(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        batch_changed = len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
        # TODO(jevinjiang): I assume we do not need to set batch_changed to true if just swapping requests.
        self._reorder_batch(scheduler_output)
        return batch_changed

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

    def _reorder_batch(self, scheduler_output: "VllmSchedulerOutput") -> int:
        """ Reorder the sheduled requests to RPA kernel friendly distribution
        (decode_only, fixed_chunked_prefill_only, mixed) and set the request
        distribution accordingly.

        Returns:
            The number of swaps in requests.
        """
        # Note(jevinjiang): currently we only consider decode_only.
        num_reqs = self.input_batch.num_reqs
        swap_cnt = 0
        if num_reqs <= 0:
            return swap_cnt
        # Use two-pointer approach to reorder the decode requests to front.
        i, j = 0, num_reqs - 1
        while i < j:
            i_req_id = self.input_batch.req_ids[i]
            j_req_id = self.input_batch.req_ids[j]

            if scheduler_output.num_scheduled_tokens[i_req_id] == 1:
                # i is a decode request, move to the next one.
                i += 1
            elif scheduler_output.num_scheduled_tokens[j_req_id] > 1:
                # j is a prefill request, move to the previous one.
                j -= 1
            else:
                # Swap i and j.
                self.input_batch.swap_states(i, j)
                i += 1
                j -= 1
                swap_cnt += 1

        num_decode = i + int(scheduler_output.num_scheduled_tokens[
            self.input_batch.req_ids[i]] == 1)

        self.input_batch.request_distribution = [
            num_decode, num_decode, num_reqs
        ]

        return swap_cnt
