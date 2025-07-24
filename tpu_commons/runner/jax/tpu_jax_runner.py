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
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request

from tpu_commons import utils_jax as utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.sharding import Sharding
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.layers.sampling import (compute_logprobs,
                                                    gather_logprobs, sample)
from tpu_commons.models.jax.model_loader import get_model
from tpu_commons.models.jax.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_commons.models.jax.utils.weight_utils import \
    transfer_state_with_mappings
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.utils import (ForbidCompile, LatencyTracker,
                                      create_kv_caches,
                                      get_padded_num_reqs_with_upper_limit,
                                      get_padded_token_len, get_req_paddings,
                                      get_token_paddings)

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8
# Block size used for kv cache updating kernel
NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK = 8

DUMMY_METADATA = AttentionMetadata(
    input_positions=[],
    seq_lens=[],
    block_tables=[],
    slot_mapping=[],
)


class TPUModelRunner():

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[Any],
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config
        self._verify_chunked_prefill_config()

        self.devices = devices
        self.dtype = self.model_config.dtype

        self._init_random()
        self._init_mesh()
        self._init_inputs()

        self.maybe_forbid_compile = ForbidCompile(
        ) if envs.VLLM_XLA_CHECK_RECOMPILATION else nullcontext()
        logger.info("TPUModelRunner created!")

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
        if os.getenv("NEW_MODEL_DESIGN", False):
            try:
                # TODO: Update override steps.
                sharding_strategy = \
                    self.vllm_config.additional_config["sharding"]["sharding_strategy"]
            except KeyError:
                logger.warning(
                    f"No sharding strategy passed! Using default of full model parallelism={len(self.devices)}"
                )
                sharding_strategy = {"tensor_parallelism": len(self.devices)}
            sharding = Sharding(strategy_dict=sharding_strategy,
                                vllm_config=self.vllm_config,
                                devices=self.devices)
            self.mesh = sharding.mesh
        else:
            axis_names = ("data", "model")
            mesh_shape = (1, len(self.devices))
            self.mesh = jax.make_mesh(mesh_shape,
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
        self.num_tokens_paddings = get_token_paddings(
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
        self.num_reqs_paddings = get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        self.temperatures_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ps_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ks_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def select_hidden_states_fn(self, hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

    def load_model(self):
        self.model_fn, self.compute_logits_fn, self.state = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )
        self.rng_params_for_sampling = nnx.Rngs(
            jax.random.key(self.model_config.seed)).params()

        logger.info(f"Init model | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}Gb")

    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        import torch
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config

        # Pad num_kv_heads to multiple of TP size.
        num_kv_heads = utils.get_padded_num_heads(
            model_config.get_total_num_kv_heads(), self.mesh.shape["model"])

        # Pad head_dim to multiple of 128.
        head_size = model_config.get_head_size()
        head_size = utils.get_padded_head_dim(head_size)

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
        self.kv_caches = create_kv_caches(
            num_blocks=kv_cache_config.num_blocks,
            block_size=kv_cache_spec.block_size,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            head_size=kv_cache_spec.head_size,
            mesh=self.mesh,
            layer_names=layer_names,
            devices=self.devices,
        )

        logger.info(jax.lib.xla_bridge.get_backend().platform_version)

    def _precompile_backbone(self) -> None:
        for num_tokens in self.num_tokens_paddings:
            input_ids = np.ones((num_tokens, ), dtype=np.int32)
            positions = np.ones((num_tokens, ), dtype=np.int32)
            padded_num_slices = _get_padded_num_kv_cache_update_slices(
                num_tokens, self.max_num_reqs, self.block_size)
            slot_mapping_metadata = np.ones((3, padded_num_slices),
                                            dtype=np.int32)
            block_tables = self.block_table_cpu[:self.max_num_reqs]
            seq_lens = np.ones((self.max_num_reqs, ), dtype=np.int32)
            query_start_loc = np.ones((self.max_num_reqs + 1, ),
                                      dtype=np.int32)
            num_seqs = np.array([self.max_num_reqs], dtype=np.int32)
            num_slices = np.array([1], dtype=np.int32)

            (input_ids, positions, slot_mapping_metadata, num_slices,
             block_tables, query_start_loc, seq_lens,
             num_seqs) = self._device_array(
                 (input_ids, positions, slot_mapping_metadata, num_slices,
                  block_tables, query_start_loc, seq_lens, num_seqs))
            logger.info(f"Precompile backbone --> num_tokens={num_tokens}")
            inputs = (
                self.kv_caches,
                input_ids,
                AttentionMetadata(
                    input_positions=positions,
                    slot_mapping=slot_mapping_metadata,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    num_seqs=num_seqs,
                    num_slices=num_slices,
                ),
            )
            start = time.perf_counter()
            self.kv_caches, hidden_states = self.model_fn(
                self.state, *inputs[:3])
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
            logits = self._device_array(logits)
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

        self._precompile_backbone()
        self._precompile_select_hidden_states()
        self._precompile_compute_logits()
        self._precompile_sampling()
        self._precompile_gather_logprobs()

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
            _, num_kv_heads, head_dim = layer_kv_cache_slices.shape
            padding_config = ((0, block_numbers.shape[0] * block_size -
                               layer_kv_cache_slices.shape[0]), (0, 0), (0, 0))
            layer_kv_cache_slices = jnp.pad(layer_kv_cache_slices,
                                            pad_width=padding_config)
            layer_kv_cache_slices = layer_kv_cache_slices.reshape(
                -1, block_size, num_kv_heads, head_dim)
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

        with LatencyTracker("BatchedGatherKVSlices-for-blocks"):
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
        sharding = NamedSharding(self.mesh, PartitionSpec(None, "model", None))
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
        block_numbers.extend([max_num_blocks + 1] * padding_size)

        padded_block_numbers = jnp.array(block_numbers, dtype=jnp.int32)

        # Call the JIT-compiled function to perform the scatter operation
        # for all layers in a single, fused kernel.
        with LatencyTracker(f"JittedInsertKVCache-b{padded_num_blocks}"):
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
            mm_inputs=getattr(request, "mm_inputs", []),
            mm_hashes=[],
            mm_positions=getattr(request, "mm_positions", []),
            pooling_params=getattr(request, "pooling_params", None),
            generator=None,
        )

        self.requests[req_id] = req_state
        self.input_batch.add_request(req_state)

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
            self.maybe_setup_kv_connector(scheduler_output)

            # Return empty ModelRunnerOutput if there's no work to do.
            # TODO(fhzhang): We rely on empty cycles to remove requests in input batch. Fix it to reduce overhead.
            logger.debug(f"Nothing scheduled: {scheduler_output}!")
            if len(scheduler_output.finished_req_ids) == 0:
                raise Exception(
                    "Should not schedule a request that does nothing!")
            return DUMMY_METADATA, EMPTY_MODEL_RUNNER_OUTPUT,

        inputs = self._prepare_inputs(scheduler_output)
        with self.maybe_forbid_compile:
            self.maybe_setup_kv_connector(scheduler_output)
            self.kv_caches, hidden_states = self.model_fn(
                self.state, *inputs[:3])
            self.maybe_wait_for_kv_save()

            hidden_states = self.select_hidden_states_fn(
                hidden_states, inputs[4])
            logits = self.compute_logits_fn(self.state, hidden_states)
            tpu_sampling_metadata = inputs[3]
            next_tokens = sample(
                self.rng_params_for_sampling,
                self.mesh,
                logits,
                tpu_sampling_metadata,
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

        next_tokens = np.asarray(jax.device_get(next_tokens))
        selected_token_ids = np.expand_dims(next_tokens[:num_reqs], 1)
        valid_sampled_token_ids = selected_token_ids.tolist()
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        # Append sampled tokens
        for i, req_state, seq_len in request_seq_lens:
            token_id = valid_sampled_token_ids[i][0]
            self.input_batch.token_ids_cpu[i, seq_len] = token_id
            req_state.output_token_ids.append(token_id)
            self.input_batch.num_tokens[i] += 1

        if logprobs is not None:
            logprobs_lists = logprobs.tolists()
        else:
            logprobs_lists = None

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )
        return inputs[2], model_runner_output

    # TODO(xiang): check this after implementing TPU connector
    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "VllmSchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv()

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

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
        padded_total_num_scheduled_tokens = get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0

        # Inputs
        input_ids = self.input_ids_cpu[:padded_total_num_scheduled_tokens]
        positions = self.positions_cpu[:padded_total_num_scheduled_tokens]
        slot_mapping_metadata = self._get_slot_mapping_metadata(
            num_reqs, num_scheduled_tokens_per_req)
        num_slices = np.array([slot_mapping_metadata.shape[0]])
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total_num_scheduled_tokens, self.max_num_reqs,
            self.block_size)
        slot_mapping_metadata = np.pad(
            slot_mapping_metadata,
            [[0, padded_num_slices - len(slot_mapping_metadata)], [0, 0]],
            constant_values=0)
        slot_mapping_metadata = np.transpose(slot_mapping_metadata)
        block_tables = self.block_table_cpu[:self.max_num_reqs]
        block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs])
        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs + 1]
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs]
        padded_num_reqs = get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
        logits_indices = self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1
        num_seqs = np.array([num_reqs])

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.\
            from_input_batch(self.mesh, self.input_batch, padded_num_reqs)
        (input_ids, positions, slot_mapping_metadata, num_slices, block_tables,
         query_start_loc, seq_lens, num_seqs,
         logits_indices) = self._device_array(
             (input_ids, positions, slot_mapping_metadata, num_slices,
              block_tables, query_start_loc, seq_lens, num_seqs,
              logits_indices))

        return (
            self.kv_caches,
            input_ids,
            AttentionMetadata(
                input_positions=positions,
                slot_mapping=slot_mapping_metadata,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                num_slices=num_slices,
            ),
            sampling_metadata,
            logits_indices,
        )

    def _get_slot_mapping_metadata(self, num_reqs,
                                   num_scheduled_tokens_per_req):
        """
        Computes metadata for mapping slots to blocks in the key-value (KV)
        cache for a batch of requests.

        This function determines, for each request in the batch, how the
        scheduled tokens are distributed across memory blocks, and generates
        metadata needed to map slices of tokens to their corresponding positions
        in the KV cache.

        Args:
            num_reqs (int): Number of requests in the current batch.
            num_scheduled_tokens_per_req (int or np.ndarray): Number of tokens
            to be scheduled for each request.

        Returns:
            np.ndarray: A 2D array of shape (total_block_len, 3), where each row
            contains:
                - kv_cache_start_index (int): The starting index in the KV cache
                    for the corresponding slice.
                - new_kv_start_index (int): The starting index in the new KV
                    cache for the corresponding slice.
                - slice_len (int): The length of the slice.
        """
        slices_start = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        slices_end = self.input_batch.num_computed_tokens_cpu[:num_reqs] + \
            num_scheduled_tokens_per_req
        local_block_start_idx = slices_start // self.block_size
        local_block_end_idx = (slices_end - 1) // self.block_size
        no_repeat_req_indices = self.arange_cpu[:num_reqs]
        global_block_start_idx = (
            no_repeat_req_indices * self.max_num_blocks_per_req +
            local_block_start_idx)
        block_lens = local_block_end_idx - local_block_start_idx + 1
        global_block_start_idx = np.repeat(global_block_start_idx, block_lens)
        slice_arange = np.concatenate(
            [self.arange_cpu[:n] for n in block_lens])
        global_block_indices = global_block_start_idx + slice_arange
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[global_block_indices]
        total_block_len = np.sum(block_lens)
        slot_mapping_slices = np.repeat(np.array([[0, self.block_size]],
                                                 dtype=np.int32),
                                        total_block_len,
                                        axis=0)
        cu_block_lens = np.zeros(len(block_lens) + 1, dtype=np.int32)
        np.cumsum(block_lens, out=cu_block_lens[1:])
        for req_idx in range(num_reqs):
            slot_mapping_slices[cu_block_lens[req_idx]][
                0] = slices_start[req_idx] % self.block_size
            slot_mapping_slices[
                cu_block_lens[req_idx + 1] -
                1][1] = (slices_end[req_idx] - 1) % self.block_size + 1
        slice_lens = slot_mapping_slices[:, 1] - slot_mapping_slices[:, 0]
        cu_slices_lens = np.zeros(len(slice_lens) + 1, dtype=np.int32)
        np.cumsum(slice_lens, out=cu_slices_lens[1:])
        kv_cache_start_indices = slot_mapping_slices[:, 0] + \
            (block_numbers * self.block_size)
        new_kv_start_indices = cu_slices_lens[:-1]
        slot_mapping_metadata = np.stack(
            [kv_cache_start_indices, new_kv_start_indices, slice_lens], axis=1)
        return slot_mapping_metadata

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

    def _update_states(self, scheduler_output: "VllmSchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
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
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
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
            self.input_batch.block_table.append_row(new_block_ids, req_index)

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

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

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


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    padded_num_slices = (
        padded_num_slices + NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK - 1
    ) // NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK * \
        NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK
    return padded_num_slices
