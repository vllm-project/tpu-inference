import os
from dataclasses import asdict
from typing import Any, List, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as envs
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from tpu_commons import utils_jax as utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.sharding import Sharding
from tpu_commons.models.jax.model_loader import get_model
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.tpu_torch_xla_runner import (_get_padded_token_len,
                                                     _get_req_paddings,
                                                     _get_token_paddings)

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8
# Block size used for kv cache updating kernel
NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK = 8


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
                                vllm_config=self.vllm_config)
            self.mesh = sharding.mesh
        else:
            axis_names = ("data", "model")
            mesh_shape = (1, len(self.devices))
            self.mesh = jax.make_mesh(mesh_shape,
                                      axis_names,
                                      devices=self.devices)

            # In case we are in disagg mode, the number of devices can exceed 8.
            # TODO(fhzhang): fix this properly as we implement disagg serving.
            if len(self.devices) > 8:
                self.devices = self.devices[:8]
            mesh_shape = (1, len(self.devices))
            self.mesh = jax.make_mesh(mesh_shape,
                                      axis_names,
                                      devices=self.devices)
        logger.warning(f"Init mesh | mesh={self.mesh}")

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
        self.num_tokens_paddings = _get_token_paddings(
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
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        self.temperatures_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ps_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)
        self.top_ks_cpu = np.zeros(self.max_num_tokens, dtype=np.float32)

    def load_model(self):
        self.model_fn = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )
        logger.info(f"Init model | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}Gb")

    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        import torch
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model_config.get_total_num_kv_heads(),
                head_size=model_config.get_head_size(),
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
        cache_dtype = jnp.bfloat16
        # TODO(xiang): fix this together with get_kv_cache_spec
        # cache_dtype = kv_cache_spec.dtype

        cache_shape = (
            kv_cache_config.num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads * 2,
            kv_cache_spec.head_size,
        )

        # Shard the num_kv_heads dim along the 'model' axis.
        sharding = NamedSharding(self.mesh, PartitionSpec(None, None, "model"))

        def _allocate() -> Any:
            return jnp.empty(
                shape=cache_shape,
                dtype=cache_dtype,
            )

        sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
        for _ in layer_names:
            self.kv_caches.append(sharded_allocate())

        logger.info(jax.lib.xla_bridge.get_backend().platform_version)
        logger.info(f"Init kv-cache | "
                    f"shape={len(layer_names)} * {cache_shape} | "
                    f"sharding={sharding} | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}Gb")

    def capture_model(self) -> None:
        pass

    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        inputs = self._prepare_inputs(scheduler_output)
        self.kv_caches, next_tokens, _ = self.model_fn(*inputs)

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

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )
        return model_runner_output

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
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0

        # Sampling
        req_ids = self.arange_cpu[:num_reqs]
        do_sampling = _do_sampling(self.input_batch.top_k_cpu[req_ids],
                                   self.input_batch.temperature_cpu[req_ids])
        if do_sampling:
            self.temperatures_cpu[:
                                  total_num_scheduled_tokens] = self.input_batch.temperature_cpu[
                                      req_indices]
            self.top_ps_cpu[:
                            total_num_scheduled_tokens] = self.input_batch.top_p_cpu[
                                req_indices]
            self.top_ks_cpu[:
                            total_num_scheduled_tokens] = self.input_batch.top_k_cpu[
                                req_indices]
            temperatures = self.temperatures_cpu[:
                                                 padded_total_num_scheduled_tokens]
            top_ps = self.top_ps_cpu[:padded_total_num_scheduled_tokens]
            top_ks = self.top_ks_cpu[:padded_total_num_scheduled_tokens]
            (temperatures, top_ps, top_ks) = self._device_array(
                (temperatures, top_ps, top_ks))
        else:
            temperatures = None
            top_ps = None
            top_ks = None

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
        num_reqs = np.array([num_reqs])

        (input_ids, positions, slot_mapping_metadata, num_slices, block_tables,
         query_start_loc, seq_lens, num_reqs) = self._device_array(
             (input_ids, positions, slot_mapping_metadata, num_slices,
              block_tables, query_start_loc, seq_lens, num_reqs))

        return (
            False,
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(
                input_positions=positions,
                seq_lens=seq_lens,
                block_indices=block_tables,
                kv_cache_write_indices=slot_mapping_metadata,
                num_prefill_seqs=num_slices,
                prefill_query_start_offsets=query_start_loc,
                num_decode_seqs=num_reqs,
                chunked_prefill_enabled=True,
            ),
            temperatures,
            top_ps,
            top_ks,
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
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_block_ids in zip(req_state.block_ids,
                                                    req_data.new_block_ids,
                                                    strict=True):
                    block_ids.extend(new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)

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


def _do_sampling(top_ks: np.ndarray, temperatures: np.ndarray) -> bool:
    return np.any(top_ks != 1) and np.any(temperatures != 0.0)
