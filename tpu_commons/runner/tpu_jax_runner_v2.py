# Here we try to bring as much code as possible from Hex-LLM, instead of `tpu_torch_xla_runner.py` -> jax conversion.
import functools
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as envs
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput

from tpu_commons.logger import init_logger
from tpu_commons.runner.tpu_torch_xla_runner import _get_token_paddings
from tpu_commons.worker.input_batch_jax import CachedRequestState, InputBatch

logger = init_logger(__name__)

MIN_NUM_SEQS = 8
# When chunked prefill is enabled, this is the max number of prefill segments that
# could scheduled in one token batch.
MAX_PREFILL_SEQS_PER_TOKEN_BATCH = 5
MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.


class TPUModelRunner():

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[jax.Device],
        random_key: jax.random.PRNGKey,
    ):
        self.vllm_config = vllm_config
        self.eviction_algorithm = None
        self.devices = devices
        self.random_key = random_key

        self.cache_config = self.vllm_config.cache_config
        self.scheduler_config = self.vllm_config.scheduler_config
        self.model_config = self.vllm_config.model_config
        self.max_model_len = self.model_config.max_model_len
        self.block_size = self.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=self.scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.vocab_size = self.model_config.get_vocab_size()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        self.encoder_cache: dict[str, dict[int, jax.Array]] = {}

        self.params = None

        # TODO(pooyam): Fix these defaults.
        if not getattr(self.scheduler_config, 'chunked_prefill_tokens_padding',
                       None):
            self.scheduler_config.chunked_prefill_tokens_padding = 8
        if not getattr(self.cache_config, 'sink_size', None):
            self.cache_config.sink_size = None
        if not getattr(self.scheduler_config, 'decode_blocks_padding', None):
            self.scheduler_config.decode_blocks_padding = 8

        self.perplexity_reference_text = None

        self.kv_caches: List[Tuple[jax.Array, jax.Array]] = []
        self._init_mesh()
        self._init_jit()

    def load_model(self):
        pass

    def get_kv_cache_spec(self):
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config

        # TODO(xiang): this hack tricks engine core to init successfully
        import torch
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
        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=None,
            pin_memory=None,
            vocab_size=self.model_config.get_vocab_size(),
            kv_cache_config=kv_cache_config,
        )

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

        # TODO(xiang): vllm's pallas backend uses a different shape:
        # (num_blocks, block_size, num_kv_heads * 2, head_size)
        # need to figure out the performance diff.
        cache_shape = (
            kv_cache_spec.num_kv_heads,
            kv_cache_config.num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.head_size,
        )
        # Shard the num_kv_heads dim along the 'model' axis.
        sharding = NamedSharding(self.mesh, PartitionSpec("model"))

        def _allocate() -> Any:
            return jnp.empty(
                shape=cache_shape,
                dtype=cache_dtype,
            )

        sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
        for _ in layer_names:
            k_cache = sharded_allocate()
            v_cache = sharded_allocate()
            self.kv_caches.append((k_cache, v_cache))

    def capture_model(self) -> None:
        pass

    def model_fn(self, is_prefill: bool, do_sampling: bool, kv_caches: Any,
                 input_ids: jax.Array, *args, **kwargs) -> None:

        # batch_size = input_ids.shape[0]

        # next_tokens = np.zeros((batch_size, 1), dtype=np.int32)
        # logits = np.zeros((batch_size, 1), dtype=np.float32)

        # return self.kv_caches, next_tokens, logits, None
        return self.kv_caches, None, None, None

    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        req_id_to_index = {}
        req_ids = []
        prompt_logprobs_dict = {}

        #At first step of this implementation, let's do prefill and decode seperately for maximum code reuse and simpler debug/verification.
        prefill_inputs = self._prepare_prefill(scheduler_output)
        if prefill_inputs is not None:
            model_inputs, (running_indices,
                           output_token_indices) = prefill_inputs
            self.kv_caches, next_tokens_prefill, logits_prefill, single_step_attn_scores_prefill = self.model_fn(
                self.params, *model_inputs)
        decode_inputs = self._prepare_decode(scheduler_output)
        if decode_inputs is not None:
            model_inputs, (running_indices,
                           output_token_indices) = decode_inputs
            self.kv_caches, next_tokens_decode, logits_decode, single_step_attn_scores_decode = self.model_fn(
                self.params, *model_inputs)

        # TODO(pooyam): merge output of prefill and decode and/or add _prepare_chunked_prefill.
        # The goal here is to make sure we are using vllm scheduler output correctly. When we verify that, we can also add `_prepare_chunked_prefill`.

        all_reqs = scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs

        for i, seq in enumerate(all_reqs):
            req_id_to_index[seq.req_id] = i
            req_ids.append(seq.req_id)
            prompt_logprobs_dict[seq.req_id] = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            prompt_logprobs_dict=prompt_logprobs_dict,
            logprobs=None,
            spec_token_ids=None,
            sampled_token_ids=[[0] for _ in range(len(req_ids))],
        )

    def _init_mesh(self) -> None:
        mesh_shape = [1, len(self.devices)]
        axis_names = ["data", "model"]
        self.mesh = Mesh(
            devices=mesh_utils.create_device_mesh(mesh_shape),
            axis_names=axis_names,
        )
        logger.info(f"Init mesh | mesh={self.mesh}")

    def _init_jit(self) -> None:
        pass

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

    # Copy-pasted from `tpu_torch_xla_runner.py`. This is for state management.
    # TODO(pooyam): Consider extraction and reuse instead of copy-paste as this is common logic.
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
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
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

    # Modified from https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py;drc=3ed287d21d5f95a053cb5fe3b249373064ac2f23;l=803.
    def _prepare_decode(self, scheduler_output: VllmSchedulerOutput) -> Any:
        if not len(scheduler_output.scheduled_cached_reqs):
            return None

        num_seqs = len(scheduler_output.scheduled_cached_reqs)
        block_size = self.cache_config.block_size
        sliding_window = self.model_config.get_sliding_window()
        sink_size = self.cache_config.sink_size
        cache_kv_before_rope = self.eviction_algorithm is not None
        max_model_len = self.model_config.max_model_len

        if sliding_window is not None:
            max_num_blocks = ((sliding_window + sink_size) //
                              block_size if sink_size else sliding_window //
                              block_size)
        else:
            max_possible_num_blocks = math.ceil(max_model_len / block_size)
            max_num_blocks = 0
            for seq in scheduler_output.scheduled_cached_reqs:
                seq_index = self.input_batch.req_id_to_index[seq.req_id]
                max_num_blocks = max(
                    max_num_blocks, self.input_batch.block_table.
                    block_tables[0].num_blocks_per_row[seq_index])

            max_num_blocks = pad_to_multiple(
                max_num_blocks,
                self.scheduler_config.decode_blocks_padding,
                max_possible_num_blocks,
                keep_one=True,
            )

        # TODO(pooyam): Fix this. Not padding right now to debug easier.
        # keep_one = not (
        #     (
        #         self.model_config.is_mistral()
        #         and max_num_blocks > MAX_ALLOWED_PAGE_INDICES_N
        #     )
        #     or self.model_config.is_moe()
        # )
        # batch_size = pad_to_multiple(
        #     num_seqs,
        #     self.scheduler_config.decode_seqs_padding,
        #     self.scheduler_config.max_decode_seqs,
        #     keep_one=keep_one,
        # )
        batch_size = num_seqs

        do_sampling = False
        running_indices = np.full((batch_size, ), -1, dtype=np.int32)
        input_token_indices = np.full((batch_size, ), -1, dtype=np.int32)
        input_positions = np.zeros([batch_size, 1], dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        input_ids = np.zeros((batch_size, ), dtype=np.int32)
        block_indices = np.zeros((batch_size, max_num_blocks), dtype=np.int32)
        kv_cache_write_indices = np.full((batch_size, ), -1, dtype=np.int32)
        temperatures = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ps = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ks = np.full((batch_size, ), 1, dtype=np.int32)
        output_token_indices = np.full((batch_size, ), -1, dtype=np.int32)
        kv_cache_position_indices = (np.zeros(
            (batch_size, max_num_blocks *
             block_size), dtype=np.int32) if cache_kv_before_rope else None)
        # Physical indices of the evicted tokens and the replacement tokens.
        evict_write_indices = (np.full((batch_size, ), -1, dtype=np.int32)
                               if self.eviction_algorithm is not None
                               and self.eviction_algorithm != "streamingllm"
                               else None)
        replacement_write_indices = (
            np.full((batch_size, ), -1, dtype=np.int32)
            if self.eviction_algorithm is not None
            and self.eviction_algorithm != "streamingllm" else None)

        for i, seq in enumerate(scheduler_output.scheduled_cached_reqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            seq_len = self.input_batch.num_computed_tokens_cpu[seq_index]

            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            position = seq_len - 1

            running_indices[i] = seq_index
            #input_token_indices[i] = seq.get_decoded_len() - 1
            # TODO(pooyam): Make sure this is correct
            input_token_indices[i] = self.input_batch.num_computed_tokens_cpu[
                seq_index] - self.input_batch.num_prompt_tokens[seq_index]
            assert input_token_indices[i] >= 0

            if self.eviction_algorithm in ["streamingllm", "h2o"]:
                input_positions[i][:] = [
                    min(position, sliding_window + sink_size - 1)
                ]
            else:
                input_positions[i][:] = [position]

            block_indices[i][:len(block_table)] = block_table

            if sliding_window is None:
                seq_lens[i] = seq_len
                assert position // block_size == len(block_table) - 1
                block_id = -1
                block_offset = position % block_size
            else:
                # Let's remove for now for simplicity.
                raise NotImplementedError("Refer to original impl in hex-llm.")

            kv_cache_write_indices[i] = (block_table[block_id] * block_size +
                                         block_offset)

            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]
            output_token_indices[i] = input_token_indices[i] + 1
            if top_ks[i] != 1:
                do_sampling = True

            # TODO(pooyam): Check this.
            input_ids[i] = self.input_batch.token_ids_cpu[seq_index][seq_len -
                                                                     1]

        running_indices = self._device_array(running_indices)
        input_token_indices = self._device_array(input_token_indices)

        # TODO(pooyam): Check this.
        # input_ids = self.read_outputs(
        #     self.output_cache, running_indices, input_token_indices
        # )

        # For perplexity experiments
        if self.perplexity_reference_text is not None:
            raise NotImplementedError("Not implemented.")
            assert len(scheduler_output.scheduled_seqs) == 1
            seq = scheduler_output.scheduled_seqs[0]
            seq_len = seq.get_prompt_len() + seq.get_decoded_len()
            input_ids = input_ids.at[0, 0].set(
                self.perplexity_reference_text[seq_len - 1])

        (
            input_positions,
            seq_lens,
            block_indices,
            kv_cache_write_indices,
            temperatures,
            top_ps,
            top_ks,
            output_token_indices,
        ) = self._device_array((
            input_positions,
            seq_lens,
            block_indices,
            kv_cache_write_indices,
            temperatures,
            top_ps,
            top_ks,
            output_token_indices,
        ))

        if kv_cache_position_indices is not None:
            kv_cache_position_indices = self._device_array(
                kv_cache_position_indices)
        if evict_write_indices is not None:
            evict_write_indices = self._device_array(evict_write_indices)
        eviction_score_mask = None
        return (
            False,  # is prefill
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(input_positions, seq_lens, block_indices,
                              kv_cache_write_indices),
            temperatures,
            top_ps,
            top_ks,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
        ), (
            running_indices,
            output_token_indices,
        )

    def _prepare_prefill(self, scheduler_output: VllmSchedulerOutput) -> Any:
        if not len(scheduler_output.scheduled_new_reqs):
            return None

        block_size = self.vllm_config.cache_config.block_size
        sliding_window = self.vllm_config.model_config.get_sliding_window()

        # TODO: pad batch_size
        batch_size = len(scheduler_output.scheduled_new_reqs)

        # Full prompt length.
        max_prompt_len = max([
            len(seq.prompt_token_ids)
            for seq in scheduler_output.scheduled_new_reqs
        ])

        # Unfilled prompt length.
        # TODO: Fix this for prefix caching.
        max_unfilled_prompt_len = max_prompt_len

        # Padded full prompt length.
        # TODO: Fix padding.
        padded_prompt_len = max_prompt_len

        # Padded unfilled prompt length.
        # TODO: Fix padding.
        padded_unfilled_prompt_len = max_unfilled_prompt_len

        images_flattened = []

        if sliding_window:
            raise NotImplementedError("Sliding window not implemented.")
        else:
            padded_num_blocks = math.ceil(padded_prompt_len / block_size)
            # same as `padded_num_blocks` when no cache hit.
            padded_num_unfilled_blocks = math.ceil(padded_unfilled_prompt_len /
                                                   block_size)

        do_sampling = False
        input_ids = np.zeros((batch_size, padded_unfilled_prompt_len),
                             dtype=np.int32)
        input_positions = np.zeros((batch_size, padded_unfilled_prompt_len),
                                   dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        image_lens = np.zeros((batch_size, ), dtype=np.int32)
        block_indices = np.zeros((batch_size, padded_num_blocks),
                                 dtype=np.int32)
        kv_cache_write_indices = np.zeros(
            (batch_size, padded_num_unfilled_blocks), dtype=np.int32)
        temperatures = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ps = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ks = np.full((batch_size, ), 1, dtype=np.int32)
        running_indices = np.full((batch_size, ), -1, dtype=np.int32)
        output_token_indices = np.full((batch_size, ), -1, dtype=np.int32)

        eviction_score_mask = None

        for i, seq in enumerate(scheduler_output.scheduled_new_reqs):
            effective_cached_prompt_len = 0
            num_effective_cached_blocks = 0
            prompt_token_ids = seq.prompt_token_ids[
                effective_cached_prompt_len:]
            prompt_len = len(prompt_token_ids)
            input_ids[i][:prompt_len] = prompt_token_ids
            input_positions[i][:prompt_len] = list(
                range(
                    effective_cached_prompt_len,
                    effective_cached_prompt_len + prompt_len,
                ))
            seq_lens[i] = prompt_len

            # Full prompt associated block indices.
            block_table = seq.block_ids[0]
            assert len(block_table) <= padded_num_blocks
            block_indices[i][:len(block_table)] = block_table
            # Unfilled prompt associated block indices.
            num_unfilled_blocks = len(
                block_table) - num_effective_cached_blocks
            assert num_unfilled_blocks <= padded_num_unfilled_blocks
            kv_cache_write_indices[i][:num_unfilled_blocks] = block_table[
                num_effective_cached_blocks:]
            temperatures[i] = seq.sampling_params.temperature
            top_ps[i] = seq.sampling_params.top_p
            top_ks[i] = seq.sampling_params.top_k

            # TODO(pooyam): How to get this?
            running_indices[i] = 0  # seq.running_id

            # TODO(pooyam): double check this.
            output_token_indices[i] = seq.num_computed_tokens

            if seq.sampling_params.top_k != 1:
                do_sampling = True
            if eviction_score_mask is not None:
                raise NotImplementedError("Evication not implemented.")

        input_ids = self._device_array(input_ids)
        input_positions = self._device_array(input_positions)
        seq_lens = self._device_array(seq_lens)
        temperatures = self._device_array(temperatures).astype(jnp.bfloat16)
        top_ps = self._device_array(top_ps).astype(jnp.bfloat16)
        top_ks = self._device_array(top_ks)
        block_indices = self._device_array(block_indices)
        kv_cache_write_indices = self._device_array(kv_cache_write_indices)
        running_indices = self._device_array(running_indices)
        output_token_indices = self._device_array(output_token_indices)
        kv_cache_position_indices = None
        evict_write_indices = None
        replacement_write_indices = None
        eviction_score_mask = (self._device_array(eviction_score_mask)
                               if eviction_score_mask is not None else None)

        return (
            True,
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(input_positions, seq_lens, block_indices,
                              kv_cache_write_indices),
            temperatures,
            top_ps,
            top_ks,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
            images_flattened,
            image_lens,
        ), (
            running_indices,
            output_token_indices,
        )


def pad_to_multiple(x: int,
                    multiple: int = 8,
                    max_limit: Optional[int] = None,
                    keep_one: bool = False) -> int:
    assert x > 0
    if keep_one and x == 1:
        return x
    x = x + (-x % multiple)
    if max_limit is not None:
        x = min(x, max_limit)
    return x


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "seq_lens",
        "block_indices",
        "kv_cache_write_indices",
        "decode_lengths",
        "decode_page_indices",
        "num_decode_seqs",
        "prefill_lengths",
        "prefill_page_indices",
        "prefill_query_start_offsets",
        "num_prefill_seqs",
    ],
    meta_fields=["chunked_prefill_enabled"],
)
@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]
    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]
