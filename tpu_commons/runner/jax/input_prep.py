import math
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.runner.utils import determine_do_sampling, pad_to_multiple

# When chunked prefill is enabled, this is the max number of prefill segments that
# could scheduled in one token batch.
MAX_PREFILL_SEQS_PER_TOKEN_BATCH = None
MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.
KVCaches = List[Tuple[jax.Array, jax.Array]]


class InputPrep:

    def __init__(self, vllm_config: VllmConfig, mesh, input_batch,
                 max_num_reqs, jitted_read_outputs):

        self.prefill_seqs_padding = 8
        self.decode_seqs_padding = 8

        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.input_batch = input_batch
        self.max_num_reqs = max_num_reqs
        self.perplexity_reference_text = None
        self.mesh = mesh
        self.jitted_read_outputs = jitted_read_outputs
        self.eviction_algorithm = None

        global MAX_PREFILL_SEQS_PER_TOKEN_BATCH
        # NOTE(pooyam): Currently we don't have a logic in vLLM scheduler to enfore certain upper-bound for number of prefilling seqs.
        # We can remove this and make it configurable once we have such thing in vLLM scheduler.
        # Also gxd@ mentioned to me he had not benchmarked the previous number which was `5` so it's not clear even if it's needed or not.
        MAX_PREFILL_SEQS_PER_TOKEN_BATCH = self.scheduler_config.max_num_seqs

        # https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py#:~:text=143-,144,-145
        # Prepare buffers used by chunk prefill
        max_num_running_seq = self.scheduler_config.max_num_seqs
        num_blocks_per_seq = (pad_to_multiple(
            self.model_config.max_model_len,
            self.cache_config.block_size,
        ) // self.cache_config.block_size)
        self.decode_seq_lens = np.zeros((max_num_running_seq, ),
                                        dtype=np.int32)
        self.decode_block_indices = np.zeros(
            (max_num_running_seq, num_blocks_per_seq), dtype=np.int32)
        self.prefill_seq_lens = np.zeros((MAX_PREFILL_SEQS_PER_TOKEN_BATCH, ),
                                         dtype=np.int32)
        self.prefill_block_indices = np.zeros(
            (MAX_PREFILL_SEQS_PER_TOKEN_BATCH, num_blocks_per_seq),
            dtype=np.int32)
        self.prefill_query_start_offsets = np.zeros(
            (MAX_PREFILL_SEQS_PER_TOKEN_BATCH + 1, ), dtype=np.int32)

    def prepare_inputs(self, scheduler_output: SchedulerOutput,
                       kv_caches: KVCaches, output_cache: jax.Array):
        # NOTE(pooyam): Different kernels have different performance for prefill, decode, mixed model.
        # We should prioritize performance first, but once performance is equal, we should opt in for fewer kernels for simplicity.
        # Current assumption from older data is that ragged kernel is good for mixed inference, prefill kernel is good for prefill-only and paged kernel is good for decode-only.
        new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs = self._get_prefill_and_decode_seqs(
            scheduler_output)

        has_new_full = bool(new_full_prefill_seqs)
        has_new_partial = bool(new_partial_prefill_seqs)
        has_subsequent_partial = bool(subsequent_partial_prefill_seqs)
        has_decoding = bool(decoding_seqs)

        if not (has_new_full or has_new_partial or has_subsequent_partial
                or has_decoding):
            return None

        if not new_full_prefill_seqs and not new_partial_prefill_seqs and not subsequent_partial_prefill_seqs:
            return self._prepare_decode(decoding_seqs, scheduler_output,
                                        kv_caches, output_cache)

        def _tokens_after_padding_in_prefill_mode(scheduler_output, reqs):
            batch_size = len(reqs)
            batch_size = pad_to_multiple(batch_size, self.prefill_seqs_padding,
                                         self.scheduler_config.max_num_seqs,
                                         True)
            max_prompt_len = max([
                scheduler_output.num_scheduled_tokens[req.req_id]
                for req in reqs
            ])
            padded_prompt_len = pad_to_multiple(
                max_prompt_len, self.scheduler_config.prefill_len_padding,
                self.model_config.max_model_len)

            return batch_size * padded_prompt_len

        # NOTE(pooyam): Based on my benchmark, ragged kernel itself had superior performance compared to flash attention and splash attention across 1K, 2K, 4K, and 8K lengths.
        # However, step time for < 4K tokens is higher for ragged kernel due to inefficient update cache.
        # After it got optimized, we can simplify the following conditions.
        if (new_full_prefill_seqs or new_partial_prefill_seqs
            ) and not subsequent_partial_prefill_seqs and not decoding_seqs:
            if _tokens_after_padding_in_prefill_mode(
                    scheduler_output,
                    new_full_prefill_seqs + new_partial_prefill_seqs) <= 2048:
                return self._prepare_prefill(
                    new_full_prefill_seqs + new_partial_prefill_seqs,
                    scheduler_output, kv_caches, output_cache)
            else:
                return self._prepare_chunked_prefill(scheduler_output,
                                                     kv_caches, output_cache)

        # All other cases fall into the "chunked prefill" category
        return self._prepare_chunked_prefill(scheduler_output, kv_caches,
                                             output_cache)

    # Modified from https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py;drc=3ed287d21d5f95a053cb5fe3b249373064ac2f23;l=803.
    def _prepare_decode(self, reqs: List[NewRequestData | CachedRequestData],
                        scheduler_output: SchedulerOutput, kv_caches: KVCaches,
                        output_cache: jax.Array) -> Any:
        if not len(reqs):
            return None

        num_seqs = len(reqs)
        block_size = self.cache_config.block_size
        sliding_window = self.model_config.get_sliding_window()
        sink_size = self.cache_config.sink_size
        cache_kv_before_rope = False
        max_model_len = self.model_config.max_model_len

        if sliding_window is not None:
            max_num_blocks = ((sliding_window + sink_size) //
                              block_size if sink_size else sliding_window //
                              block_size)
        else:
            max_possible_num_blocks = math.ceil(max_model_len / block_size)
            max_num_blocks = 0
            for seq in reqs:
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

        is_moe = hasattr(self.model_config.hf_config, "num_local_experts")
        is_mistral = self.model_config.hf_config.model_type == "mistral"
        keep_one = not (
            (is_mistral and max_num_blocks > MAX_ALLOWED_PAGE_INDICES_N)
            or is_moe)
        batch_size = pad_to_multiple(
            num_seqs,
            self.decode_seqs_padding,
            self.max_num_reqs,
            keep_one=keep_one,
        )

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

        for i, seq in enumerate(reqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            # num cached tokens + token of this decode
            seq_len = self.input_batch.num_computed_tokens_cpu[seq_index] + 1

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
            do_sampling = determine_do_sampling(top_ks[i], temperatures[i])

        running_indices = self._device_array(running_indices)
        input_token_indices = self._device_array(input_token_indices)

        input_ids = self.jitted_read_outputs(output_cache, running_indices,
                                             input_token_indices)

        # For perplexity experiments
        if self.perplexity_reference_text is not None:
            raise NotImplementedError("Not implemented.")

        for seq in reqs:
            req_id = seq.req_id
            seq_index = self.input_batch.req_id_to_index[req_id]

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
            kv_caches,
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

    def _prepare_prefill(self, reqs: List[NewRequestData | CachedRequestData],
                         scheduler_output: SchedulerOutput,
                         kv_caches: KVCaches, output_cache: jax.Array) -> Any:
        if not len(reqs):
            return None

        block_size = self.cache_config.block_size
        sliding_window = self.model_config.get_sliding_window()

        batch_size = len(reqs)
        batch_size = pad_to_multiple(
            batch_size,
            self.prefill_seqs_padding,
            self.scheduler_config.max_num_seqs,
            True,
        )

        # Full prompt length.
        max_prompt_len = max([
            scheduler_output.num_scheduled_tokens[req.req_id] for req in reqs
        ])

        # Unfilled prompt length.
        # TODO: Fix this for prefix caching.
        max_unfilled_prompt_len = max_prompt_len

        # Padded full prompt length.
        padded_prompt_len = pad_to_multiple(
            max_prompt_len,
            self.scheduler_config.prefill_len_padding,
            self.model_config.max_model_len,
        )

        # Padded unfilled prompt length.
        padded_unfilled_prompt_len = pad_to_multiple(
            max_unfilled_prompt_len,
            self.scheduler_config.prefill_len_padding,
            self.model_config.max_model_len,
        )

        images_flattened = None

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

        for i, seq in enumerate(reqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]

            effective_cached_prompt_len = 0
            num_effective_cached_blocks = 0
            prompt_token_ids = seq.prompt_token_ids[
                effective_cached_prompt_len:]
            scheduled_tokens = scheduler_output.num_scheduled_tokens[
                seq.req_id]
            assert scheduled_tokens <= len(prompt_token_ids)
            prompt_len = scheduled_tokens
            input_ids[i][:prompt_len] = prompt_token_ids[:prompt_len]
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
            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]

            running_indices[i] = seq_index

            # TODO(pooyam): double check this.
            output_token_indices[i] = seq.num_computed_tokens

            do_sampling = determine_do_sampling(
                seq.sampling_params.top_k, seq.sampling_params.temperature)
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
            kv_caches,
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

    def _prepare_chunked_prefill(self, scheduler_output: SchedulerOutput,
                                 kv_caches: KVCaches,
                                 output_cache: jax.Array) -> Any:
        block_size = self.cache_config.block_size
        # in vLLMs scheduler output, scheduled_cached_reqs can mean two things: Subsequent prefill of an already seen request / or decode.

        new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs = self._get_prefill_and_decode_seqs(
            scheduler_output)
        num_decode_seqs = len(decoding_seqs)
        num_prefill_seqs = len(new_full_prefill_seqs) + len(
            new_partial_prefill_seqs) + len(subsequent_partial_prefill_seqs)
        assert num_prefill_seqs > 0
        assert num_prefill_seqs <= MAX_PREFILL_SEQS_PER_TOKEN_BATCH

        if self.scheduler_config.page_aligned_scheduling:
            # NOTE(pooyam): It's possible to remove this loop by approximating by upper bound.
            num_tokens_scheduled = pad_to_multiple(
                num_decode_seqs, block_size) if num_decode_seqs > 0 else 0
            for seq in new_full_prefill_seqs + new_partial_prefill_seqs + subsequent_partial_prefill_seqs:
                num_tokens_scheduled += pad_to_multiple(
                    scheduler_output.num_scheduled_tokens[seq.req_id],
                    block_size)
            num_tokens_scheduled = pad_to_multiple(
                num_tokens_scheduled,
                self.scheduler_config.chunked_prefill_tokens_padding)
        else:
            num_tokens_scheduled = pad_to_multiple(
                scheduler_output.total_num_scheduled_tokens,
                self.scheduler_config.chunked_prefill_tokens_padding)

        if num_decode_seqs > 0:
            decode_input_token_indices = np.full((num_tokens_scheduled, ),
                                                 -1,
                                                 dtype=np.int32)
        else:
            decode_input_token_indices = None

        do_sampling = False
        decode_seq_lens = self.decode_seq_lens
        decode_seq_lens[num_decode_seqs:] = 0
        decode_block_indices = self.decode_block_indices
        input_positions = np.zeros((1, num_tokens_scheduled), dtype=np.int32)
        decode_kv_cache_write_indices = np.full((num_tokens_scheduled, ),
                                                -1,
                                                dtype=np.int32)
        temperatures = np.full((num_tokens_scheduled, ), 1.0, dtype=np.float32)
        top_ps = np.full((num_tokens_scheduled, ), 1.0, dtype=np.float32)
        top_ks = np.full((num_tokens_scheduled, ), 1, dtype=np.int32)
        running_indices = np.full((num_tokens_scheduled, ), -1, dtype=np.int32)
        output_token_indices = np.full((num_tokens_scheduled, ),
                                       -1,
                                       dtype=np.int32)

        # Fill the token batch with decode tokens first.
        for i, seq in enumerate(decoding_seqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            seq_len = self.input_batch.num_computed_tokens_cpu[seq_index] + 1

            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            position = seq_len - 1

            running_indices[i] = seq_index
            decode_input_token_indices[
                i] = self.input_batch.num_computed_tokens_cpu[
                    seq_index] - self.input_batch.num_prompt_tokens[seq_index]
            assert decode_input_token_indices[i] >= 0
            input_positions[:, i] = position
            decode_block_indices[i][:len(block_table)] = block_table

            decode_seq_lens[i] = seq_len
            assert position // block_size == len(block_table) - 1
            block_id = -1
            block_offset = position % block_size
            decode_kv_cache_write_indices[i] = (
                block_table[block_id] * block_size + block_offset)
            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]
            output_token_indices[i] = decode_input_token_indices[i] + 1
            do_sampling = determine_do_sampling(top_ks[i], temperatures[i])

        token_offset = num_decode_seqs

        if self.scheduler_config.page_aligned_scheduling:
            if num_decode_seqs > 0 and num_prefill_seqs > 0:
                # Add padding tokens so that prefill segments are paged aligned
                if num_decode_seqs % block_size != 0:
                    token_offset = pad_to_multiple(num_decode_seqs, block_size)

        # Then fill the token batch with prefill tokens.
        prefill_seq_lens = self.prefill_seq_lens
        prefill_seq_lens[num_prefill_seqs:] = 0
        prefill_block_indices = self.prefill_block_indices
        prefill_query_start_offsets = self.prefill_query_start_offsets

        if self.scheduler_config.page_aligned_scheduling:
            # One cache update index per page for prefill.
            assert num_tokens_scheduled % block_size == 0
            prefill_kv_cache_write_indices = np.full(
                (num_tokens_scheduled // block_size, ), -1, dtype=np.int32)
        else:
            prefill_kv_cache_write_indices = np.full((num_tokens_scheduled, ),
                                                     -1,
                                                     dtype=np.int32)

        prefill_input_ids = np.zeros((1, num_tokens_scheduled), dtype=np.int32)
        for i, seq in enumerate(new_full_prefill_seqs +
                                new_partial_prefill_seqs +
                                subsequent_partial_prefill_seqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            num_tokens = self.input_batch.num_tokens[seq_index]
            prefill_len = scheduler_output.num_scheduled_tokens[seq.req_id]
            num_prefilled_tokens = self.input_batch.num_computed_tokens_cpu[
                seq_index]
            # TODO(pooyam): How to make sure we are not reading beyond prefill?
            prefill_input_ids[:, token_offset:token_offset +
                              prefill_len] = (self.input_batch.token_ids_cpu[
                                  seq_index,
                                  num_prefilled_tokens:num_prefilled_tokens +
                                  prefill_len])
            input_positions[:, token_offset:token_offset + prefill_len] = list(
                range(
                    num_prefilled_tokens,
                    num_prefilled_tokens + prefill_len,
                ))

            if self.scheduler_config.page_aligned_scheduling:
                prefill_seq_lens[i] = pad_to_multiple(
                    num_prefilled_tokens + prefill_len, block_size)
            else:
                prefill_seq_lens[i] = num_prefilled_tokens + prefill_len

            prefill_query_start_offsets[i] = token_offset
            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            prefill_block_indices[i][:len(block_table)] = block_table

            if self.scheduler_config.page_aligned_scheduling:
                prefill_kv_cache_write_indices[
                    token_offset // block_size:math.ceil(
                        (token_offset + prefill_len) /
                        block_size)] = block_table[num_prefilled_tokens //
                                                   block_size:math.ceil(
                                                       (num_prefilled_tokens +
                                                        prefill_len) /
                                                       block_size)]
            else:
                for j in range(prefill_len):
                    position = num_prefilled_tokens + j
                    block_index = position // block_size
                    assert block_index <= len(block_table) - 1
                    block_offset = position % block_size
                    prefill_kv_cache_write_indices[
                        token_offset + j] = block_table[
                            block_index] * block_size + block_offset

            assert num_prefilled_tokens + prefill_len <= num_tokens

            if num_prefilled_tokens + prefill_len == num_tokens:
                # only in this case, a new decode token will be generated
                last_prefill_token_idx = token_offset + prefill_len - 1
                running_indices[last_prefill_token_idx] = seq_index

                # Hex-LLM equivalent: output_token_indices[last_prefill_token_idx] = seq.get_decoded_len()
                # NOTE(pooyam): Is there a case where this shouldn't be 0?
                output_token_indices[last_prefill_token_idx] = 0

                temperatures[
                    last_prefill_token_idx] = self.input_batch.temperature_cpu[
                        seq_index]
                top_ps[last_prefill_token_idx] = self.input_batch.top_p_cpu[
                    seq_index]
                top_ks[last_prefill_token_idx] = self.input_batch.top_k_cpu[
                    seq_index]

            do_sampling = determine_do_sampling(
                self.input_batch.top_k_cpu[seq_index],
                self.input_batch.temperature_cpu[seq_index])

            if self.scheduler_config.page_aligned_scheduling:
                # Add padding tokens so that prefill segments are paged aligned
                token_offset = pad_to_multiple(token_offset + prefill_len,
                                               block_size)
            else:
                token_offset += prefill_len

        prefill_query_start_offsets[num_prefill_seqs:] = token_offset

        if self.scheduler_config.page_aligned_scheduling:
            # Concat the kv cache write indices for decode tokens and prefill tokens
            kv_cache_write_indices = np.concatenate(
                (decode_kv_cache_write_indices,
                 prefill_kv_cache_write_indices))
        else:
            kv_cache_write_indices = jnp.where(
                jnp.arange(num_tokens_scheduled) < num_decode_seqs,
                decode_kv_cache_write_indices, prefill_kv_cache_write_indices)
        num_decode_seqs_arr = np.array([num_decode_seqs], np.int32)
        num_prefill_seqs_arr = np.array([num_prefill_seqs], np.int32)

        (
            prefill_input_ids,
            decode_input_token_indices,
            input_positions,
            temperatures,
            top_ps,
            top_ks,
            kv_cache_write_indices,
            running_indices,
            output_token_indices,
            decode_seq_lens,
            decode_block_indices,
            num_decode_seqs_arr,
            prefill_seq_lens,
            prefill_block_indices,
            prefill_query_start_offsets,
            num_prefill_seqs_arr,
        ) = self._device_array((
            prefill_input_ids,
            decode_input_token_indices,
            input_positions,
            temperatures,
            top_ps,
            top_ks,
            kv_cache_write_indices,
            running_indices,
            output_token_indices,
            decode_seq_lens,
            decode_block_indices,
            num_decode_seqs_arr,
            prefill_seq_lens,
            prefill_block_indices,
            prefill_query_start_offsets,
            num_prefill_seqs_arr,
        ))
        # Merge decode tokens with prefill tokens.
        if num_decode_seqs == 0:
            input_ids = prefill_input_ids
        else:
            decode_input_ids = jnp.swapaxes(
                self.jitted_read_outputs(output_cache, running_indices,
                                         decode_input_token_indices),
                0,
                1,
            )
            input_ids = jnp.where(
                jnp.arange(num_tokens_scheduled) < num_decode_seqs,
                decode_input_ids,
                prefill_input_ids,
            )
        kv_cache_position_indices = None
        evict_write_indices = None
        replacement_write_indices = None
        eviction_score_mask = None
        return (
            False,  # when chunked prefill is enabled, `is_prefill` is just a dummy value.
            do_sampling,
            kv_caches,
            input_ids,
            AttentionMetadata(
                input_positions=input_positions,
                seq_lens=None,  # use decode_lengths / prefill_lengths instead
                block_indices=
                None,  # use decode_page_indices / prefill_page_indices instead
                kv_cache_write_indices=kv_cache_write_indices,
                chunked_prefill_enabled=True,
                decode_lengths=decode_seq_lens,
                decode_page_indices=decode_block_indices,
                num_decode_seqs=num_decode_seqs_arr,
                prefill_lengths=prefill_seq_lens,
                prefill_page_indices=prefill_block_indices,
                prefill_query_start_offsets=prefill_query_start_offsets,
                num_prefill_seqs=num_prefill_seqs_arr,
                page_aligned_update=self.scheduler_config.
                page_aligned_scheduling,
            ),
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

    def _get_prefill_and_decode_seqs(
        self, scheduler_output: SchedulerOutput
    ) -> Tuple[List[NewRequestData], List[CachedRequestData],
               List[CachedRequestData]]:
        # NOTE(pooyam): We categorize sequences into 4 different categories to have freedom to choose which kernel to use for each of them:
        # 1. full prefill: A sequence which is scheduled for a full prefill -> We can use prefill or chunked prefill kernel.
        # 2. new partial prefill: A sequence that its first chunk is scheduled for prefill -> We can use prefill or chunked prefill kernel.
        # 3. subsequent partial prefill: A sequence that its subsequent chunks are scheduled for prefill -> We should use chunked prefill kernel.
        # 4. decode: A sequence that has been completely prefilled already => We should use decode or chunked prefill kernel.
        # NOTE(pooyam): If God helps and we benchmark that ragged attention is superior in every scenario, we can simplify all these and get rid of this misery.

        new_full_prefill_seqs = []
        new_partial_prefill_seqs = []
        subsequent_partial_prefill_seqs = []
        decoding_seqs = []

        # NOTE(pooyam): The lesson learned was that `num_prompt_tokens` is not a reliable field to decide prefilling state based on solely.
        # The reason is that once a request is preempated and added back later, `num_prompt_tokens` still points to original number of prompts.
        # However, once the request is added back, we need to prefill previous prompt tokens AND previous output tokens again.
        # We should use `num_scheduled_tokens`, `num_computed_tokens` and **`num_tokens`** in most of the logics.
        for seq in (scheduler_output.scheduled_new_reqs +
                    scheduler_output.scheduled_cached_reqs):
            index = self.input_batch.req_id_to_index[seq.req_id]
            num_tokens = self.input_batch.num_tokens[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                seq.req_id]
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[
                index]

            assert num_computed_tokens + num_scheduled_tokens <= num_tokens

            if num_computed_tokens + num_scheduled_tokens < num_tokens:
                if num_computed_tokens == 0:
                    new_partial_prefill_seqs.append(seq)
                else:
                    subsequent_partial_prefill_seqs.append(seq)
            else:
                if num_computed_tokens == 0:
                    new_full_prefill_seqs.append(seq)
                elif num_scheduled_tokens != 1:
                    subsequent_partial_prefill_seqs.append(seq)
                elif num_scheduled_tokens == 1:
                    # We need to distinguish the following cases:
                    # Case 1: not preempted at all:  prompt: 10 | output tokens: 4 | num_computed: 13 | num scheduled: 1 | num tokens: 14 => decoding and it has written sth before => write to position i=4 in output cache.
                    # Case 2: preempted and added back:  prompt: 10 | output tokens: 4 | num_computed: 13 | num_scheduled: 1 | num tokens: 14 => prefilling and it has not written sth before => write to position 0 in output cache.
                    # As you see, metadata related to Case 1 and Case 2 are exactly the same. Then how to distinguish? One idea is to use `resumed_from_preemption` field.
                    is_cached_req = isinstance(seq, CachedRequestData)
                    is_resumed_from_preemption = is_cached_req and seq.resumed_from_preemption
                    if not is_resumed_from_preemption:
                        if num_computed_tokens < self.input_batch.num_prompt_tokens[
                                index]:
                            subsequent_partial_prefill_seqs.append(seq)
                        else:
                            decoding_seqs.append(seq)
                    else:
                        subsequent_partial_prefill_seqs.append(seq)
                else:
                    raise ValueError("This should not happen.")

        return new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)
