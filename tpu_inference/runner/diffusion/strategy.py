# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from vllm.forward_context import set_forward_context
from vllm.v1.outputs import ModelRunnerOutput

from tpu_inference.layers.common.attention_metadata import (AttentionMaskKind,
                                                            AttentionMaskSpec,
                                                            AttentionMetadata)
from tpu_inference.logger import init_logger
from tpu_inference.runner.diffusion.algorithm import (
    get_commit_algorithm, get_commit_diagnostics_algorithm)
from tpu_inference.runner.diffusion.batch import (
    PendingBlockOutput, complete_seeded_decode_block, diffusion_batch_sizes,
    flush_partial_block_output, needs_block_anchor, plan_seeded_prompt,
    required_cache_end, select_diffusion_batch_size,
    start_partial_block_output, trim_generation_mask)
from tpu_inference.runner.diffusion.config import (CanvasPolicy,
                                                   DiffusionConfig,
                                                   NextBlockPolicy,
                                                   PromptRemainderPolicy)
from tpu_inference.runner.diffusion.program import (
    DUAL_CACHE_Q32, DUAL_CACHE_Q8, DualCacheAcceptanceTrace, denoise_block,
    denoise_block_dual_cache, select_aligned_hidden_states)
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

DUAL_CACHE_ACCEPTANCE_TRACE_PREFIX = \
    "Fast-dLLM DualCache acceptance step: "


def _eligible_finite_values(values: np.ndarray,
                            eligible: np.ndarray) -> list[float | None]:
    return [
        float(value) if is_eligible and np.isfinite(value) else None
        for value, is_eligible in zip(values, eligible, strict=True)
    ]


def _log_dual_cache_acceptance_trace(trace: DualCacheAcceptanceTrace) -> None:
    trace_host = jax.device_get(trace)
    count = int(trace_host.count)
    block_start = int(trace_host.block_start)
    forward_kinds = {
        DUAL_CACHE_Q32: "q32",
        DUAL_CACHE_Q8: "q8",
    }
    for index in range(count):
        eligible = np.asarray(trace_host.row0_eligible[index], dtype=bool)
        record = {
            "block_start":
            block_start,
            "subblock_start":
            int(trace_host.sub_block_starts[index]),
            "iteration":
            int(trace_host.iterations[index]),
            "forward_kind":
            forward_kinds[int(trace_host.forward_kinds[index])],
            "row0_token_ids":
            np.asarray(trace_host.row0_token_ids[index]).tolist(),
            "row0_eligible":
            eligible.tolist(),
            "row0_commit":
            np.asarray(trace_host.row0_commit[index], dtype=bool).tolist(),
            "row0_remaining":
            np.asarray(trace_host.row0_remaining[index], dtype=bool).tolist(),
            "row0_forced_anchor":
            bool(trace_host.row0_forced_anchor[index]),
            "row0_selected_log_confidence":
            _eligible_finite_values(
                np.asarray(trace_host.row0_selected_log_confidence[index]),
                eligible,
            ),
            "row0_threshold_margin":
            _eligible_finite_values(
                np.asarray(trace_host.row0_threshold_margin[index]),
                eligible,
            ),
        }
        logger.info(
            "%s%s",
            DUAL_CACHE_ACCEPTANCE_TRACE_PREFIX,
            json.dumps(record,
                       allow_nan=False,
                       separators=(",", ":"),
                       sort_keys=True),
        )


class BlockDiffusionStrategy:

    def __init__(self, runner: "TPUModelRunner",
                 config: DiffusionConfig) -> None:
        self.runner = runner
        self.config = config
        self._pending_outputs: dict[str, PendingBlockOutput] = {}
        self._last_denoise_trace: dict[str, Any] | None = None
        self._last_prefill_trace: dict[str, Any] | None = None
        self._forward_fn = self._model_forward
        self._full_subblock_forward_fn = self._model_forward_full_subblock
        self._partial_subblock_forward_fn = self._model_forward_partial_subblock
        self._final_forward_fn = self._model_forward_final
        self._commit_fn = get_commit_algorithm(config.runtime.algorithm)
        self._commit_diagnostics_fn = get_commit_diagnostics_algorithm(
            config.runtime.algorithm)

        model = config.model
        if model.canvas_policy is not CanvasPolicy.SEED_AND_MASK:
            raise ValueError(
                "The TPU serving strategy currently requires seed_and_mask "
                "canvas semantics")
        if model.prompt_remainder_policy is not \
                PromptRemainderPolicy.INCLUDE_IN_FIRST_CANVAS:
            raise ValueError(
                "The TPU serving strategy currently requires prompt "
                "remainders in the first canvas")
        if model.next_block_policy is not NextBlockPolicy.LAST_LOGIT_ANCHOR:
            raise ValueError(
                "The seed_and_mask serving strategy requires a last-logit "
                "next-block anchor")
        if config.runtime.temperature != 0.0:
            raise ValueError(
                "Stochastic diffusion sampling is not supported yet; set "
                "diffusion.temperature to 0")

    @property
    def block_size(self) -> int:
        return self.config.model.block_size

    @property
    def batch_size(self) -> int:
        configured = (self.runner.dp_size *
                      self.runner.scheduler_config.max_num_seqs)
        return max(1, min(configured, self.runner.max_num_reqs))

    def _validate_runner_capabilities(self) -> None:
        runner = self.runner
        if runner.dp_size != 1:
            raise ValueError(
                "Block diffusion currently supports data_parallel_size=1")
        if "dcp" in runner.mesh.shape and runner.mesh.shape["dcp"] > 1:
            raise ValueError("Block diffusion does not support DCP")
        if len(runner.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError(
                "Block diffusion requires exactly one KV cache group")
        if runner.kv_cache_config.has_mamba_layers:
            raise ValueError(
                "Block diffusion does not support hybrid or Mamba models")
        if runner.cache_config.enable_prefix_caching:
            raise ValueError("Block diffusion does not support prefix caching")
        if getattr(runner.vllm_config, "kv_transfer_config", None) is not None:
            raise ValueError(
                "Block diffusion does not support KV transfer connectors")

    def _validate_requests(self, req_ids: list[str]) -> None:
        input_batch = self.runner.input_batch
        for req_id in req_ids:
            req_index = input_batch.req_id_to_index[req_id]
            request = self.runner.requests[req_id]
            sampling_params = request.sampling_params
            if (req_id in input_batch.num_logprobs
                    or req_id in input_batch.num_prompt_logprobs):
                raise ValueError(
                    "Block diffusion does not support token logprobs yet")
            if req_id in input_batch.has_allowed_token_ids:
                raise ValueError(
                    "Block diffusion does not support allowed_token_ids")
            if req_index in input_batch.bad_words_token_ids:
                raise ValueError(
                    "Block diffusion does not support bad_words filtering")
            if input_batch.logit_bias[req_index]:
                raise ValueError(
                    "Block diffusion does not support per-request logit_bias")
            if req_id in input_batch.random_reqs:
                raise ValueError(
                    "Block diffusion currently requires greedy sampling; set "
                    "temperature=0")
            if (sampling_params.presence_penalty != 0.0
                    or sampling_params.frequency_penalty != 0.0
                    or sampling_params.repetition_penalty != 1.0):
                raise ValueError(
                    "Block diffusion does not support sampling penalties: "
                    f"presence_penalty={sampling_params.presence_penalty}, "
                    f"frequency_penalty={sampling_params.frequency_penalty}, "
                    f"repetition_penalty={sampling_params.repetition_penalty}")
            if sampling_params.min_tokens != 0:
                raise ValueError(
                    "Block diffusion does not support min_tokens yet")

            if sampling_params.max_tokens is None:
                raise ValueError(
                    "Block diffusion requires an explicit max_tokens limit")
            max_tokens = int(sampling_params.max_tokens)
            max_tokens = min(
                max_tokens,
                max(0, self.runner.max_model_len - request.num_prompt_tokens),
            )
            cache_end = required_cache_end(request.num_prompt_tokens,
                                           max_tokens, self.block_size)
            if cache_end > self.runner.max_model_len:
                raise ValueError(
                    f"Block diffusion request {req_id!r} requires cache "
                    f"through position {cache_end}, beyond "
                    f"max_model_len={self.runner.max_model_len}")

    def _build_batch(
        self,
        req_ids: list[str],
        block_starts: list[int],
        canvases: list[list[int]],
        masks: list[list[bool]] | None = None,
        *,
        padded_batch_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, tuple[
            AttentionMetadata, AttentionMetadata]]:
        runner = self.runner
        block_size = self.block_size
        sub_block_size = self.config.model.sub_block_size
        num_active = len(req_ids)
        capacity = self.batch_size
        batch_size = (select_diffusion_batch_size(num_active, capacity)
                      if padded_batch_size is None else padded_batch_size)
        if not max(1, num_active) <= batch_size <= capacity:
            raise ValueError(
                "Diffusion padded batch size must fit the active requests and "
                "configured capacity")
        if num_active > capacity:
            raise ValueError("Diffusion batch exceeds max_num_reqs")

        canvas = np.zeros((batch_size, block_size), dtype=np.int32)
        mask = np.zeros((batch_size, block_size), dtype=np.bool_)
        positions = np.zeros((batch_size, block_size), dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        active_rows = np.zeros((batch_size, ), dtype=np.bool_)
        query_start_loc = np.full((batch_size + 1, ),
                                  num_active * block_size,
                                  dtype=np.int32)
        query_start_loc[:num_active + 1] = np.arange(
            num_active + 1, dtype=np.int32) * block_size
        partial_query_start_loc = np.full((batch_size + 1, ),
                                          num_active * sub_block_size,
                                          dtype=np.int32)
        partial_query_start_loc[:num_active + 1] = np.arange(
            num_active + 1, dtype=np.int32) * sub_block_size

        source_block_tables = runner.input_batch.block_table[0].get_cpu_tensor(
        )
        block_tables = np.zeros((batch_size, source_block_tables.shape[1]),
                                dtype=source_block_tables.dtype)
        offsets = np.arange(block_size, dtype=np.int32)
        for row, (req_id, block_start,
                  row_canvas) in enumerate(zip(req_ids, block_starts,
                                               canvases)):
            if len(row_canvas) != block_size:
                raise ValueError("Every diffusion canvas must be one block")
            req_index = runner.input_batch.req_id_to_index[req_id]
            block_end = block_start + block_size
            if block_end > runner.max_model_len:
                raise ValueError(
                    f"Diffusion block for request {req_id!r} ends at "
                    f"{block_end}, beyond max_model_len={runner.max_model_len}"
                )
            cache_block_size = runner.block_size
            required_cache_blocks = (block_end + cache_block_size -
                                     1) // cache_block_size
            allocated_cache_blocks = int(runner.input_batch.block_table[0].
                                         num_blocks_per_row[req_index])
            if required_cache_blocks > allocated_cache_blocks:
                raise ValueError(
                    f"Diffusion block for request {req_id!r} needs "
                    f"{required_cache_blocks} KV blocks, but the scheduler "
                    f"allocated {allocated_cache_blocks}")
            canvas[row] = row_canvas
            if masks is not None:
                mask[row] = masks[row]
            positions[row] = block_start + offsets
            seq_lens[row] = block_start + block_size
            active_rows[row] = True
            block_tables[row] = source_block_tables[req_index]

        request_distribution = np.array([0, 0, num_active], dtype=np.int32)
        (canvas, mask, positions, seq_lens, active_rows, query_start_loc,
         partial_query_start_loc, request_distribution,
         block_tables) = device_array(self.runner.mesh, (
             canvas,
             mask,
             positions,
             seq_lens,
             active_rows,
             query_start_loc,
             partial_query_start_loc,
             request_distribution,
             block_tables.reshape(-1),
         ))
        full_metadata = AttentionMetadata(
            input_positions=positions.reshape(-1),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            padded_num_reqs=batch_size,
            attention_mask_spec=AttentionMaskSpec(
                AttentionMaskKind.BIDIRECTIONAL),
        )
        partial_metadata = AttentionMetadata(
            input_positions=positions[:, :sub_block_size].reshape(-1),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=partial_query_start_loc,
            request_distribution=request_distribution,
            padded_num_reqs=batch_size,
            attention_mask_spec=AttentionMaskSpec(
                AttentionMaskKind.BIDIRECTIONAL),
            replace_cached_kv=True,
            fp32_rpa_accumulator=self.config.runtime.fp32_partial_rpa,
        )
        return (canvas, mask, positions, active_rows, (full_metadata,
                                                       partial_metadata))

    def _model_forward(
        self,
        state_leaves: Any,
        canvas: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        forward_context: tuple[AttentionMetadata, AttentionMetadata],
    ) -> tuple[jax.Array, Any]:
        del active_rows
        attention_metadata, _ = forward_context
        kv_caches, hidden_states = self._run_model(
            state_leaves,
            kv_caches,
            canvas.reshape(-1),
            positions.reshape(-1),
            attention_metadata,
        )
        logits = self.runner.compute_logits_fn(state_leaves, hidden_states,
                                               None)
        return logits.reshape(canvas.shape[0], canvas.shape[1], -1), kv_caches

    def _run_model(
        self,
        state_leaves: Any,
        kv_caches: Any,
        input_ids: jax.Array,
        input_positions: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> tuple[Any, jax.Array]:
        runner = self.runner
        attention_metadata = replace(attention_metadata,
                                     input_positions=input_positions)
        kv_caches, hidden_states, _, _ = runner.model_fn_no_options(
            state_leaves,
            kv_caches,
            input_ids,
            attention_metadata,
            None,
            input_positions,
            tuple(runner.layer_name_to_kvcache_index.items()),
            None,
            None,
            runner.is_first_rank,
            runner.is_last_rank,
        )
        return kv_caches, hidden_states

    def _model_forward_full_subblock(
        self,
        state_leaves: Any,
        canvas: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        forward_context: tuple[AttentionMetadata, AttentionMetadata],
        start: jax.Array,
    ) -> tuple[jax.Array, Any]:
        del active_rows
        full_metadata, _ = forward_context
        kv_caches, hidden_states = self._run_model(
            state_leaves,
            kv_caches,
            canvas.reshape(-1),
            positions.reshape(-1),
            full_metadata,
        )
        selected = select_aligned_hidden_states(
            hidden_states,
            canvas.shape[0],
            canvas.shape[1],
            start,
            self.config.model.sub_block_size,
            self.config.model.logit_alignment,
            local_alignment=False,
        )
        logits = self.runner.compute_logits_fn(state_leaves, selected, None)
        return logits.reshape(canvas.shape[0],
                              self.config.model.sub_block_size, -1), kv_caches

    def _model_forward_partial_subblock(
        self,
        state_leaves: Any,
        canvas: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        forward_context: tuple[AttentionMetadata, AttentionMetadata],
        start: jax.Array,
    ) -> tuple[jax.Array, Any]:
        _, partial_metadata = forward_context
        sub_block_size = self.config.model.sub_block_size
        partial_canvas = jax.lax.dynamic_slice(
            canvas, (0, start), (canvas.shape[0], sub_block_size))
        partial_positions = jax.lax.dynamic_slice(
            positions, (0, start), (positions.shape[0], sub_block_size))
        partial_metadata = replace(partial_metadata,
                                   cache_update_active_rows=active_rows)
        kv_caches, hidden_states = self._run_model(
            state_leaves,
            kv_caches,
            partial_canvas.reshape(-1),
            partial_positions.reshape(-1),
            partial_metadata,
        )
        selected = select_aligned_hidden_states(
            hidden_states,
            canvas.shape[0],
            sub_block_size,
            jnp.array(0, dtype=jnp.int32),
            sub_block_size,
            self.config.model.logit_alignment,
            local_alignment=True,
        )
        logits = self.runner.compute_logits_fn(state_leaves, selected, None)
        return logits.reshape(canvas.shape[0], sub_block_size, -1), kv_caches

    def _model_forward_final(
        self,
        state_leaves: Any,
        canvas: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        forward_context: tuple[AttentionMetadata, AttentionMetadata],
    ) -> tuple[jax.Array, Any]:
        del active_rows
        full_metadata, _ = forward_context
        kv_caches, hidden_states = self._run_model(
            state_leaves,
            kv_caches,
            canvas.reshape(-1),
            positions.reshape(-1),
            full_metadata,
        )
        final_hidden = hidden_states.reshape(canvas.shape[0], canvas.shape[1],
                                             -1)[:, -1, :]
        logits = self.runner.compute_logits_fn(state_leaves, final_hidden,
                                               None)
        return logits, kv_caches

    def _build_prompt_batch(
        self,
        req_ids: list[str],
        prompt_tokens: list[list[int]],
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        if not req_ids or len(req_ids) != len(prompt_tokens):
            raise ValueError(
                "Prompt prefill requires matching non-empty request and token lists"
            )
        sequence_length = len(prompt_tokens[0])
        if sequence_length == 0 or sequence_length % self.block_size != 0:
            raise ValueError(
                "Block-causal prompt prefill requires complete diffusion blocks"
            )
        if any(len(tokens) != sequence_length for tokens in prompt_tokens):
            raise ValueError(
                "Requests in one prompt prefill batch must have equal lengths")

        runner = self.runner
        num_active = len(req_ids)
        batch_size = select_diffusion_batch_size(num_active, self.batch_size)
        total_tokens = num_active * sequence_length
        padded_tokens = next(
            (size
             for size in runner.num_tokens_paddings if size >= total_tokens),
            None)
        if padded_tokens is None:
            raise ValueError(
                f"Prompt prefill needs {total_tokens} tokens, beyond the "
                "configured TPU token buckets")

        input_ids = np.full((padded_tokens, ),
                            runner.pad_token_id,
                            dtype=np.int32)
        positions = np.zeros((padded_tokens, ), dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        query_start_loc = np.full((batch_size + 1, ),
                                  total_tokens,
                                  dtype=np.int32)
        query_start_loc[:num_active + 1] = np.arange(
            num_active + 1, dtype=np.int32) * sequence_length
        final_hidden_indices = np.full((batch_size, ),
                                       total_tokens - 1,
                                       dtype=np.int32)

        source_block_tables = runner.input_batch.block_table[0].get_cpu_tensor(
        )
        block_tables = np.zeros((batch_size, source_block_tables.shape[1]),
                                dtype=source_block_tables.dtype)
        token_positions = np.arange(sequence_length, dtype=np.int32)
        for row, (req_id, tokens) in enumerate(zip(req_ids, prompt_tokens)):
            req_index = runner.input_batch.req_id_to_index[req_id]
            required_cache_blocks = (sequence_length + runner.block_size -
                                     1) // runner.block_size
            allocated_cache_blocks = int(runner.input_batch.block_table[0].
                                         num_blocks_per_row[req_index])
            if required_cache_blocks > allocated_cache_blocks:
                raise ValueError(
                    f"Prompt prefill for request {req_id!r} needs "
                    f"{required_cache_blocks} KV blocks, but the scheduler "
                    f"allocated {allocated_cache_blocks}")
            start = row * sequence_length
            end = start + sequence_length
            input_ids[start:end] = tokens
            positions[start:end] = token_positions
            seq_lens[row] = sequence_length
            final_hidden_indices[row] = end - 1
            block_tables[row] = source_block_tables[req_index]

        request_distribution = np.array([0, num_active, num_active],
                                        dtype=np.int32)
        (input_ids, positions, seq_lens, query_start_loc, final_hidden_indices,
         request_distribution,
         block_tables) = device_array(runner.mesh, (
             input_ids,
             positions,
             seq_lens,
             query_start_loc,
             final_hidden_indices,
             request_distribution,
             block_tables.reshape(-1),
         ))
        metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            padded_num_reqs=batch_size,
            attention_mask_spec=AttentionMaskSpec(
                AttentionMaskKind.BLOCK_CAUSAL,
                block_size=self.block_size,
            ),
            rpa_static_query_len=sequence_length,
        )
        return input_ids, positions, final_hidden_indices, metadata

    def _forward_prompt_blocks(
        self,
        req_ids: list[str],
        prompt_tokens: list[list[int]],
    ) -> tuple[np.ndarray, int]:
        input_ids, positions, final_hidden_indices, metadata = \
            self._build_prompt_batch(req_ids, prompt_tokens)
        self.runner.kv_caches, hidden_states = self._run_model(
            self.runner.state_leaves,
            self.runner.kv_caches,
            input_ids,
            positions,
            metadata,
        )
        final_hidden = hidden_states[final_hidden_indices]
        logits = self.runner.compute_logits_fn(self.runner.state_leaves,
                                               final_hidden, None)
        logits = np.asarray(jax.device_get(logits[:len(req_ids)]))
        return logits, input_ids.shape[0]

    def _denoise_blocks(
        self,
        req_ids: list[str],
        block_starts: list[int],
        canvases: list[list[int]],
        masks: list[list[bool]],
        needs_final_forward: list[bool] | None = None,
        stop_on_eos: list[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        canvas, mask, positions, active_rows, metadata = self._build_batch(
            req_ids, block_starts, canvases, masks)
        batch_size = canvas.shape[0]
        if needs_final_forward is None:
            needs_final_forward = [True] * len(req_ids)
        if len(needs_final_forward) != len(req_ids):
            raise ValueError("needs_final_forward must match req_ids")
        if stop_on_eos is None:
            stop_on_eos = [False] * len(req_ids)
        if len(stop_on_eos) != len(req_ids):
            raise ValueError("stop_on_eos must match req_ids")
        output_candidate_positions = sum(
            sum(row_mask) for row_mask in masks) + sum(needs_final_forward)
        final_rows = np.zeros((batch_size, ), dtype=np.bool_)
        final_rows[:len(req_ids)] = needs_final_forward
        final_rows = device_array(self.runner.mesh, final_rows)
        stop_rows = np.zeros((batch_size, ), dtype=np.bool_)
        stop_rows[:len(req_ids)] = stop_on_eos
        stop_rows = device_array(self.runner.mesh, stop_rows)
        eos_token_ids = tuple(
            int(token_id)
            for token_id in np.atleast_1d(self.runner.eos_token_id))
        thresholds = jnp.full(
            (batch_size, ),
            self.config.runtime.confidence_threshold,
            dtype=jnp.float32,
        )
        temperatures = jnp.zeros((batch_size, ), dtype=jnp.float32)
        if self.config.runtime.use_dual_cache:
            output = denoise_block_dual_cache(
                self._full_subblock_forward_fn,
                self._partial_subblock_forward_fn,
                self._final_forward_fn,
                self._commit_fn,
                self.runner.state_leaves,
                canvas,
                mask,
                positions,
                self.runner.kv_caches,
                active_rows,
                thresholds,
                temperatures,
                metadata,
                mask_token_id=self.config.model.mask_token_id,
                logit_alignment=self.config.model.logit_alignment,
                next_block_policy=self.config.model.next_block_policy,
                sub_block_size=self.config.model.sub_block_size,
                max_denoise_steps=self.config.runtime.max_denoise_steps,
                needs_final_forward=final_rows,
                stop_on_eos_rows=stop_rows,
                eos_token_ids=eos_token_ids,
                commit_diagnostics_fn=self._commit_diagnostics_fn,
                trace_acceptance_steps=(
                    self.config.runtime.trace_acceptance_steps),
                q8_log_confidence_bias=(
                    self.config.runtime.q8_log_confidence_bias),
                force_q32_anchor_commit=(
                    self.config.runtime.force_q32_anchor_commit),
            )
        else:
            output = denoise_block(
                self._forward_fn,
                self._commit_fn,
                self.runner.state_leaves,
                canvas,
                mask,
                positions,
                self.runner.kv_caches,
                active_rows,
                thresholds,
                temperatures,
                metadata,
                mask_token_id=self.config.model.mask_token_id,
                logit_alignment=self.config.model.logit_alignment,
                next_block_policy=self.config.model.next_block_policy,
                sub_block_size=self.config.model.sub_block_size,
                max_denoise_steps=self.config.runtime.max_denoise_steps,
                needs_final_forward=final_rows,
                stop_on_eos_rows=stop_rows,
                eos_token_ids=eos_token_ids,
            )
        self.runner.kv_caches = output.kv_caches
        if self.config.runtime.use_dual_cache:
            device_outputs = (
                output.canvas[:len(req_ids)],
                output.next_anchor[:len(req_ids)],
                output.denoise_steps[:len(req_ids)],
                output.stopped_rows[:len(req_ids)],
                output.q32_forward_calls,
                output.q8_forward_calls,
                output.final_q32_forward_calls,
                output.forced_q32_anchor_commits,
            )
            if self.config.runtime.trace_acceptance_steps:
                assert output.acceptance_trace is not None
                device_outputs += (output.acceptance_trace, )
            host_outputs = jax.device_get(device_outputs)
            (canvas_host, anchors_host, denoise_steps_host, stopped_rows_host,
             q32_calls_host, q8_calls_host, final_q32_calls_host,
             forced_q32_anchor_commits_host) = (host_outputs[:8])
            q32_calls = int(q32_calls_host)
            q8_calls = int(q8_calls_host)
            final_q32_calls = int(final_q32_calls_host)
            static_transformer_positions = batch_size * (
                q32_calls * self.block_size +
                q8_calls * self.config.model.sub_block_size)
            static_lm_head_positions = batch_size * (
                (q32_calls - final_q32_calls + q8_calls) *
                self.config.model.sub_block_size + final_q32_calls)
            denoise_iterations = q32_calls + q8_calls - final_q32_calls
            useful_row_iterations = int(np.asarray(denoise_steps_host).sum())
            active_row_iterations = len(req_ids) * denoise_iterations
            static_row_iterations = batch_size * denoise_iterations
            self._last_denoise_trace = {
                "active_requests":
                len(req_ids),
                "static_batch_rows":
                batch_size,
                "q32_forward_calls":
                q32_calls,
                "q8_forward_calls":
                q8_calls,
                "final_q32_forward_calls":
                final_q32_calls,
                "force_q32_anchor_commit":
                self.config.runtime.force_q32_anchor_commit,
                "forced_q32_anchor_commits":
                int(forced_q32_anchor_commits_host),
                "output_candidate_positions":
                output_candidate_positions,
                "trim_final_block_candidates":
                self.config.runtime.trim_final_block_candidates,
                "confidence_threshold":
                self.config.runtime.confidence_threshold,
                "q8_log_confidence_bias":
                self.config.runtime.q8_log_confidence_bias,
                "q8_effective_confidence_threshold":
                (self.config.runtime.confidence_threshold *
                 np.exp(-self.config.runtime.q8_log_confidence_bias)
                 if 0.0 < self.config.runtime.confidence_threshold < 1.0 else
                 self.config.runtime.confidence_threshold),
                "static_transformer_positions":
                static_transformer_positions,
                "static_lm_head_positions":
                static_lm_head_positions,
                "useful_row_iterations":
                useful_row_iterations,
                "static_row_iterations":
                static_row_iterations,
                "padding_row_iterations":
                (static_row_iterations - active_row_iterations),
                "straggler_row_iterations":
                (active_row_iterations - useful_row_iterations),
                "wasted_row_iterations":
                (static_row_iterations - useful_row_iterations),
                "denoise_steps":
                np.asarray(denoise_steps_host).tolist(),
                "eos_stopped_rows":
                np.asarray(stopped_rows_host).tolist(),
            }
            logger.info("Fast-dLLM DualCache trace: %s",
                        self._last_denoise_trace)
            if self.config.runtime.trace_acceptance_steps:
                _log_dual_cache_acceptance_trace(host_outputs[8])
        else:
            canvas_host, anchors_host = jax.device_get((
                output.canvas[:len(req_ids)],
                output.next_anchor[:len(req_ids)],
            ))
            self._last_denoise_trace = None
        return np.asarray(canvas_host), np.asarray(anchors_host)

    def _process_prefill(self, req_ids: list[str]) -> dict[str, list[int]]:
        if not req_ids:
            return {}
        plans = {}
        for req_id in req_ids:
            request = self.runner.requests[req_id]
            scheduled = self._scheduler_output.num_scheduled_tokens[req_id]
            history_token_ids = [
                *request.prompt_token_ids,
                *request.output_token_ids,
            ]
            if request.num_computed_tokens != 0 or scheduled != len(
                    history_token_ids):
                raise ValueError(
                    "Block diffusion requires a full unchunked prefill or "
                    "recompute pass "
                    f"for request {req_id!r}")
            self._pending_outputs.pop(req_id, None)
            plans[req_id] = plan_seeded_prompt(
                history_token_ids,
                self.block_size,
                self.config.model.mask_token_id,
            )

        aligned_seeds: dict[str, int] = {}
        full_block_groups: dict[int, list[str]] = {}
        for req_id in req_ids:
            block_count = len(plans[req_id].full_blocks)
            if block_count:
                full_block_groups.setdefault(block_count, []).append(req_id)

        full_prompt_tokens = 0
        padded_transformer_positions = 0
        for block_count, group in full_block_groups.items():
            prompts = [[
                token for block in plans[req_id].full_blocks for token in block
            ] for req_id in group]
            logits, padded_tokens = self._forward_prompt_blocks(group, prompts)
            full_prompt_tokens += len(group) * block_count * self.block_size
            padded_transformer_positions += padded_tokens
            anchors = np.argmax(logits, axis=-1)
            for row, req_id in enumerate(group):
                plan = plans[req_id]
                if plan.remainder_size == 0:
                    aligned_seeds[req_id] = int(anchors[row])

        self._last_prefill_trace = {
            "active_requests":
            len(req_ids),
            "requests_with_full_blocks":
            sum(len(group) for group in full_block_groups.values()),
            "forward_calls":
            len(full_block_groups),
            "full_prompt_tokens":
            full_prompt_tokens,
            "padded_transformer_positions":
            padded_transformer_positions,
            "prompt_block_counts": {
                block_count: len(group)
                for block_count, group in full_block_groups.items()
            },
        }
        logger.info("Fast-dLLM block-causal prefill trace: %s",
                    self._last_prefill_trace)

        outputs = {req_id: [aligned_seeds[req_id]] for req_id in aligned_seeds}
        partial_group = [
            req_id for req_id in req_ids
            if plans[req_id].partial_canvas is not None
        ]
        if partial_group:
            canvases = [
                list(plans[req_id].partial_canvas) for req_id in partial_group
            ]
            masks = []
            needs_final_forward = []
            stop_on_eos = []
            for req_id in partial_group:
                plan = plans[req_id]
                assert plan.partial_mask is not None
                mask = list(plan.partial_mask)
                if self.config.runtime.trim_final_block_candidates:
                    mask = trim_generation_mask(
                        mask,
                        self._remaining_output_capacity(req_id),
                    )
                masks.append(mask)
                needs_final_forward.append(
                    needs_block_anchor(
                        mask,
                        self._remaining_output_capacity(req_id),
                    ))
                stop_on_eos.append(not self.runner.requests[req_id].
                                   sampling_params.ignore_eos)
            starts = [
                len(plans[req_id].full_blocks) * self.block_size
                for req_id in partial_group
            ]
            committed, anchors = self._denoise_blocks(
                partial_group,
                starts,
                canvases,
                masks,
                needs_final_forward,
                stop_on_eos,
            )
            for row, req_id in enumerate(partial_group):
                output, pending = start_partial_block_output(
                    committed[row].tolist(),
                    plans[req_id].remainder_size,
                    int(anchors[row]),
                )
                outputs[req_id] = output
                self._pending_outputs[req_id] = pending
        return outputs

    def _process_decode(self, req_ids: list[str]) -> dict[str, list[int]]:
        outputs: dict[str, list[int]] = {}
        denoise_group = []
        for req_id in req_ids:
            pending = self._pending_outputs.pop(req_id, None)
            if pending is not None:
                outputs[req_id] = flush_partial_block_output(pending)
            else:
                denoise_group.append(req_id)

        if not denoise_group:
            return outputs

        canvases = []
        masks = []
        needs_final_forward = []
        stop_on_eos = []
        starts = []
        for req_id in denoise_group:
            request = self.runner.requests[req_id]
            block_start = request.num_computed_tokens
            seed = request.get_token_id(block_start)
            canvases.append([seed] + [self.config.model.mask_token_id] *
                            (self.block_size - 1))
            mask = [False] + [True] * (self.block_size - 1)
            if self.config.runtime.trim_final_block_candidates:
                mask = trim_generation_mask(
                    mask,
                    self._remaining_output_capacity(req_id),
                )
            masks.append(mask)
            needs_final_forward.append(
                needs_block_anchor(
                    mask,
                    self._remaining_output_capacity(req_id),
                ))
            stop_on_eos.append(not request.sampling_params.ignore_eos)
            starts.append(block_start)

        committed, anchors = self._denoise_blocks(
            denoise_group,
            starts,
            canvases,
            masks,
            needs_final_forward,
            stop_on_eos,
        )
        for row, req_id in enumerate(denoise_group):
            outputs[req_id] = complete_seeded_decode_block(
                committed[row].tolist(), int(anchors[row]))
        return outputs

    def _remaining_output_capacity(self, req_id: str) -> int:
        request = self.runner.requests[req_id]
        assert request.sampling_params.max_tokens is not None
        max_tokens = int(request.sampling_params.max_tokens)
        output_remaining = max(0, max_tokens - len(request.output_token_ids))
        context_remaining = max(0,
                                self.runner.max_model_len - request.num_tokens)
        return min(output_remaining, context_remaining)

    def _truncate_output(self, req_id: str, tokens: list[int]) -> list[int]:
        request = self.runner.requests[req_id]
        remaining = self._remaining_output_capacity(req_id)
        tokens = tokens[:remaining]

        if not request.sampling_params.ignore_eos:
            eos_token_ids = np.atleast_1d(self.runner.eos_token_id)
            for index, token in enumerate(tokens):
                if token in eos_token_ids:
                    tokens = tokens[:index + 1]
                    self._pending_outputs.pop(req_id, None)
                    break
        if len(tokens) == remaining:
            self._pending_outputs.pop(req_id, None)
        return tokens

    def _append_outputs(self, outputs: dict[str,
                                            list[int]]) -> list[list[int]]:
        runner = self.runner
        sampled_token_ids = [[] for _ in range(runner.input_batch.num_reqs)]
        for req_id, tokens in outputs.items():
            tokens = self._truncate_output(req_id, tokens)
            req_index = runner.input_batch.req_id_to_index[req_id]
            sampled_token_ids[req_index] = tokens
            if not tokens:
                continue
            start = runner.input_batch.num_tokens_no_spec[req_index]
            end = start + len(tokens)
            if end > runner.max_model_len:
                raise ValueError(
                    f"Diffusion output for request {req_id!r} exceeds "
                    "max_model_len")
            runner.input_batch.token_ids_cpu[req_index, start:end] = tokens
            runner.input_batch.num_tokens_no_spec[req_index] = end
            runner.input_batch.num_tokens[req_index] = end
            runner.requests[req_id].output_token_ids.extend(tokens)
        return sampled_token_ids

    def execute(self, scheduler_output: "SchedulerOutput") -> None:
        self._validate_runner_capabilities()
        self._scheduler_output = scheduler_output

        scheduled_req_ids = [
            req_id for req_id in
            self.runner.input_batch.req_ids[:self.runner.input_batch.num_reqs]
            if req_id in scheduler_output.num_scheduled_tokens
        ]
        if getattr(scheduler_output, "has_structured_output_requests", False):
            raise ValueError(
                "Block diffusion does not support structured output")
        self._validate_requests(scheduled_req_ids)
        prefill_req_ids = [
            req_id for req_id in scheduled_req_ids
            if self.runner.requests[req_id].num_computed_tokens <
            self.runner.requests[req_id].num_prompt_tokens
        ]
        decode_req_ids = [
            req_id for req_id in scheduled_req_ids
            if req_id not in set(prefill_req_ids)
        ]

        with self.runner.maybe_forbid_compile, \
             set_forward_context(None, self.runner.vllm_config), \
             self.runner.maybe_get_kv_connector_output(
                 scheduler_output) as kv_connector_output:
            outputs = self._process_prefill(prefill_req_ids)
            outputs.update(self._process_decode(decode_req_ids))

        sampled_token_ids = self._append_outputs(outputs)
        num_reqs = self.runner.input_batch.num_reqs
        self.runner._generation_strategy_output = ModelRunnerOutput(
            req_ids=self.runner.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.runner.input_batch.req_id_to_index.copy(),
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

    def on_scheduler_update(self, finished_req_ids: set[str]) -> None:
        for req_id in finished_req_ids:
            self._pending_outputs.pop(req_id, None)

    def precompile(self) -> None:
        self._validate_runner_capabilities()
        eos_token_ids = tuple(
            int(token_id)
            for token_id in np.atleast_1d(self.runner.eos_token_id))
        for batch_size in diffusion_batch_sizes(self.batch_size):
            canvas, mask, positions, active_rows, metadata = self._build_batch(
                [], [], [], [], padded_batch_size=batch_size)
            logits, self.runner.kv_caches = self._final_forward_fn(
                self.runner.state_leaves,
                canvas,
                positions,
                self.runner.kv_caches,
                active_rows,
                metadata,
            )
            thresholds = jnp.full(
                (batch_size, ),
                self.config.runtime.confidence_threshold,
                dtype=jnp.float32,
            )
            if self.config.runtime.use_dual_cache:
                output = denoise_block_dual_cache(
                    self._full_subblock_forward_fn,
                    self._partial_subblock_forward_fn,
                    self._final_forward_fn,
                    self._commit_fn,
                    self.runner.state_leaves,
                    canvas,
                    mask,
                    positions,
                    self.runner.kv_caches,
                    active_rows,
                    thresholds,
                    jnp.zeros_like(thresholds),
                    metadata,
                    mask_token_id=self.config.model.mask_token_id,
                    logit_alignment=self.config.model.logit_alignment,
                    next_block_policy=self.config.model.next_block_policy,
                    sub_block_size=self.config.model.sub_block_size,
                    max_denoise_steps=self.config.runtime.max_denoise_steps,
                    stop_on_eos_rows=jnp.zeros_like(active_rows),
                    eos_token_ids=eos_token_ids,
                    commit_diagnostics_fn=self._commit_diagnostics_fn,
                    trace_acceptance_steps=(
                        self.config.runtime.trace_acceptance_steps),
                    q8_log_confidence_bias=(
                        self.config.runtime.q8_log_confidence_bias),
                    force_q32_anchor_commit=(
                        self.config.runtime.force_q32_anchor_commit),
                )
            else:
                output = denoise_block(
                    self._forward_fn,
                    self._commit_fn,
                    self.runner.state_leaves,
                    canvas,
                    mask,
                    positions,
                    self.runner.kv_caches,
                    active_rows,
                    thresholds,
                    jnp.zeros_like(thresholds),
                    metadata,
                    mask_token_id=self.config.model.mask_token_id,
                    logit_alignment=self.config.model.logit_alignment,
                    next_block_policy=self.config.model.next_block_policy,
                    sub_block_size=self.config.model.sub_block_size,
                    max_denoise_steps=self.config.runtime.max_denoise_steps,
                    stop_on_eos_rows=jnp.zeros_like(active_rows),
                    eos_token_ids=eos_token_ids,
                )
            self.runner.kv_caches = output.kv_caches
            jax.block_until_ready((logits, output.canvas))
