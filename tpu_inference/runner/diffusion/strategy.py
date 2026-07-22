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
from tpu_inference.runner.diffusion.algorithm import get_commit_algorithm
from tpu_inference.runner.diffusion.batch import (PendingBlockOutput,
                                                  complete_seeded_decode_block,
                                                  flush_partial_block_output,
                                                  plan_seeded_prompt,
                                                  required_cache_end,
                                                  start_partial_block_output)
from tpu_inference.runner.diffusion.config import (CanvasPolicy,
                                                   DiffusionConfig,
                                                   NextBlockPolicy,
                                                   PromptRemainderPolicy)
from tpu_inference.runner.diffusion.program import denoise_block
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner


class BlockDiffusionStrategy:

    def __init__(self, runner: "TPUModelRunner",
                 config: DiffusionConfig) -> None:
        self.runner = runner
        self.config = config
        self._pending_outputs: dict[str, PendingBlockOutput] = {}
        self._forward_fn = self._model_forward
        self._commit_fn = get_commit_algorithm(config.runtime.algorithm)

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
                    "Block diffusion does not support sampling penalties")
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        runner = self.runner
        batch_size = runner.max_num_reqs
        block_size = self.block_size
        num_active = len(req_ids)
        if num_active > batch_size:
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

        source_block_tables = runner.input_batch.block_table[0].get_cpu_tensor(
        )
        block_tables = np.zeros_like(source_block_tables)
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
         request_distribution,
         block_tables) = device_array(self.runner.mesh, (
             canvas,
             mask,
             positions,
             seq_lens,
             active_rows,
             query_start_loc,
             request_distribution,
             block_tables.reshape(-1),
         ))
        metadata = AttentionMetadata(
            input_positions=positions.reshape(-1),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            padded_num_reqs=batch_size,
            attention_mask_spec=AttentionMaskSpec(
                AttentionMaskKind.BIDIRECTIONAL),
        )
        return canvas, mask, positions, active_rows, metadata

    def _model_forward(
        self,
        state_leaves: Any,
        canvas: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> tuple[jax.Array, Any]:
        del active_rows
        runner = self.runner
        flat_canvas = canvas.reshape(-1)
        flat_positions = positions.reshape(-1)
        attention_metadata = replace(attention_metadata,
                                     input_positions=flat_positions)
        kv_caches, hidden_states, _, _ = runner.model_fn_no_options(
            state_leaves,
            kv_caches,
            flat_canvas,
            attention_metadata,
            None,
            flat_positions,
            tuple(runner.layer_name_to_kvcache_index.items()),
            None,
            None,
            runner.is_first_rank,
            runner.is_last_rank,
        )
        logits = runner.compute_logits_fn(state_leaves, hidden_states, None)
        return logits.reshape(canvas.shape[0], canvas.shape[1], -1), kv_caches

    def _forward_blocks(
        self,
        req_ids: list[str],
        block_starts: list[int],
        canvases: list[list[int]],
    ) -> np.ndarray:
        canvas, _, positions, active_rows, metadata = self._build_batch(
            req_ids, block_starts, canvases)
        logits, self.runner.kv_caches = self._forward_fn(
            self.runner.state_leaves,
            canvas,
            positions,
            self.runner.kv_caches,
            active_rows,
            metadata,
        )
        return np.asarray(jax.device_get(logits[:len(req_ids)]))

    def _denoise_blocks(
        self,
        req_ids: list[str],
        block_starts: list[int],
        canvases: list[list[int]],
        masks: list[list[bool]],
    ) -> tuple[np.ndarray, np.ndarray]:
        canvas, mask, positions, active_rows, metadata = self._build_batch(
            req_ids, block_starts, canvases, masks)
        batch_size = self.runner.max_num_reqs
        thresholds = jnp.full(
            (batch_size, ),
            self.config.runtime.confidence_threshold,
            dtype=jnp.float32,
        )
        temperatures = jnp.zeros((batch_size, ), dtype=jnp.float32)
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
            logit_alignment=self.config.model.logit_alignment,
            next_block_policy=self.config.model.next_block_policy,
            sub_block_size=self.config.model.sub_block_size,
            max_denoise_steps=self.config.runtime.max_denoise_steps,
        )
        self.runner.kv_caches = output.kv_caches
        canvas_host, anchors_host = jax.device_get(
            (output.canvas[:len(req_ids)], output.next_anchor[:len(req_ids)]))
        return np.asarray(canvas_host), np.asarray(anchors_host)

    def _process_prefill(self, req_ids: list[str]) -> dict[str, list[int]]:
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
        max_full_blocks = max(
            (len(plan.full_blocks) for plan in plans.values()), default=0)
        for block_index in range(max_full_blocks):
            group = [
                req_id for req_id in req_ids
                if block_index < len(plans[req_id].full_blocks)
            ]
            canvases = [
                list(plans[req_id].full_blocks[block_index])
                for req_id in group
            ]
            logits = self._forward_blocks(
                group,
                [block_index * self.block_size] * len(group),
                canvases,
            )
            anchors = np.argmax(logits[:, -1, :], axis=-1)
            for row, req_id in enumerate(group):
                plan = plans[req_id]
                if (plan.remainder_size == 0
                        and block_index == len(plan.full_blocks) - 1):
                    aligned_seeds[req_id] = int(anchors[row])

        outputs = {req_id: [aligned_seeds[req_id]] for req_id in aligned_seeds}
        partial_group = [
            req_id for req_id in req_ids
            if plans[req_id].partial_canvas is not None
        ]
        if partial_group:
            canvases = [
                list(plans[req_id].partial_canvas) for req_id in partial_group
            ]
            masks = [
                list(plans[req_id].partial_mask) for req_id in partial_group
            ]
            starts = [
                len(plans[req_id].full_blocks) * self.block_size
                for req_id in partial_group
            ]
            committed, anchors = self._denoise_blocks(partial_group, starts,
                                                      canvases, masks)
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
        starts = []
        for req_id in denoise_group:
            request = self.runner.requests[req_id]
            block_start = request.num_computed_tokens
            seed = request.get_token_id(block_start)
            canvases.append([seed] + [self.config.model.mask_token_id] *
                            (self.block_size - 1))
            masks.append([False] + [True] * (self.block_size - 1))
            starts.append(block_start)

        committed, anchors = self._denoise_blocks(denoise_group, starts,
                                                  canvases, masks)
        for row, req_id in enumerate(denoise_group):
            outputs[req_id] = complete_seeded_decode_block(
                committed[row].tolist(), int(anchors[row]))
        return outputs

    def _truncate_output(self, req_id: str, tokens: list[int]) -> list[int]:
        request = self.runner.requests[req_id]
        assert request.sampling_params.max_tokens is not None
        max_tokens = int(request.sampling_params.max_tokens)
        output_remaining = max(0, max_tokens - len(request.output_token_ids))
        context_remaining = max(0,
                                self.runner.max_model_len - request.num_tokens)
        remaining = min(output_remaining, context_remaining)
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
        canvas, mask, positions, active_rows, metadata = self._build_batch([],
                                                                           [],
                                                                           [],
                                                                           [])
        logits, self.runner.kv_caches = self._forward_fn(
            self.runner.state_leaves,
            canvas,
            positions,
            self.runner.kv_caches,
            active_rows,
            metadata,
        )
        thresholds = jnp.full(
            (self.runner.max_num_reqs, ),
            self.config.runtime.confidence_threshold,
            dtype=jnp.float32,
        )
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
            logit_alignment=self.config.model.logit_alignment,
            next_block_policy=self.config.model.next_block_policy,
            sub_block_size=self.config.model.sub_block_size,
            max_denoise_steps=self.config.runtime.max_denoise_steps,
        )
        self.runner.kv_caches = output.kv_caches
        jax.block_until_ready((logits, output.canvas))
