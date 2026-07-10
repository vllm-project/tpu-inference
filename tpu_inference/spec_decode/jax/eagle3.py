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
"""Implements the Eagle3 proposer for speculative decoding on JAX/TPU."""
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig

from tpu_inference import envs
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import (
    get_model, resolve_model_architecture)
from tpu_inference.utils import device_array

logger = init_logger(__name__)


class Eagle3Proposer:
    """A proposer for speculative decoding using the Eagle3 method.

    This class is responsible for loading the draft model and generating draft
    tokens based on the target model's outputs.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            runner: Any,  # TPUModelRunner
    ):
        """Initializes the Eagle3Proposer.

        Args:
            vllm_config: The vLLM configuration.
            runner: The TPUModelRunner instance.
        """
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.mesh = runner.mesh
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens
        self.token_arange = jnp.arange(self.max_num_tokens)
        self.constant_draft_positions = self.speculative_config.use_gemma4_mtp(
        )

    def load_model(self, target_model: Any) -> None:
        """Loads the draft model."""
        shared_names = [
            "vllm_model.language_model.model.embed_tokens.weight",
            "vllm_model.model.embed_tokens.weight"
        ]
        shared_params = {
            k: v
            for k, v in self.runner.state.items() if k in shared_names
        }

        model = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
            is_draft_model=True,
            shared_params=shared_params,
        )

        self.model_fn = model.model_fn
        self.compute_logits_fn = model.compute_logits_fn
        self.pooler_fn = model.pooler_fn
        self.combine_hidden_states_fn = model.combine_hidden_states_fn
        self.state = model.state
        self.state_leaves = model.state_leaves
        self.model = model.model

        draft_model_impl = envs.DRAFT_MODEL_IMPL_TYPE
        target_model_impl = envs.MODEL_IMPL_TYPE
        if draft_model_impl == 'auto':
            draft_model_impl = resolve_model_architecture(
                self.vllm_config, True)
        if target_model_impl == 'auto':
            target_model_impl = resolve_model_architecture(
                self.vllm_config, False)
        if draft_model_impl != target_model_impl:
            raise ValueError(
                "The implementation of the draft model must be the same as the target model."
            )
        # TODO(ranlihao): Handles the case where the draft model and target model have different implementations. This may require converting the parameters of the target model to match the draft model's format.
        # Reuse the target model's embedding if the draft model doesn't have its own or if they are identical, to save memory.
        # TODO(ranlihao): Unify the weight loading process for jax and torchax path.
        if draft_model_impl == "flax_nnx":
            from tpu_inference.models.jax.utils.weight_utils import get_param

            # 1. Resolve draft embed param in self.state
            draft_embed_param = None
            for path in [
                    "model.embed_tokens.weight", "model.embed.embedding",
                    "model.embed_tokens.embedding"
            ]:
                try:
                    draft_embed_param = get_param(self.state, path)
                    logger.info(f"Resolved draft model embedding path: {path}")
                    break
                except ValueError:
                    continue

            # 2. Resolve target embed param in target_model (which is already the target state)
            target_embed_param = None
            for path in [
                    "model.language_model.embed_tokens.weight",
                    "language_model.embed_tokens.weight",
                    "model.embed_tokens.weight", "model.embed.embedding",
                    "model.embed_tokens.embedding"
            ]:
                try:
                    target_embed_param = get_param(target_model, path)
                    logger.info(
                        f"Resolved target model embedding path: {path}")
                    break
                except ValueError:
                    continue

            # 3. Check and share embedding values directly in the State trees
            if draft_embed_param is not None and target_embed_param is not None:
                if not jnp.any(draft_embed_param.value):
                    logger.info(
                        "Draft model does not have embedding. Setting draft model's embed_tokens to target model's embed"
                    )
                    draft_embed_param.value = target_embed_param.value
                elif self.speculative_config.use_gemma4_mtp():
                    logger.info(
                        "Setting draft model's embed_tokens to target model's embed unconditionally (Gemma4-MTP/KV-sharing layout)"
                    )
                    draft_embed_param.value = target_embed_param.value
                elif jnp.array_equal(draft_embed_param.value,
                                     target_embed_param.value):
                    logger.info(
                        "Draft model's embed_tokens is identical to target model's embed. Sharing the embedding."
                    )
                    draft_embed_param.value = target_embed_param.value
                else:
                    logger.info("Draft model has its own embed_tokens.")
            else:
                logger.warning(
                    "Failed to locate draft or target embedding parameter in State objects."
                )

        # The embed_tokens assignment above may have mutated `self.state`;
        # re-derive `state_leaves` so the dispatch-side view matches.
        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state

    def _prepare_input_ids(
            self, query_start_loc: jax.Array, target_token_ids: jax.Array,
            next_token_ids: jax.Array,
            num_reqs: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled helper for preparing the input IDs for the draft model."""

        def _sharded_prepare_input_ids(query_start_loc, target_token_ids,
                                       next_token_ids, num_reqs):
            last_token_indices = query_start_loc[1:] - 1
            # Shift the input ids by one token.
            rolled_input_ids = jnp.roll(target_token_ids, -1, axis=0)

            # To make the update JIT-compatible with a dynamic `num_reqs`, we
            # perform a scatter update of a static size, using a mask to handle
            # the dynamic part.
            max_num_reqs = last_token_indices.shape[0]
            mask = jnp.arange(max_num_reqs) < num_reqs
            last_token_indices = jnp.where(mask, last_token_indices,
                                           last_token_indices[num_reqs - 1])

            # Mask out the update for the padded requests (where mask is False).
            values_to_set = jnp.where(mask, next_token_ids,
                                      next_token_ids[num_reqs - 1])

            input_ids = rolled_input_ids.at[last_token_indices].set(
                values_to_set)

            return input_ids, last_token_indices

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        return jax.shard_map(
            _sharded_prepare_input_ids,
            mesh=self.mesh,
            in_specs=(data_spec, data_spec, data_spec, data_spec),
            out_specs=(data_spec, data_spec),
        )(query_start_loc, target_token_ids, next_token_ids, num_reqs)

    def _update_inputs_for_loop_speculation(
        self, positions: jax.Array, seq_lens: jax.Array,
        block_tables: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """JIT-compiled helper for preparing inputs in the loop of prediction."""

        def _sharded_update_inputs_for_loop_speculation(
                positions, seq_lens, block_tables):
            positions += 1
            exceeds_max_model_len = positions >= self.runner.max_model_len
            if exceeds_max_model_len.ndim == 2:
                exceeds_max_model_len_reduced = jnp.any(exceeds_max_model_len,
                                                        axis=0)
            else:
                exceeds_max_model_len_reduced = exceeds_max_model_len

            clamped_positions = jnp.where(exceeds_max_model_len, 0, positions)

            new_seq_lens = seq_lens + 1
            new_seq_lens = jnp.minimum(new_seq_lens, self.runner.max_model_len)
            new_seq_lens = jnp.where(exceeds_max_model_len_reduced, 1,
                                     new_seq_lens)

            num_reqs = seq_lens.shape[0]
            query_start_loc = jnp.arange(num_reqs + 1)

            # Compute the slot mapping.
            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            max_num_blocks_per_req = block_tables.shape[0] // num_reqs
            expanded_exceeds_mask = jnp.repeat(exceeds_max_model_len_reduced,
                                               max_num_blocks_per_req)
            new_block_tables = jnp.where(expanded_exceeds_mask, -1,
                                         block_tables)

            return (positions, clamped_positions, new_seq_lens,
                    query_start_loc, new_block_tables)

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        positions_spec = (PartitionSpec(None, ShardingAxisName.ATTN_DATA)
                          if positions.ndim == 2 else data_spec)
        (positions, clamped_positions, new_seq_lens, query_start_loc,
         new_block_tables) = jax.shard_map(
             _sharded_update_inputs_for_loop_speculation,
             mesh=self.mesh,
             in_specs=(positions_spec, data_spec, data_spec),
             out_specs=(positions_spec, positions_spec, data_spec, data_spec,
                        data_spec),
         )(positions, seq_lens, block_tables)
        return positions, clamped_positions, new_seq_lens, query_start_loc, new_block_tables

    def _get_loop_query_start_loc(self, positions: jax.Array) -> jax.Array:
        """JIT-compiled helper for generating query_start_loc inside speculation loop."""

        def _sharded_get(positions):
            num_reqs = positions.shape[0]
            return jnp.arange(num_reqs + 1)

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        return jax.shard_map(
            _sharded_get,
            mesh=self.mesh,
            in_specs=(data_spec, ),
            out_specs=data_spec,
        )(positions)

    def _stack_draft_token_ids(
            self, draft_token_ids_list: list[jax.Array]) -> jnp.ndarray:
        """JIT-compiled helper for stacking draft token IDs."""
        return jnp.stack(draft_token_ids_list, axis=1)

    def _prepare_hidden_states_and_input_ids(
        self,
        state_leaves: Any,
        aux_hidden_states: tuple[jax.Array, ...],
        query_start_loc: jax.Array,
        target_token_ids: jax.Array,
        next_token_ids: jax.Array,
        num_reqs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.method == "mtp":
            target_hidden_states = aux_hidden_states[0]
        else:
            target_hidden_states = jnp.concatenate(aux_hidden_states, axis=-1)
            target_hidden_states = self.combine_hidden_states_fn(
                state_leaves, target_hidden_states)

        input_ids, last_token_indices = self._prepare_input_ids(
            query_start_loc, target_token_ids, next_token_ids, num_reqs)
        # NOTE(pooyam): For now, we don't support multimodal.

        return target_hidden_states, input_ids, last_token_indices

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare drafter inputs based on target forward outputs.

        Mirrors the GPU reference logic but adapted to TPU/JAX types:
        - When no rejection happened, select the first N scheduled tokens.
        - When rejections happened, trim the per-request tail tokens and
          update attention metadata accordingly.
        - Build the EAGLE3 hidden input by concatenating auxiliary hidden
          states along the last dimension.

        Returns updated AttentionMetadata (positions, query_start_loc, seq_lens)
        and the selected `target_token_ids` and `target_hidden_states`.
        """
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0, (
            f"{self.method} requires auxiliary hidden states from the target model."
        )

        # The last KV cache group is for the draft model.
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
        block_tables = device_array(self.mesh,
                                    block_tables,
                                    sharding=PartitionSpec(
                                        ShardingAxisName.ATTN_DATA))
        num_reqs = num_reqs_dp
        return self._prepare_inputs(
            state_leaves=self.state_leaves,
            num_reqs=num_reqs,
            block_tables=block_tables,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            aux_hidden_states=aux_hidden_states,
            last_sampled_token_id=last_sampled_token_id,
            next_prompt_token_id=next_prompt_token_id,
            is_in_prefill=is_in_prefill,
            num_rejected_tokens=num_rejected_tokens)

    @jax.jit(static_argnums=(0, ))
    def _prepare_inputs(
        self,
        state_leaves: Any,
        num_reqs: jax.Array,
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare drafter inputs based on target forward outputs.

        Mirrors the GPU reference logic but adapted to TPU/JAX types:
        - When no rejection happened, select the first N scheduled tokens.
        - When rejections happened, trim the per-request tail tokens and
          update attention metadata accordingly.
        - Build the EAGLE3 hidden input by concatenating auxiliary hidden
          states along the last dimension.

        Returns updated AttentionMetadata (positions, query_start_loc, seq_lens)
        and the selected `target_token_ids` and `target_hidden_states`.
        """
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0, (
            "EAGLE3 requires auxiliary hidden states from the target model.")

        def _compute_token_indices(is_in_prefill, next_prompt_token_id,
                                   last_sampled_token_id, query_start_loc,
                                   seq_lens, num_reqs, num_rejected_tokens,
                                   input_ids):
            next_token_ids = jnp.where(is_in_prefill, next_prompt_token_id,
                                       last_sampled_token_id)

            # Rejection-aware path: compute new per-request lengths and token
            # indices.
            # query_len_per_req = [q1, q2, ...]
            query_len_per_req = (query_start_loc[1:] - query_start_loc[:-1])

            # query_start_loc and consequentaly query_len_per_req are padded
            # For padded requests, the query length should be 0.
            query_len_per_req = jnp.where(
                jnp.arange(query_len_per_req.shape[0]) < num_reqs,
                query_len_per_req, 0)
            num_rejected_tokens = jnp.where(
                jnp.arange(num_rejected_tokens.shape[0]) < num_reqs,
                num_rejected_tokens, 0)
            # num_tokens_per_req = [q1 - n1, q2 - n2, ...]
            num_tokens_per_req = (query_len_per_req - num_rejected_tokens)

            # new_query_start_loc = [0, q1-n1, q1+q2-n1-n2, ...]
            new_query_start_loc = jnp.cumsum(num_tokens_per_req)
            new_query_start_loc = jnp.pad(new_query_start_loc, (1, 0),
                                          constant_values=0)
            total_num_tokens = input_ids.shape[0]

            # Expand request starts: [0, 0, q1-n1, ...,]
            expanded_new_query_start_loc = jnp.repeat(
                new_query_start_loc[:-1],
                num_tokens_per_req,
                total_repeat_length=total_num_tokens)
            # Offsets within each request window: [0,1,2, 0,1,2,3, ...]
            token_offsets = jnp.arange(total_num_tokens, dtype=np.int32)
            token_offsets -= expanded_new_query_start_loc

            # Map into old flat indices by adding original request starts.
            old_query_start_loc_expanded = jnp.repeat(
                query_start_loc[:-1],
                num_tokens_per_req,
                total_repeat_length=total_num_tokens)

            token_indices = token_offsets + old_query_start_loc_expanded

            # Update seq_lens for active requests only: new_seq_lens = s - n.
            new_seq_lens = seq_lens - num_rejected_tokens

            return (next_token_ids, token_indices, new_query_start_loc,
                    new_seq_lens)

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        (next_token_ids, token_indices, new_query_start_loc,
         new_seq_lens) = jax.shard_map(
             _compute_token_indices,
             mesh=self.mesh,
             in_specs=(data_spec, ) * 8,
             out_specs=(data_spec, ) * 4,
         )(is_in_prefill, next_prompt_token_id, last_sampled_token_id,
           attn_metadata.query_start_loc, attn_metadata.seq_lens, num_reqs,
           num_rejected_tokens, input_ids)

        attn_metadata = replace(attn_metadata, block_tables=block_tables)
        return self._filter_token_and_prepare_initial_inputs(
            state_leaves, token_indices, new_query_start_loc, new_seq_lens,
            input_ids, aux_hidden_states, attn_metadata, next_token_ids,
            num_reqs)

    def _filter_token_and_prepare_initial_inputs(
        self,
        state_leaves: Any,
        token_indices: jax.Array,
        query_start_loc: jax.Array,
        seq_lens: jax.Array,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        attn_metadata: AttentionMetadata,
        next_token_ids: jax.Array,
        num_reqs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:

        # Select tokens and hidden states.
        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        positions_spec = (PartitionSpec(None, ShardingAxisName.ATTN_DATA)
                          if attn_metadata.input_positions.ndim == 2 else
                          data_spec)

        def _sharded_select_target_tokens_and_hidden_states(
                input_ids, token_indices, input_positions, aux_hidden_states):
            target_token_ids = input_ids[token_indices]
            # Update positions to match the selected tokens.
            if input_positions.ndim == 2:
                input_positions = input_positions[:, token_indices]
            else:
                input_positions = input_positions[token_indices]
            aux_states_processed = [
                h[token_indices] for h in aux_hidden_states
            ]
            return target_token_ids, input_positions, aux_states_processed

        target_token_ids, input_positions, aux_states_processed = jax.shard_map(
            _sharded_select_target_tokens_and_hidden_states,
            mesh=self.mesh,
            in_specs=(data_spec, data_spec, positions_spec, data_spec),
            out_specs=(data_spec, positions_spec, data_spec),
        )(input_ids, token_indices, attn_metadata.input_positions,
          aux_hidden_states)

        attn_metadata = replace(
            attn_metadata,
            input_positions=input_positions,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )

        target_hidden_states, input_ids, last_token_indices = self._prepare_hidden_states_and_input_ids(
            state_leaves, aux_states_processed, query_start_loc,
            target_token_ids, next_token_ids, num_reqs)

        return target_hidden_states, input_ids, last_token_indices, attn_metadata

    def _select_draft_token_ids(
        self,
        state_leaves: Any,
        hidden_states: jax.Array,
        last_token_indices: jax.Array,
    ) -> jax.Array:

        def _select(hidden_states, last_token_indices):
            return hidden_states[last_token_indices]

        sample_hidden_states = jax.shard_map(
            _select,
            mesh=self.mesh,
            in_specs=(PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA)),
            out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA),
        )(hidden_states, last_token_indices)
        return self._get_draft_token_ids(state_leaves, sample_hidden_states)

    def _get_draft_token_ids(self, state_leaves: Any,
                             hidden_states: jax.Array) -> jax.Array:
        lora_metadata = None
        logits = self.compute_logits_fn(state_leaves, hidden_states,
                                        lora_metadata)
        draft_token_ids = jnp.argmax(logits, axis=-1)
        return lax.with_sharding_constraint(
            draft_token_ids,
            NamedSharding(self.mesh,
                          PartitionSpec(ShardingAxisName.ATTN_DATA)))

    def _select_inputs_for_loop_speculation(
            self, state_leaves: Any, positions: jax.Array, residual: jax.Array,
            hidden_states: jax.Array,
            last_token_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
        draft_token_ids = self._select_draft_token_ids(state_leaves,
                                                       hidden_states,
                                                       last_token_indices)

        def _select(positions, residual, hidden_states, last_token_indices):
            if self.method == "mtp":
                # We need a separate branch for MTP because:
                # 1. MTP uses final target output hidden states directly as inputs for the next step,
                #    whereas Eagle uses intermediate residuals.
                # 2. M-RoPE positions are 2D (3, total_tokens), requiring specific dim-1 slicing
                #    which _select_inputs_for_loop_speculation does not support.
                if positions.ndim == 2:
                    positions = positions[:, last_token_indices]
                else:
                    positions = positions[last_token_indices]
                residual = residual[last_token_indices]
                return positions, residual

            positions = positions[last_token_indices]
            residual = residual[last_token_indices]
            return positions, residual

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        # M-RoPE positions are 2D (3, total_tokens); shard the token axis.
        positions_spec = (PartitionSpec(None, ShardingAxisName.ATTN_DATA)
                          if positions.ndim == 2 else data_spec)
        positions, residual = jax.shard_map(
            _select,
            mesh=self.mesh,
            in_specs=(positions_spec, data_spec, data_spec, data_spec),
            out_specs=(positions_spec, data_spec),
        )(positions, residual, hidden_states, last_token_indices)
        return positions, residual, draft_token_ids

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        return self._propose(
            state_leaves=self.state_leaves,
            kv_caches=kv_caches,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            last_token_indices=last_token_indices,
            target_hidden_states=target_hidden_states,
            num_speculative_tokens=self.num_speculative_tokens,
            layer_name_to_kvcache_index=tuple(
                self.runner.layer_name_to_kvcache_index.items()))

    @jax.jit(
        donate_argnames=("kv_caches", ),
        out_shardings=(
            None,  # kv_caches - keep original sharding
            None,  # draft_token_ids
        ),
        compiler_options={
            "xla_tpu_all_gather_collective_matmul_mode":
            "post_spmd_conservative",
            "xla_tpu_reduce_scatter_collective_matmul_mode":
            "post_spmd_conservative"
        },
        static_argnums=(
            0,
            7,
            8,
        ),
    )
    def _propose(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices,
        target_hidden_states,
        num_speculative_tokens: int,
        layer_name_to_kvcache_index: tuple,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Proposes draft tokens using the draft model.
        Returns:
            A tuple containing the updated KV caches and a tensor of proposed
            draft token IDs.
        """

        kv_caches, hidden_states, residual, _ = self.model_fn(
            state_leaves,
            kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
            layer_name_to_kvcache_index,
            spec_step_idx=0,
        )

        if num_speculative_tokens == 1:
            return kv_caches, self._select_draft_token_ids(
                state_leaves, hidden_states, last_token_indices)

        positions, hidden_states, draft_token_ids = self._select_inputs_for_loop_speculation(
            state_leaves, attn_metadata.input_positions, residual[0],
            hidden_states, last_token_indices)
        draft_token_ids_list = [draft_token_ids]

        for i in range(num_speculative_tokens - 1):
            input_ids_loop = draft_token_ids_list[-1]

            if self.constant_draft_positions:
                # For Gemma4-MTP sharing verifier caches: positions, sequence lengths, and block tables remain constant.
                clamped_positions = positions
                new_seq_lens = attn_metadata.seq_lens
                query_start_loc = self._get_loop_query_start_loc(positions)
                new_block_tables = attn_metadata.block_tables
            else:
                # Eagle3: advance positions sequentially
                positions, clamped_positions, new_seq_lens, query_start_loc, new_block_tables = self._update_inputs_for_loop_speculation(
                    positions, attn_metadata.seq_lens,
                    attn_metadata.block_tables)

            attn_metadata = replace(
                attn_metadata,
                input_positions=clamped_positions,
                seq_lens=new_seq_lens,
                query_start_loc=query_start_loc,
                block_tables=new_block_tables,
            )
            kv_caches, new_hidden_states, residual, _ = self.model_fn(
                state_leaves,
                kv_caches,
                input_ids_loop,
                hidden_states,
                attn_metadata,
                layer_name_to_kvcache_index,
                spec_step_idx=i + 1,
            )
            hidden_states = residual[0]
            draft_token_ids = self._get_draft_token_ids(
                state_leaves, new_hidden_states)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = self._stack_draft_token_ids(draft_token_ids_list)

        return kv_caches, draft_token_ids
