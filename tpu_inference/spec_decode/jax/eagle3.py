"""Implements the Eagle3 proposer for speculative decoding on JAX/TPU."""
import functools
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.runner import utils as runner_utils
from tpu_inference.utils import device_array


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

    def load_model(self, target_model: Any) -> None:
        """Loads the draft model."""
        self.model_fn, self.compute_logits_fn, self.combine_hidden_states_fn, _, _, _, self.state, _, _ = get_model(
            self.vllm_config, self.rng_key, self.mesh, is_draft_model=True)
        del self.state.model['embed_tokens']
        self.state.model.embed_tokens = target_model.model.embed

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _concate_hidden_states(self, aux_hidden_states):
        """JIT-compiled helper for concatenating auxiliary hidden states."""
        # Concat aux hidden states along feature dim.
        return jnp.concatenate(aux_hidden_states, axis=-1)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _select_target_hidden_states(self, aux_hidden_states, token_indices):
        """JIT-compiled helper for selecting target hidden states."""
        return jnp.concatenate([h[token_indices] for h in aux_hidden_states],
                               axis=-1)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _prepare_input_ids(self, query_start_loc: jax.Array,
                           target_token_ids: jax.Array,
                           next_token_ids: jax.Array,
                           num_reqs: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled helper for preparing the input IDs for the draft model."""

        last_token_indices = query_start_loc[1:] - 1
        # Shift the input ids by one token.
        rolled_input_ids = jnp.roll(target_token_ids, -1, axis=0)

        # To make the update JIT-compatible with a dynamic `num_reqs`, we perform a
        # scatter update of a static size, using a mask to handle the dynamic part.
        max_num_reqs = last_token_indices.shape[0]
        mask = jnp.arange(max_num_reqs) < num_reqs

        # For padded requests (where mask is False), we use the original value from
        # the rolled array, making the update a no-op for them.
        original_values_at_indices = rolled_input_ids[last_token_indices]
        values_to_set = jnp.where(mask, next_token_ids,
                                  original_values_at_indices)

        input_ids = rolled_input_ids.at[last_token_indices].set(values_to_set)

        return input_ids, last_token_indices

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _prepare_input_loop(self, positions, seq_lens, block_tables):
        """JIT-compiled helper for preparing inputs in the loop of prediction."""

        positions += 1
        exceeds_max_model_len = positions >= self.runner.max_model_len
        clamped_positions = jnp.where(exceeds_max_model_len, 0, positions)

        new_seq_lens = seq_lens + 1
        new_seq_lens = jnp.minimum(new_seq_lens, self.runner.max_model_len)
        new_seq_lens = jnp.where(exceeds_max_model_len, 1, new_seq_lens)

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
        expanded_exceeds_mask = jnp.repeat(exceeds_max_model_len,
                                           max_num_blocks_per_req)
        new_block_tables = jnp.where(expanded_exceeds_mask, -1, block_tables)

        return positions, clamped_positions, new_seq_lens, query_start_loc, new_block_tables

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _reshape_block_tables(self, block_tables: jax.Array) -> jax.Array:
        """JIT-compiled helper for reshaping block tables."""
        return block_tables.reshape(-1)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _get_draft_token_ids(self, logits: jax.Array) -> jnp.ndarray:
        """JIT-compiled helper for getting draft token IDs from logits."""
        return jnp.argmax(logits, axis=-1)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _stack_draft_token_ids(
            self, draft_token_ids_list: list[jax.Array]) -> jnp.ndarray:
        """JIT-compiled helper for stacking draft token IDs."""
        return jnp.stack(draft_token_ids_list, axis=1)

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[AttentionMetadata, jnp.ndarray, jnp.ndarray]:
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

        if num_rejected_tokens is None:
            return attn_metadata, input_ids, self._concate_hidden_states(
                aux_hidden_states)

        # Number of active requests in this step (un-padded count).
        num_reqs = self.runner.input_batch.num_reqs

        # Host copies from the metadata prepared by the runner.
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        assert query_start_loc_cpu is not None and seq_lens_cpu is not None

        # Rejection-aware path: compute new per-request lengths and token indices.
        # Convert to host numpy for efficient prefix-sum and repeat ops.
        nrt_cpu = jax.device_get(num_rejected_tokens).astype("int32")

        # query_len_per_req = [q1, q2, ...]
        query_len_per_req = (query_start_loc_cpu[1:] -
                             query_start_loc_cpu[:-1])

        # query_start_loc_cpu and consequentaly query_len_per_req are padded
        # For padded requests, the query length should be 0.
        query_len_per_req[num_reqs:] = 1
        # num_tokens_per_req = [q1 - n1, q2 - n2, ...]
        num_tokens_per_req = (query_len_per_req - nrt_cpu)
        assert (num_tokens_per_req
                >= 0).all(), ("num_tokens_per_req must be non-negative")

        # new_query_start_loc = [0, q1-n1, q1+q2-n1-n2, ...]
        # Use numpy for cumsum and then convert back.
        new_query_start_loc_cpu = np.zeros_like(query_start_loc_cpu)
        np.cumsum(num_tokens_per_req, out=new_query_start_loc_cpu[1:])

        # Build token indices selecting the kept tokens from each request.
        total_num_tokens = int(new_query_start_loc_cpu[-1])

        # Pad to total_num_tokens.
        padded_total_num_tokens = runner_utils.get_padded_token_len(
            self.runner.num_tokens_paddings, total_num_tokens)
        pad_width = padded_total_num_tokens - total_num_tokens
        assert pad_width >= 0, (
            f"total_num_tokens {total_num_tokens} exceeds "
            f"num_tokens_paddings {self.runner.num_tokens_paddings}")

        # Expand request starts: [0, 0, q1-n1, ...,]
        expanded_new_query_start_loc = np.repeat(new_query_start_loc_cpu[:-1],
                                                 num_tokens_per_req)
        # Offsets within each request window: [0,1,2, 0,1,2,3, ...]
        token_offsets = np.arange(total_num_tokens, dtype=np.int32)
        token_offsets -= expanded_new_query_start_loc
        # Map into old flat indices by adding original request starts.
        old_query_start_loc_expanded = np.repeat(query_start_loc_cpu[:-1],
                                                 num_tokens_per_req)

        token_indices_cpu = token_offsets + old_query_start_loc_expanded
        token_indices_cpu = np.pad(token_indices_cpu, (0, pad_width),
                                   "constant",
                                   constant_values=0)
        token_indices = jnp.asarray(token_indices_cpu, dtype=jnp.int32)
        # Select tokens and hidden states.
        target_token_ids = self.runner._select_from_array_fn(
            input_ids, token_indices)
        target_hidden_states = self._select_target_hidden_states(
            aux_hidden_states, token_indices)
        # Update positions to match the selected tokens.
        if attn_metadata.input_positions.ndim == 2:
            input_positions = attn_metadata.input_positions[:, token_indices]
        else:
            input_positions = self.runner._select_from_array_fn(
                attn_metadata.input_positions, token_indices)

        # Update seq_lens for active requests only: new_seq_lens = s - n.
        new_seq_lens_cpu = seq_lens_cpu - nrt_cpu

        query_start_loc, seq_lens = device_array(self.mesh, (
            new_query_start_loc_cpu,
            new_seq_lens_cpu,
        ))

        # Return updated metadata with positions, qsl, and seq_lens.
        updated_attn = AttentionMetadata(
            input_positions=input_positions,
            block_tables=attn_metadata.block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=attn_metadata.request_distribution,
        )
        return updated_attn, target_token_ids, target_hidden_states

    def propose(
        self,
        kv_caches: list[jax.Array],
        next_token_ids: jnp.ndarray,  # [batch_size]
        attn_metadata: AttentionMetadata,
        target_token_ids,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Proposes draft tokens using the draft model.
        Returns:
            A tuple containing the updated KV caches and a tensor of proposed
            draft token IDs.
        """

        target_hidden_states = self.combine_hidden_states_fn(
            self.state, target_hidden_states)

        input_ids, last_token_indices = self._prepare_input_ids(
            attn_metadata.query_start_loc, target_token_ids, next_token_ids,
            self.runner.input_batch.num_reqs)
        # NOTE(pooyam): For now, we don't support multimodal.

        # The last KV cache group is for the draft model.
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_device_tensor()
        block_tables = self._reshape_block_tables(block_tables)
        attn_metadata = replace(attn_metadata, block_tables=block_tables)

        kv_caches, hidden_states, residual = self.model_fn(
            self.state,
            kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
        )
        sample_hidden_states = self.runner._select_from_array_fn(
            hidden_states, last_token_indices)
        lora_metadata = None
        logits = self.compute_logits_fn(self.state, sample_hidden_states,
                                        lora_metadata)
        draft_token_ids = self._get_draft_token_ids(logits)

        draft_token_ids_list = [draft_token_ids]
        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            return kv_caches, self._stack_draft_token_ids(draft_token_ids_list)

        positions = self.runner._select_from_array_fn(
            attn_metadata.input_positions, last_token_indices)
        hidden_states = self.runner._select_from_array_fn(
            residual[0], last_token_indices)

        for _ in range(self.num_speculative_tokens - 1):
            input_ids_loop = draft_token_ids_list[-1]

            positions, clamped_positions, new_seq_lens, query_start_loc, new_block_tables = self._prepare_input_loop(
                positions, attn_metadata.seq_lens, attn_metadata.block_tables)

            attn_metadata = replace(
                attn_metadata,
                input_positions=clamped_positions,
                seq_lens=new_seq_lens,
                query_start_loc=query_start_loc,
                block_tables=new_block_tables,
            )
            kv_caches, new_hidden_states, residual = self.model_fn(
                self.state,
                kv_caches,
                input_ids_loop,
                hidden_states,  # This should be the hidden_states from previous step
                attn_metadata,
            )
            hidden_states = residual[0]
            logits = self.compute_logits_fn(self.state, new_hidden_states,
                                            lora_metadata)
            draft_token_ids = self._get_draft_token_ids(logits)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = self._stack_draft_token_ids(draft_token_ids_list)

        return kv_caches, draft_token_ids
