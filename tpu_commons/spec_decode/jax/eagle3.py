"""Implements the Eagle proposer for speculative decoding on JAX/TPU."""
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from vllm.config import VllmConfig

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.model_loader import get_model


class EagleProposer:
    """A proposer for speculative decoding using the Eagle method.

    This class is responsible for loading the draft model and generating draft
    tokens based on the target model's outputs.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            runner: Any,  # TPUModelRunner
    ):
        """Initializes the EagleProposer.

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

    def load_model(self) -> None:
        """Loads the draft model."""
        # TODO(ranlihao): Sharing embedding and lm_head weights between the target and draft
        self.model_fn, self.compute_logits_fn, _, _, self.state, _, _ = get_model(
            self.vllm_config, self.rng_key, self.mesh, is_draft_model=True)

    def prepare_inputs(self, ) -> tuple[jnp.ndarray]:
        """Prepares inputs for the speculative decoding step.

        This function updates the common attention metadata to account for
        rejected tokens and newly sampled tokens. It also returns the token
        indices of the tokens that should be fed to the speculator.
        """
        raise NotImplementedError("prepare_inputs is not implemented yet.")

    def propose(
        self,
        target_token_ids: jnp.ndarray,  # [num_tokens]
        target_positions: jnp.ndarray,  # [num_tokens]
        target_hidden_states: jnp.ndarray,  # [num_tokens, hidden_size]
        next_token_ids: jnp.ndarray,  # [batch_size]
        attn_metadata: AttentionMetadata,
    ) -> jnp.ndarray:
        """Proposes draft tokens using the draft model.

        Args:
            target_token_ids: The token IDs from the target model.
            target_positions: The positions of the tokens from the target model.
            target_hidden_states: The hidden states from the target model.
            next_token_ids: The next token IDs sampled from the target model.
            attn_metadata: The attention metadata for the draft model.

        Returns:
            A tensor of proposed draft token IDs.
        """
        last_token_indices = attn_metadata.query_start_loc[1:] - 1

        # Shift the input ids by one token.
        input_ids = jnp.roll(target_token_ids, -1)

        # Replace the last token of each sequence with the next token.
        input_ids = input_ids.at[last_token_indices].set(next_token_ids)

        # NOTE(pooyam): For now, we don't support multimodal.
        _, hidden_states = self.model(
            kv_caches=None,
            input_ids=input_ids,
            hidden_states=target_hidden_states,
            positions=target_positions,
            attention_metadata=attn_metadata,
        )
        sample_hidden_states = hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = jnp.argmax(logits, axis=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.reshape(-1, 1)

        draft_token_ids_list = [draft_token_ids]
        positions = attn_metadata.seq_lens
        hidden_states = hidden_states[last_token_indices]

        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            input_ids = draft_token_ids_list[-1].astype(jnp.int32)
            positions += 1

            # TODO(pooyam): We don't handle the case where the draft model
            # generates tokens beyond the max model length.

            # Update attention metadata for the new token
            new_seq_lens = attn_metadata.seq_lens + 1
            block_numbers = positions // self.block_size
            block_ids = jnp.take_along_axis(attn_metadata.block_tables,
                                            block_numbers[:, None],
                                            axis=1).squeeze(axis=1)
            new_input_positions = (block_ids * self.block_size +
                                   positions % self.block_size)
            attn_metadata = replace(
                attn_metadata,
                seq_lens=new_seq_lens,
                input_positions=new_input_positions,
            )

            # Run the model.
            _, hidden_states = self.model(
                kv_caches=None,
                input_ids=input_ids,
                hidden_states=hidden_states,
                positions=positions,
                attention_metadata=attn_metadata,
            )
            logits = self.model.compute_logits(hidden_states)
            draft_token_ids = jnp.argmax(logits, axis=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = jnp.stack(draft_token_ids_list, axis=1)
        return draft_token_ids
