"""Implements the Eagle proposer for speculative decoding on JAX/TPU."""
from typing import Any

import jax.numpy as jnp
from vllm.config import VllmConfig

from tpu_commons.models.jax.attention_metadata import AttentionMetadata


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

    def load_model(self) -> None:
        """Loads the draft model."""
        # TODO: Add impelemntation
        raise NotImplementedError("load_model is not implemented yet.")

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
        raise NotImplementedError("propose is not implemented yet.")
