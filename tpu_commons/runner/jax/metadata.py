from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on JAX/TPU, containing all necessary indices."""
    draft_token_ids: jnp.ndarray
    draft_lengths: jnp.ndarray
    target_logits_indices: jnp.ndarray
    bonus_logits_indices: jnp.ndarray
    final_logits_indices: jnp.ndarray
    max_spec_len: int
