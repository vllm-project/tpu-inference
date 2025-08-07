"""
JAX-based rejection sampler for speculative decoding on TPU.

This implementation follows the same algorithm as the GPU version but is
designed for JAX/TPU compatibility. It currently only supports greedy sampling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from tpu_commons.models.jax.layers.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32

# Placeholder token ID for rejected tokens
PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler:
    """
    JAX-based rejection sampler for speculative decoding.

    The implementation follows the algorithm described in
    https://arxiv.org/abs/2211.17192.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        # [batch_size, max_spec_len] - batched format
        draft_token_ids: jnp.ndarray,
        # [batch_size] - number of draft tokens per request
        num_draft_tokens: jnp.ndarray,
        max_spec_len: int,
        # [batch_size, max_spec_len, vocab_size] - batched format
        draft_probs: Optional[jnp.ndarray],
        # [batch_size, max_spec_len, vocab_size] - batched format
        target_probs: jnp.ndarray,
        # [batch_size, 1]
        bonus_token_ids: jnp.ndarray,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> jnp.ndarray:
        """
        Perform rejection sampling on draft tokens with batched inputs.

        Args:
            draft_token_ids: Draft token IDs in batched format [batch_size, max_spec_len].
            num_draft_tokens: Number of draft tokens per request [batch_size].
            max_spec_len: Maximum number of speculative tokens.
            draft_probs: Draft probabilities in batched format [batch_size, max_spec_len, vocab_size].
            target_probs: Target probabilities in batched format [batch_size, max_spec_len, vocab_size].
            bonus_token_ids: Bonus token IDs [batch_size, 1].
            sampling_metadata: Additional metadata needed for sampling.

        Returns:
            output_token_ids: A tensor containing the final output token IDs.
        """
        return self.forward(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=max_spec_len,
            draft_probs=draft_probs,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )

    def forward(
        self,
        # [batch_size, max_spec_len] - batched format
        draft_token_ids: jnp.ndarray,
        # [batch_size] - number of draft tokens per request
        num_draft_tokens: jnp.ndarray,
        max_spec_len: int,
        # [batch_size, max_spec_len, vocab_size] - batched format
        draft_probs: Optional[jnp.ndarray],
        # [batch_size, max_spec_len, vocab_size] - batched format
        target_probs: jnp.ndarray,
        # [batch_size, 1]
        bonus_token_ids: jnp.ndarray,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> jnp.ndarray:
        """
        Perform rejection sampling on draft tokens with batched inputs.

        Args:
            draft_token_ids: Draft token IDs in batched format [batch_size, max_spec_len].
            num_draft_tokens: Number of draft tokens per request [batch_size].
            max_spec_len: Maximum number of speculative tokens.
            draft_probs: Draft probabilities in batched format [batch_size, max_spec_len, vocab_size].
            target_probs: Target probabilities in batched format [batch_size, max_spec_len, vocab_size].
            bonus_token_ids: Bonus token IDs [batch_size, 1].
            sampling_metadata: Additional metadata needed for sampling.

        Returns:
            output_token_ids: A tensor containing the final output token IDs.
        """
        assert max_spec_len <= MAX_SPEC_LEN

        output_token_ids = rejection_sample(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids

    @staticmethod
    def parse_output(
        output_token_ids: jnp.ndarray,
        vocab_size: int,
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.

        Returns:
            A list of lists of token IDs.
        """
        # Convert JAX array to numpy for easier manipulation
        output_token_ids_np = np.asarray(output_token_ids)
        # Create mask for valid tokens.
        valid_mask = ((output_token_ids_np != PLACEHOLDER_TOKEN_ID) &
                      (output_token_ids_np < vocab_size))
        outputs = [
            row[valid_mask[i]].tolist()
            for i, row in enumerate(output_token_ids_np)
        ]
        return outputs


def rejection_sample(
    # [batch_size, max_spec_len] - batched format
    draft_token_ids: jnp.ndarray,
    # [batch_size] - JAX array
    num_draft_tokens: jnp.ndarray,
    max_spec_len: int,
    # [batch_size, max_spec_len, vocab_size] - batched format
    draft_probs: Optional[jnp.ndarray],
    # [batch_size, max_spec_len, vocab_size] - batched format
    target_probs: jnp.ndarray,
    # [batch_size, 1]
    bonus_token_ids: jnp.ndarray,
    sampling_metadata: TPUSupportedSamplingMetadata,
) -> jnp.ndarray:
    """
    Perform rejection sampling on draft tokens with batched inputs.

    Args:
        draft_token_ids: Draft token IDs in batched format [batch_size, max_spec_len].
        num_draft_tokens: Number of draft tokens per request [batch_size].
        max_spec_len: Maximum number of speculative tokens.
        draft_probs: Draft probabilities in batched format [batch_size, max_spec_len, vocab_size].
        target_probs: Target probabilities in batched format [batch_size, max_spec_len, vocab_size].
        bonus_token_ids: Bonus token IDs [batch_size, 1].
        sampling_metadata: Sampling metadata.

    Returns:
        output_token_ids: Output token IDs [batch_size, max_spec_len + 1].
    """
    # For greedy sampling, use JIT-compiled scan-based approach
    output_token_ids = _greedy_rejection_sample_with_scan(
        draft_token_ids,
        num_draft_tokens,
        target_probs,
        bonus_token_ids,
    )

    return output_token_ids


# TODO(pooyam): Benchmark against the following approaches:
# - Using `jax.lax.segment_xyz`` to work with flattened inputs instead of batched inputs.
# - Using vectorized implementation using `cumprod` and other masking tricks.
# - A pallas kernel similar to the Triton implementation.
# Overall, I expect XLA to optimize the scan-based approach pretty well, but
# it would be good to compare performance against other methods.
def greedy_scan_step(carry, inputs):
    """
    Scan step function for greedy rejection sampling.

    Args:
        carry: The 'rejected' state from the previous step.
               Shape: (batch_size,) boolean.
        inputs: A slice of the data for the current token position.
                A tuple of (draft_k, target_k, mask_k)
                Each has shape (batch_size,)

    Returns:
        new_rejected: Updated rejected state for next step (batch_size,)
        output_k: Output token for this step (batch_size,)
    """
    prev_rejected = carry
    draft_k, target_k, mask_k = inputs

    # Determine the output for this step.
    # If the sequence was already rejected OR if the current position is padding,
    # the output is a placeholder. Otherwise, it's the target token.
    output_k = jnp.where(prev_rejected | ~mask_k, PLACEHOLDER_TOKEN_ID,
                         target_k)

    # Determine the new 'rejected' state for the next step
    # A sequence becomes rejected if it was already rejected OR if there's a new mismatch.
    is_mismatch = (draft_k != target_k) & mask_k
    new_rejected = prev_rejected | is_mismatch

    return new_rejected, output_k


@jax.jit
def _greedy_rejection_sample_with_scan(
        draft_token_ids: jnp.ndarray,  # [batch_size, max_spec_len]
        num_draft_tokens: jnp.ndarray,  # [batch_size]
        target_probs: jnp.ndarray,  # [batch_size, max_spec_len, vocab_size]
        bonus_token_ids: jnp.ndarray,  # [batch_size, 1]
) -> jnp.ndarray:
    """
    JIT-compiled greedy rejection sampling using jax.lax.scan.

    Args:
        draft_token_ids: Draft token IDs in batched format [batch_size, max_spec_len].
        num_draft_tokens: Number of draft tokens per request [batch_size].
        target_probs: Target probabilities in batched format [batch_size, max_spec_len, vocab_size].
        bonus_token_ids: Bonus token IDs [batch_size, 1].

    Returns:
        output_token_ids: Output token IDs [batch_size, max_spec_len + 1].
    """
    batch_size = draft_token_ids.shape[0]
    max_spec_len = draft_token_ids.shape[1]

    # 1. Prepare draft_padded, target_padded, and mask
    # draft_padded and target_padded are already in the right format
    draft_padded = draft_token_ids  # [batch_size, max_spec_len]

    # Get the argmax for greedy selection - shape [batch_size, max_spec_len]
    target_padded = jnp.argmax(target_probs,
                               axis=-1)  # [batch_size, max_spec_len]

    # Create mask: True for valid positions, False for padding
    mask = jnp.arange(max_spec_len)[
        None, :] < num_draft_tokens[:, None]  # [batch_size, max_spec_len]

    # 2. Initial state: nothing is rejected yet.
    initial_rejected_state = jnp.zeros(batch_size, dtype=jnp.bool_)

    # 3. Transpose inputs for scan: jax.lax.scan expects (max_spec_len, batch_size)
    draft_transposed = draft_padded.T  # [max_spec_len, batch_size]
    target_transposed = target_padded.T  # [max_spec_len, batch_size]
    mask_transposed = mask.T  # [max_spec_len, batch_size]

    # 4. jax.lax.scan will loop over the `max_spec_len` dimension
    # and apply `greedy_scan_step` at each slice.
    final_rejected_state, output_padded = jax.lax.scan(
        f=greedy_scan_step,
        init=initial_rejected_state,
        xs=(draft_transposed, target_transposed, mask_transposed))

    # `output_padded` will have the shape (max_spec_len, batch_size)
    # Transpose it back to (batch_size, max_spec_len)
    output_tokens = output_padded.T  # [batch_size, max_spec_len]

    # 5. Handle bonus tokens
    # Check which sequences had all tokens accepted (final_rejected_state is False)
    all_accepted = ~final_rejected_state  # [batch_size]

    # Create output buffer with space for bonus tokens
    output_token_ids = jnp.full((batch_size, max_spec_len + 1),
                                PLACEHOLDER_TOKEN_ID,
                                dtype=jnp.int32)

    # Copy the selected tokens to the output
    output_token_ids = output_token_ids.at[:, :max_spec_len].set(output_tokens)

    # Add bonus tokens for sequences where all tokens were accepted
    # Bonus tokens go at position num_draft_tokens for each sequence
    bonus_positions = num_draft_tokens  # [batch_size]

    # Create a mask for where to place bonus tokens
    position_grid = jnp.arange(max_spec_len +
                               1)[None, :]  # [1, max_spec_len + 1]
    bonus_mask = (position_grid == bonus_positions[:, None]
                  )  # [batch_size, max_spec_len + 1]

    # Apply bonus tokens where all tokens were accepted
    output_token_ids = jnp.where(all_accepted[:, None] & bonus_mask,
                                 bonus_token_ids[:, 0, None], output_token_ids)

    return output_token_ids
