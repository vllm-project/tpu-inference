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
        # [num_tokens] - flattened format
        draft_token_ids: jnp.ndarray,
        # [batch_size] - number of draft tokens per request
        num_draft_tokens: jnp.ndarray,
        max_spec_len: int,
        # [num_tokens, vocab_size] - flattened format
        draft_probs: Optional[jnp.ndarray],
        # [num_tokens, vocab_size] - flattened format
        target_probs: jnp.ndarray,
        # [batch_size]
        bonus_token_ids: jnp.ndarray,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> jnp.ndarray:
        """
        Perform rejection sampling on draft tokens with flattened inputs.

        Args:
            draft_token_ids: Draft token IDs in flattened format [num_tokens].
            num_draft_tokens: Number of draft tokens per request [batch_size].
            max_spec_len: Maximum number of speculative tokens.
            draft_probs: Draft probabilities in flattened format [num_tokens, vocab_size].
            target_probs: Target probabilities in flattened format [num_tokens, vocab_size].
            bonus_token_ids: Bonus token IDs [batch_size].
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
        # [num_tokens] - flattened format
        draft_token_ids: jnp.ndarray,
        # [batch_size] - number of draft tokens per request
        num_draft_tokens: jnp.ndarray,
        max_spec_len: int,
        # [num_tokens, vocab_size] - flattened format
        draft_probs: Optional[jnp.ndarray],
        # [num_tokens, vocab_size] - flattened format
        target_probs: jnp.ndarray,
        # [batch_size]
        bonus_token_ids: jnp.ndarray,
        sampling_metadata: TPUSupportedSamplingMetadata,
    ) -> jnp.ndarray:
        """
        Perform rejection sampling on draft tokens with flattened inputs.

        Args:
            draft_token_ids: Draft token IDs in flattened format [num_tokens].
            num_draft_tokens: Number of draft tokens per request [batch_size].
            max_spec_len: Maximum number of speculative tokens.
            draft_probs: Draft probabilities in flattened format [num_tokens, vocab_size].
            target_probs: Target probabilities in flattened format [num_tokens, vocab_size].
            bonus_token_ids: Bonus token IDs [batch_size].
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
        num_draft_tokens_cpu: np.ndarray,
        batch_size: int,
        padded_tokens_length: int,
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        Args:
            output_token_ids: The sampled token IDs in shape
                [num_tokens + batch_size]. The first num_tokens elements are
                the main tokens, and the last batch_size elements are bonus tokens.
                Rejected tokens are replaced with `PLACEHOLDER_TOKEN_ID`.
            vocab_size: The size of the vocabulary.
            num_draft_tokens_cpu: Number of draft tokens per request [batch_size]
                as a numpy array on CPU.
            batch_size: The number of requests in the batch.
            padded_tokens_length: The padded length of the main tokens in the output.

        Returns:
            A list of lists of token IDs.
        """
        # Convert JAX array to numpy for easier manipulation
        output_token_ids_np = np.asarray(output_token_ids)

        # Split main tokens and bonus tokens
        main_tokens = output_token_ids_np[:
                                          padded_tokens_length]  # [num_tokens]
        bonus_tokens = output_token_ids_np[
            padded_tokens_length:]  # [batch_size]

        # Reconstruct per-sequence outputs
        outputs = []
        start_idx = 0

        for i in range(batch_size):
            seq_length = int(num_draft_tokens_cpu[i])
            end_idx = start_idx + seq_length

            # Get main tokens for this sequence
            seq_main_tokens = main_tokens[start_idx:end_idx]

            # Filter out placeholder tokens
            valid_main_tokens = seq_main_tokens[
                (seq_main_tokens != PLACEHOLDER_TOKEN_ID)
                & (seq_main_tokens < vocab_size)]

            # Add bonus token if it's valid
            bonus_token = bonus_tokens[i]
            if bonus_token != PLACEHOLDER_TOKEN_ID and bonus_token < vocab_size:
                seq_tokens = np.concatenate([valid_main_tokens, [bonus_token]])
            else:
                seq_tokens = valid_main_tokens

            outputs.append(seq_tokens.tolist())
            start_idx = end_idx

        return outputs


def rejection_sample(
    # [num_tokens] - flattened format
    draft_token_ids: jnp.ndarray,
    # [batch_size] - JAX array
    num_draft_tokens: jnp.ndarray,
    max_spec_len: int,
    # [num_tokens, vocab_size] - flattened format
    draft_probs: Optional[jnp.ndarray],
    # [num_tokens, vocab_size] - flattened format
    target_probs: jnp.ndarray,
    # [batch_size]
    bonus_token_ids: jnp.ndarray,
    sampling_metadata: TPUSupportedSamplingMetadata,
) -> jnp.ndarray:
    """
    Perform rejection sampling on draft tokens with flattened inputs.

    Args:
        draft_token_ids: Draft token IDs in flattened format [num_tokens].
        num_draft_tokens: Number of draft tokens per request [batch_size].
        max_spec_len: Maximum number of speculative tokens.
        draft_probs: Draft probabilities in flattened format [num_tokens, vocab_size].
        target_probs: Target probabilities in flattened format [num_tokens, vocab_size].
        bonus_token_ids: Bonus token IDs [batch_size].
        sampling_metadata: Sampling metadata.

    Returns:
        output_token_ids: Output token IDs [num_tokens + batch_size].
    """
    # Use segment-based approach with flattened inputs directly
    output_token_ids = _greedy_rejection_sample_with_segment(
        draft_token_ids,
        target_probs,
        num_draft_tokens,
        bonus_token_ids,
    )

    return output_token_ids


# TODO(pooyam): Optimize/Profile this implementation further. Currently, I just want working e2e. There might be overheads with `parse_output` that can be optimized on TPU.
# I should Benchmark against the following approaches:
# - Using `jax.lax.segment_xyz`` to work with flattened inputs instead of batched inputs.
# - Using vectorized implementation using `cumprod` and other masking tricks.
# - A pallas kernel similar to the Triton implementation.
# - Scan based approach.
# Overall, I expect XLA to optimize the scan-based approach pretty well, but
# it would be good to compare performance against other methods.
@jax.jit
def _greedy_rejection_sample_with_segment(
    draft_token_ids: jax.Array,
    target_probs: jax.Array,
    num_draft_tokens: jax.Array,
    bonus_token_ids: jax.Array,
) -> jax.Array:
    """
    Performs greedy speculative decoding validation in a vectorized, jittable manner.

    This function compares draft tokens with the target model's outputs. For each
    sequence in the batch, it accepts tokens as long as the draft and target match.
    When a mismatch occurs, it takes the target model's token and invalidates the
    rest of the tokens in that sequence by setting them to -1.

    Args:
        draft_token_ids: A 1D JAX array (num_tokens,) of integers representing the
                         concatenated draft tokens for all sequences in the batch.
        target_probs: A 2D JAX array (num_tokens, vocab_size) of floats representing
                      the concatenated target model's probabilities.
        num_draft_tokens: A 1D JAX array (batch_size,) of integers specifying the
                          number of draft tokens for each sequence in the batch.
        bonus_token_ids: A 1D JAX array (batch_size,) of integers representing the
                         bonus token for each sequence.

    Returns:
        A 1D JAX array (num_tokens + batch_size,) containing the validated token
        sequence followed by bonus tokens (or -1 if not accepted).
    """
    # Get target argmax
    target_logits_argmax = jnp.argmax(target_probs, axis=-1)

    # --- Step 1: Create Segment IDs and Per-Segment Indices ---
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]

    # `segment_ids` assigns a unique ID to each token, corresponding to its
    # sequence in the batch. E.g., [0, 0, 0, 1, 1, 2, 2, 2, 2] for sequences [3, 2, 4].
    segment_ids = jnp.repeat(jnp.arange(batch_size),
                             num_draft_tokens,
                             total_repeat_length=total_tokens)

    # `group_indices` creates a within-segment index for each token.
    # E.g., [0, 1, 2, 0, 1, 0, 1, 2, 3] for the example above.
    segment_starts = jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(num_draft_tokens)[:-1]])
    broadcast_starts = jnp.repeat(segment_starts,
                                  num_draft_tokens,
                                  total_repeat_length=total_tokens)
    group_indices = jnp.arange(total_tokens) - broadcast_starts

    # --- Step 2: Find the First Mismatch in Each Segment ---

    # Find all mismatches between draft and target tokens.
    mismatches = draft_token_ids != target_logits_argmax

    # To find the *first* mismatch, we use a trick with segment_min.
    # We create an array where mismatched positions hold their `group_index`
    # and matched positions hold a large value.
    large_value = total_tokens
    mismatch_indices = jnp.where(mismatches, group_indices, large_value)

    # `segment_min` finds the minimum `mismatch_index` for each segment. This
    # effectively gives us the `group_index` of the first mismatch.
    # For sequences with no mismatches, the result will be `large_value`.
    first_mismatch_idx_per_segment = jax.ops.segment_min(
        data=mismatch_indices.astype(jnp.int32),
        segment_ids=segment_ids,
        num_segments=batch_size,
        indices_are_sorted=True,
    )

    # Handle empty segments (where num_draft_tokens is 0). `segment_min` returns
    # the dtype's max value for empty segments; we replace it with our large_value
    # for consistency.
    max_int = jnp.iinfo(jnp.int32).max
    first_mismatch_idx_per_segment = jnp.where(
        first_mismatch_idx_per_segment == max_int, large_value,
        first_mismatch_idx_per_segment)

    # --- Step 3: Broadcast Mismatch Info and Generate Main Token Output ---

    # Broadcast the first mismatch index back to the original token dimension.
    first_mismatch_idx_broadcast = jnp.repeat(first_mismatch_idx_per_segment,
                                              num_draft_tokens,
                                              total_repeat_length=total_tokens)

    # The final logic for main tokens:
    # A token is valid if its `group_index` is less than or equal to the
    # index of the first mismatch in its segment.
    # - If `group_index < first_mismatch_idx`, the draft was correct.
    # - If `group_index == first_mismatch_idx`, this is the correction token.
    # - If `group_index > first_mismatch_idx`, the token is invalid (-1).
    main_tokens = jnp.where(group_indices <= first_mismatch_idx_broadcast,
                            target_logits_argmax, PLACEHOLDER_TOKEN_ID)

    # --- Step 4: Handle Bonus Tokens ---

    # A sequence gets its bonus token if there were no mismatches
    # (first_mismatch_idx_per_segment == large_value)
    all_accepted = first_mismatch_idx_per_segment == large_value

    # For sequences with no draft tokens, we should still give them the bonus token
    # since there's nothing to reject
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    bonus_tokens = jnp.where(should_get_bonus, bonus_token_ids,
                             PLACEHOLDER_TOKEN_ID)

    # --- Step 5: Concatenate Main Tokens and Bonus Tokens ---

    output = jnp.concatenate([main_tokens, bonus_tokens])

    return output
