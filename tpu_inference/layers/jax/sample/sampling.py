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

import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

_SAMPLING_EPS = 1e-5


@functools.partial(
    jax.jit,
    static_argnames=["mesh"],
)
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))
    greedy_sampled = jnp.argmax(logits, axis=-1)
    if not tpu_sampling_metadata.do_sampling:
        return greedy_sampled

    logits = logits.astype(jnp.float32)

    # If current sequence length < min_tokens, we must mask out stop tokens.
    if (tpu_sampling_metadata.min_tokens is not None
            and tpu_sampling_metadata.stop_token_ids is not None):

        # [TRACE LOG] This prints ONCE when the model first loads/compiles.
        # It confirms the code is actually being included in the graph.
        print("DEBUG: JAX is compiling the min_tokens logic path!")

        # Check which requests haven't reached min_tokens yet
        # (B,) boolean array
        should_enforce_min = tpu_sampling_metadata.seq_lens < tpu_sampling_metadata.min_tokens

        # [RUNTIME LOG] This prints EVERY time the model generates a token.
        # We limit it to the first request [0] to avoid spamming 64 lines per step.
        jax.debug.print("DEBUG RUNTIME: Len={x} Min={y} Enforce={z}",
                        x=tpu_sampling_metadata.seq_lens[0],
                        y=tpu_sampling_metadata.min_tokens[0],
                        z=should_enforce_min[0])

        # We need to mask logits for stop tokens.
        # Since scattering (at[].set) on sharded arrays can be tricky,
        # we define a vectorized function to apply per-row.
        def _mask_stop_tokens(logit_row, stop_ids, enforce):
            # If enforce is False, return row unchanged
            # If enforce is True, set stop_ids to -inf

            # Helper to perform the update
            def _apply_mask(l_row):
                # Filter out padding (-1) in stop_ids by only updating valid indices
                # Note: jax.numpy.at ignores negative indices usually, but to be safe:
                valid_mask = stop_ids != -1
                # We essentially want: l_row[stop_ids] = -inf where valid
                # We use 'min' to avoid overwriting if it's already -inf? No, just set.
                return l_row.at[stop_ids].set(
                    jnp.where(valid_mask, -1e12, l_row.at[stop_ids].get()))

            return jax.lax.cond(enforce, _apply_mask, lambda x: x, logit_row)

        # Vectorize over the batch
        logits = jax.vmap(_mask_stop_tokens)(
            logits, tpu_sampling_metadata.stop_token_ids, should_enforce_min)

    # Only apply top-k masking if k > 0 for each token
    top_k = tpu_sampling_metadata.top_k
    should_apply_topk = jnp.expand_dims(top_k > 0, axis=-1)
    topk_masked = topk_mask(logits, top_k, replace_val=-1e12)
    logits = jnp.where(should_apply_topk, topk_masked, logits)

    # Only apply top-p masking if p < 1.0 for each token
    top_p = tpu_sampling_metadata.top_p
    should_apply_topp = jnp.expand_dims(top_p < 1.0, axis=-1)
    topp_masked = topp_mask(logits, top_p, replace_val=-1e12)
    logits = jnp.where(should_apply_topp, topp_masked, logits)

    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    next_tokens = jax.random.categorical(rng, logits)
    # Note: avoid using the sample result when temperature < _SAMPLING_EPS
    # If temperature < 0, logits /= temperatures will flip the result, causing error.
    return jnp.where(tpu_sampling_metadata.temperature < _SAMPLING_EPS,
                     greedy_sampled, next_tokens)


def compute_logprobs(logits: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)


def gather_logprobs(
    logprobs: jax.Array,
    token_ids: jax.Array,
    num_logprobs: int,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
        num_logprobs: minimum number of logprobs to
                    retain per token


    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    # Find the topK values.
    topk_logprobs, topk_indices = jax.lax.top_k(logprobs, k=num_logprobs)

    # Get with the logprob of the prompt or sampled token.
    token_ids = jnp.expand_dims(token_ids, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, token_ids, axis=-1)

    # Compute the ranks of the actual token.
    token_ranks = jnp.sum(logprobs >= token_logprobs, axis=-1)

    # Concatenate together with the topk.
    indices = jnp.concatenate((token_ids, topk_indices), axis=1)
    logprobs = jnp.concatenate((token_logprobs, topk_logprobs), axis=1)

    # Use int32 to reduce the tensor size.
    indices = jnp.int32(indices)

    return LogprobsTensors(indices, logprobs, token_ranks)
