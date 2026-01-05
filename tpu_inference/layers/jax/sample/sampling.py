import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference import envs
from tpu_inference.kernels.sampling import topk_topp_and_sample as pallas_sample
from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

SAMPLING_EPS = 1e-5
REPLACE_VAL = -1e12


def _sample(
    rng: jax.Array,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # Compute greedy sample before applying temperature
    greedy_sampled = jnp.argmax(logits, axis=-1)

    logits = logits.astype(jnp.float32)

    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    logits = topk_mask(logits, tpu_sampling_metadata.top_k, replace_val=REPLACE_VAL)
    logits = topp_mask(logits, tpu_sampling_metadata.top_p, replace_val=REPLACE_VAL)

    # (batch_size,)
    next_tokens = jax.random.categorical(rng, logits)
    # Note: avoid using the sample result when temperature < SAMPLING_EPS
    # If temperature < 0, logits /= temperatures will flip the result, causing error.
    return jnp.where(tpu_sampling_metadata.temperature < SAMPLING_EPS,
                     greedy_sampled, next_tokens)

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
    if tpu_sampling_metadata.use_pallas_kernel:
        return pallas_sample(
            rng, logits, tpu_sampling_metadata,
            max_k=envs.PALLAS_SAMPLING_TOPK_THRESHOLD,
            sampling_eps=SAMPLING_EPS,
            replace_val=REPLACE_VAL,
        )
    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
    greedy_sampled = jnp.argmax(logits, axis=-1)
    if not tpu_sampling_metadata.do_sampling:
        return greedy_sampled
    return _sample(rng, logits, tpu_sampling_metadata)


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
