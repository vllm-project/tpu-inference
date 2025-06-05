import jax
import jax.numpy as jnp
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.layers.binary_search import topk_mask, topp_mask


def sample(
    is_prefill: bool,
    do_sampling: bool,
    rng: PRNGKey,
    mesh: Mesh,
    logits: jax.Array,
    seq_lens: jax.Array,
    temperatures: jax.Array,
    top_ps: jax.Array,
    top_ks: jax.Array,
    chunked_prefill_enabled: bool = False,
) -> jax.Array:
    # (batch_size, vocab_size)
    if do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        # TODO: shard the batch axis without affecting unembedding
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(None, None)))

    if chunked_prefill_enabled:
        batch_size = logits.shape[0]
        assert batch_size == 1
        logits = jnp.squeeze(logits, 0)
    elif is_prefill:
        batch_size = logits.shape[0]
        batch_indices = jnp.arange(batch_size)
        logits = logits[batch_indices, seq_lens - 1, :]
    else:
        logits = jnp.squeeze(logits, 1)

    if not do_sampling:
        return jnp.argmax(logits, axis=-1)

    logits = logits.astype(jnp.float32)
    logits = topk_mask(logits, top_ks, replace_val=-1e12)
    logits = topp_mask(logits, top_ps, replace_val=-1e12)

    temperatures = temperatures.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    next_tokens = jax.random.categorical(rng, logits)
    return next_tokens
