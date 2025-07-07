import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.layers.binary_search import topk_mask, topp_mask


def sample(
    do_sampling: bool,
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    temperatures: jax.Array,
    top_ps: jax.Array,
    top_ks: jax.Array,
) -> jax.Array:
    # (B, vocab_size)
    if do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(None, None)))

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
