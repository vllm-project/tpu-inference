import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.layers.binary_search import topk_mask, topp_mask
from tpu_commons.sample.metadata_jax import TPUSupportedSamplingMetadata


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
            logits, NamedSharding(mesh, P(None, None)))

    if not tpu_sampling_metadata.do_sampling:
        return jnp.argmax(logits, axis=-1)

    logits = logits.astype(jnp.float32)
    logits = topk_mask(logits, tpu_sampling_metadata.top_k, replace_val=-1e12)
    logits = topp_mask(logits, tpu_sampling_metadata.top_p, replace_val=-1e12)

    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    next_tokens = jax.random.categorical(rng, logits)
    return next_tokens
