import math
from typing import Any, Dict

import jax
import jax.numpy as jnp


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_theta: float = 10000,
    rope_scaling: Dict[str, Any] = None,
) -> jax.Array:
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = rope_theta**fraction
    timescale = 1.0 / timescale

    if rope_scaling:
        timescale = apply_rope_scaling(timescale, rope_scaling)

    sinusoid_inp = positions[..., jnp.newaxis] * timescale[jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[:, jnp.newaxis, ...]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    # Some models pad the inputs head_dim with zeros,
    # so we need to split the inputs using the head_dim before padding.
    padded_head_dim = inputs.shape[-1]
    first_half = inputs[..., :head_dim // 2]
    second_half = inputs[..., head_dim // 2:head_dim]
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    if padded_head_dim > head_dim:
        out = jnp.pad(out, ((0, 0), (0, 0), (0, 0),
                            (0, padded_head_dim - head_dim)))
    return out.astype(inputs.dtype)


def apply_longrope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_scaling: Dict[str, Any],
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    rope_theta: float = 10000,
) -> jax.Array:
    # LongRoPE implementation specific to Phi-3
    # Implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py#L197-L235

    scale = max_position_embeddings / original_max_position_embeddings
    if scale <= 1.0:
        mscale = 1.0
    else:
        mscale = jnp.sqrt(1 + (jnp.log(scale) /
                               jnp.log(original_max_position_embeddings)))

    seq_len = inputs.shape[2]
    if seq_len > original_max_position_embeddings:
        long_factor = jnp.array(rope_scaling.get("long_factor"))
        timescale = 1.0 / (long_factor * (rope_theta**(
            (2 * jnp.arange(0, head_dim // 2)) / head_dim)))
    else:
        short_factor = jnp.array(rope_scaling.get("short_factor"))
        timescale = 1.0 / (short_factor * (rope_theta**(
            (2 * jnp.arange(0, head_dim // 2)) / head_dim)))

    # Calculate RoPE positions
    sinusoid_inp = positions[..., jnp.newaxis] * timescale[jnp.newaxis,
                                                           jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[:, jnp.newaxis, ...]
    sin = jnp.sin(sinusoid_inp) * mscale
    cos = jnp.cos(sinusoid_inp) * mscale

    # Padding logic
    padded_head_dim = inputs.shape[-1]

    # Apply RoPE mechanism
    first_half = inputs[..., :head_dim // 2]
    second_half = inputs[..., head_dim // 2:head_dim]

    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)

    if padded_head_dim > head_dim:
        out = jnp.pad(out, ((0, 0), (0, 0), (0, 0),
                            (0, padded_head_dim - head_dim)))

    return out.astype(inputs.dtype)


def apply_rope_scaling(freqs: jax.Array, rope_scaling: Dict[str,
                                                            Any]) -> jax.Array:
    # Values obtained from grid search
    scale_factor = rope_scaling.get("scale_factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings",
                                       8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / freqs
    smooth = (old_context_len / wavelen -
              low_freq_factor) / (high_freq_factor - low_freq_factor)

    high_freqs = jnp.where(wavelen < high_freq_wavelen, freqs, 0)
    low_freqs = jnp.where(wavelen > low_freq_wavelen, freqs / scale_factor, 0)
    mid_freqs = jnp.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * freqs / scale_factor + smooth * freqs,
        0,
    )
    new_freqs = high_freqs + low_freqs + mid_freqs
    return new_freqs
