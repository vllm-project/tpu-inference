import math

import jax
from jax import numpy as jnp


class RotaryEmbedding:
    """
    An implementation of the original rotary positional embedding.
    """

    def __init__(self, rotary_dim: int, rope_theta: float,
                 original_max_position_embeddings: int):
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.original_max_position_embeddings = original_max_position_embeddings
        self.sin_cos_cache = self._compute_sin_cos()

    def _compute_inv_freq(self):
        fractions = jnp.arange(0, self.rotary_dim, 2,
                               dtype=jnp.float32) / self.rotary_dim
        inv_freq = 1.0 / (self.rope_theta**fractions)
        return inv_freq

    def _compute_sin_cos(self):
        inv_freq = self._compute_inv_freq()
        t = jnp.arange(self.original_max_position_embeddings,
                       dtype=jnp.float32)

        freqs = jnp.einsum("...T,k->...Tk",
                           t,
                           inv_freq,
                           precision=jax.lax.Precision.HIGHEST)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache

    def apply_rope(self, positions: jax.Array, x: jax.Array):
        # positions should be 1d according to vllm.
        assert x.ndim == 3
        cos_sin = self.sin_cos_cache[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)
        assert sin.ndim == 2 and cos.ndim == 2
        cos, sin = cos[:, None, :], sin[:, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        assert cos.shape[0] == x1.shape[0] and cos.shape[-1] == x1.shape[-1]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                               axis=-1)


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """
    Rotary Embedding for deepseek, with scaling and YaRN method.
    """

    def __init__(self,
                 rotary_dim: int,
                 rope_theta: float,
                 original_max_position_embeddings: int,
                 scaling_factor: float,
                 beta_fast: int = 32,
                 beta_slow: int = 1,
                 mscale: float = 1,
                 mscale_all_dim: float = 0):
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.mscale = _yarn_get_mscale(scaling_factor,
                                       mscale) / _yarn_get_mscale(
                                           scaling_factor, mscale_all_dim)
        super().__init__(rotary_dim, rope_theta,
                         original_max_position_embeddings)

    def _compute_inv_freq(self):
        fractions = jnp.arange(0, self.rotary_dim, 2,
                               dtype=jnp.float32) / self.rotary_dim
        inv_freq_extrapolation = 1.0 / (self.rope_theta**fractions)
        inv_freq_interpolation = 1.0 / (self.scaling_factor *
                                        self.rope_theta**fractions)
        low, high = _yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.rotary_dim, self.rope_theta,
            self.original_max_position_embeddings)

        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2).astype(jnp.float32)
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_sin_cos(self):
        inv_freq = self._compute_inv_freq()
        t = jnp.arange(self.original_max_position_embeddings *
                       self.scaling_factor,
                       dtype=jnp.float32)
        freqs = jnp.einsum("...T,k->...Tk", t, inv_freq)
        sin, cos = jnp.sin(freqs) * self.mscale, jnp.cos(freqs) * self.mscale
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache


def _yarn_get_mscale(scale, mscale):
    return jnp.where(scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0)


def _yarn_find_correction_dim(num_rotations,
                              dim,
                              base=10000,
                              max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


def _yarn_find_correction_range(low_rot,
                                high_rot,
                                dim,
                                base=10000,
                                max_position_embeddings=2048):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim) - min) / (max - min)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func
