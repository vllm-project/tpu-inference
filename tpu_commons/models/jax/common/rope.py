import math

import jax
from jax import numpy as jnp


class RotaryEmbedding:
    """
    An implementation of the original rotary positional embedding.
    """

    def __init__(self, rotary_dim: int, rope_theta: float,
                 original_max_position_embeddings: int, dtype: jnp.dtype):
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.original_max_position_embeddings = original_max_position_embeddings
        self.sin_cos_cache = self._compute_sin_cos()
        self.dtype = dtype

    def _compute_inv_freq(self):
        fractions_H = jnp.arange(0, self.rotary_dim, 2,
                                 dtype=jnp.float32) / self.rotary_dim
        inv_freq_H = 1.0 / (self.rope_theta**fractions_H)
        return inv_freq_H

    def _compute_sin_cos(self):
        inv_freq_H = self._compute_inv_freq()
        t = jnp.arange(self.original_max_position_embeddings,
                       dtype=jnp.float32)

        freqs = jnp.einsum("...T,k->...Tk",
                           t,
                           inv_freq_H,
                           precision=jax.lax.Precision.HIGHEST)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache

    def apply_rope(self, positions: jax.Array, x_TNH: jax.Array):
        assert x_TNH.ndim == 3
        cos_sin_TH = self.sin_cos_cache[positions]
        # cos, sin: (T, H/2)
        cos_TH, sin_TH = jnp.split(cos_sin_TH, 2, axis=-1)
        assert sin_TH.ndim == 2 and cos_TH.ndim == 2
        # cos, sin: (T, 1, H/2)
        cos_T1H, sin_T1H = cos_TH[:, None, :], sin_TH[:, None, :]
        # first_half, second_half: (T, N, H/2)
        first_half_TNH, second_half_TNH = jnp.split(x_TNH, 2, axis=-1)
        combined = jnp.concatenate([
            first_half_TNH * cos_T1H - second_half_TNH * sin_T1H,
            second_half_TNH * cos_T1H + first_half_TNH * sin_T1H
        ],
                                   axis=-1)
        return combined.astype(self.dtype)


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """
    Rotary Embedding for deepseek, with scaling and YaRN method.
    """

    def __init__(
        self,
        rotary_dim: int,
        rope_theta: float,
        original_max_position_embeddings: int,
        scaling_factor: float,
        dtype: jnp.dtype,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.dtype = dtype

        self.mscale = _yarn_get_mscale(scaling_factor,
                                       mscale) / _yarn_get_mscale(
                                           scaling_factor, mscale_all_dim)
        super().__init__(rotary_dim, rope_theta,
                         original_max_position_embeddings, dtype)

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
        inv_freq_H = self._compute_inv_freq()
        t = jnp.arange(self.original_max_position_embeddings *
                       self.scaling_factor,
                       dtype=jnp.float32)
        freqs = jnp.einsum("...T,k->...Tk", t, inv_freq_H)
        sin, cos = jnp.sin(freqs) * self.mscale, jnp.cos(freqs) * self.mscale
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache

    def apply_rope(self, positions: jax.Array, x_TNH: jax.Array):
        assert x_TNH.ndim == 3
        cos_sin_TH = self.sin_cos_cache[positions]
        # cos, sin: (T, H/2)
        cos_TH, sin_TH = jnp.split(cos_sin_TH, 2, axis=-1)
        assert sin_TH.ndim == 2 and cos_TH.ndim == 2
        # cos, sin: (T, 1, H/2)
        cos_T1H, sin_T1H = cos_TH[:, None, :], sin_TH[:, None, :]
        # even, odd: (T, N, H/2)
        even_TNH, odd_TNH = x_TNH[..., ::2], x_TNH[..., 1::2]
        combined_TNH = jnp.stack([
            even_TNH * cos_T1H - odd_TNH * sin_T1H,
            odd_TNH * cos_T1H + even_TNH * sin_T1H
        ],
                                 axis=-1).reshape(x_TNH.shape)
        return combined_TNH.astype(self.dtype)


# Calculates the temperature scaling factor for YaRN to adjust
# RoPE embedding magnitudes.
def _yarn_get_mscale(scale, mscale):
    return jnp.where(scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0)


# Inverses dim formula to find dim based on number of rotations.
def _yarn_find_correction_dim(num_rotations,
                              dim,
                              base=10000,
                              max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Finds dim range bounds based on rotations.
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


# Creates a 1D mask that ramps linearly from 0 to 1 between min and max indices.
def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min) / (max - min)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func
