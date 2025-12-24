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

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
from flax import nnx
from jax import numpy as jnp
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import NamedSharding, PartitionSpec


@dataclass(kw_only=True)
class RotaryEmbedding(nnx.Module):
    """
    An implementation of the original rotary positional embedding.
    """
    rotary_dim: int
    rope_theta: float
    original_max_position_embeddings: int
    dtype: jnp.dtype
    sin_cos_cache: Optional[jax.Array] = field(init=False, default=None)

    def initialize_cache(self):
        """Computes and caches the sin/cos embeddings."""
        if self.sin_cos_cache is None:
            self.sin_cos_cache = self._compute_sin_cos()

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
        assert self.sin_cos_cache is not None, "RoPE cache not initialized."
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


@dataclass(kw_only=True)
class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """
    Rotary Embedding for deepseek, with scaling and YaRN method.
    """
    scaling_factor: float
    beta_fast: int = 32
    beta_slow: int = 1
    mscale_value: float = 1
    mscale_all_dim: float = 0

    def initialize_cache(self, mesh: jax.sharding.Mesh):
        """Computes and caches the sin/cos embeddings."""
        # The second condition is for the Qwix case, where we need to call `initialize_cache` on
        # the abstract model.  Thus, when we go to call `initialize_cache` on the concrete model,
        # this method will have been called already, but we need to recompute the cache so that
        # it's concrete (otherwise, it'll still be a jax.ShapeDtypeStruct).
        if self.sin_cos_cache is not None and not isinstance(
                self.sin_cos_cache, jax.ShapeDtypeStruct):
            return
        mscale_val = _yarn_get_mscale(
            self.scaling_factor, self.mscale_value) / _yarn_get_mscale(
                self.scaling_factor, self.mscale_all_dim)
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self.mscale = jax.device_put(mscale_val, replicated_sharding)
        self.sin_cos_cache = self._compute_sin_cos()

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

    @jax.jit
    def _compute_sin_cos(self):
        inv_freq_H = self._compute_inv_freq()
        t = jnp.arange(self.original_max_position_embeddings *
                       self.scaling_factor,
                       dtype=jnp.float32)
        freqs = jnp.einsum("...T,k->...Tk", t, inv_freq_H)
        sin, cos = jnp.sin(freqs) * self.mscale, jnp.cos(freqs) * self.mscale
        cache = jnp.concatenate((cos, sin), axis=-1)
        H = cache.shape[1]
        target_dim = ((H - 1) // 128 + 1) * 128
        padding_amount = target_dim - self.rotary_dim
        pad_width = ((0, 0), (0, padding_amount))
        cache_padded = jnp.pad(cache, pad_width, mode='constant')
        desired_layout = Layout(major_to_minor=(1, 0))
        cache_padded = with_layout_constraint(cache_padded, desired_layout)
        return cache_padded

    def apply_rope(self, positions: jax.Array, x_TNH: jax.Array):
        assert x_TNH.ndim == 3
        assert self.sin_cos_cache is not None, "RoPE cache not initialized."
        cos_sin_padded = self.sin_cos_cache[positions]
        cos_sin_TH = cos_sin_padded[:, :self.rotary_dim]
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


@dataclass(kw_only=True)
class GptOssRotaryEmbedding(nnx.Module):
    """
    JAX implementation of the Rotary Positional Embedding with YaRN scaling.
    """
    head_dim: int
    rope_theta: float
    dtype: jnp.dtype
    initial_context_length: int = 4096
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    def _compute_concentration_and_inv_freq(self) -> Tuple[float, jax.Array]:
        """
        Computes the inverse frequencies and concentration factor for YaRN.
        See YaRN paper: https://arxiv.org/abs/2309.00071
        """
        freq = self.rope_theta**(
            jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)

        if self.rope_scaling_factor > 1.0:
            concentration = 0.1 * jnp.log(self.rope_scaling_factor) + 1.0

            d_half = self.head_dim / 2
            # NTK by parts
            low = (d_half * jnp.log(self.initial_context_length /
                                    (self.rope_ntk_beta * 2 * jnp.pi)) /
                   jnp.log(self.rope_theta))
            high = (d_half * jnp.log(self.initial_context_length /
                                     (self.rope_ntk_alpha * 2 * jnp.pi)) /
                    jnp.log(self.rope_theta))

            interpolation = 1.0 / (self.rope_scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
            mask = 1 - jnp.clip(ramp, 0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self,
                         positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Computes cosine and sine embeddings for given positions."""
        concentration, inv_freq_H = self._compute_concentration_and_inv_freq()

        # freqs: (T, H/2)
        freqs = jnp.einsum("T,H->TH",
                           positions.astype(jnp.float32),
                           inv_freq_H,
                           precision=jax.lax.Precision.HIGHEST)

        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration
        return cos, sin

    def __call__(self, query_TNH: jax.Array, key_TNH: jax.Array,
                 positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Applies rotary embeddings to query and key tensors.
        Args:
            query_TNH: Query tensor with shape (num_tokens, num_heads, head_dim)
            key_TNH: Key tensor with shape (num_tokens, num_kv_heads, head_dim)
            positions: A 1D array of token positions.
        """
        # cos, sin: (T, H/2)
        cos_TH, sin_TH = self._compute_cos_sin(positions)

        # Reshape for broadcasting: (T, 1, H/2)
        cos_T1H = cos_TH[:, None, :]
        sin_T1H = sin_TH[:, None, :]

        def _apply_rotation(x_TNH: jax.Array) -> jax.Array:
            # Split the last dimension
            first_half, second_half = jnp.split(x_TNH, 2, axis=-1)

            # Apply rotation
            rotated_x = jnp.concatenate([
                first_half * cos_T1H - second_half * sin_T1H,
                second_half * cos_T1H + first_half * sin_T1H
            ],
                                        axis=-1)
            return rotated_x.astype(self.dtype)

        rotated_query = _apply_rotation(query_TNH)
        rotated_key = _apply_rotation(key_TNH)

        return rotated_query, rotated_key
