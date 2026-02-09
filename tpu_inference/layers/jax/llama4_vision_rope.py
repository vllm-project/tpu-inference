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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(kw_only=True)
class Llama4VisionRotaryEmbedding(nnx.Module):
    """
    Calculates and stores the 2D Rotary Positional Embedding (RoPE) frequencies
    with Float32 precision to match PyTorch/HF reference values.
    """
    image_size: int
    patch_size: int
    hidden_size: int
    num_attention_heads: int
    rope_theta: float = 10000.0
    dtype: jnp.dtype = jnp.bfloat16

    def __post_init__(self):
        # 1. Setup Grid Dimensions
        idx = self.image_size // self.patch_size
        num_patches = idx**2

        # 2. Create 2D Position Grid
        # Shape: (num_patches, 1)
        img_idx = jnp.arange(num_patches,
                             dtype=jnp.int32).reshape(num_patches, 1)

        # Add the CLS token
        img_idx = jnp.concatenate([img_idx, img_idx[:1]], axis=0)

        # Determine X and Y coordinates
        frequencies_x = img_idx % idx
        frequencies_y = img_idx // idx

        frequencies_x = frequencies_x.at[-1, -1].set(-2)
        frequencies_y = frequencies_y.at[-1, -1].set(-2)

        # 3. Calculate Inverse Frequencies
        freq_dim = self.hidden_size // self.num_attention_heads // 2

        t_indices = jnp.arange(0, freq_dim, 2,
                               dtype=jnp.float32)[:(freq_dim // 2)]
        inv_freq = 1.0 / (self.rope_theta**(t_indices / freq_dim))

        # 4. Create Frequency Bands
        freqs_x = (frequencies_x + 1).astype(jnp.float32) * inv_freq[None, :]
        freqs_y = (frequencies_y + 1).astype(jnp.float32) * inv_freq[None, :]

        # Repeat interleaving to match Complex number format (Real, Imag)
        freqs_x = jnp.repeat(freqs_x, 2, axis=-1)
        freqs_y = jnp.repeat(freqs_y, 2, axis=-1)

        # 5. Concatenate and Format
        freqs = jnp.concatenate([freqs_x, freqs_y], axis=-1)

        mask_cond = img_idx < 0
        freqs = jnp.where(mask_cond, 0.0, freqs)

        # 6. Construct Complex Rotary Embeddings
        freqs_rad = freqs[..., ::2]

        cos_freqs = jnp.cos(freqs_rad)
        sin_freqs = jnp.sin(freqs_rad)

        # Stack to (Seq, Dim/2, 2) which represents (Real, Imag)
        # This matches the shape expected by apply_rope: (S, D_rot, 2)
        freqs_cis_stacked = jnp.stack([cos_freqs, sin_freqs], axis=-1)

        # Store as parameter - Cast to model dtype only at the very end
        self.freqs_cis_stacked = freqs_cis_stacked.astype(jnp.float32)

    def __call__(self) -> jax.Array:
        return self.freqs_cis_stacked