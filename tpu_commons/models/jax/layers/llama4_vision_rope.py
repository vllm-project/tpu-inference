from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


# This is the class that generates the static 2D positional frequencies
@dataclass(kw_only=True)
class Llama4VisionRotaryEmbedding(nnx.Module):
    """
    Calculates and stores the 2D Rotary Positional Embedding (RoPE) frequencies
    based on the Vision Transformer (ViT) patch grid structure.
    """
    image_size: int
    patch_size: int
    hidden_size: int
    num_attention_heads: int
    rope_theta: float = 10000.0
    dtype: jnp.dtype = jnp.bfloat16
    rngs: nnx.Rngs

    # Stored complex frequency tensor
    freqs_ci: jnp.ndarray = nnx.field(init=False)

    def __post_init__(self):
        # Calculate the size of the patch grid (e.g., 336 / 14 = 24)
        idx = self.image_size // self.patch_size
        num_patches = idx * idx

        # Calculate head_dim
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Total number of tokens: Patches + 1 CLS token
        total_tokens = num_patches + 1

        # 1. Create 1D index including the CLS token
        # The CLS token is handled at index 0 in the PyTorch reference, but often at the end in JAX.
        # We model the grid coordinates for the patches (idx*idx) and add a dummy position for the CLS token.

        # --- COORDINATE GENERATION ---

        # Generate 1D coordinates (0 to num_patches - 1)
        patch_indices_1d = jnp.arange(num_patches, dtype=jnp.int32)

        # Convert to 2D coordinates (x and y)
        frequencies_x = patch_indices_1d % idx  # X coordinate (width)
        frequencies_y = patch_indices_1d // idx  # Y coordinate (height)

        # Add a coordinate placeholder for the CLS token (e.g., coordinate -1, or 0)
        # We set CLS token index to -1 as per HF reference to exclude it from rotation factor calculation
        cls_token_idx = jnp.array([-1], dtype=jnp.int32)

        # Create full coordinate arrays including CLS placeholder (total_tokens, 1)
        freqs_x_full = jnp.concatenate([frequencies_x, cls_token_idx],
                                       axis=0)[:, jnp.newaxis]
        freqs_y_full = jnp.concatenate([frequencies_y, cls_token_idx],
                                       axis=0)[:, jnp.newaxis]

        # --- FREQUENCY CALCULATION ---

        # RoPE applies to half the head dimension. We need two sets (for x and y).
        freq_dim_per_coord = self.head_dim // 2

        # Calculate the inverse frequencies (timescale) shared by x and y
        # We split the head_dim/2 across x and y (i.e., head_dim/4 for each coord if concatenated)
        inv_freq = 1.0 / (self.rope_theta**(
            jnp.arange(0, freq_dim_per_coord, 2).astype(jnp.float32) /
            freq_dim_per_coord))

        # We must repeat the inv_freq pattern to fill the required dimensions for rotation
        # The HF code uses concatenation and repeats for X and Y features.

        # 1. Calculate frequencies for X and Y dimensions
        freqs_x = freqs_x_full * inv_freq[jnp.newaxis, :]
        freqs_y = freqs_y_full * inv_freq[jnp.newaxis, :]

        # 2. Concatenate X and Y frequencies to form the final frequency list
        # Resulting shape: (total_tokens, head_dim // 2)
        final_freqs = jnp.concatenate([freqs_x, freqs_y], axis=1)

        # 3. Create complex rotation factors (cos(theta) + i*sin(theta))
        cos_freqs = jnp.cos(final_freqs)
        sin_freqs = jnp.sin(final_freqs)

        # The complex format is complex_number = real_part + imag_part * i
        # We stack the real (cos) and imaginary (sin) parts to pass to jnp.complex64 later.
        # Shape: (total_tokens, head_dim/2, 2)
        freqs_cis_stacked = jnp.stack([cos_freqs, sin_freqs], axis=-1)

        # Mask out the CLS token frequency (where freqs_x_full < 0)
        cls_mask = freqs_x_full < 0
        cls_mask = cls_mask[:, jnp.newaxis, :]
        freqs_cis_stacked = jnp.where(cls_mask, 0, freqs_cis_stacked)

        # Store as parameter/attribute for checkpointing
        self.freqs_ci = freqs_cis_stacked.astype(self.dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Returns the pre-calculated complex rotation factors."""
        # The input x is needed only to get the batch dimensions if necessary, but here
        # we return the static frequencies calculated in __init__.

        # The final rotation array must be expanded to handle the batch dimension of x.
        # x is typically (B, S, D). freqs_ci is (S, D_rot)

        # We return the static frequencies. The calling attention layer (Llama4VisionAttention)
        # is responsible for handling batch and head dimensions.
        return self.freqs_ci
