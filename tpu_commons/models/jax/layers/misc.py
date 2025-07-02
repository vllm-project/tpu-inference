import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.layers.params import sharding_init


def shard_put(x: jax.Array, sharding_names: Tuple[str, ...] | P,
              mesh: jax.sharding.Mesh) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, jax.devices()[0])
    return jax.device_put(x, NamedSharding(mesh, P(*sharding_names)))


class Einsum(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )
        return jnp.einsum(eqn, x, w)


class EinsumBias(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh

    # We need this because not every EinsumBias usage is for qkv_proj.
    bias_shape: Optional[Tuple[int, ...]] = None
    bias_named_axes: Optional[Tuple[str, ...]] = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        bias_shape = self.bias_shape or (
            self.shape[0],
            self.shape[2],
        )  # (num_heads or num_kv_heads, head_dim)
        bias_named_axes = self.bias_named_axes or ("model", None)

        b = self.param(
            "bias",
            sharding_init(bias_named_axes, self.mesh),
            bias_shape,
            self.dtype,
        )

        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )

        return jnp.einsum(eqn, x, w), b


class Embedder(nn.Module):
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        self.input_embedding_table = self.param(
            "weight",
            sharding_init(
                ("model", None),
                self.mesh,
            ),
            (self.vocab_size, self.hidden_size),
            self.dtype,
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x, )]
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)


class RMSNorm(nn.Module):
    rms_norm_eps: float
    dtype: jnp.dtype
    mesh: Mesh

    @nn.compact
    def __call__(self, x) -> jax.Array:
        scale = self.param(
            "weight",
            sharding_init((None, ), self.mesh),
            (x.shape[-1], ),
            self.dtype,
        )
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(
            x * jnp.reciprocal(jnp.sqrt(var + self.rms_norm_eps)))
        normed_inputs = normed_inputs * scale
        return normed_inputs
