from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from tpu_commons.models.jax.param_init import sharding_init

INT8_ABSMAX_SCALER = 127


class Int8Einsum(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh
    hidden_dim: int

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        # N: num_heads
        # D: hidden_size
        # H: head_dim

        # [num_heads, hidden_size, head_dim] for qkv_proj (hidden_dim=1).
        # [num_heads, head_dim, hidden_size] for o_proj (hidden_dim=2).
        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            self.shape,
            jnp.int8,
        )
        if self.hidden_dim == 1:
            # qkv_proj: Dimension matches num_heads * head_dim.
            s = self.param(
                "SCB",
                sharding_init((None, ), self.mesh, use_constant=True),
                (self.shape[0] * self.shape[2], ),
                self.dtype,
            )
        else:  # self.hidden_dim == 2
            # o_proj: Dimension matches hidden_size.
            s = self.param(
                "SCB",
                sharding_init((None, ), self.mesh, use_constant=True),
                (self.shape[self.hidden_dim], ),
                self.dtype,
            )

        if self.hidden_dim == 1:
            scaled_weight = (w *
                             jnp.reshape(s,
                                         (self.shape[0], 1, self.shape[2])) /
                             INT8_ABSMAX_SCALER)
        else:
            scaled_weight = w * jnp.expand_dims(s, (0, 1)) / INT8_ABSMAX_SCALER
        return jnp.einsum(eqn, x, scaled_weight)


class Int8MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype
    mesh: Mesh
    act: str

    @nn.compact
    def __call__(self, x) -> jax.Array:
        # D: hidden_size
        # I: intermediate_size

        gate_proj_w = self.param(
            "gate_proj_weight",
            sharding_init((None, "model"), self.mesh, use_constant=True),
            (self.hidden_size, self.intermediate_size),
            jnp.int8,
        )
        gate_proj_s = self.param(
            "gate_proj_SCB",
            sharding_init((None, ), self.mesh, use_constant=True),
            (self.intermediate_size, ),
            self.dtype,
        )
        up_proj_w = self.param(
            "up_proj_weight",
            sharding_init((None, "model"), self.mesh, use_constant=True),
            (self.hidden_size, self.intermediate_size),
            jnp.int8,
        )
        up_proj_s = self.param(
            "up_proj_SCB",
            sharding_init((None, ), self.mesh, use_constant=True),
            (self.intermediate_size, ),
            self.dtype,
        )
        down_proj_w = self.param(
            "down_proj_weight",
            sharding_init(("model", None), self.mesh, use_constant=True),
            (self.intermediate_size, self.hidden_size),
            jnp.int8,
        )
        down_proj_s = self.param(
            "down_proj_SCB",
            sharding_init(("model", ), self.mesh, use_constant=True),
            (self.hidden_size, ),
            self.dtype,
        )

        gate_proj = gate_proj_w * jnp.expand_dims(gate_proj_s,
                                                  0) / INT8_ABSMAX_SCALER
        up_proj = up_proj_w * jnp.expand_dims(up_proj_s,
                                              0) / INT8_ABSMAX_SCALER
        down_proj = down_proj_w * jnp.expand_dims(down_proj_s,
                                                  0) / INT8_ABSMAX_SCALER

        gate = jnp.dot(x, gate_proj)
        gate = modeling_flax_utils.ACT2FN[self.act](gate)
        up = jnp.dot(x, up_proj)
        fuse = gate * up
        outputs = jnp.dot(fuse, down_proj)
        return outputs
