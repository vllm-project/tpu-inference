# Copyright 2026 Google LLC
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
"""Shared gated feed-forward network.

Extracted from the DeepSeek-V3 model file (``DeepseekV3MLP``) so the
MLA/MoE-family models can share one implementation. Beyond DeepSeek-V3 this
serves the Kimi-Linear / Kimi-K3 family, which needs the ``situ`` activation
(soft-clamps the up branch as well as the gate branch) and per-sub-module
quantization prefixes.
"""

from dataclasses import InitVar, dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.activation import (SITU_BETA, SITU_LINEAR_BETA,
                                                 apply_gated_activation)
from tpu_inference.layers.jax.base import sharded_initializer
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


@dataclass(kw_only=True)
class GatedMLP(JaxModule):
    """A gated feed-forward network: ``down(act(gate(x), up(x)))``.

    Attributes:
        hidden_act: Name of the gated activation. ``situ`` uses both
            ``situ_beta`` and ``situ_linear_beta``; every other activation is
            unary on the gate branch.
        prefix: Quantization prefix. Empty (the DeepSeek-V3 default) leaves the
            sub-modules unprefixed, matching the pre-extraction behaviour.
    """
    dtype: jnp.dtype
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    df_sharding: P = P()
    fd_sharding: P = P()
    activation_ffw_td: P = P()
    random_init: bool = False
    quant_config: Optional[QuantizationConfig] = None
    situ_beta: float = SITU_BETA
    situ_linear_beta: float | None = SITU_LINEAR_BETA
    prefix: str = ""

    rngs: InitVar[nnx.Rngs]

    def __call__(self, x_TD):
        """Performs the forward pass of the FFW layer.

        Args:
            x_TD: The input tensor of shape either `(sequence, d_model)`

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = lax.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("wi_0"):
            gating_TF = self.gate_proj(x_TD)
        with jax.named_scope("wi_1"):
            up_proj_TF = self.up_proj(x_TD)
        fuse_TF = apply_gated_activation(self.hidden_act, gating_TF,
                                         up_proj_TF, self.situ_beta,
                                         self.situ_linear_beta)
        with jax.named_scope("wo"):
            output_TD = self.down_proj(fuse_TF)

        return output_TD

    def _sub_prefix(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else ""

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size
        weight_init = _weight_init(self.random_init)

        self.gate_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
            prefix=self._sub_prefix("gate_proj"),
        )
        self.up_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
            prefix=self._sub_prefix("up_proj"),
        )
        self.down_proj = JaxEinsum(
            einsum_str="TF,FD->TD",
            kernel_shape=(F, D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.fd_sharding),
            prefix=self._sub_prefix("down_proj"),
        )
