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

from typing import Optional, Sequence

import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights)
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.utils import get_mesh_shape_product


class UnquantizedLinearMethod:
    """Implements the forward method for unquantized linear layers.

    This class will be shared in both vLLM and jax path.
    """

    def __init__(self, linear_config: QuantLinearConfig):
        self.linear_config = linear_config

    def _apply_fused(self,
                     x_jax: jax.Array,
                     weight_jax: jax.Array,
                     bias_jax: Optional[jax.Array] = None,
                     einsum_str: str = "mn,pn->mp") -> jax.Array:
        outs = jnp.einsum(einsum_str, x_jax, weight_jax)
        if bias_jax is not None:
            outs += bias_jax

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return out

    def _apply_split(
            self,
            x_jax: jax.Array,
            weights: Sequence[jax.Array],
            bias_jax: Optional[Sequence[jax.Array]] = None) -> jax.Array:
        outs = []
        for i, weight_jax in enumerate(weights):
            out = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
            if bias_jax is not None:
                out += bias_jax[i]

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return out


@jax.jit(static_argnames=('mesh', 'activation', 'moe_backend'))
def process_unquantized_moe_weights(
    *,
    mesh: Mesh,
    moe_backend: MoEBackend,
    activation: MoEActivation,
    w13_weight: jax.Array,
    w13_bias: jax.Array | None,
    w2_weight: jax.Array,
    w2_bias: jax.Array | None,
) -> FusedMoEWeights:
    """Jit'ed version to process unquantized moe weights. See `process_moe_weights` for details.
    """
    w13_interleave = activation == MoEActivation.SWIGLUOAI
    w13_reorder_size = get_mesh_shape_product(mesh,
                                              ShardingAxisName.MLP_TENSOR)

    return process_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=None,
            w13_bias=w13_bias,
            w2_weight=w2_weight,
            w2_weight_scale=None,
            w2_bias=w2_bias,
        ),
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
    )
