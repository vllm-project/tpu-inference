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

import functools
import math
from typing import Optional, Sequence

import jax
from jax import numpy as jnp
from jax.sharding import Mesh

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights)
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation


class Fp8LinearMethod:
    """Implements the forward method for fp8 linear layers.

    This class will be shared in both vLLM and jax path.
    """

    def __init__(self, linear_config: QuantLinearConfig):
        self.linear_config = linear_config

    def _apply_fused(self, x: jax.Array, weight_jax: jax.Array,
                     weight_scale_jax: jax.Array,
                     bias: Optional[jax.Array]) -> jax.Array:
        outs = sharded_quantized_matmul(x,
                                        weight_jax,
                                        weight_scale_jax,
                                        self.linear_config.weight_sharding,
                                        mesh=self.linear_config.mesh)

        if bias is not None:
            outs += bias
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return jnp.concatenate(outs, axis=-1)

    def _apply_split(self,
                     x: jax.Array,
                     weight_and_scale: Sequence[tuple[jax.Array, jax.Array]],
                     bias: Optional[Sequence[jax.Array]] = None,
                     mesh: Optional[Mesh] = None) -> jax.Array:

        outs = []
        for i, (weight, weight_scale) in enumerate(weight_and_scale):

            out = sharded_quantized_matmul(x,
                                           weight,
                                           weight_scale,
                                           self.linear_config.weight_sharding,
                                           mesh=mesh)

            if bias is not None:
                out += bias[i]
            outs.append(out)
        return jnp.concatenate(outs, axis=-1)


@functools.partial(jax.jit,
                   static_argnames=('linear_config', 'weight_block_size'))
def process_blockwise_fp8_linear_weights(
    weight: jax.Array,
    weight_scale: jax.Array,
    *,
    bias: jax.Array | None,
    weight_block_size: Sequence[int],
    linear_config,
) -> LinearWeights:
    weights = []
    weight_scales = []
    original_block_size = weight_block_size[0]
    if linear_config.enable_quantized_matmul_kernel:
        if linear_config.requant_block_size:
            requant_block_size = linear_config.requant_block_size
        else:
            requant_block_size = weight.shape[1]
    start = 0
    for output_size in linear_config.output_sizes:
        end = start + output_size

        weight_slice = weight[start:end]
        weight_scale_slice = weight_scale[start // original_block_size:math.
                                          ceil(end / original_block_size)]
        dequantized_weight = dequantize_tensor(
            weight_slice,
            weight_scale_slice,
            (0, 1),
            block_size=weight_block_size,
        )
        weight_slice, weight_scale_slice = quantize_tensor(
            linear_config.requant_weight_dtype,
            dequantized_weight,
            block_size=requant_block_size)

        weights.append(weight_slice)
        weight_scales.append(weight_scale_slice)

        start = end

    weight = jnp.concat(weights, axis=0)
    weight_scale = jnp.concat(weight_scales, axis=0)

    return process_linear_weights(
        LinearWeights(
            weight=weight,
            weight_scale=weight_scale,
            zero_point=None,
            bias=bias,
        ),
        fused=linear_config.fuse_matmuls,
        output_sizes=linear_config.output_sizes,
        reorder_size=linear_config.n_shards,
    )
