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
"""JAX-native int8 W8A8 (compressed-tensors ``int-quantized``) linear method.

Supports the llm-compressor / RedHatAI-style scheme:
  - weights: int8, symmetric, static, per-channel scale (``weight_scale``)
  - activations: int8, symmetric, dynamic per-token (quantized at runtime)

The numeric path reuses ``sharded_quantized_matmul`` ->
``xla_quantized_matmul``, which already handles integer weight dtypes
(int32 accumulation + per-token activation quantization).

NOTE: this class intentionally mirrors ``Fp8TensorwiseLinearMethod``
(layers/jax/quantization/fp8.py) with the weight dtype swapped to int8.
Kept as a standalone copy for now to leave the fp8 path untouched; when
upstreaming, extract a shared channelwise base class (fp8-tensorwise /
int8 family) instead of inheriting across dtypes. fp4/mx formats are a
different family (packed storage + block scales) and should not share
this base.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.linear import sharded_quantized_batched_matmul
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantLinearConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import \
    load_nnx_param_from_reshaped_torch

logger = init_logger(__name__)


class Int8ChannelwiseLinearMethod(QuantizeMethodBase,
                                  common_fp8.Fp8LinearMethod):
    """int8 W8A8 (channelwise weight scale, dynamic per-token acts)."""

    weight_dtype = jnp.int8

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)

        self.einsum_str = layer.einsum_str

        self.output_shape = linear_config.out_features
        self.batch_features = linear_config.batch_features
        self.batch_sharding = linear_config.batch_sharding
        out_features = math.prod(self.output_shape)
        in_features = math.prod(linear_config.in_features)
        self.weight_sharding = linear_config.weight_sharding
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (in_features, out_features)

        self.in_features = in_features

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)

        layer.weight = create_param(rngs,
                                    shape=self.kernel_shape,
                                    dtype=self.weight_dtype,
                                    sharding=self.weight_sharding)

        layer.weight.set_metadata(
            'weight_loader',
            partial(load_nnx_param_from_reshaped_torch,
                    permute_dims=(1, 0),
                    param_name=layer.prefix + ".weight"))

        # compressed-tensors serializes the channelwise dequant scale as
        # "weight_scale" with shape [out, 1]; reshape to 1D on load.
        scale_sharding = None
        if self.batch_features:
            if self.batch_sharding:
                scale_sharding = None  # replicated scale for simplicity
        elif isinstance(self.weight_sharding, P) and len(
                self.weight_sharding) > 0:
            scale_sharding = P(self.weight_sharding[0])
        elif isinstance(self.weight_sharding,
                        (tuple, list)) and len(self.weight_sharding) > 0:
            scale_sharding = (self.weight_sharding[0], )

        layer.weight_scale = create_param(rngs,
                                          shape=(out_features, ),
                                          dtype=jnp.float32,
                                          sharding=scale_sharding)
        layer.weight_scale.set_metadata(
            'weight_loader',
            partial(load_nnx_param_from_reshaped_torch,
                    reshape_dims=(out_features, ),
                    permute_dims=None,
                    param_name=layer.prefix + ".weight_scale"))

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        bias = layer.bias[...] if layer.bias is not None else None

        if self.batch_features:
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight[...],
                layer.weight_scale[...],
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            if bias is not None:
                out += bias
            return out

        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        out = self._apply_fused(x,
                                layer.weight[...],
                                layer.weight_scale[...],
                                bias=bias)
        out = out.reshape(out.shape[:-1] + self.output_shape)
        return out
