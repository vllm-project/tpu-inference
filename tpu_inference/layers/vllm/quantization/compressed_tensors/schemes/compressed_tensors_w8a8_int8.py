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

from typing import Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationStrategy
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import \
    CompressedTensorsW8A8Int8
from vllm.model_executor.layers.quantization.utils.w8a8_utils import \
    convert_to_channelwise

from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.linear import sharded_quantized_matmul
from tpu_inference.layers.vllm.process_weights.linear_weights import (
    LinearWeights, process_lienar_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger

P = PartitionSpec
logger = init_logger(__name__)


class VllmCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8):

    def __init__(self, strategy: str, is_static_input_scheme: bool,
                 input_symmetric: bool, linear_config: VllmQuantLinearConfig):
        super().__init__(strategy, is_static_input_scheme, input_symmetric)

        self.linear_config = linear_config
        self.is_channelwise = (self.strategy == QuantizationStrategy.CHANNEL)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = t2j(layer.weight, use_dlpack=False)
        delattr(layer, "weight")

        weight_scale = layer.weight_scale
        is_fused_module = len(layer.logical_widths) > 1
        if is_fused_module and not self.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale,
                                                  layer.logical_widths)
        weight_scale = weight_scale.squeeze(-1)
        weight_scale = t2j(weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_int8_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            return process_lienar_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_int8_linear_weights(weight, weight_scale, bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            layer.weight_scale = Parameter(weights.weight_scale,
                                           requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            layer.weight_scale = to_parameter_list(weights.weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

        # TODO(kyuyeunk): Support static range input quantization.
        assert getattr(layer, "input_scale", None) is None
        assert getattr(layer, "input_zero_point", None) is None
        assert getattr(layer, "azp_adj", None) is None

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)
        weight_scale_jax = jax_view(layer.weight_scale)

        outs = sharded_quantized_matmul(
            x_jax,
            weight_jax,
            weight_scale_jax,
            self.linear_config.mesh,
            self.linear_config.weight_sharding,
        )
        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (weight, weight_scale) in enumerate(
                zip(layer.weight, layer.weight_scale)):
            weight_jax = jax_view(weight)
            weight_scale_jax = jax_view(weight_scale)

            out = sharded_quantized_matmul(
                x_jax,
                weight_jax,
                weight_scale_jax,
                self.linear_config.mesh,
                self.linear_config.weight_sharding,
            )
            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)
