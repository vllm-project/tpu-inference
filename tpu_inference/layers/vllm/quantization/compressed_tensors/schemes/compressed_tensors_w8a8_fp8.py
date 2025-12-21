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
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    CompressedTensorsW8A8Fp8
from vllm.model_executor.layers.quantization.utils.w8a8_utils import \
    per_tensor_dequantize

from tpu_inference.layers.vllm.linear_common import (
    sharded_quantized_matmul, slice_sharded_tensor_for_concatenation,
    torch_to_jax_param)
from tpu_inference.layers.vllm.quantization.common import JaxCommonLinearConfig

P = PartitionSpec


def requantize_with_max_scale(
        weight: torch.Tensor, weight_scale: torch.Tensor,
        logical_widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = weight.dtype
    dtype_info = torch.finfo(dtype)
    maxval = float(dtype_info.max)
    minval = float(dtype_info.min)

    max_w_scale = weight_scale.max()

    unfused_module_in_checkpoint = (weight_scale[-1]
                                    > torch.finfo(torch.float8_e4m3fn).min)

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            # Skip any component with zero width.
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :],
                                              weight_scale[idx])
            weight_q = weight_dq / max_w_scale
            weight[start:end, :] = weight_q.clamp(minval, maxval).to(dtype)
            start = end

    return max_w_scale, weight


class VllmCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        is_static_input_scheme: bool,
        jax_config: JaxCommonLinearConfig,
    ):
        super().__init__(weight_quant, is_static_input_scheme)

        self.jax_config = jax_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale

        if self.is_static_input_scheme:
            # In static quant, all input_scales share the same value.
            assert layer.input_scale.min() == layer.input_scale.max()
            input_scale_first = layer.input_scale[0]

            input_scale = jax.device_put(
                t2j(input_scale_first, use_dlpack=False),
                NamedSharding(self.jax_config.mesh, P()))
            input_scale = torch.nn.Parameter(torch_view(input_scale),
                                             requires_grad=False)
            delattr(layer, "input_scale")
            layer.input_scale = input_scale

            # TODO(kyuyeunk): Investigate performance gain from merging scales.
            # By merging input and weight scales, we reduce the number of muls
            # required for dequantization from 2 (for each scales) to 1.
            # weight_scale *= input_scale_first

        if self.strategy == QuantizationStrategy.TENSOR:
            weight_scale, weight = requantize_with_max_scale(
                weight, weight_scale, self.jax_config.output_sizes)
            weight_scale = jax.device_put(
                t2j(weight_scale, use_dlpack=False),
                NamedSharding(self.jax_config.mesh, P()))
            weight_scale = torch.nn.Parameter(torch_view(weight_scale),
                                              requires_grad=False)
        else:
            weight_scale = weight_scale.squeeze(-1)
            weight_scale = torch_to_jax_param(
                weight_scale,
                NamedSharding(self.jax_config.mesh,
                              self.jax_config.bias_sharding),
                self.jax_config.output_sizes, self.jax_config.n_shards,
                self.jax_config.fuse_matmuls)
        delattr(layer, "weight_scale")
        layer.weight_scale = weight_scale

        weight = torch_to_jax_param(
            layer.weight,
            NamedSharding(self.jax_config.mesh,
                          self.jax_config.weight_sharding),
            self.jax_config.output_sizes, self.jax_config.n_shards,
            self.jax_config.fuse_matmuls)
        delattr(layer, "weight")
        layer.weight = weight

        if layer.bias is not None:
            bias = torch_to_jax_param(
                layer.bias,
                NamedSharding(self.jax_config.mesh,
                              self.jax_config.bias_sharding),
                self.jax_config.output_sizes, self.jax_config.n_shards,
                self.jax_config.fuse_matmuls)
            delattr(layer, "bias")
            layer.bias = bias

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.jax_config.fuse_matmuls:
                return self._apply_fused(layer, x, bias)
            else:
                return self._apply_split(layer, x, bias)

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)
        weight_scale_jax = jax_view(layer.weight_scale)

        if self.is_static_input_scheme:
            # TODO(kyuyeunk): Add kernel support for static quant
            input_scale = jax_view(layer.input_scale)
            dtype_info = jnp.finfo(weight_jax.dtype)
            maxval = float(dtype_info.max)
            minval = float(dtype_info.min)
            x_q = jnp.clip(x_jax / input_scale.astype(x_jax.dtype), minval,
                           maxval).astype(weight_jax.dtype)

            outs = jax.lax.dot_general(
                x_q,
                weight_jax,
                (((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32,
            )
            outs *= weight_scale_jax
            outs = outs.astype(x_jax.dtype)
        else:
            outs = sharded_quantized_matmul(x_jax, weight_jax,
                                            weight_scale_jax,
                                            self.jax_config.mesh,
                                            self.jax_config.weight_sharding)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.jax_config.output_sizes, self.jax_config.n_shards)
        return torch_view(jnp.concatenate(outs, axis=-1))

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (weight, weight_scale) in enumerate(
                zip(layer.weight, layer.weight_scale)):
            weight_jax = jax_view(weight)
            weight_scale_jax = jax_view(weight_scale)

            if self.is_static_input_scheme:
                # TODO(kyuyeunk): Add kernel support for static quant
                input_scale = jax_view(layer.input_scale)
                dtype_info = jnp.finfo(weight_jax.dtype)
                maxval = float(dtype_info.max)
                minval = float(dtype_info.min)
                x_q = jnp.clip(x_jax / input_scale.astype(x_jax.dtype), minval,
                               maxval).astype(weight_jax.dtype)

                out = jax.lax.dot_general(
                    x_q,
                    weight_jax,
                    (((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                # TODO(kyuyeunk): Investigate performance gain from merging scales.
                # out *= weight_scale_jax
                out *= weight_scale_jax * input_scale
                out = out.astype(x_jax.dtype)
            else:
                out = sharded_quantized_matmul(x_jax, weight_jax,
                                               weight_scale_jax,
                                               self.jax_config.mesh,
                                               self.jax_config.weight_sharding)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
