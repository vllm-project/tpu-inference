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

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from jax.sharding import NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    CompressedTensorsW8A8Fp8
from vllm.model_executor.layers.quantization.utils.w8a8_utils import \
    per_tensor_dequantize

from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.linear import sharded_quantized_matmul
from tpu_inference.layers.vllm.process_weights.linear_weights import (
    process_lienar_weights, shard_linear_weights, to_parameter_list)
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

P = PartitionSpec

logger = init_logger(__name__)


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
        linear_config: VllmQuantLinearConfig,
    ):
        super().__init__(weight_quant, is_static_input_scheme)

        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # TODO: Move scale processing logic into a consolidated file.
        weight = layer.weight
        weight_scale = layer.weight_scale
        if self.strategy == QuantizationStrategy.TENSOR:
            weight_scale, weight = requantize_with_max_scale(
                weight, weight_scale, self.linear_config.output_sizes)
        else:
            weight_scale = weight_scale.squeeze(-1)

        weight = t2j(weight, use_dlpack=False)
        delattr(layer, "weight")
        weight_scale = t2j(weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        # TODO: Use jit to speedup weight loading.
        # @jax.jit
        def wrapper(weight, weight_scale, bias):
            return process_lienar_weights(
                weight,
                weight_scale,
                None,
                bias,
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        shard_weights = functools.partial(
            shard_linear_weights,
            mesh=self.linear_config.mesh,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )

        # If per-tensor quantization is used, skip processing for scale.
        if self.strategy == QuantizationStrategy.TENSOR:
            weights = wrapper(weight, None, bias)
            weight, _, _, bias = torch_view(shard_weights(weights))
            weight_scale = torch_view(weight_scale)
        else:
            weights = wrapper(weight, weight_scale, bias)
            weight, weight_scale, _, bias = torch_view(shard_weights(weights))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weight)
            layer.weight_scale = to_parameter_list(weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(bias)

        if self.is_static_input_scheme:
            # In static quant, all input_scales share the same value.
            assert layer.input_scale.min() == layer.input_scale.max()
            input_scale_first = layer.input_scale[0]

            input_scale = jax.device_put(
                t2j(input_scale_first, use_dlpack=False),
                NamedSharding(self.linear_config.mesh, P()))
            input_scale = torch.nn.Parameter(torch_view(input_scale),
                                             requires_grad=False)
            delattr(layer, "input_scale")
            layer.input_scale = input_scale

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
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
                                            self.linear_config.mesh,
                                            self.linear_config.weight_sharding)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
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
                out = sharded_quantized_matmul(
                    x_jax, weight_jax, weight_scale_jax,
                    self.linear_config.mesh,
                    self.linear_config.weight_sharding)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
