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
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    CompressedTensorsW8A8Fp8

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.common.quantization.fp8 import \
    process_blockwise_fp8_linear_weights
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


class VllmCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        is_static_input_scheme: bool,
        linear_config: VllmQuantLinearConfig,
    ):
        # Per https://github.com/vllm-project/vllm/pull/32929,
        # init_fp8_linear_kernel is now called by super().__init__
        # but does not support TPU backends as expected.
        # use_marlin was also changed to be determined via isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel).
        # We need to monkeypatch init_fp8_linear_kernel and explicitly set use_marlin = True
        # in order to bypass using native vLLM's vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py:scaled_quantize.
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
            compressed_tensors_w8a8_fp8 as vllm_ct_fp8
        vllm_ct_fp8.init_fp8_linear_kernel = lambda *args, **kwargs: None

        CompressedTensorsW8A8Fp8.__init__(
            self,
            weight_quant=weight_quant,
            is_static_input_scheme=is_static_input_scheme)
        self.use_marlin = True

        self.linear_config = linear_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Per https://github.com/vllm-project/vllm/pull/33892, use_marlin is set again
        # in vllm/model_executor/layers/quantization/fp8.py `create_weights`.
        # The flag is set on whether a specific type of GPU kernel is being used which
        # means it is set to False for TPU.
        # We need to return it back to True here.
        super().create_weights(layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs)
        self.use_marlin = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = t2j(layer.weight, use_dlpack=False)
        delattr(layer, "weight")
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        per_tensor = self.strategy == QuantizationStrategy.TENSOR

        @jax.jit
        def process_fp8_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            if per_tensor:
                weights = []
                start = 0
                # Multiple weights may have been concatenated. Loop through
                # each weight and perform dequantization.
                for i, output_size in enumerate(
                        self.linear_config.output_sizes):
                    end = start + output_size
                    weights.append(
                        dequantize_tensor(weight[start:end], weight_scale[i]))
                    start = end
                weight = jnp.concat(weights, axis=0)
                # Requantize into per-tensor.
                weight, weight_scale = quantize_tensor(jnp.float8_e4m3fn,
                                                       weight, None)
            else:
                # Handle 2D blockwise scales correctly for quantized_matmul_kernel
                # which expects 3D dims for blockwise ops.
                # In standard per-channel quantization, the weight_scale is
                # typically 2D with a singleton dimension which can be safely
                # squeezed to 1D. However, in blockwise quantization, the
                # scale is truly 2D.
                if weight_scale.ndim == 2:
                    weight_scale = jnp.squeeze(weight_scale)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                per_tensor=per_tensor,
            )

        if self.weight_block_size is not None:
            weights = process_blockwise_fp8_linear_weights(
                weight=weight,
                weight_scale=weight_scale,
                bias=bias,
                weight_block_size=tuple(self.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards)
        else:
            weights = process_fp8_linear_weights(weight, weight_scale, bias)

        if self.linear_config.enable_quantized_matmul_kernel and weights.weight_scale.ndim == 2:
            if weights.weight_scale.shape[
                    0] > 1 and weights.weight_scale.shape[1] > 1:
                # The quantized_matmul_kernel expects weight scales shaped (n_blocks, 1, n_out_features) for blockwisze quantization.
                if self.weight_block_size is not None:
                    # process_blockwise_fp8_linear_weights returns weight_scale shaped (n_out_features, n_blocks)
                    # We need it shaped (n_blocks, 1, n_out_features)
                    weights.weight_scale = jnp.expand_dims(
                        jnp.transpose(weights.weight_scale),
                        axis=1,
                    )
                else:
                    weights.weight_scale = jnp.expand_dims(
                        jnp.transpose(weights.weight_scale),
                        axis=1,
                    )

        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
                per_tensor=per_tensor,
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
            outs = sharded_quantized_matmul(x_jax,
                                            weight_jax,
                                            weight_scale_jax,
                                            self.linear_config.weight_sharding,
                                            mesh=self.linear_config.mesh)

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
                    x_jax,
                    weight_jax,
                    weight_scale_jax,
                    self.linear_config.weight_sharding,
                    mesh=self.linear_config.mesh)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
