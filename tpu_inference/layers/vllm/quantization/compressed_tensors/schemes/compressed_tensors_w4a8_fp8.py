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

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationArgs,
                                             QuantizationStrategy)
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8 import (
    W4A8_SUPPORTED_TYPES_MAP, CompressedTensorsW4A8Fp8)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    unpack_quantized_values_into_int32
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.scalar_type import scalar_types

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger

P = PartitionSpec
logger = init_logger(__name__)


class VllmCompressedTensorsW4A8Fp8(CompressedTensorsW4A8Fp8):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        is_static_input_scheme: bool,
        linear_config: VllmQuantLinearConfig,
    ):
        # Skips calling super().__init__() because the parent currently requires
        # group_size = 128, which is often disadvantageous on TPU.
        self.pack_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.num_bits = weight_quant.num_bits
        self.symmetric = weight_quant.symmetric
        self.actorder = weight_quant.actorder
        self.group_size = -1 if weight_quant.group_size is None else weight_quant.group_size
        self.has_g_idx = weight_quant.actorder == ActivationOrdering.GROUP

        if self.num_bits == 4:
            self.wtype = scalar_types.uint4
        else:
            raise ValueError(
                f"Unsupported num_bits = {weight_quant.num_bits}. "
                f"Supported num_bits = {W4A8_SUPPORTED_TYPES_MAP.keys()}")

        if is_static_input_scheme:
            raise NotImplementedError(
                "Static input scheme is not yet supported for W4A8.")

        if not weight_quant.symmetric:
            raise ValueError(
                "Scheme W4A8Fp8 only supports symmetric quantization.")
        self.quant_type = W4A8_SUPPORTED_TYPES_MAP[weight_quant.num_bits]

        self.weight_quant = weight_quant
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        self.linear_config = linear_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition

        partition_scales = self.weight_quant.strategy == "group" and not row_parallel

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        weight_scale_args = {
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader,
        }

        if partition_scales:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        unpacked_weights = unpack_quantized_values_into_int32(
            layer.weight_packed, self.wtype, packed_dim=1)
        uint_weight = t2j(unpacked_weights, use_dlpack=False)
        delattr(layer, "weight_packed")
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if getattr(layer, "bias",
                   None) is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        per_tensor = self.strategy == QuantizationStrategy.TENSOR

        @jax.jit
        def process_uint4_linear_weights(
            uint_weight: jax.Array,
            weight_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            # Convert from uint4 to int4.
            weight = (uint_weight - 8).astype(jnp.int4)

            if weight_scale.shape[-1] == 1:
                weight_scale = jnp.squeeze(weight_scale, -1)

            processed_weights = process_linear_weights(
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

            # W4A8 scale comes as (out_features, num_blocks), but the kernel
            # expects (num_blocks, 1, out_features). We format the weight scales
            # here after processing so that slicing on dim=0 succeeds.
            if isinstance(processed_weights.weight_scale, list):
                raise ValueError("Unexpected weight scale format.")

            if processed_weights.weight_scale is not None and processed_weights.weight_scale.ndim == 2:
                processed_weights.weight_scale = jnp.expand_dims(
                    jnp.transpose(processed_weights.weight_scale, (1, 0)), 1)

            return processed_weights

        weights = process_uint4_linear_weights(uint_weight, weight_scale, bias)
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

        outs = sharded_quantized_matmul(x_jax,
                                        weight_jax,
                                        weight_scale_jax,
                                        self.linear_config.weight_sharding,
                                        mesh=self.linear_config.mesh,
                                        x_q_dtype=jnp.float8_e4m3fn)

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

            out = sharded_quantized_matmul(
                x_jax,
                weight_jax,
                weight_scale_jax,
                self.linear_config.weight_sharding,
                mesh=self.linear_config.mesh,
                x_q_dtype=jnp.float8_e4m3fn,
                acc_dtype=jnp.float32,
            )

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
