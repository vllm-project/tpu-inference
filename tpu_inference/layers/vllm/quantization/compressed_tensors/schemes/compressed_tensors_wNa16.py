# compressed_tensors_wNa16.py
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

from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import ActivationOrdering
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.quantization import ct_u32_unpack_u4
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec
logger = init_logger(__name__)

WNA16_SUPPORTED_BITS = [4, 8]


class VllmCompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        linear_config: VllmQuantLinearConfig,
        group_size: int | None = None,
        symmetric: bool = True,
        actorder: ActivationOrdering | None = None,
    ):
        self.pack_factor = 32 // num_bits
        self.num_bits = num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.linear_config = linear_config

        if num_bits not in WNA16_SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_BITS}")

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        output_size: int,
        input_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = row_parallel and self.group_size != -1

        scales_and_zp_size = input_size // group_size
        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }
        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args)
        layer.register_parameter("weight_scale", weight_scale)

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            zeros_args = {
                "weight_loader": weight_loader,
                "data": torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    scales_and_zp_size,
                    dtype=torch.int32,
                ),
            }
            if not partition_scales:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            else:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            layer.register_parameter("weight_zero_point", qzeros)

        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.int32),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

    def _unpack_packed_tensor(self, packed: jax.Array) -> jax.Array:
        """Unpack int32-packed weights to individual values."""
        if self.num_bits == 4:
            return ct_u32_unpack_u4(packed)
        u8 = jax.lax.bitcast_convert_type(packed, jnp.uint8)
        return jnp.reshape(u8, u8.shape[:-2] + (-1,))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        sort_indices = None
        if hasattr(layer, "weight_g_idx"):
            g_idx_tensor = layer.weight_g_idx
            if not (g_idx_tensor < 0).any():
                g_idx = t2j(g_idx_tensor.to(torch.int32), use_dlpack=False)
                # Optimization: Sort weights by group at load time
                sort_indices = jnp.argsort(g_idx)
            delattr(layer, "weight_g_idx")

        if hasattr(layer, "weight_shape"):
            delattr(layer, "weight_shape")

        weight = t2j(layer.weight_packed, use_dlpack=False)
        delattr(layer, "weight_packed")

        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        zero_point = None
        if not self.symmetric and hasattr(layer, "weight_zero_point"):
            zero_point = t2j(layer.weight_zero_point, use_dlpack=False)
            delattr(layer, "weight_zero_point")

        bias = None
        if (hasattr(layer, 'bias') and layer.bias is not None
                and not layer.skip_bias_add):
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")

        @jax.jit
        def process_wna16_weights(weight, weight_scale, zero_point, bias, sort_indices):
            unpacked_weight = self._unpack_packed_tensor(weight).astype(jnp.int8)

            if sort_indices is not None:
                unpacked_weight = unpacked_weight[:, sort_indices]
                
            unpacked_zp = None
            if zero_point is not None:
                unpacked_zp = self._unpack_packed_tensor(zero_point.T).T.astype(jnp.int8)

            return process_linear_weights(
                LinearWeights(
                    weight=unpacked_weight,
                    weight_scale=weight_scale,
                    zero_point=unpacked_zp,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_wna16_weights(weight, weight_scale, zero_point, bias, sort_indices)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        # Reassign to new attribute names to avoid shape mismatch issues 
        # (since original tensors were int32 packed, and these are now int8 unpacked)
        if self.linear_config.fuse_matmuls:
            layer.weight_unpacked = Parameter(weights.weight, requires_grad=False)
            if weights.weight_scale is not None:
                layer.weight_scale = Parameter(weights.weight_scale, requires_grad=False)
            if weights.zero_point is not None:
                layer.weight_zero_point_unpacked = Parameter(weights.zero_point, requires_grad=False)
            if weights.bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight_unpacked = to_parameter_list(weights.weight)
            if weights.weight_scale is not None:
                layer.weight_scale = to_parameter_list(weights.weight_scale)
            if weights.zero_point is not None:
                layer.weight_zero_point_unpacked = to_parameter_list(weights.zero_point)
            if weights.bias is not None:
                layer.bias = to_parameter_list(weights.bias)
                
        # Save sort_indices to permute the input 'x' in the forward pass
        if sort_indices is not None:
            layer.sort_indices = Parameter(torch_view(sort_indices), requires_grad=False)

    def _dequantize_to_bf16(self, weight_unpacked: jax.Array, weight_scale: jax.Array, 
                           zero_point_unpacked: Optional[jax.Array]) -> jax.Array:
        """Standard sequential group dequantization directly from unpacked int8 tensors."""
        out_features, in_features = weight_unpacked.shape
        w_bf16 = weight_unpacked.astype(jnp.bfloat16)

        effective_gs = in_features if self.group_size == -1 else self.group_size
        num_groups = in_features // effective_gs

        # Reshape to expose the groups: (out_features, num_groups, group_size)
        w_grouped = w_bf16.reshape((out_features, num_groups, effective_gs))
        scale_expanded = jnp.expand_dims(weight_scale, -1)

        if zero_point_unpacked is not None:
            zp_bf16 = jnp.expand_dims(zero_point_unpacked.astype(jnp.bfloat16), -1)
            w_deq_grouped = (w_grouped - zp_bf16) * scale_expanded
        else:
            # Shift by implicit offset if symmetric
            offset = jnp.array(1 << (self.num_bits - 1), dtype=jnp.bfloat16)
            w_deq_grouped = (w_grouped - offset) * scale_expanded
        
        return w_deq_grouped.reshape((out_features, in_features))

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
        
        # If we permuted weights at load time, we must permute x at runtime
        if hasattr(layer, "sort_indices"):
            sort_idx = jax_view(layer.sort_indices)
            x_jax = x_jax[..., sort_idx]

        weight_unpacked = jax_view(layer.weight_unpacked)
        w_scale = jax_view(layer.weight_scale)
        zp = jax_view(layer.weight_zero_point_unpacked) if hasattr(layer, "weight_zero_point_unpacked") else None
        
        weight_bf16 = self._dequantize_to_bf16(weight_unpacked, w_scale, zp)
        outs = jnp.einsum("bd,fd->bf", x_jax, weight_bf16)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes,
            self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight_unpacked, torch.nn.ParameterList)
        x_jax = jax_view(x)
        
        if hasattr(layer, "sort_indices"):
            sort_idx = jax_view(layer.sort_indices)
            x_jax = x_jax[..., sort_idx]

        outs = []
        for i in range(len(layer.weight_unpacked)):
            weight_unpacked = jax_view(layer.weight_unpacked[i])
            w_scale = jax_view(layer.weight_scale[i])
            zp = jax_view(layer.weight_zero_point_unpacked[i]) if hasattr(layer, "weight_zero_point_unpacked") else None
            
            weight_bf16 = self._dequantize_to_bf16(weight_unpacked, w_scale, zp)
            out = jnp.einsum("bd,fd->bf", x_jax, weight_bf16)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)

        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)