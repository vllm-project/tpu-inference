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

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "WNA16 requires group quantization or channelwise "
                "quantization, but found no group size and strategy "
                "is not channelwise.")

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_packed = t2j(layer.weight_packed, use_dlpack=False)
        delattr(layer, "weight_packed")

        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if hasattr(layer, "weight_shape"):
            delattr(layer, "weight_shape")

        g_idx = None
        if hasattr(layer, "weight_g_idx"):
            g_idx_torch = layer.weight_g_idx
            if not (g_idx_torch < 0).any():
                g_idx = t2j(g_idx_torch, use_dlpack=False)
            delattr(layer, "weight_g_idx")

        zero_point = None
        if not self.symmetric and hasattr(layer, "weight_zero_point"):
            zero_point = t2j(layer.weight_zero_point, use_dlpack=False)
            delattr(layer, "weight_zero_point")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        group_size = self.group_size
        symmetric = self.symmetric
        num_bits = self.num_bits
        has_g_idx = g_idx is not None

        if has_g_idx:
            perm = jnp.argsort(g_idx)
            inv_perm = jnp.argsort(perm)
        else:
            perm = None
            inv_perm = None

        @jax.jit
        def process_wna16_linear_weights(
            weight_packed: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array | None,
            bias: jax.Array | None,
            perm: jax.Array | None,
            inv_perm: jax.Array | None,
        ) -> LinearWeights:
            weight = ct_u32_unpack_u4(weight_packed)

            if has_g_idx:
                weight = weight[:, perm]

            input_size = weight.shape[1]
            effective_group_size = input_size if group_size == -1 else group_size
            num_groups = input_size // effective_group_size
            weight = weight.reshape(
                (weight.shape[0], num_groups, effective_group_size))

            if not symmetric and zero_point is not None:
                zero_point = ct_u32_unpack_u4(zero_point)

            scales = jnp.expand_dims(weight_scale, -1)
            weight_f = weight.astype(jnp.bfloat16)
            if not symmetric and zero_point is not None:
                zp = jnp.expand_dims(zero_point.astype(jnp.bfloat16), -1)
                weight_deq = (weight_f - zp) * scales
            else:
                offset = jnp.array(1 << (num_bits - 1), dtype=jnp.bfloat16)
                weight_deq = (weight_f - offset) * scales
            weight_deq = weight_deq.reshape((weight_deq.shape[0], -1))

            if has_g_idx:
                weight_deq = weight_deq[:, inv_perm]

            return process_linear_weights(
                LinearWeights(
                    weight=weight_deq,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_wna16_linear_weights(
            weight_packed, weight_scale, zero_point, bias, perm, inv_perm)

        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

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
        weight = jax_view(layer.weight)

        outs = jnp.einsum("bd,fd->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, w in enumerate(layer.weight):
            weight = jax_view(w)
            out = jnp.einsum("bd,fd->bf", x_jax, weight)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)
