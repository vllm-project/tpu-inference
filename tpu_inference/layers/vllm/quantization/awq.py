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

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.awq import (AWQConfig,
                                                         AWQLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

from tpu_inference.layers.common.quant_methods import AWQ, get_tpu_quant_method
from tpu_inference.layers.vllm.linear_common import (
    slice_sharded_tensor_for_concatenation, torch_to_jax_param)
from tpu_inference.layers.vllm.quantization.common import (
    JaxCommonConfig, JaxCommonLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod

P = PartitionSpec
logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(AWQ))
class VllmAWQConfig(AWQConfig, JaxCommonConfig):

    @classmethod
    def get_name(cls):
        return AWQ

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOTE: AWQ checkpoint was quantized with float16. But on TPUs, using
        # bfloat16 is significantly preferred over float16. This might lead to
        # some numeric output change.
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(prefix, self.modules_to_not_convert):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmAWQLinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            raise NotImplementedError(
                "AWQ FusedMoE is currently not supported in torchax-jax")
        return None


class VllmAWQLinearMethod(AWQLinearMethod):

    def __init__(self, quant_config: VllmAWQConfig,
                 jax_config: JaxCommonLinearConfig):
        super().__init__(quant_config)
        self.jax_config = jax_config

        out_sharding, in_sharding = self.jax_config.weight_sharding[:]
        self.jax_config.weight_sharding = P(in_sharding, None, out_sharding)
        self.jax_config.scale_sharding = P(in_sharding, out_sharding)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight = layer.qweight
        qweight = unpack_awq_weight(qweight, qweight.packed_dim)

        group_size = self.quant_config.group_size
        # Reshape so that each qweight[i] were quantized with same scales[i].
        qweight = qweight.reshape((-1, group_size, layer.output_size))
        qweight = torch_to_jax_param(qweight,
                                     NamedSharding(
                                         self.jax_config.mesh,
                                         self.jax_config.weight_sharding),
                                     self.jax_config.output_sizes,
                                     self.jax_config.n_shards,
                                     self.jax_config.fuse_matmuls,
                                     dim=2,
                                     jax_dtype=jnp.uint4)
        delattr(layer, "qweight")
        layer.qweight = qweight

        qzeros = layer.qzeros
        qzeros = unpack_awq_weight(qzeros, qzeros.packed_dim)
        qzeros = torch_to_jax_param(qzeros,
                                    NamedSharding(
                                        self.jax_config.mesh,
                                        self.jax_config.scale_sharding),
                                    self.jax_config.output_sizes,
                                    self.jax_config.n_shards,
                                    self.jax_config.fuse_matmuls,
                                    dim=1,
                                    jax_dtype=jnp.uint4)
        delattr(layer, "qzeros")
        layer.qzeros = qzeros

        scales = torch_to_jax_param(layer.scales,
                                    NamedSharding(
                                        self.jax_config.mesh,
                                        self.jax_config.scale_sharding),
                                    self.jax_config.output_sizes,
                                    self.jax_config.n_shards,
                                    self.jax_config.fuse_matmuls,
                                    dim=1)
        delattr(layer, "scales")
        layer.scales = scales

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")

            bias = torch_to_jax_param(
                layer.bias,
                NamedSharding(self.jax_config.mesh,
                              self.jax_config.bias_sharding),
                self.jax_config.output_sizes,
                self.jax_config.n_shards,
                self.jax_config.fuse_matmuls,
            )
            delattr(layer, "bias")
            layer.bias = bias

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            if self.jax_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)

        qweight = jax_view(layer.qweight)
        qzeros = jnp.expand_dims(jax_view(layer.qzeros), 1)
        scales = jnp.expand_dims(jax_view(layer.scales), 1)

        qweight = qweight.astype(jnp.int8)
        qzeros = qzeros.astype(jnp.int8)

        weight = (qweight - qzeros) * scales
        weight = weight.reshape((-1, weight.shape[-1]))
        outs = jnp.einsum("bd,df->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.jax_config.output_sizes, self.jax_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.qweight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        params = zip(layer.qweight, layer.qzeros, layer.scales)
        outs = []
        for i, (qweight, qzeros, scales) in enumerate(params):
            qweight = jax_view(qweight)
            scales = jnp.expand_dims(jax_view(scales), 1)
            qzeros = jnp.expand_dims(jax_view(qzeros), 1)

            qweight = qweight.astype(jnp.int8)
            qzeros = qzeros.astype(jnp.int8)

            weight = (qweight - qzeros) * scales
            weight = weight.reshape((-1, weight.shape[-1]))
            out = jnp.einsum("bd,df->bf", x_jax, weight)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)


def unpack_awq_weight(weight: torch.Tensor, packed_dim: int):
    weight = unpack_quantized_values_into_int32(weight, scalar_types.uint4,
                                                packed_dim)

    # AWQ packs 8 uint4 into 32-bits in this order: (0, 2, 4, 6, 1, 3, 5, 7).
    # Following list maps the order used by AWQ into an ascending order.
    reverse_awq_order = (0, 4, 1, 5, 2, 6, 3, 7)

    orig_shape = weight.shape
    weight = weight.reshape(orig_shape[:-1] + (-1, 8))
    return weight[..., reverse_awq_order].reshape(orig_shape)
