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
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.gptq import (GPTQConfig,
                                                          GPTQLinearMethod)

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.quant_methods import GPTQ
from tpu_inference.layers.common.quantization import gptq_i32_unpack_u4
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(GPTQ)
class VllmGPTQConfig(GPTQConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return GPTQ

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOTE: GPTQ checkpoints typically use float16 scales/zeros. On TPUs,
        # bfloat16 is significantly preferred over float16. This may lead to
        # minor numeric differences but is acceptable given 4-bit weight
        # quantization.
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            return VllmGPTQLinearMethod(self, linear_config)
        return None


class VllmGPTQLinearMethod(GPTQLinearMethod):

    def __init__(self, quant_config: VllmGPTQConfig,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(quant_config)
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # GPTQ qweight: packed_dim=0 (input dim packed)
        assert layer.qweight.packed_dim == 0
        weight = t2j(layer.qweight, use_dlpack=False)
        delattr(layer, "qweight")

        weight_scale = t2j(layer.scales, use_dlpack=False)
        delattr(layer, "scales")

        # GPTQ qzeros: packed_dim=1 (output dim packed)
        assert layer.qzeros.packed_dim == 1
        zero_point = t2j(layer.qzeros, use_dlpack=False)
        delattr(layer, "qzeros")

        g_idx = t2j(layer.g_idx, use_dlpack=False)
        delattr(layer, "g_idx")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_gptq_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array,
            g_idx: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            # GPTQ qweight is packed along dim 0 (input dimension).
            # Shape: (input/8, output). Transpose so packed dim is last,
            # unpack, then transpose back to (input, output).
            weight = gptq_i32_unpack_u4(weight.T).T

            # GPTQ qzeros is packed along dim 1 (output dimension).
            # Shape: (num_groups, output/8) -> (num_groups, output)
            zero_point = gptq_i32_unpack_u4(zero_point)

            # GPTQ v1 format stores zero_point as (true_zero - 1).
            # AutoGPTQ does `zeros -= 1` before packing, so we add 1
            # to recover the true zero point.
            if not self.use_v2_format:
                zero_point = zero_point.astype(jnp.int8) + jnp.int8(1)

            # Fully dequantize using g_idx to map each input row to its
            # quantization group. This correctly handles both desc_act=True
            # (non-contiguous group assignments) and desc_act=False
            # (sequential groups). g_idx[i] gives the group index for
            # input channel i, so we gather per-row scales and zeros.
            scales_per_row = weight_scale[g_idx]  # (input, output)
            zeros_per_row = zero_point[g_idx]  # (input, output)

            weight = ((weight.astype(jnp.int8) -
                       zeros_per_row.astype(jnp.int8)).astype(jnp.float32) *
                      scales_per_row.astype(jnp.float32))
            weight = weight.astype(jnp.bfloat16)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                transposed=False,
            )

        weights = process_gptq_linear_weights(weight, weight_scale, zero_point,
                                              g_idx, bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
                transposed=False,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)
        weight = jax_view(layer.weight)

        outs = jnp.einsum("bd,df->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, w in enumerate(layer.weight):
            w_jax = jax_view(w)
            out = jnp.einsum("bd,df->bf", x_jax, w_jax)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)
