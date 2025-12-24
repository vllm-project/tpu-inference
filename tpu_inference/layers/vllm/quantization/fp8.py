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
import torch
from jax.sharding import PartitionSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8LinearMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.quant_methods import FP8, get_tpu_quant_method
from tpu_inference.layers.vllm.quantization.common import (
    JaxCommonConfig, JaxCommonLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod

P = PartitionSpec
logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(FP8))
class VllmFp8Config(Fp8Config, JaxCommonConfig):

    @classmethod
    def get_name(cls):
        return FP8

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(prefix, self.ignored_layers):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmFp8LinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            raise NotImplementedError(
                "FP8 FusedMoE is currently not supported in torchax-jax")
        return None


class VllmFp8LinearMethod(Fp8LinearMethod):

    def __init__(self, quant_config: VllmFp8Config,
                 jax_config: JaxCommonLinearConfig):
        super().__init__(quant_config)
        self.jax_config = jax_config
        self._configure_sharding()

    def _configure_sharding(self) -> None:

        raise NotImplementedError(
            "Configure PartitionSpec for weight_sharding and scale_sharding "
            "based on layer type (RowParallel/ColumnParallel)")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        raise NotImplementedError(
            "Convert layer.weight, layer.weight_scale, and optionally "
            "layer.input_scale and layer.bias from torch tensors to JAX arrays "
            "using torch_to_jax_param() with appropriate sharding")

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

        raise NotImplementedError(
            "Implement single matmul for fused outputs: "
            "quantize input to fp8, perform fp8 matmul with weight and scales, "
            "dequantize output, and add bias if present")

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        raise NotImplementedError(
            "Implement separate matmuls per output partition: "
            "split weight/scale by output_sizes, perform fp8 matmul for each, "
            "concatenate results, and add bias if present")
