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

import torch
from jax.sharding import PartitionSpec
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase  # noqa: E501
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig, CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod, CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, should_ignore_layer)

from tpu_inference.layers.common.quant_methods import (COMPRESSED_TENSORS,
                                                       get_tpu_quant_method)
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe import \
    VllmCompressedTensorsMoEMethod
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    VllmCompressedTensorsW8A8Fp8
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import \
    VllmCompressedTensorsW8A8Int8
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedConfig
from tpu_inference.logger import init_logger

P = PartitionSpec
logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(COMPRESSED_TENSORS))
class VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return COMPRESSED_TENSORS

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")

        if weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        # TODO(kyuyeunk): Add support for different act_quant_format

        linear_config = self.get_linear_config(layer)
        if self._is_fp8_w8a8(weight_quant, input_quant):
            is_static_input_scheme = input_quant and not input_quant.dynamic
            return VllmCompressedTensorsW8A8Fp8(
                weight_quant=weight_quant,
                is_static_input_scheme=is_static_input_scheme,
                linear_config=linear_config,
            )
        if self._is_dynamic_token_w8a8(weight_quant, input_quant):
            return VllmCompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric,
                linear_config=linear_config,
            )
        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return VllmUnquantizedConfig.get_quant_method(
                    self, layer, prefix)
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, FusedMoE):
            layer.moe_config = self.get_moe_config(layer)
            return VllmCompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        return None
