# Copyright 2026 Google LLC
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

import torch
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

from tpu_inference.layers.common.quant_methods import DSV4_FP8
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.fp8 import (VllmFp8LinearMethod,
                                                        VllmFp8MoEMethod)
from tpu_inference.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(DSV4_FP8)
class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):
    """TPU quantization config for "deepseek_v4_fp8" format.
    Registered under "deepseek_v4_fp8".
    Dispatches MoE quant methods based on the checkpoint's expert_dtype:
      - "fp4" → VllmMxfp4MoEMethod
      - "fp8" → VllmFp8MoEMethod
    Linear layers always use VllmFp8LinearMethod (FP8 block-quant).
    """

    @classmethod
    def get_name(cls) -> str:
        return DSV4_FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[Union[vllm_linear.LinearMethodBase, QuantizeMethodBase]]:
        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmFp8LinearMethod(self, linear_config)
            case RoutedExperts():
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedFusedMoEMethod(layer.moe_config)

                if self.expert_dtype == "fp4":
                    if self.moe_quant_algo == "NVFP4":
                        # TODO: support NVFP4.
                        return NotImplementedError(
                            "NVFP4 is not supported yet.")

                    moe_config = self.get_moe_config(layer)
                    return VllmMxfp4MoEMethod(moe_config, self.mesh)
                else:
                    layer.moe_config = self.get_moe_config(layer)
                    return VllmFp8MoEMethod(self, layer, self.mesh)

            case _:
                return None
