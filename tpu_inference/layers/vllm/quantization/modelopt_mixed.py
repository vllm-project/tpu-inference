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
"""ModelOpt MIXED_PRECISION quantization support for TPU.

A MIXED_PRECISION checkpoint quantizes different layers with different
algorithms. The per-layer algorithm is listed in ``quantized_layers`` of the
quantization config; e.g. nvidia/Qwen3.8-2.4T-A95B-NVFP4 uses per-tensor FP8
for the attention and linear-attention projections and NVFP4 for the routed
experts. Upstream ``ModelOptMixedPrecisionConfig`` resolves the algorithm for
a vLLM layer prefix (including fused layers such as ``qkv_proj``); this config
maps each algorithm to the existing TPU method:

 - FP8 (per-tensor) linear      -> VllmFp8LinearMethod
 - NVFP4 linear                 -> VllmNvfp4LinearMethod
 - NVFP4 routed experts         -> VllmNvfp4MoEMethod
 - unlisted or excluded layers  -> the TPU unquantized methods

Other ModelOpt algorithms (block FP8, MXFP8, W4A16_NVFP4, FP8 MoE) raise
NotImplementedError instead of silently loading as unquantized.
"""

from typing import Optional

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import \
    ModelOptMixedPrecisionConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from tpu_inference.layers.common.quant_methods import MODELOPT_MIXED
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.fp8 import (VllmFp8Config,
                                                        VllmFp8LinearMethod)
from tpu_inference.layers.vllm.quantization.nvfp4 import (
    VllmNvfp4Config, VllmNvfp4LinearMethod, VllmNvfp4MoEMethod)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(MODELOPT_MIXED)
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):
    """ModelOpt MIXED_PRECISION config for TPU.

    Inherits config parsing and per-layer algorithm resolution from upstream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU counterparts of the upstream per-algorithm sub-configs. ModelOpt
        # FP8 is per-tensor with static input scales and no block size.
        self.tpu_fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            weight_block_size=None,
        )
        self.tpu_nvfp4_config = VllmNvfp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=self.kv_cache_quant_method,
            exclude_modules=[],
            group_size=self.nvfp4_config.group_size,
        )

    @classmethod
    def get_name(cls):
        return MODELOPT_MIXED

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, Attention):
            if self.kv_cache_quant_method:
                logger.warning_once(
                    "ModelOpt KV cache quantization is not implemented on "
                    "TPU; use --kv-cache-dtype instead. Skipping "
                    "quantization for attention layers.")
            return None

        if self.is_layer_excluded(prefix):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)

        if isinstance(layer, LinearBase):
            quant_algo = self._resolve_quant_algo(prefix)
            if quant_algo == "FP8":
                return VllmFp8LinearMethod(self.tpu_fp8_config,
                                           self.get_linear_config(layer))
            if quant_algo == "NVFP4":
                return VllmNvfp4LinearMethod(self.tpu_nvfp4_config,
                                             self.get_linear_config(layer))
            if quant_algo is not None:
                raise NotImplementedError(
                    f"ModelOpt {quant_algo} linear layers are not supported "
                    f"on TPU yet (layer {prefix}). Supported: FP8, NVFP4.")
        elif isinstance(layer, RoutedExperts):
            quant_algo = self._resolve_quant_algo(prefix)
            if quant_algo == "NVFP4":
                layer.moe_config = self.get_moe_config(layer)
                return VllmNvfp4MoEMethod(self.tpu_nvfp4_config, layer,
                                          self.mesh)
            if quant_algo is not None:
                raise NotImplementedError(
                    f"ModelOpt {quant_algo} MoE layers are not supported on "
                    f"TPU yet (layer {prefix}). Supported: NVFP4.")
        elif isinstance(layer, ParallelLMHead):
            quant_algo = self._resolve_quant_algo(prefix)
            if quant_algo is not None:
                raise NotImplementedError(
                    f"ModelOpt {quant_algo} lm_head is not supported on TPU "
                    f"yet (layer {prefix}).")

        return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
