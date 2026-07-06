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

import copy
from collections.abc import Callable
from typing import Type

from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from tpu_inference.layers.common import quant_methods
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import VllmUnquantizedConfig

ConfigFactory = Callable[[], Type[QuantizationConfig]]


def _get_compressed_tensors_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import (
        VllmCompressedTensorsConfig,
    )

    return VllmCompressedTensorsConfig


def _get_awq_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.awq import VllmAWQConfig

    return VllmAWQConfig


def _get_fp8_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.fp8 import VllmFp8Config

    return VllmFp8Config


def _get_mxfp4_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.mxfp4 import VllmMxfp4Config

    return VllmMxfp4Config


def _get_nvfp4_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.nvfp4 import VllmNvfp4Config

    return VllmNvfp4Config


def _get_deepseek_v4_fp8_config() -> Type[QuantizationConfig]:
    from tpu_inference.layers.vllm.quantization.deepseek_v4_fp8 import (
        VllmDeepseekV4Fp8Config,
    )

    return VllmDeepseekV4Fp8Config


def get_tpu_quantization_config(
    vllm_config: VllmConfig, mesh: Mesh
) -> QuantizationConfig:
    model_config = copy.deepcopy(vllm_config.model_config)
    method_to_config: dict[str | None, ConfigFactory] = {
        None: lambda: VllmUnquantizedConfig,
        quant_methods.COMPRESSED_TENSORS: _get_compressed_tensors_config,
        quant_methods.AWQ: _get_awq_config,
        quant_methods.FP8: _get_fp8_config,
        quant_methods.MXFP4: _get_mxfp4_config,
        quant_methods.NVFP4: _get_nvfp4_config,
        quant_methods.DSV4_FP8: _get_deepseek_v4_fp8_config,
    }
    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported."
            f" Supported methods are {tuple(method_to_config.keys())}"
        )
    quant_config = method_to_config[model_config.quantization]()
    assert issubclass(quant_config, VllmQuantConfig)
    quant_config.set_configs(vllm_config, mesh)

    model_config.quantization = quant_config.get_name()
    return VllmConfig.get_quantization_config(model_config, vllm_config.load_config)
