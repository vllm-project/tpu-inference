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

import copy
from abc import ABC, abstractmethod

import jax
from vllm.config import VllmConfig

from tpu_inference.layers.jax import JaxModule


def get_tpu_quantization_config(vllm_config: VllmConfig):
    from tpu_inference.layers.common.quant_methods import FP8
    from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig

    model_config = copy.deepcopy(vllm_config.model_config)
    method_to_config: dict[str | None, type] = {
        None: UnquantizedConfig,
        FP8: Fp8Config,
    }
    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported."
            f" Supported methods are {method_to_config.keys()}")
    quant_config = method_to_config[model_config.quantization]
    return quant_config()


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    def create_weights_jax(self, layer: JaxModule, *weight_args,
                           **extra_weight_attrs):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        pass

    @abstractmethod
    def apply_jax(self, layer: JaxModule, *args, **kwargs) -> jax.Array:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError
