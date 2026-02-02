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

import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod


def attach_incremental_weight_loader(model: torch.nn.Module) -> None:
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a module have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Loading the weight
            res = original_loader(param, loaded_weight, *args, **kwargs)

            # Processing and sharding
            # For now, only handle unquantized linear and moe layers.
            quant_method = layer.quant_method
            if isinstance(quant_method, VllmQuantizationMethod):
                quant_method.maybe_process_weights(layer, param_name, args,
                                                   kwargs)

            return res

        return weight_loader_wrapper

    for name, module in model.named_modules():
        # Weight loader will be invoked multiple times for module. In order to determine when all the weights are loaded,
        # we need to keep track of the loaded weights for each module.
        module._loaded_weights = set()
        for param_name, param in module.named_parameters(recurse=False):
            # Omit parameters that do not have a weight_loader
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            setattr(
                param, "weight_loader",
                create_weight_loader(module, original_loader, name,
                                     param_name))


@register_model_loader("tpu_streaming_loader")
class IncrementalModelLoader(DefaultModelLoader):
    """
    Model loader that supports incremental weight loading and sharding.

    This loader is needed to inject the `attach_incremental_weight_loader` logic
    before the actual weight loading begins. This allows us to wrap the
    parameter weight loaders so that weights are sharded to TPU and freed from
    CPU memory as soon as a layer is fully loaded, rather than waiting for the
    entire model to be loaded into CPU memory first.
    """

    def __init__(self, load_config: LoadConfig):
        load_config.load_format = "auto"
        super().__init__(load_config)

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()

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

import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase, QKVParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    process_unquantized_linear_weight, process_unquantized_moe_weight)


def attach_incremental_weight_loader(model: torch.nn.Module) -> None:
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a parameter have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Loading the weight
            res = original_loader(param, loaded_weight, *args, **kwargs)

            # Processing and sharding
            # For now, only handle unquantized linear and moe layers.
            if isinstance(layer, LinearBase) and isinstance(
                    layer.quant_method, UnquantizedLinearMethod):
                if isinstance(layer, QKVParallelLinear):
                    assert len(
                        args) == 1, "Expecting shard_id as the only argument"
                    shard_id = args[0]
                    # Keep track of loaded weights for QKVLinear, e.g. (('weight', 'q'), ('bias', 'q'), ('weight', 'k'), ('bias', 'k'), ...)
                    layer._loaded_weights.add((param_name, shard_id))
                    if len(layer._loaded_weights) == len(
                        ('q', 'k', 'v')) * len(
                            dict(layer.named_parameters(recurse=False))):
                        process_unquantized_linear_weight(layer)
                else:
                    # Keep track of loaded weights for other linear layers, e.g. ('weight', 'bias')
                    layer._loaded_weights.add(param_name)
                    if len(layer._loaded_weights) == len(
                            dict(layer.named_parameters(recurse=False))):
                        process_unquantized_linear_weight(layer)
            if isinstance(layer, FusedMoE) and isinstance(
                    layer.quant_method, UnquantizedFusedMoEMethod):
                expert_id = kwargs.get('expert_id')
                shard_id = kwargs.get('shard_id')
                assert expert_id is not None, "Expecting expert_id argument"
                assert shard_id is not None, "Expecting shard_id argument"
                # Keep track of loaded weights for MoE layers, e.g. (('0', 'w1'), ('0', 'w2'), ('0', 'w3'), ('1', 'w1'), ...)
                layer._loaded_weights.add((expert_id, shard_id))
                if len(layer._loaded_weights
                       ) == layer.global_num_experts * len(('w1', 'w2', 'w3')):
                    process_unquantized_moe_weight(layer)

            return res

        return weight_loader_wrapper

    for name, module in model.named_modules():
        # Weight loader will be invoked multiple times for module. In order to determine when all the weights are loaded,
        # we need to keep track of the loaded weights for each module.
        module._loaded_weights = set()
        for param_name, param in module.named_parameters(recurse=False):
            # Omit parameters that do not have a weight_loader
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            setattr(
                param, "weight_loader",
                create_weight_loader(module, original_loader, name,
                                     param_name))


@register_model_loader("incremental")
class IncrementalModelLoader(DefaultModelLoader):
    """
    Model loader that supports incremental weight loading and sharding.

    This loader is needed to inject the `attach_incremental_weight_loader` logic
    before the actual weight loading begins. This allows us to wrap the
    parameter weight loaders so that weights are sharded to TPU and freed from
    CPU memory as soon as a layer is fully loaded, rather than waiting for the
    entire model to be loaded into CPU memory first.
    """

    def __init__(self, load_config: LoadConfig):
        load_config.load_format = "auto"
        super().__init__(load_config)

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
