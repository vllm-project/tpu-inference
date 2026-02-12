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

from abc import ABC, abstractmethod
from typing import Optional

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase


class QuantizationConfig(ABC):

    def __init__(self, hf_quant_config: dict):
        pass

    @abstractmethod
    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError

    @classmethod
    def get_from_keys(cls, config: dict, keys: list, *args):
        """Get value from config using the first matching key.'
        
        Return default value if no key is found and default is provided.
        Raise KeyError if no key is found and no default is provided.
        """
        assert len(args) <= 1, "Only one default value is allowed."
        for key in keys:
            if key in config:
                return config[key]
        if args:
            return args[0]
        raise KeyError(f"None of the keys {keys} found in config.")

    @classmethod
    def is_layer_skipped(
        cls,
        prefix: str,
        *,
        ignored_layers: list[str],
        fused_mapping: dict = dict()) -> bool:
        """Check if a layer should be skipped from quantization.

        Follows: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py#L418
        """

        def prefix_full_match(prefix: str, ignored_layers: list[str]) -> bool:
            return prefix in ignored_layers

        match_func = prefix_full_match

        proj_name = prefix.split(".")[-1]

        if fused_mapping and proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = match_func(shard_prefix, ignored_layers)

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "must have the same precision.")
        elif "experts" in prefix:
            expert_ignore_layers = [
                layer_name for layer_name in ignored_layers
                if "experts" in layer_name
            ]
            is_skipped = any(prefix in layer_name
                             for layer_name in expert_ignore_layers)
        else:
            is_skipped = match_func(prefix, ignored_layers)

        return is_skipped
