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

from abc import ABC, abstractmethod

import torch


class VllmQuantizationMethod(ABC):

    def maybe_process_linear_weights(
        self,
        layer: torch.nn.Module,
        param_name: str,
        args,
        kwargs,
        num_proj: int,
        log_prefix: str = "",
    ):
        """Shared tracking logic for incremental sharding of linear layers."""
        if isinstance(layer, vllm_linear.QKVParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                layer._loaded_weights.add((param_name, "q"))
                layer._loaded_weights.add((param_name, "k"))
                layer._loaded_weights.add((param_name, "v"))
        elif isinstance(layer, vllm_linear.MergedColumnParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                for i in range(len(layer.output_sizes)):
                    layer._loaded_weights.add((param_name, i))
        else:
            layer._loaded_weights.add(param_name)

        expected_count = num_proj * len(
            dict(layer.named_parameters(recurse=False)))
        if len(layer._loaded_weights) == expected_count:
            prefix_str = f"[{log_prefix}] " if log_prefix else ""
            logger.debug(
                f"{prefix_str}Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(
                f"{prefix_str}Complete sharding weights for layer {type(layer)}"
            )

    @abstractmethod
    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError
