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

    # Whether this method's matmul honors ``linear_config.defer_all_reduce``
    # (skipping its own all-reduce over the contracting axis and returning
    # per-shard partial sums for the caller to reduce later). Methods must NOT
    # claim support unless their apply() actually threads the flag through:
    # a plain einsum under GSPMD is always all-reduced by the partitioner, so
    # deferral silently fails and downstream merged reduces double-count.
    # Opt in per method class after validating end-to-end numerics.
    supports_defer_all_reduce: bool = False

    @abstractmethod
    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        raise NotImplementedError
