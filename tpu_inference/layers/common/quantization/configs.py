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

from jax.sharding import PartitionSpec as P


class QuantLinearConfig:

    def __init__(self, *, enable_sp: bool, output_sizes: list[int]):
        self.output_sizes = output_sizes
        self.weight_sharding = P(None, None)
        self.fuse_matmuls = True
        self.enable_sp = enable_sp
        self.input_sharding = None
        self.output_sharding = None

        self.bias_sharding = P(self.weight_sharding[0])
        self.n_shards = len(output_sizes)
