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

import jax
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.utils import get_mesh_shape_product, to_jax_dtype


class QuantLinearConfig:

    def __init__(self,
                 *,
                 enable_sp: bool,
                 output_sizes: list[int],
                 weight_sharding: P | None = None
                 defer_all_reduce: bool = False):
        # Output size across all TP ranks.
        self.output_sizes = output_sizes
        self.weight_sharding = weight_sharding if weight_sharding is not None else P(
            None, None)
        self.fuse_matmuls = True
        self.enable_sp = enable_sp
        self.input_sharding = None
        self.output_sharding = None
        self.mesh = None

        # If True, defer the all-reduce (psum) over the contracting (in) axis of
        # the matmul: it is not performed here even when that axis is sharded.
        # The matmul then returns per-shard partial sums and the caller is
        # responsible for reducing them later (e.g. fusing the reduction with a
        # downstream collective).
        self.defer_all_reduce = defer_all_reduce

        self.bias_sharding = P(self.weight_sharding[1])
        # n_shards is always the TP degree for the weight's output axis, derived
        # from the active mesh.  get_mesh_shape_product returns 1 when the axis
        # is None or absent from the mesh, so no explicit fallback is needed.
        self.n_shards = get_mesh_shape_product(
            jax.sharding.get_abstract_mesh(), self.weight_sharding[1])
        self.enable_quantized_matmul_kernel = envs.ENABLE_QUANTIZED_MATMUL_KERNEL
        self.requant_block_size = envs.REQUANTIZE_BLOCK_SIZE
        self.requant_weight_dtype = to_jax_dtype(envs.REQUANTIZE_WEIGHT_DTYPE)
