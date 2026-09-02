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
                 weight_sharding: P,
                 input_sharding: P,
                 output_sharding: P,
                 deferred_output_sharding: P | None = None,
                 defer_all_reduce: bool = False):
        # Output size across all TP ranks.
        self.output_sizes = output_sizes
        # shard_map specs of the matmul's weight, x and result; a row-parallel
        # layer that may hand out its partial sums instead of reducing them
        # (see set_defer_all_reduce) also gives that output's spec.
        self.weight_sharding = weight_sharding
        self.input_sharding = input_sharding
        self.output_sharding = output_sharding
        self._reduced_output_sharding = output_sharding
        self.deferred_output_sharding = deferred_output_sharding
        # Attention q/k/v takes the residual stream (replicated over pcp,
        # DENSE_DATA) but works on tokens split over pcp (ATTN_DATA). The
        # layer pins the input replicated before the matmul slices it, so
        # the split layout does not propagate back into the residual (see
        # VllmColumnParallelLinear.forward). False for every other layer.
        self.pin_input_replicated = False
        self.fuse_matmuls = True
        self.enable_sp = enable_sp
        # Sequence parallelism: layouts the layer's activations are resharded
        # to around the matmul (None = leave them as they arrive).
        self.sp_input_sharding = None
        self.sp_output_sharding = None
        self.mesh = None

        # If True, defer the all-reduce (psum) over the contracting (in) axis of
        # the matmul: it is not performed here even when that axis is sharded.
        # The matmul then returns the per-shard partial sums stacked under a
        # leading axis, and the caller reduces them later (sum_partials).
        self.defer_all_reduce = False
        if defer_all_reduce:
            self.set_defer_all_reduce(True)

        self.bias_sharding = P(self.weight_sharding[1])
        # n_shards is always the TP degree for the weight's output axis, derived
        # from the active mesh.  get_mesh_shape_product returns 1 when the axis
        # is None or absent from the mesh, so no explicit fallback is needed.
        self.n_shards = get_mesh_shape_product(
            jax.sharding.get_abstract_mesh(), self.weight_sharding[1])
        self.enable_quantized_matmul_kernel = envs.ENABLE_QUANTIZED_MATMUL_KERNEL
        self.requant_block_size = envs.REQUANTIZE_BLOCK_SIZE
        self.requant_weight_dtype = to_jax_dtype(envs.REQUANTIZE_WEIGHT_DTYPE)

    def set_defer_all_reduce(self, defer_all_reduce: bool) -> None:
        """Whether a row-parallel matmul hands out its partial sums instead
        of reducing them; the output spec follows."""
        if defer_all_reduce:
            assert self.deferred_output_sharding is not None, (
                "no deferred_output_sharding for this layer")
            self.output_sharding = self.deferred_output_sharding
        else:
            self.output_sharding = self._reduced_output_sharding
        self.defer_all_reduce = defer_all_reduce
