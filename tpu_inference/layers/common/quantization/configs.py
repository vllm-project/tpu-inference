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

from dataclasses import dataclass

import jax
from flax import nnx
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.utils import to_jax_dtype


class QuantLinearConfig:

    @dataclass
    class LinearOpAdaptInfo:
        # Non-contracting, non-batch axis sizes from the weight.
        # E.g. for "TD,DNH->TNH", out_features = (N, H).
        # Used for reshaping right before return from __call__.
        out_features: tuple[int, ...]

        # Sharding for out_features, extracted from weight sharding.
        # E.g. for "TD,DNH->TNH", if weight sharding is ('x', None, 'y'),
        # adapted out_features_sharding will be ('y', ) because the sharding on
        # N and H axis are fused.
        out_features_sharding: tuple

        # Contracting axis sizes from the weight (axes shared between both
        # operands but NOT in the output).
        # E.g. for "TD,DNH->TNH", in_features = (D,).
        in_features: tuple[int, ...]

        # Sharding for in_features, extracted from weight sharding.
        # E.g. for "TNH,NHD->TD", if weight sharding is ('x', None, 'y'),
        # adapted in_features_sharding will be ('x', ) because the sharding on
        # N and H axis are fused.
        in_features_sharding: tuple

        # Batch axis sizes from the weight (axes shared between both operands
        # AND present in the output). E.g. for "TNH,ANH->TNA", batch_features
        # = (N,). Empty tuple means no batch dims (standard 2D matmul).
        batch_features: tuple[int, ...] = ()

        # Sharding for batch dims, extracted from weight sharding.
        batch_sharding: tuple = ()

    @classmethod
    def get_adapt_info(cls, *, einsum_str: str,
                       weight: nnx.Param) -> LinearOpAdaptInfo:
        # Parse the einsum string to classify axes:
        #   - contracting: in both operands but NOT in output (summed over)
        #   - batch: in both operands AND in output (paired/indexed)
        #   - free: in only one operand and in output
        lhs, output_axis = einsum_str.replace(" ", "").split("->")
        x_axis, w_axis = lhs.split(",")

        shared_axes = set(x_axis) & set(w_axis)
        batch_axes = shared_axes & set(output_axis)
        contracting_axes = shared_axes - batch_axes

        in_features = tuple(weight.value.shape[i] for i, c in enumerate(w_axis)
                            if c in contracting_axes)

        # Extract and fuse sharding per axis category.
        spec = getattr(weight, "sharding", ())
        if isinstance(spec, jax.NamedSharding):
            spec = spec.spec
        elif isinstance(spec, jax.sharding.SingleDeviceSharding):
            spec = ()
        sharding = spec + (None, ) * (len(weight.value.shape) - len(spec))

        in_sharding = set(s for i, s in enumerate(sharding)
                          if w_axis[i] in contracting_axes and s is not None)
        out_sharding = set(
            s for i, s in enumerate(sharding)
            if w_axis[i] not in (contracting_axes
                                 | batch_axes) and s is not None)
        batch_sharding_set = set(s for i, s in enumerate(sharding)
                                 if w_axis[i] in batch_axes and s is not None)

        assert len(in_sharding) <= 1 and len(out_sharding) <= 1, \
            f"Cannot fuse sharding {getattr(weight, 'sharding', ())=} into 2D weight sharding for {einsum_str}"

        out_features = tuple(
            weight.value.shape[i] for i, c in enumerate(w_axis)
            if c not in contracting_axes and c not in batch_axes)
        batch_features = tuple(weight.value.shape[i]
                               for i, c in enumerate(w_axis)
                               if c in batch_axes)
        batch_sharding_tuple = tuple(batch_sharding_set)

        return cls.LinearOpAdaptInfo(
            out_features=out_features,
            out_features_sharding=(next(iter(out_sharding), None), ),
            in_features=in_features,
            in_features_sharding=(next(iter(in_sharding), None), ),
            batch_features=batch_features,
            batch_sharding=batch_sharding_tuple,
        )

    def __init__(self, *, enable_sp: bool, output_sizes: list[int]):
        # Output size across all TP ranks.
        self.output_sizes = output_sizes
        self.weight_sharding = P(None, None)
        self.fuse_matmuls = True
        self.enable_sp = enable_sp
        self.input_sharding = None
        self.output_sharding = None
        self.mesh = None

        self.bias_sharding = P(self.weight_sharding[0])
        self.n_shards = len(output_sizes)
        self.enable_quantized_matmul_kernel = envs.ENABLE_QUANTIZED_MATMUL_KERNEL
        self.requant_block_size = envs.REQUANTIZE_BLOCK_SIZE
        self.requant_weight_dtype = to_jax_dtype(envs.REQUANTIZE_WEIGHT_DTYPE)
