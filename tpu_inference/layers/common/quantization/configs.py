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
        # Adapted weight sharding. E.g. for "TD,DNH->TNH", if weight sharding is ('x', None, 'y'), adapted weight sharding will be ('x', 'y') becuase the sharding on N and H axis are fused.
        weight_sharding: tuple
        # Non-contracting axis size. E.g. for "TD,DNH->TNH", out_features will be (N, H). This is used for reshaping right before return from __call__.
        out_features: tuple[int, ...]
        # Contracting axis size. E.g. for "TNH,NHD->TD", in_features will be (N, H). This is used for reshaping input at the beginning of __call__.
        in_features: tuple[int, ...]

    @classmethod
    def get_adapt_info(cls, *, einsum_str: str,
                       weight: nnx.Param) -> LinearOpAdaptInfo:
        # HF model stores weight in 2-D shape. E.g. for "TD,DNH->TNH", weight shape in HF is (NH, D)
        x_axis, w_axis = einsum_str.split("->")[0].split(",")
        contracting_axis = set(x_axis) & set(w_axis)
        in_features = tuple(weight.value.shape[i] for i, c in enumerate(w_axis)
                            if c in contracting_axis)

        # E.g. if weight shape is (NH, D), sharding is ('x', None, 'y'), we need to fuse sharding to ('x', 'y')
        spec = weight.sharding
        if isinstance(weight.sharding, jax.NamedSharding):
            spec = weight.sharding.spec
        elif isinstance(weight.sharding, jax.sharding.SingleDeviceSharding):
            spec = ()
        sharding = spec + (None, ) * (len(weight.value.shape) - len(spec))
        in_sharding = set(s for i, s in enumerate(sharding)
                          if w_axis[i] in contracting_axis and s is not None)
        out_sharding = set(
            s for i, s in enumerate(sharding)
            if w_axis[i] not in contracting_axis and s is not None)
        assert len(in_sharding) <= 1 and len(out_sharding) <= 1, \
            f"Cannot fuse sharding {weight.sharding} into 2D weight sharding for {einsum_str}"

        weight_sharding = (next(iter(in_sharding),
                                None), next(iter(out_sharding), None))
        out_features = tuple([
            weight.value.shape[i] for i, c in enumerate(w_axis)
            if c not in contracting_axis
        ])
        return cls.LinearOpAdaptInfo(weight_sharding=weight_sharding,
                                     out_features=out_features,
                                     in_features=in_features)

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
