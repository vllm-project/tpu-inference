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

import math
from abc import ABC, abstractmethod
from typing import Optional

import jax

from tpu_inference.layers.common.quantization.configs import \
    QuantLinearConfig as CommonQuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
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


def _to_partition_spec(sharding) -> jax.sharding.PartitionSpec:
    """Convert a sharding value to a PartitionSpec.

    Handles NamedSharding (extracts .spec), raw tuples/lists from
    nnx.with_partitioning, and passthrough for existing PartitionSpec.
    """
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding.spec
    if isinstance(sharding, jax.sharding.PartitionSpec):
        return sharding
    if isinstance(sharding, (tuple, list)):
        return jax.sharding.PartitionSpec(*sharding)
    return jax.sharding.PartitionSpec()


class QuantLinearConfig(CommonQuantLinearConfig):
    """Quantization config for jax linear layers."""

    def __init__(self, layer: JaxEinsum, *, enable_sp: bool):
        # Update config attributes by parsing einsum string and weight sharding.
        # Parse the einsum string to classify axes:
        #   - contracting: in both operands but NOT in output (summed over)
        #   - batch: in both operands AND in output (paired/indexed)
        #   - free: in only one operand and in output
        einsum_str = layer.einsum_str
        weight = layer.weight

        lhs, output_axis = einsum_str.replace(" ", "").split("->")
        x_axis, w_axis = lhs.split(",")

        shared_axes = set(x_axis) & set(w_axis)
        batch_axes = shared_axes & set(output_axis)
        contracting_axes = shared_axes - batch_axes

        self.in_features = tuple(weight.shape[i] for i, c in enumerate(w_axis)
                                 if c in contracting_axes)

        # Extract and fuse sharding per axis category.
        sharding = _to_partition_spec(getattr(weight, "sharding", ()))
        sharding = sharding + (None, ) * (len(weight.shape) - len(sharding))

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

        self.out_features = tuple(
            weight.shape[i] for i, c in enumerate(w_axis)
            if c not in contracting_axes and c not in batch_axes)
        self.batch_features = tuple(weight.shape[i]
                                    for i, c in enumerate(w_axis)
                                    if c in batch_axes)

        self.out_features_sharding = (next(iter(out_sharding), None), )
        self.in_features_sharding = (next(iter(in_sharding), None), )
        self.batch_sharding = tuple(batch_sharding_set)

        output_sizes = [math.prod(self.out_features)]
        super().__init__(enable_sp=enable_sp, output_sizes=output_sizes)

        # Update weight_sharding and bias_sharding for 2D matmul compatibility
        if self.batch_features:
            self.weight_sharding = _to_partition_spec(
                weight.get_metadata().get("sharding", ()))
        else:
            self.weight_sharding = jax.sharding.PartitionSpec(
                *(self.out_features_sharding + self.in_features_sharding))

        self.bias_sharding = jax.sharding.PartitionSpec(
            *self.out_features_sharding)
