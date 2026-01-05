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

from dataclasses import dataclass, fields

import jax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
from torchax.tensor import Tensor

from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.logger import init_logger

P = PartitionSpec

logger = init_logger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class LinearWeights:
    weight: jax.Array | Tensor | list[jax.Array | Tensor]
    weight_scale: jax.Array | Tensor | list[jax.Array | Tensor] | None
    zero_point: jax.Array | Tensor | list[jax.Array | Tensor] | None
    bias: jax.Array | Tensor | list[jax.Array | Tensor] | None


MODEL_MATMUL_FUSION_TRUTH_TABLE = {
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "QKVParallelLinear"):
    True,
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "QKVParallelLinear"):
    False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "MergedColumnParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "QKVParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "QKVParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "MergedColumnParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 1024, 1, "QKVParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 2048, 1, "QKVParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 2048, 1, "MergedColumnParallelLinear"):
    False,
}


def to_parameter_list(tensor: list[torch.Tensor]):
    tensor = [Parameter(t, requires_grad=False) for t in tensor]
    return ParameterList(tensor)


def get_model_matmul_fusion_assignment(model_name: str, batch_size: int,
                                       tp_size: int, layer_name: str):
    key = (model_name, batch_size, tp_size, layer_name)
    return MODEL_MATMUL_FUSION_TRUTH_TABLE.get(key, True)


def process_lienar_weights(
    weights: LinearWeights,
    fused: bool = False,
    output_sizes: list[int] | None = None,
    reorder_size: int | None = None,
    transposed: bool = True,
    per_tensor: bool = False,
) -> LinearWeights:
    weight = weights.weight
    weight_scale = weights.weight_scale
    zero_point = weights.zero_point
    bias = weights.bias

    dim = 0 if transposed else -1
    if output_sizes is None:
        output_sizes = [weight.shape[dim]]

    if fused:
        assert reorder_size is not None
        weight = reorder_concatenated_tensor_for_sharding(
            weight, output_sizes, reorder_size, dim)

        if weight_scale is not None and not per_tensor:
            weight_scale = reorder_concatenated_tensor_for_sharding(
                weight_scale, output_sizes, reorder_size, dim)
        if zero_point is not None:
            zero_point = reorder_concatenated_tensor_for_sharding(
                zero_point, output_sizes, reorder_size, dim)
        if bias is not None:
            bias = reorder_concatenated_tensor_for_sharding(
                bias, output_sizes, reorder_size, dim)
    else:

        def slice_tensor(tensor):
            tensors = []
            start = 0
            for size in output_sizes:
                end = start + size
                tensor_split = jax.lax.slice_in_dim(tensor,
                                                    start,
                                                    end,
                                                    axis=dim)
                tensors.append(tensor_split)
                start = end
            return tensors

        weight = slice_tensor(weight)
        if weight_scale is not None and not per_tensor:
            weight_scale = slice_tensor(weight_scale)
        if zero_point is not None:
            zero_point = slice_tensor(zero_point)
        if bias is not None:
            bias = slice_tensor(bias)

    return LinearWeights(
        weight=weight,
        weight_scale=weight_scale,
        zero_point=zero_point,
        bias=bias,
    )


def shard_linear_weights(
    weights: LinearWeights,
    mesh: Mesh,
    weight_p_spec: PartitionSpec,
    bias_p_spec: PartitionSpec,
    transposed: bool = True,
    per_tensor: bool = False,
) -> LinearWeights:

    if not transposed:
        # By defualt, we use transposed weights. If it is not transposed,
        # we need to transpose the sharding as well.
        weight_p_spec = PartitionSpec(*weight_p_spec[::-1])
        bias_p_spec = PartitionSpec(weight_p_spec[0])

    weight_sharding = NamedSharding(mesh, weight_p_spec)
    bias_sharding = NamedSharding(mesh, bias_p_spec)

    weight_shardings = LinearWeights(
        weight=weight_sharding,
        weight_scale=NamedSharding(mesh, P()) if per_tensor else bias_sharding,
        zero_point=bias_sharding,
        bias=bias_sharding,
    )

    for field in fields(LinearWeights):
        key = field.name
        if (weight := getattr(weights, key, None)) is not None:
            sharding = getattr(weight_shardings, key)
            weight = jax.device_put(weight, sharding)
            setattr(weights, key, weight)
    return weights
