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
"""Opt-in native JAX groupwise W4A16 compressed-tensors implementation."""

import functools
import math

import jax
import jax.numpy as jnp
from compressed_tensors.config import CompressionFormat
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.model_executor.layers.quantization.compressed_tensors.utils import \
    should_ignore_layer

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    cpu_mesh_context, reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.configs import QuantLinearConfig
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param, jax_array_from_reshaped_torch)


def _unpack_uint4b8(weight_packed: jax.Array) -> jax.Array:
    """Unpack CT uint4b8 weights from ``[K / 8, N]`` to ``[K, N]``.

    ``pack-quantized`` stores eight consecutive K-axis values in each INT32.
    Symmetric INT4 uses a bias of 8, so the stored nibbles [0, 15] represent
    signed values [-8, 7].
    """
    shifts = jnp.arange(0, 32, 4, dtype=jnp.uint32)
    packed = weight_packed.astype(jnp.uint32)
    nibbles = (packed[..., None] >> shifts) & jnp.uint32(0xF)
    # [K/8, N, 8] -> [K/8, 8, N] -> [K, N]
    unpacked = jnp.transpose(nibbles,
                             (0, 2, 1)).reshape(weight_packed.shape[0] * 8,
                                                weight_packed.shape[1])
    return unpacked.astype(jnp.int8) - jnp.int8(8)


def _sharded_w4a16_matmul(x: jax.Array,
                          weight_packed: jax.Array,
                          weight_scale: jax.Array,
                          weight_sharding: P | NamedSharding,
                          group_size: int,
                          *,
                          mesh=None) -> jax.Array:
    """Reference native-JAX groupwise W4A16 matmul.

    Packed weights remain resident in HBM and are unpacked/dequantized inside each
    shard immediately before ``dot_general``.
    """
    if isinstance(weight_sharding, NamedSharding):
        mesh = mesh or weight_sharding.mesh
        weight_spec = weight_sharding.spec
    else:
        weight_spec = weight_sharding

    in_axis, out_axis = weight_spec
    x_spec = P(ShardingAxisName.ATTN_DATA, in_axis)
    scale_spec = P(in_axis, out_axis)
    out_spec = P(ShardingAxisName.ATTN_DATA, out_axis)
    x = jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, x_spec) if mesh else x_spec)

    def wrapper(x_local, packed_local, scale_local):
        weight_q = _unpack_uint4b8(packed_local)
        num_groups = scale_local.shape[0]
        if weight_q.shape[0] != num_groups * group_size:
            raise ValueError(
                "W4A16 weight K dimension must equal num_groups * group_size: "
                f"{weight_q.shape[0]} != {num_groups} * {group_size}")
        weight = (weight_q.reshape(num_groups, group_size,
                                   weight_q.shape[1]).astype(x_local.dtype) *
                  scale_local[:, None, :].astype(x_local.dtype)).reshape(
                      weight_q.shape)
        out = jax.lax.dot_general(x_local,
                                  weight,
                                  dimension_numbers=(((1, ), (0, )), ((), ())),
                                  preferred_element_type=jnp.float32).astype(
                                      x_local.dtype)
        if in_axis:
            out = jax.lax.psum(out, axis_name=in_axis)
        return out

    return jax.shard_map(wrapper,
                         mesh=mesh,
                         in_specs=(x_spec, weight_spec, scale_spec),
                         out_specs=out_spec,
                         check_vma=False)(x, weight_packed, weight_scale)


class W4A16LinearMethod(QuantizeMethodBase):
    """Native-JAX compressed-tensors symmetric groupwise INT4 linear."""

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig,
                 group_size: int):
        if group_size <= 0:
            raise ValueError("W4A16 requires a positive group_size")
        self.linear_config = linear_config
        self.group_size = group_size
        self.in_features = math.prod(linear_config.in_features)
        self.out_features = sum(linear_config.output_sizes)
        if self.in_features % 8:
            raise ValueError("W4A16 input size must be divisible by 8")
        if self.in_features % group_size:
            raise ValueError("W4A16 group_size must divide the input size")
        if linear_config.batch_features:
            raise NotImplementedError(
                "W4A16 is not yet supported for batched JaxEinsum weights")

    @staticmethod
    def _load_param(param,
                    torch_tensor,
                    _shard_id=-1,
                    *,
                    permute_dims,
                    param_name):
        value = jax_array_from_reshaped_torch(torch_tensor,
                                              permute_dims=permute_dims)
        assign_and_shard_param(param, value, param_name=param_name)

    @staticmethod
    def _load_merged_param(param,
                           torch_tensor,
                           shard_id=-1,
                           *,
                           output_sizes,
                           n_shards,
                           permute_dims,
                           param_name):
        shards = param.get_metadata("_merged_shards")
        with cpu_mesh_context():
            if shard_id == -1:
                merged = jax_array_from_reshaped_torch(
                    torch_tensor, permute_dims=permute_dims)
            else:
                shards[shard_id] = torch_tensor
                if any(shard is None for shard in shards):
                    return
                merged = jnp.concatenate([
                    jax_array_from_reshaped_torch(shard,
                                                  permute_dims=permute_dims)
                    for shard in shards
                ],
                                         axis=1)
            merged = reorder_concatenated_tensor_for_sharding(merged,
                                                              output_sizes,
                                                              n_shards,
                                                              dim=1)
        assign_and_shard_param(param, merged, param_name=param_name)

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        del layer.weight
        sharding = self.linear_config.weight_sharding
        layer.weight_packed = nnx.Param(jnp.zeros(
            (self.in_features // 8, self.out_features), dtype=jnp.int32),
                                        out_sharding=sharding,
                                        init_fn=jnp.zeros)
        layer.weight_scale = create_param(
            rngs,
            shape=(self.in_features // self.group_size, self.out_features),
            dtype=layer.dtype,
            sharding=sharding)
        # CT serializes this metadata tensor. It is not needed after the JAX
        # parameters have been created, but registering it keeps loading strict.
        layer.weight_shape = nnx.Param(jnp.zeros((2, ), dtype=jnp.int32),
                                       out_sharding=(),
                                       init_fn=jnp.zeros)

        if isinstance(layer, JaxMergedColumnParallelLinear):
            n_proj = len(layer.output_sizes)
            for param in (layer.weight_packed, layer.weight_scale):
                param.set_metadata("_merged_shards", [None] * n_proj)
            loader = functools.partial(
                self._load_merged_param,
                output_sizes=self.linear_config.output_sizes,
                n_shards=self.linear_config.n_shards)
            layer.weight_packed.set_metadata(
                "weight_loader",
                functools.partial(loader,
                                  permute_dims=(1, 0),
                                  param_name=layer.prefix + ".weight_packed"))
            layer.weight_scale.set_metadata(
                "weight_loader",
                functools.partial(loader,
                                  permute_dims=(1, 0),
                                  param_name=layer.prefix + ".weight_scale"))
        else:
            layer.weight_packed.set_metadata(
                "weight_loader",
                functools.partial(self._load_param,
                                  permute_dims=(1, 0),
                                  param_name=layer.prefix + ".weight_packed"))
            layer.weight_scale.set_metadata(
                "weight_loader",
                functools.partial(self._load_param,
                                  permute_dims=(1, 0),
                                  param_name=layer.prefix + ".weight_scale"))
        layer.weight_shape.set_metadata(
            "weight_loader",
            functools.partial(self._load_param,
                              permute_dims=None,
                              param_name=layer.prefix + ".weight_shape"))

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        # A JaxEinsum can contract more than one trailing input axis. Gemma 4's
        # o_proj, for example, contracts N and H in ``TNH,NHD->TD``. Preserve
        # only the non-contracting prefix before flattening those axes.
        original_shape = x.shape[:-len(self.linear_config.in_features)]
        x = x.reshape(-1, self.in_features)
        out = _sharded_w4a16_matmul(x,
                                    layer.weight_packed[...],
                                    layer.weight_scale[...],
                                    self.linear_config.weight_sharding,
                                    self.group_size,
                                    mesh=self.linear_config.mesh)
        if layer.bias is not None:
            out += layer.bias[...]
        outs = slice_sharded_tensor_for_concatenation(
            out, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return out.reshape(original_shape +
                           tuple(self.linear_config.out_features))


class W4A16CompressedTensorsConfig(CompressedTensorsConfig):
    """Extend the existing config only for explicitly enabled W4A16 linears."""

    def get_quant_method(self, layer: JaxModule, prefix: str):
        if not isinstance(layer, JaxEinsum):
            return super().get_quant_method(layer, prefix)
        if should_ignore_layer(prefix,
                               ignore=self._ignore,
                               fused_mapping=self._fused_mapping):
            return super().get_quant_method(layer, prefix)
        scheme = self._match_target(layer, prefix)
        if scheme is None:
            return super().get_quant_method(layer, prefix)
        weight_quant = scheme.get("weights")
        input_quant = scheme.get("input_activations")
        if not (self._ct._is_wNa16_group_channel(weight_quant, input_quant)
                and self._ct.quant_format
                == CompressionFormat.pack_quantized.value):
            return super().get_quant_method(layer, prefix)
        if weight_quant.num_bits != 4 or not weight_quant.symmetric:
            raise NotImplementedError("W4A16 requires symmetric 4-bit weights")
        if weight_quant.group_size != 32:
            raise NotImplementedError("W4A16 requires group size 32")
        if getattr(weight_quant, "actorder", None) is not None:
            raise NotImplementedError(
                "W4A16 does not support activation ordering")
        return W4A16LinearMethod(layer,
                                 QuantLinearConfig(layer, enable_sp=False),
                                 weight_quant.group_size)
