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
"""Weight-only int4 (W4A16) linear method for the JAX path.

Serves compressed-tensors ``pack-quantized`` checkpoints with symmetric int4
weights (group or channel strategy) and unquantized activations, e.g.
``google/gemma-4-31B-it-qat-w4a16-ct``.

The checkpoint ships three tensors per linear layer: ``weight_packed`` (int32,
``[out, ceil(in / 8)]``), ``weight_scale`` (``[out, in // group_size]``) and
``weight_shape`` (``[out, in]``). They are loaded onto the host, unpacked to
int4 ``[in, out]`` there, and only the int4 weight and its scale go to the
device, so HBM holds half a byte per weight.

At apply time grouped weights go through the gmm_v2 kernel, which dequantizes
each int4 tile in VMEM and multiplies it with the unquantized activation, so
the full-precision weight never exists in HBM. Groups at least as wide as the
MXU, channelwise included, use the XLA reference path instead, which is only
supported on layers whose input dim is not sharded.
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, shard_linear_weights)
from tpu_inference.layers.common.quantization import unpack_wna16_linear_weight
from tpu_inference.layers.common.utils import (
    cpu_mesh, cpu_mesh_context, reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantLinearConfig
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param, jax_array_from_reshaped_torch,
    load_nnx_param_from_reshaped_torch)

# The three tensors a compressed-tensors pack-quantized linear layer ships.
_CHECKPOINT_PARAMS = ("weight_packed", "weight_scale", "weight_shape")
_PACK_FACTOR = 8  # int4 values per int32 word
# MXU width assumed when no TPU is attached (tests, CPU tracing): the narrowest
# current MXU (128 columns before v6e, 256 on v6e and later).
_FALLBACK_MXU_COLUMNS = 128


def _mxu_column_size() -> int:
    try:
        return pltpu.get_tpu_info().mxu_column_size
    except Exception:  # pylint: disable=broad-except
        return _FALLBACK_MXU_COLUMNS


def _uses_kernel(group_size: int) -> bool:
    """Whether gmm_v2 serves this group size through its validated branch.

    gmm_v2 dequantizes an rhs tile in VMEM before the matmul only when the quant
    block is strictly narrower than the MXU; wider groups take its
    dequantize-after-matmul branch, which is not validated for W4A16. This is
    the kernel's own test, evaluated on the attached chip.
    """
    return group_size < _mxu_column_size()


class WNA16LinearMethod(QuantizeMethodBase):
    """Symmetric int4 weight, unquantized activation, for ``JaxEinsum``."""

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig,
                 group_size: Optional[int]):
        self.linear_config = linear_config
        if linear_config.batch_features:
            raise NotImplementedError(
                f"wNa16 is not supported for batched einsum "
                f"'{layer.einsum_str}' ({layer.prefix}).")
        self.in_features = math.prod(linear_config.in_features)
        self.output_shape = linear_config.out_features
        self.out_features = sum(linear_config.output_sizes)
        # Channelwise checkpoints carry group_size None or -1: one group
        # spanning the whole input dim.
        if group_size is None or group_size <= 0:
            group_size = self.in_features
        if self.in_features % group_size:
            raise ValueError(
                f"{layer.prefix}: in_features {self.in_features} is not a "
                f"multiple of group_size {group_size}.")
        self.group_size = group_size
        self.num_groups = self.in_features // group_size
        self.use_kernel = _uses_kernel(group_size)
        in_sharding = getattr(linear_config, "in_features_sharding", (None, ))
        if not self.use_kernel and in_sharding[0] is not None:
            # The XLA path's scale is [in // group_size, out] and inherits the
            # weight's sharding; with few groups (one, for channelwise) the
            # contracting axis cannot be split across the mesh.
            raise NotImplementedError(
                f"{layer.prefix}: wNa16 with group_size {group_size} is not "
                "supported on a layer whose input dim is sharded; groups "
                f"narrower than the MXU ({_mxu_column_size()}) are.")

    def _host_param(self, shape, dtype, loader) -> nnx.Param:
        # Load onto the host so process_weights_after_loading can unpack there;
        # only the unpacked int4 weight and its scale are put on the device.
        param = nnx.Param(jnp.zeros(shape, dtype),
                          weight_loader=loader,
                          eager_sharding=False)
        param.set_metadata("out_sharding", ())
        param.set_metadata("mesh", cpu_mesh())
        return param

    def _check_weight_shape(self,
                            param: nnx.Param,
                            torch_tensor,
                            shard_id: int = -1,
                            *,
                            param_name: str):
        out_features, in_features = (int(v) for v in torch_tensor.tolist())
        expected_out = (self.out_features if shard_id == -1 else
                        self.linear_config.output_sizes[shard_id])
        if (out_features, in_features) != (expected_out, self.in_features):
            raise ValueError(
                f"{param_name} is {[out_features, in_features]}, expected "
                f"{[expected_out, self.in_features]}.")
        param.set_metadata("_is_loaded", True)

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)
        if layer.bias is not None:
            raise NotImplementedError(
                f"wNa16 with a bias is not supported yet ({layer.prefix}).")
        # JaxEinsum created a full-precision kernel under the name the
        # checkpoint would use for an unquantized layer; this checkpoint ships
        # weight_packed/weight_scale/weight_shape instead.
        delattr(layer, "weight")
        packed_cols = -(-self.in_features // _PACK_FACTOR)
        for name, shape, dtype in (
            ("weight_packed", (self.out_features, packed_cols), jnp.int32),
            ("weight_scale", (self.out_features, self.num_groups),
             jnp.bfloat16),
        ):
            setattr(
                layer, name,
                self._host_param(
                    shape, dtype,
                    self._tensor_loader(layer.prefix + "." + name)))
        layer.weight_shape = self._host_param(
            (2, ), jnp.int32,
            functools.partial(self._check_weight_shape,
                              param_name=layer.prefix + ".weight_shape"))

    def _tensor_loader(self, param_name: str):
        # Keep checkpoint layout ([out, cols], no transpose); unpacking and
        # transposing happen together in process_weights_after_loading.
        return functools.partial(load_nnx_param_from_reshaped_torch,
                                 permute_dims=(0, 1),
                                 param_name=param_name)

    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        if not all(
                getattr(layer, name).get_metadata("_is_loaded", False)
                for name in _CHECKPOINT_PARAMS):
            # The three tensors can be spread across checkpoint files.
            return False

        with cpu_mesh_context():
            weight, scale = unpack_wna16_linear_weight(
                layer.weight_packed[...], layer.weight_scale[...],
                self.in_features)
            output_sizes = self.linear_config.output_sizes
            n_shards = self.linear_config.n_shards
            if len(output_sizes) > 1 and n_shards > 1:
                # Interleave the fused projections per shard, the inverse of
                # slice_sharded_tensor_for_concatenation in apply_jax.
                weight = reorder_concatenated_tensor_for_sharding(weight,
                                                                  output_sizes,
                                                                  n_shards,
                                                                  dim=1)
                scale = reorder_concatenated_tensor_for_sharding(scale,
                                                                 output_sizes,
                                                                 n_shards,
                                                                 dim=1)
            if self.use_kernel:
                # [in // group_size, 1, out] is the scale layout that routes
                # sharded_quantized_matmul to the gmm_v2 kernel.
                scale = scale[:, None, :]
        for name in _CHECKPOINT_PARAMS:
            delattr(layer, name)

        weights = shard_linear_weights(
            LinearWeights(weight=weight,
                          weight_scale=scale,
                          zero_point=None,
                          bias=None),
            mesh=None,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )
        layer.weight = nnx.Param(weights.weight)
        layer.weight_scale = nnx.Param(weights.weight_scale)
        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        # A 3D scale selects the gmm_v2 kernel, a 2D one xla_quantized_matmul;
        # maybe_quantize_x=False keeps the activation in its own dtype in both.
        out = sharded_quantized_matmul(
            x,
            layer.weight[...],
            layer.weight_scale[...],
            self.linear_config.weight_sharding,
            mesh=self.linear_config.mesh,
            defer_all_reduce=self.linear_config.defer_all_reduce,
            maybe_quantize_x=False)
        if len(self.linear_config.output_sizes) > 1:
            out = jnp.concatenate(slice_sharded_tensor_for_concatenation(
                out, self.linear_config.output_sizes,
                self.linear_config.n_shards),
                                  axis=-1)
        return out.reshape(out.shape[:-1] + self.output_shape)


class WNA16MergedLinearMethod(WNA16LinearMethod):
    """wNa16 for ``JaxMergedColumnParallelLinear`` (fused gate_up).

    The loader routes each projection's tensors with its ``shard_id``; they are
    buffered and concatenated along the output dim once all have arrived.
    """

    @staticmethod
    def _load_merged_shard(param: nnx.Param,
                           torch_tensor,
                           shard_id: int = -1,
                           *,
                           param_name: str):
        shards = param.get_metadata("_merged_shards")
        with cpu_mesh_context():
            if shard_id == -1:
                merged = jax_array_from_reshaped_torch(torch_tensor,
                                                       permute_dims=(0, 1))
            else:
                shards[shard_id] = torch_tensor
                if any(s is None for s in shards):
                    return
                merged = jnp.concatenate([
                    jax_array_from_reshaped_torch(t, permute_dims=(0, 1))
                    for t in shards
                ],
                                         axis=0)
        assign_and_shard_param(param, merged, param_name=param_name)

    def _check_weight_shape(self,
                            param: nnx.Param,
                            torch_tensor,
                            shard_id: int = -1,
                            *,
                            param_name: str):
        shards = param.get_metadata("_merged_shards")
        super()._check_weight_shape(param,
                                    torch_tensor,
                                    shard_id,
                                    param_name=param_name)
        if shard_id != -1:
            shards[shard_id] = True
            param.set_metadata("_is_loaded", all(shards))

    def _tensor_loader(self, param_name: str):
        return functools.partial(self._load_merged_shard,
                                 param_name=param_name)

    def create_weights_jax(self, layer: JaxMergedColumnParallelLinear,
                           *weight_args, rngs, **extra_weight_attrs):
        assert isinstance(layer, JaxMergedColumnParallelLinear)
        super().create_weights_jax(layer,
                                   *weight_args,
                                   rngs=rngs,
                                   **extra_weight_attrs)
        n_proj = len(self.linear_config.output_sizes)
        for name in _CHECKPOINT_PARAMS:
            getattr(layer, name).set_metadata("_merged_shards",
                                              [None] * n_proj)
