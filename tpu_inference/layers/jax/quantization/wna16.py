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
"""Weight-only int4 (W4A16) linear and fused-MoE methods for the JAX path.

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

Routed experts ship the same three tensors per expert and projection
(``experts.<i>.{gate,up,down}_proj``, e.g. a compressed-tensors export of Gemma
4 26B-A4B). They are unpacked on the host, fused into the GMM layout without
requantization, and served by the same gmm_v2 dequantize-before-matmul branch.
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, shard_linear_weights)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights,
    shard_moe_weights_to_tpu)
from tpu_inference.layers.common.quantization import (
    u32_unpack_i4, unpack_wna16_linear_weight)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    cpu_mesh, cpu_mesh_context, reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantLinearConfig
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param, jax_array_from_reshaped_torch,
    load_nnx_param_from_reshaped_torch)
from tpu_inference.utils import get_mesh_shape_product

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
        if (not self.use_kernel and in_sharding[0] is not None
                and get_mesh_shape_product(
                    linear_config.mesh or jax.sharding.get_abstract_mesh(),
                    in_sharding[0]) > 1):
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

        # A row-parallel layer splits the input dim across the serving mesh.
        # Read its size here: inside cpu_mesh_context the current mesh is the
        # host's, and QuantLinearConfig carries no mesh of its own.
        in_axis = self.linear_config.in_features_sharding[0]
        in_shards = 1
        if self.use_kernel and in_axis is not None:
            in_shards = get_mesh_shape_product(
                self.linear_config.mesh or jax.sharding.get_abstract_mesh(),
                in_axis)
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
            if in_shards > 1:
                # Keep every shard on whole scale groups.
                scale, _ = _split_groups_for_shards(scale,
                                                    self.group_size,
                                                    self.in_features,
                                                    in_shards,
                                                    axis=0)
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


WNA16_MOE_SUPPORTED_BACKENDS = [MoEBackend.GMM_EP, MoEBackend.GMM_TP]

# Checkpoint projection name -> its role in the fused expert weights.
_MOE_PROJECTIONS = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}


def _split_groups_for_shards(scale: jax.Array,
                             group_size: int,
                             in_features: int,
                             num_shards: int,
                             axis: int = -1) -> tuple[jax.Array, int]:
    """Refine per-group scales so each shard of the input dim holds whole groups.

    A layer whose input dim is split across ``num_shards`` (GMM_TP's w2, a
    row-parallel linear) needs every shard to hold whole scale groups. When
    the per-shard slice is not a multiple of ``group_size`` (Gemma 4
    26B-A4B: the experts' 704 rows over 4 shards is 176, 5.5 groups of 32; the
    dense MLP's 2112 is 528, 16.5), one group straddles two shards and its
    scale cannot be split. Repeating each scale over sub-groups of
    ``gcd(group_size, in_features // num_shards)`` rows keeps every weight's
    scale unchanged and makes the group count divide evenly.

    Args:
        scale: per-group scales, ``in_features // group_size`` along ``axis``.
        group_size: The checkpoint's group size.
        in_features: The logical input size the groups run along.
        num_shards: How many ways that dim is sharded.
        axis: The scale's group axis.

    Returns:
        ``(scale, group_size)``, unchanged when the split already falls on
        group boundaries.
    """
    if in_features % num_shards:
        raise ValueError(f"input size {in_features} does not split into "
                         f"{num_shards} shards.")
    fine = math.gcd(group_size, in_features // num_shards)
    if fine == group_size:
        return scale, group_size
    return jnp.repeat(scale, group_size // fine, axis=axis), fine


class WNA16FusedMoEMethod(QuantizeMethodBase):
    """W4A16 method for routed experts on the GMM backends.

    The checkpoint holds ``experts.<i>.<proj>.weight_{packed,scale,shape}`` per
    expert. Each tensor is staged on the host, then all experts are unpacked
    to int4 ``[E, out, in]``, gate and up are fused into w13, and the result
    goes through ``process_moe_weights`` and ``shard_moe_weights`` directly.
    ``process_quantized_moe_weights`` is deliberately not used: it requantizes
    by default, which would replace the checkpoint's per-group scales with
    per-channel ones.

    Only groups narrower than the MXU are served. gmm_v2 dequantizes those rhs
    tiles in VMEM and multiplies them with the unquantized activation; wider
    blocks, channelwise included, take its dequantize-after-matmul branch,
    which quantizes the activation and would no longer be W4A16.
    """

    def __init__(self, group_size: int):
        self.group_size = group_size
        self.extra_backend_kwargs = {}

    @staticmethod
    def _shapes(layer) -> dict[str, tuple[int, int]]:
        """Logical [out, in] of each projection of one expert."""
        d, f = layer.hidden_size, layer.intermediate_size_moe
        return {"gate": (f, d), "up": (f, d), "down": (d, f)}

    @staticmethod
    def _staged(role: str, kind: str) -> str:
        return f"w4a16_{role}_{kind}"

    def _expert_shape(self, layer, role: str, kind: str) -> tuple[int, int]:
        """Checkpoint [out, cols] of one expert's packed weight or scale."""
        out, n_in = self._shapes(layer)[role]
        div = _PACK_FACTOR if kind == "packed" else self.group_size
        return out, n_in // div

    def _staged_names(self) -> list[str]:
        return [
            self._staged(role, kind) for role in ("gate", "up", "down")
            for kind in ("packed", "scale")
        ]

    def create_weights_jax(self, layer: JaxRoutedExperts, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        if layer.moe_backend not in WNA16_MOE_SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"Unsupported moe backend for W4A16 experts: "
                f"{layer.moe_backend}. Supported: "
                f"{WNA16_MOE_SUPPORTED_BACKENDS}")
        num_experts = layer.num_local_experts
        for role, (out, n_in) in self._shapes(layer).items():
            if n_in % self.group_size or n_in % _PACK_FACTOR:
                raise ValueError(
                    f"{layer.prefix}: {role} input size {n_in} is not a "
                    f"multiple of group_size {self.group_size} and of "
                    f"{_PACK_FACTOR}.")
        for name in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                     "kernel_down_proj_EFD"):
            assert isinstance(getattr(layer, name, None), nnx.Param), name
            delattr(layer, name)
        for role in self._shapes(layer):
            for kind, dtype in (("packed", jnp.int32), ("scale",
                                                        jnp.bfloat16)):
                # Staged on the host, as in the mxfp4 MoE method; per-expert
                # tensors collect in _weights_to_load and are concatenated
                # once all of them have arrived, as in the fp8 MoE method.
                # The placeholder is [E, cols, out]: the dummy loaders swap
                # the last two dims of every _weights_to_load param, so this
                # makes them produce the checkpoint's [out, cols] per expert.
                out, cols = self._expert_shape(layer, role, kind)
                param = nnx.Param(jnp.zeros((num_experts, cols, out),
                                            dtype=dtype),
                                  eager_sharding=False)
                param.set_metadata("mesh", cpu_mesh())
                param.set_metadata(_weights_to_load=[None] * num_experts)
                setattr(layer, self._staged(role, kind), param)

    def load_weights(self, *, layer: JaxRoutedExperts,
                     original_load_weights_fn, weights) -> set:
        shapes = self._shapes(layer)
        for torch_name, torch_weight in weights:
            # "<prefix>.<i>.<proj>.<tensor>" or, after Gemma4MoE strips the
            # "experts." prefix, "<i>.<proj>.<tensor>".
            parts = torch_name.split(layer.prefix)[-1].strip(".").split(".")
            if len(parts) != 3 or parts[1] not in _MOE_PROJECTIONS:
                raise ValueError(
                    f"{layer.prefix}: unexpected W4A16 expert tensor "
                    f"{torch_name}; expected <expert>.<proj>.weight_*.")
            expert_id, proj, tensor = int(parts[0]), parts[1], parts[2]
            role = _MOE_PROJECTIONS[proj]
            out, n_in = shapes[role]
            if tensor == "weight_shape":
                got = tuple(int(v) for v in torch_weight.reshape(-1).tolist())
                if got != (out, n_in):
                    raise ValueError(
                        f"{torch_name}: checkpoint weight_shape {got} does "
                        f"not match the layer's [out, in] {(out, n_in)}.")
                continue
            if tensor == "weight_packed":
                kind = "packed"
                if torch_weight.is_floating_point():
                    raise TypeError(
                        f"{torch_name}: packed int4 weights must be an "
                        f"integer dtype, got {torch_weight.dtype}.")
            elif tensor == "weight_scale":
                kind = "scale"
            else:
                raise ValueError(
                    f"{layer.prefix}: unexpected W4A16 expert tensor "
                    f"{torch_name}.")
            param = getattr(layer, self._staged(role, kind))
            expected = self._expert_shape(layer, role, kind)
            if tuple(torch_weight.shape) != expected:
                raise ValueError(
                    f"{torch_name}: shape {tuple(torch_weight.shape)}, "
                    f"expected {expected}.")
            param._weights_to_load[expert_id] = jax_array_from_reshaped_torch(
                torch_weight, reshape_dims=(1, ) + tuple(torch_weight.shape))
        return {
            name
            for name in self._staged_names() if all(
                w is not None for w in getattr(layer, name)._weights_to_load)
        }

    def process_weights_after_loading(self, layer: JaxRoutedExperts) -> bool:
        staged = {name: getattr(layer, name) for name in self._staged_names()}
        if any(
                any(w is None for w in param._weights_to_load)
                for param in staged.values()):
            # Experts can be spread across checkpoint files.
            return False

        shapes = self._shapes(layer)
        with cpu_mesh_context():

            def unpacked(role):
                packed = jnp.concatenate(staged[self._staged(
                    role, "packed")]._weights_to_load,
                                         axis=0)
                scale = jnp.concatenate(staged[self._staged(
                    role, "scale")]._weights_to_load,
                                        axis=0)
                # [E, out, in] int4 and [E, out, in // group], the layout
                # FusedMoEWeights expects before process_moe_weights.
                return (u32_unpack_i4(packed)[..., :shapes[role][1]], scale)

            w_gate, s_gate = unpacked("gate")
            w_up, s_up = unpacked("up")
            w2, s2 = unpacked("down")
            if layer.moe_backend == MoEBackend.GMM_TP:
                # GMM_TP shards w2's input dim; keep every shard on whole
                # scale groups.
                s2, _ = _split_groups_for_shards(
                    s2, self.group_size, shapes["down"][1],
                    get_mesh_shape_product(layer.mesh,
                                           ShardingAxisName.MLP_TENSOR))
            weights = FusedMoEWeights(
                w13_weight=jnp.concatenate([w_gate, w_up], axis=1),
                w13_weight_scale=jnp.concatenate([s_gate, s_up], axis=1),
                w13_bias=None,
                w2_weight=w2,
                w2_weight_scale=s2,
                w2_bias=None,
            )
            weights = process_moe_weights(
                weights,
                moe_backend=layer.moe_backend,
                w13_reorder_size=get_mesh_shape_product(
                    layer.mesh, ShardingAxisName.MLP_TENSOR),
                w13_interleave=False,
            )
            for name in staged:
                delattr(layer, name)

        # Copy to the devices first, as process_quantized_moe_weights does:
        # shard_moe_weights puts with a layout, which under the TPU context
        # mesh becomes a jitted reshard that cannot take a host array.
        weights = shard_moe_weights_to_tpu(weights,
                                           layer.mesh,
                                           source_mesh=cpu_mesh())
        weights = shard_moe_weights(weights,
                                    moe_backend=layer.moe_backend,
                                    mesh=layer.mesh)
        layer.kernel_gating_upproj_EDF = nnx.Param(weights.w13_weight)
        layer.kernel_gating_upproj_EDF_weight_scale = nnx.Param(
            weights.w13_weight_scale)
        layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight)
        layer.kernel_down_proj_EFD_weight_scale = nnx.Param(
            weights.w2_weight_scale)
        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        assert isinstance(layer, (JaxMoE, JaxRoutedExperts))
        if layer.moe_backend not in WNA16_MOE_SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"Unsupported moe backend for W4A16 experts: "
                f"{layer.moe_backend}.")
        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))
        weights = FusedMoEWeights(
            w13_weight=layer.kernel_gating_upproj_EDF[...],
            w13_weight_scale=layer.kernel_gating_upproj_EDF_weight_scale[...],
            w13_bias=None,
            w2_weight=layer.kernel_down_proj_EFD[...],
            w2_weight_scale=layer.kernel_down_proj_EFD_weight_scale[...],
            w2_bias=None,
        )
        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)
