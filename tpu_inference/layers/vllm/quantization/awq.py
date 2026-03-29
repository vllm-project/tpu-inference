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

import functools
from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import \
    FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.awq import (AWQConfig,
                                                         AWQLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import AWQ
from tpu_inference.layers.common.quantization import awq_u32_unpack_u4
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.layers.vllm.interface.moe import (
    MoEBackend, select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

P = PartitionSpec

logger = init_logger(__name__)


def _dense_gmm_local(x, weight, scale, *, group_size):
    """Run GMM kernel on a local shard for dense matmul.

    Treats the dense linear as a single-group GMM, allowing the kernel
    to handle int8 weights with subchannel scale application.

    Args:
        x: Input activations, shape (batch, K_local).
        weight: Int8 weight (zero-point subtracted), shape (K_local, N_local).
        scale: Per-group scales, shape (num_groups_local, N_local).
        group_size: Number of input channels per quantization group.
    """
    batch = x.shape[0]
    rhs = weight[jnp.newaxis, :, :]  # (1, K_local, N_local)
    rhs_scale = scale[jnp.newaxis, :,
                      jnp.newaxis, :]  # (1, G_local, 1, N_local)
    group_sizes = jnp.array([batch], dtype=jnp.int32)

    return gmm_v2(
        lhs=x,
        rhs=rhs,
        rhs_scale=rhs_scale,
        group_sizes=group_sizes,
        zero_initialize=False,
        maybe_quantize_lhs=False,
    )


@register_quantization_config(AWQ)
class VllmAWQConfig(AWQConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return AWQ

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOTE: AWQ checkpoint was quantized with float16. But on TPUs, using
        # bfloat16 is significantly preferred over float16. This might lead to
        # some numeric output change.
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        match layer:
            case LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(prefix, self.modules_to_not_convert):
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmAWQLinearMethod(self, linear_config)
            case FusedMoE():
                layer.moe_config = self.get_moe_config(layer)
                return VllmAWQMoEMethod(self, layer, self.mesh)
            case _:
                return None


class VllmAWQLinearMethod(AWQLinearMethod):

    def __init__(self, quant_config: VllmAWQConfig,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(quant_config)
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer.qweight.packed_dim == layer.qweight.ndim - 1
        weight = t2j(layer.qweight, use_dlpack=False)
        delattr(layer, "qweight")

        weight_scale = t2j(layer.scales, use_dlpack=False)
        delattr(layer, "scales")

        assert layer.qzeros.packed_dim == layer.qzeros.ndim - 1
        zero_point = t2j(layer.qzeros, use_dlpack=False)
        delattr(layer, "qzeros")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_awq_gmm_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            # Unpack uint4 from int32
            weight = awq_u32_unpack_u4(weight)
            zero_point = awq_u32_unpack_u4(zero_point)

            group_size = self.quant_config.group_size

            # Reshape weight to (num_groups, group_size, N), subtract zero
            # point per group, then flatten back to (K, N) int8.
            weight = weight.reshape((-1, group_size, weight.shape[-1]))
            zero_point = zero_point[:, jnp.newaxis, :]  # (num_groups, 1, N)
            weight = weight.astype(jnp.int8) - zero_point.astype(jnp.int8)
            weight = weight.reshape((-1, weight.shape[-1]))

            # weight is now (K, N) int8, zero-adjusted.
            # weight_scale is (num_groups, N) float32 -- maps directly to
            # GMM subchannel quantization parameters.

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=False,  # Always split to avoid VMEM OOM on large N
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                transposed=False,
            )

        weights = process_awq_gmm_weights(weight, weight_scale, zero_point,
                                          bias)

        # Manually shard weights. shard_linear_weights does not correctly
        # handle 2D scale tensors with transposed=False.
        mesh = self.linear_config.mesh
        orig_p_spec = self.linear_config.weight_sharding
        # Reverse for non-transposed layout: (K, N) instead of (N, K)
        weight_p_spec = P(*orig_p_spec[::-1])
        weight_sharding = NamedSharding(mesh, weight_p_spec)
        scale_sharding = NamedSharding(mesh, weight_p_spec)
        bias_p_spec = P(weight_p_spec[-1])
        bias_sharding = NamedSharding(mesh, bias_p_spec)

        def shard(arr, sharding):
            if arr is None:
                return None
            if isinstance(arr, list):
                return [general_device_put(a, sharding) for a in arr]
            return general_device_put(arr, sharding)

        sharded_weight = shard(weights.weight, weight_sharding)
        sharded_scale = shard(weights.weight_scale, scale_sharding)
        sharded_bias = shard(weights.bias, bias_sharding)

        # Always use split path
        layer.weight = to_parameter_list(
            [torch_view(w) for w in sharded_weight])
        layer.weight_scale = to_parameter_list(
            [torch_view(s) for s in sharded_scale])
        if sharded_bias is not None:
            layer.bias = to_parameter_list(
                [torch_view(b) for b in sharded_bias])

    def _get_tp_axis(self):
        """Determine which mesh axis is used for tensor parallelism."""
        p_spec = self.linear_config.weight_sharding
        for axis in p_spec:
            if axis is not None:
                return axis
        return None

    def _is_row_parallel(self):
        """Check if this is a row-parallel linear (K dimension sharded)."""
        p_spec = self.linear_config.weight_sharding
        return p_spec[0] is None and p_spec[1] is not None

    def _gmm_matmul(self, x_jax: jax.Array, weight: jax.Array,
                    scale: jax.Array) -> jax.Array:
        """Perform quantized matmul using GMM V2 kernel via shard_map."""
        mesh = self.linear_config.mesh
        tp_axis = self._get_tp_axis()
        is_row_parallel = self._is_row_parallel()

        weight_p_spec = P(*self.linear_config.weight_sharding[::-1])

        if is_row_parallel:
            x_p_spec = P(None, tp_axis)
            scale_p_spec = P(tp_axis, None)
            out_p_spec = P()

            def _local_fn(x, w, s):
                out = _dense_gmm_local(x,
                                       w,
                                       s,
                                       group_size=self.quant_config.group_size)
                return jax.lax.psum(out, axis_name=tp_axis)
        else:
            x_p_spec = P()
            scale_p_spec = P(None, tp_axis)
            out_p_spec = P(None, tp_axis)

            _local_fn = functools.partial(
                _dense_gmm_local, group_size=self.quant_config.group_size)

        return jax.shard_map(
            _local_fn,
            mesh=mesh,
            in_specs=(x_p_spec, weight_p_spec, scale_p_spec),
            out_specs=out_p_spec,
            check_vma=False,
        )(x_jax, weight, scale)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            out = self._apply_split(layer, x, bias)

        return out

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (w, s) in enumerate(zip(layer.weight, layer.weight_scale)):
            out = self._gmm_matmul(x_jax, jax_view(w), jax_view(s))

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)


class VllmAWQMoEMethod(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: VllmAWQConfig,
        layer: torch.nn.Module,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update({
            "is_transposed":
            True,
            "quant_method":
            FusedMoeWeightScaleSupported.GROUP.value,
        })

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition //
                self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts,
                        num_groups_w2,
                        hidden_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition //
                self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if self.moe.has_bias:
            w13_bias = Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_qweight = t2j(layer.w13_qweight, use_dlpack=False)
        delattr(layer, "w13_qweight")

        w2_qweight = t2j(layer.w2_qweight, use_dlpack=False)
        delattr(layer, "w2_qweight")

        w13_scales = t2j(layer.w13_scales, use_dlpack=False)
        delattr(layer, "w13_scales")

        w2_scales = t2j(layer.w2_scales, use_dlpack=False)
        delattr(layer, "w2_scales")

        w13_qzeros = t2j(layer.w13_qzeros, use_dlpack=False)
        delattr(layer, "w13_qzeros")

        w2_qzeros = t2j(layer.w2_qzeros, use_dlpack=False)
        delattr(layer, "w2_qzeros")

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
            delattr(layer, "w13_bias")
            delattr(layer, "w2_bias")
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_awq_moe_weights(
            w13_qweight: jax.Array,
            w13_scales: jax.Array,
            w13_qzeros: jax.Array,
            w13_bias: jax.Array | None,
            w2_qweight: jax.Array,
            w2_scales: jax.Array,
            w2_qzeros: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            w13_qweight = awq_u32_unpack_u4(w13_qweight).astype(jnp.int8)
            w13_qzeros = awq_u32_unpack_u4(w13_qzeros).astype(jnp.int8)
            w2_qweight = awq_u32_unpack_u4(w2_qweight).astype(jnp.int8)
            w2_qzeros = awq_u32_unpack_u4(w2_qzeros).astype(jnp.int8)

            w13_weight = (w13_qweight.reshape(w13_qweight.shape[0], -1,
                                              self.quant_config.group_size,
                                              w13_qweight.shape[-1]) -
                          w13_qzeros[:, :, jnp.newaxis, :]).reshape(
                              w13_qweight.shape)
            w2_weight = (w2_qweight.reshape(w2_qweight.shape[0], -1,
                                            self.quant_config.group_size,
                                            w2_qweight.shape[-1]) -
                         w2_qzeros[:, :, jnp.newaxis, :]).reshape(
                             w2_qweight.shape)

            w13_weight = jnp.swapaxes(w13_weight, 1, 2)
            w2_weight = jnp.swapaxes(w2_weight, 1, 2)
            w13_scales = jnp.swapaxes(w13_scales, 1, 2)
            w2_scales = jnp.swapaxes(w2_scales, 1, 2)

            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_scales,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_scales,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_interleave=w13_interleave,
                w13_reorder_size=w13_reorder_size,
            )

        weights = process_awq_moe_weights(
            w13_qweight,
            w13_scales,
            w13_qzeros,
            w13_bias,
            w2_qweight,
            w2_scales,
            w2_qzeros,
            w2_bias,
        )

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None)

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
