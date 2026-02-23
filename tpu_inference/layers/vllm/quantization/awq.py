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

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
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

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import AWQ
from tpu_inference.layers.common.quantization import awq_u32_unpack_u4
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.moe import (
    MoEBackend, select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

P = PartitionSpec

logger = init_logger(__name__)


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
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(prefix, self.modules_to_not_convert):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmAWQLinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            layer.moe_config = self.get_moe_config(layer)
            return VllmAWQMoEMethod(self, layer, self.mesh)
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
        def process_awq_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            weight = awq_u32_unpack_u4(weight)
            group_size = self.quant_config.group_size
            weight = weight.reshape((-1, group_size, weight.shape[-1]))

            zero_point = awq_u32_unpack_u4(zero_point)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=zero_point,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                transposed=False,
            )

        weights = process_awq_linear_weights(weight, weight_scale, zero_point,
                                             bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
                transposed=False,
            ))

        if self.linear_config.fuse_matmuls:
            layer.qweight = Parameter(weights.weight, requires_grad=False)
            layer.scales = Parameter(weights.weight_scale, requires_grad=False)
            layer.qzeros = Parameter(weights.zero_point, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.qweight = to_parameter_list(weights.weight)
            layer.scales = to_parameter_list(weights.weight_scale)
            layer.qzeros = to_parameter_list(weights.zero_point)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)

        qweight = jax_view(layer.qweight)
        qzeros = jnp.expand_dims(jax_view(layer.qzeros), 1)
        scales = jnp.expand_dims(jax_view(layer.scales), 1)

        qweight = qweight.astype(jnp.int8)
        qzeros = qzeros.astype(jnp.int8)

        weight = (qweight - qzeros) * scales
        weight = weight.reshape((-1, weight.shape[-1]))
        outs = jnp.einsum("bd,df->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.qweight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        params = zip(layer.qweight, layer.qzeros, layer.scales)
        outs = []
        for i, (qweight, qzeros, scales) in enumerate(params):
            qweight = jax_view(qweight)
            scales = jnp.expand_dims(jax_view(scales), 1)
            qzeros = jnp.expand_dims(jax_view(qzeros), 1)

            qweight = qweight.astype(jnp.int8)
            qzeros = qzeros.astype(jnp.int8)

            weight = (qweight - qzeros) * scales
            weight = weight.reshape((-1, weight.shape[-1]))
            out = jnp.einsum("bd,df->bf", x_jax, weight)

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
        self._w13_interleave = layer.activation == "swigluoai"
        self._w13_reorder_size = get_mesh_shape_product(
            self.mesh, ShardingAxisName.MLP_TENSOR)

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)
        assert not self.moe.has_bias

        w13_qweight = t2j(layer.w13_qweight, use_dlpack=False)
        w13_qweight = jnp.swapaxes(awq_u32_unpack_u4(w13_qweight), 1, 2)
        delattr(layer, "w13_qweight")

        w2_qweight = t2j(layer.w2_qweight, use_dlpack=False)
        w2_qweight = jnp.swapaxes(awq_u32_unpack_u4(w2_qweight), 1, 2)
        delattr(layer, "w2_qweight")

        w13_scales = t2j(layer.w13_scales, use_dlpack=False)
        w13_scales = jnp.swapaxes(w13_scales, 1, 2)
        delattr(layer, "w13_scales")

        w2_scales = t2j(layer.w2_scales, use_dlpack=False)
        w2_scales = jnp.swapaxes(w2_scales, 1, 2)
        delattr(layer, "w2_scales")

        w13_qzeros = t2j(layer.w13_qzeros, use_dlpack=False)
        w13_qzeros = jnp.swapaxes(awq_u32_unpack_u4(w13_qzeros), 1, 2)
        delattr(layer, "w13_qzeros")

        w2_qzeros = t2j(layer.w2_qzeros, use_dlpack=False)
        w2_qzeros = jnp.swapaxes(awq_u32_unpack_u4(w2_qzeros), 1, 2)
        delattr(layer, "w2_qzeros")

        weights = process_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_qweight,
                w13_weight_scale=w13_scales,
                w13_weight_zero_point=w13_qzeros,
                w13_bias=None,
                w2_weight=w2_qweight,
                w2_weight_scale=w2_scales,
                w2_weight_zero_point=w2_qzeros,
                w2_bias=None,
            ),
            moe_backend=self.moe_backend,
            w13_interleave=self._w13_interleave,
            w13_reorder_size=self._w13_reorder_size,
            transposed=False,
        )

        weights = shard_moe_weights(weights, self.moe_backend, self.mesh)

        layer.w13_weight = Parameter(torch_view(weights.w13_weight),
                                     requires_grad=False)
        layer.w2_weight = Parameter(torch_view(weights.w2_weight),
                                    requires_grad=False)

        layer.w13_weight_scale = Parameter(torch_view(
            weights.w13_weight_scale),
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(torch_view(weights.w2_weight_scale),
                                          requires_grad=False)

        layer.w13_weight_zero_point = Parameter(torch_view(
            weights.w13_weight_zero_point),
                                                requires_grad=False)
        layer.w2_weight_zero_point = Parameter(torch_view(
            weights.w2_weight_zero_point),
                                               requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_weight_zero_point=jax_view(layer.w13_weight_zero_point),
            w13_bias=None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_weight_zero_point=jax_view(layer.w2_weight_zero_point),
            w2_bias=None,
        )
        return vllm_moe_apply(
            layer=layer,
            weights=weights,
            quant_method_instance=self,
            x=x,
            router_logits=router_logits,
        )
