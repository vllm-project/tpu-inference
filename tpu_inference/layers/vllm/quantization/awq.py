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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
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
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.quant_methods import AWQ
from tpu_inference.layers.common.quantization import (
    awq_u32_unpack_u4, dequantize_tensor_from_awq_packed)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    slice_sharded_tensor_for_concatenation)
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

# ---------------------------------------------------------------------------
# Helpers for non-GMM_EP backends that still dequant on the host side.
# ---------------------------------------------------------------------------


def _awq_dequant_and_format_moe_weights(
    w13_qw: jax.Array,
    w13_qz: jax.Array,
    w13_s: jax.Array,
    w2_qw: jax.Array,
    w2_qz: jax.Array,
    w2_s: jax.Array,
    group_size: int,
    moe_backend: MoEBackend,
    w13_interleave: bool,
    w13_reorder_size: int,
    mesh: Mesh,
) -> FusedMoEWeights:
    """Fully dequantize AWQ weights to bf16 and process for a given backend.

    This path is used for FUSED_MOE and GMM_TP where the kernel cannot unpack
    AWQ int4 on the fly.
    """
    # Dequantize awq int4 weights to fp32
    w13_weight = dequantize_tensor_from_awq_packed(w13_qw, w13_qz, w13_s, 1,
                                                   jnp.float32)
    w2_weight = dequantize_tensor_from_awq_packed(w2_qw, w2_qz, w2_s, 1,
                                                  jnp.float32)

    w13_weight = jnp.swapaxes(w13_weight, 1, 2)
    w2_weight = jnp.swapaxes(w2_weight, 1, 2)

    weights = quantize_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=None,
            w2_bias=None,
        ),
        jnp.float8_e4m3fn,
        None,
    )
    return process_moe_weights(
        weights,
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Linear method (unchanged from the current version)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MoE method — with on-the-fly kernel dequant for GMM_EP
# ---------------------------------------------------------------------------


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
        elif self.moe_backend == MoEBackend.GMM_EP:
            # Tell the GMM kernel to unpack AWQ int4 on the fly.
            self.extra_backend_kwargs = dict(
                awq_pack_factor=quant_config.pack_factor)

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

        if self.moe_backend == MoEBackend.GMM_EP:
            # =============================================================
            # GMM_EP: keep weights packed in u32 — the Pallas GMM kernel
            # will unpack int4 on the fly, tile by tile, in VMEM.
            # No optimization_barrier needed.
            # =============================================================
            #
            # Checkpoint shapes (all expert-leading):
            #   w13_qweight: (E, H, 2I // pack)     uint32
            #   w13_qzeros:  (E, H//G, 2I // pack)  uint32
            #   w13_scales:  (E, H//G, 2I)           bf16/fp16
            #   w2_qweight:  (E, I, H // pack)       uint32
            #   w2_qzeros:   (E, I//G, H // pack)    uint32
            #   w2_scales:   (E, I//G, H)             bf16/fp16
            #
            # The GMM kernel expects:
            #   rhs       = (E, k, n_packed)          uint32
            #   rhs_scale = (E, num_blocks, 1, n)     bf16
            #   rhs_zeros = (E, num_blocks, n_packed)  uint32
            #
            # Checkpoint layout already matches (E, in_features, out_packed)
            # so no reshape is needed for weights or zeros.

            ep_sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))

            # AWQ checkpoints store packed int4 values as int32.  The GMM
            # kernel operates on the raw bit patterns via shifts & masks so
            # we reinterpret them as uint32.
            w13_qweight = w13_qweight.astype(jnp.uint32)
            w2_qweight = w2_qweight.astype(jnp.uint32)
            w13_qzeros = w13_qzeros.astype(jnp.uint32)
            w2_qzeros = w2_qzeros.astype(jnp.uint32)

            w13_qweight = jax.device_put(w13_qweight, ep_sharding)
            w2_qweight = jax.device_put(w2_qweight, ep_sharding)
            w13_scales = jax.device_put(w13_scales, ep_sharding)
            w2_scales = jax.device_put(w2_scales, ep_sharding)
            w13_qzeros = jax.device_put(w13_qzeros, ep_sharding)
            w2_qzeros = jax.device_put(w2_qzeros, ep_sharding)

            # Format scales for GMM kernel: (E, num_blocks, 1, n_unpacked)
            w13_scales = jnp.expand_dims(w13_scales.astype(jnp.bfloat16), 2)
            w2_scales = jnp.expand_dims(w2_scales.astype(jnp.bfloat16), 2)

            # Store packed weights directly — no dequant, no barrier.
            layer.w13_weight = Parameter(torch_view(w13_qweight),
                                         requires_grad=False)
            layer.w2_weight = Parameter(torch_view(w2_qweight),
                                        requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(torch_view(w13_scales),
                                                   requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(torch_view(w2_scales),
                                                  requires_grad=False)
            layer.w13_weight_zeros = Parameter(torch_view(w13_qzeros),
                                               requires_grad=False)
            layer.w2_weight_zeros = Parameter(torch_view(w2_qzeros),
                                              requires_grad=False)

        else:
            # =============================================================
            # FUSED_MOE / GMM_TP: fully dequantize → requantize to fp8.
            # This path uses jax.jit so the dequant is folded into the
            # compilation and does NOT need optimization_barrier.
            # =============================================================
            if self.moe_backend in MoEBackend.fused_moe_backends() - {
                    MoEBackend.GMM_TP
            }:
                sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))
            else:
                sharding = NamedSharding(self.mesh, P())

            w13_qweight = jax.device_put(w13_qweight, sharding)
            w2_qweight = jax.device_put(w2_qweight, sharding)
            w13_scales = jax.device_put(w13_scales, sharding)
            w2_scales = jax.device_put(w2_scales, sharding)
            w13_qzeros = jax.device_put(w13_qzeros, sharding)
            w2_qzeros = jax.device_put(w2_qzeros, sharding)

            @jax.jit
            def _dequant_and_format(
                w13_qw,
                w13_qz,
                w13_s,
                w2_qw,
                w2_qz,
                w2_s,
            ) -> FusedMoEWeights:
                return _awq_dequant_and_format_moe_weights(
                    w13_qw,
                    w13_qz,
                    w13_s,
                    w2_qw,
                    w2_qz,
                    w2_s,
                    group_size=self.quant_config.group_size,
                    moe_backend=self.moe_backend,
                    w13_interleave=self._w13_interleave,
                    w13_reorder_size=self._w13_reorder_size,
                    mesh=self.mesh,
                )

            weights = _dequant_and_format(
                w13_qweight,
                w13_scales,
                w13_qzeros,
                w2_qweight,
                w2_scales,
                w2_qzeros,
            )
            weights = torch_view(
                shard_moe_weights(weights, self.moe_backend, self.mesh))

            layer.w13_weight = Parameter(weights.w13_weight,
                                         requires_grad=False)
            layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(weights.w13_weight_scale,
                                                   requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(weights.w2_weight_scale,
                                                  requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        if self.moe_backend == MoEBackend.GMM_EP:
            # ---------------------------------------------------------
            # GMM_EP path: pass packed u32 weights straight to the kernel.
            # The Pallas GMM kernel unpacks int4 on the fly in VMEM.
            # No optimization_barrier, no host-side dequant.
            # ---------------------------------------------------------
            weights = FusedMoEWeights(
                w13_weight=jax_view(layer.w13_weight),
                w13_weight_scale=jax_view(layer.w13_weight_scale_inv),
                w13_bias=None,
                w2_weight=jax_view(layer.w2_weight),
                w2_weight_scale=jax_view(layer.w2_weight_scale_inv),
                w2_bias=None,
                w13_weight_zeros=jax_view(layer.w13_weight_zeros),
                w2_weight_zeros=jax_view(layer.w2_weight_zeros),
            )
            return vllm_moe_apply(
                layer=layer,
                weights=weights,
                quant_method_instance=self,
                x=x,
                router_logits=router_logits,
            )

        # ---------------------------------------------------------
        # FUSED_MOE / GMM_TP: weights were already dequantized and
        # requantized to fp8 during process_weights_after_loading.
        # ---------------------------------------------------------
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale_inv),
            w13_bias=None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale_inv),
            w2_bias=None,
        )
        return vllm_moe_apply(
            layer=layer,
            weights=weights,
            quant_method_instance=self,
            x=x,
            router_logits=router_logits,
        )
