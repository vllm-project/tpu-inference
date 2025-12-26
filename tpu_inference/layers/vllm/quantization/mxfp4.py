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

from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import (Mxfp4Backend,
                                                           Mxfp4Config,
                                                           Mxfp4MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.quant_methods import (MXFP4,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.quantization import \
    dequantize_tensor_from_mxfp4_packed
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.fused_moe import (FusedMoEBackend,
                                                 fused_moe_apply,
                                                 select_moe_backend)
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import (
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

REQUANTIZED_BLOCK_SIZE = 512

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return MXFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod(linear_config)
            logger.warning_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.")
            return VllmUnquantizedLinearMethod(linear_config)
        elif isinstance(layer, FusedMoE):
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once("MXFP4 attention layer is not implemented. "
                                "Skipping quantization for this layer.")
        return None


class VllmMxfp4MoEMethod(Mxfp4MoEMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        FusedMoEMethodBase.__init__(self, moe)

        # We piggyback on triton implementation as it applies minimal hardware
        # specific post processing to the weights.
        self.mxfp4_backend = Mxfp4Backend.TRITON

        self.mesh = mesh
        self.moe_backend = select_moe_backend(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == FusedMoEBackend.FUSED_MOE:
            # When fused moe kernle is used, we pass extra arguments like
            # tuned block sizes to the kernel.
            self.extra_backend_kwargs = dict(
                subc_quant_wsz=REQUANTIZED_BLOCK_SIZE,
                ep_axis_name=ep_axis_name,
                # TODO: Use autotune table once we have it.
                bt=256,
                bf=1024,
                bd1=1024,
                bd2=1024,
                btc=256,
                bfc=1024,
                bd1c=1024,
                bd2c=1024,
            )

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, FusedMoE)
        assert layer.moe_config.has_bias, "mxfp4 quantization alwyas use bias."

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w13_bias = t2j(layer.w13_bias, use_dlpack=False)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w2_bias = t2j(layer.w2_bias, use_dlpack=False)

        @jax.jit
        def process_mxfp4_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array,
        ) -> FusedMoEWeights:
            # Dequantize fp4 weights into fp32.
            w13_weight = dequantize_tensor_from_mxfp4_packed(
                w13_weight, w13_weight_scale, 2)
            w2_weight = dequantize_tensor_from_mxfp4_packed(
                w2_weight, w2_weight_scale, 2)

            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            weights = quantize_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                jnp.float4_e2m1fn,
                REQUANTIZED_BLOCK_SIZE,
            )
            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_mxfp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_bias,
            w2_weight,
            w2_weight_scale,
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

        layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        return fused_moe_apply(
            layer,
            x,
            router_logits,
            self.moe_backend,
            self.mesh,
            self.extra_backend_kwargs,
        )
