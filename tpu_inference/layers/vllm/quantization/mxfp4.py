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
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
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

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.quant_methods import (MXFP4,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.quantization import (
    dequantize_tensor_from_mxfp4_packed, quantize_tensor)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.fused_moe import fused_moe_func
from tpu_inference.layers.vllm.linear_common import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.utils import get_mesh_shape_product

REQUANTIZED_BLOCK_SIZE = 512

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config(Mxfp4Config, JaxCommonConfig):

    @classmethod
    def get_name(cls):
        return MXFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

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

    def __init__(self,
                 moe: FusedMoEConfig,
                 mesh: Mesh,
                 ep_axis_name: str = 'model'):
        FusedMoEMethodBase.__init__(self, moe)

        # We piggyback on triton implementation as it applies minimal hardware
        # specific post processing to the weights.
        self.mxfp4_backend = Mxfp4Backend.TRITON

        self.mesh = mesh
        self.use_kernel = envs.USE_MOE_EP_KERNEL and moe.use_ep
        self.ep_axis_name = ep_axis_name
        # TODO: Use autotune table once we have it.
        self.block_size = {
            "bt": 256,
            "bf": 1024,
            "bd1": 1024,
            "bd2": 1024,
            "btc": 256,
            "bfc": 1024,
            "bd1c": 1024,
            "bd2c": 1024,
        }

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

        # Wrap functions in jit to speedup requantization.
        @jax.jit
        def wrapper(w13_weight, w13_weight_scale, w13_bias, w2_weight,
                    w2_weight_scale, w2_bias):
            # Dequantize fp4 weights into fp32.
            w13_weight = dequantize_tensor_from_mxfp4_packed(
                w13_weight, w13_weight_scale, 2)
            w2_weight = dequantize_tensor_from_mxfp4_packed(
                w2_weight, w2_weight_scale, 2)

            num_experts, orig_hidden_size, orig_intermediate_size = w2_weight.shape

            # Requantize the weights into TPU friendly block size.
            w13_weight, w13_weight_scale = quantize_tensor(
                jnp.float4_e2m1fn, w13_weight, 2, REQUANTIZED_BLOCK_SIZE, True)
            w2_weight, w2_weight_scale = quantize_tensor(
                jnp.float4_e2m1fn, w2_weight, 2, REQUANTIZED_BLOCK_SIZE, True)

            intermediate_size = w2_weight.shape[-1]
            hidden_size = w13_weight.shape[-1]

            # Dims may have been padded to align with subchannel size during
            # quantization. We pad the corresponding dim on other weight.
            # NOTE: We perform padding after quantization as padding value can
            # affect quantization numerics.
            intermediate_padding_size = 2 * (intermediate_size -
                                             orig_intermediate_size)
            w13_weight = jnp.pad(w13_weight,
                                 ((0, 0), (0, intermediate_padding_size),
                                  (0, 0)))
            w13_weight_scale = jnp.pad(w13_weight_scale,
                                       ((0, 0), (0, intermediate_padding_size),
                                        (0, 0)))
            w13_bias = jnp.pad(w13_bias,
                               ((0, 0), (0, intermediate_padding_size)))

            hidden_padding_size = hidden_size - orig_hidden_size
            w2_weight = jnp.pad(w2_weight,
                                ((0, 0), (0, hidden_padding_size), (0, 0)))
            w2_weight_scale = jnp.pad(w2_weight_scale,
                                      ((0, 0), (0, hidden_padding_size),
                                       (0, 0)))
            w2_bias = jnp.pad(w2_bias, ((0, 0), (0, hidden_padding_size)))

            if layer.activation == "swigluoai":
                # When using swigluoai, vLLM splits gmm output in a interleaved way.
                # However, interleaved split is not performant on TPU. Therefore,
                # we preprocess the weight so that splitting gmm output by middle
                # can still get the same result.
                w1_weight = w13_weight[:, ::2, :]
                w3_weight = w13_weight[:, 1::2, :]
                w13_weight = jnp.concat([w1_weight, w3_weight], axis=1)

                w1_weight_scale = w13_weight_scale[:, ::2, :]
                w3_weight_scale = w13_weight_scale[:, 1::2, :]
                w13_weight_scale = jnp.concat(
                    [w1_weight_scale, w3_weight_scale], axis=1)

                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = jnp.concat([w1_bias, w3_bias], axis=1)

            if self.use_kernel:
                # Kernel expects:
                # w13: (num_experts, 2, hidden_size, intermediate_size)
                # w2: (num_experts, intermediate_size, hidden_size)
                # Current format:
                # w13_weight: (num_experts, 2*intermediate_size, hidden_size)
                # w2_weight: (num_experts, hidden_size, intermediate_size)

                w13_weight = w13_weight.reshape(num_experts, 2,
                                                intermediate_size, hidden_size)

                w13_weight_scale = w13_weight_scale.reshape(
                    num_experts, 2, intermediate_size, 1, -1)
                w2_weight_scale = w2_weight_scale.reshape(
                    num_experts, hidden_size, 1, -1)

                w13_bias = w13_bias.astype(jnp.float32).reshape(
                    num_experts, 2, 1, intermediate_size)
                w2_bias = w2_bias.astype(jnp.float32).reshape(
                    num_experts, 1, hidden_size)

                # Transpose non-constracting dim to right most dim
                w13_weight = jnp.swapaxes(w13_weight, 2, 3)
                w2_weight = jnp.swapaxes(w2_weight, 1, 2)

                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 2, 4)
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 3)

                # Apply EP sharding
                ep_sharding = NamedSharding(self.mesh, P("model"))

                w13_weight = jax.lax.with_sharding_constraint(
                    w13_weight, Format(Layout((0, 1, 2, 3)), ep_sharding))
                w2_weight = jax.lax.with_sharding_constraint(
                    w2_weight, Format(Layout((0, 1, 2)), ep_sharding))

                w13_weight_scale = jax.lax.with_sharding_constraint(
                    w13_weight_scale,
                    Format(Layout((0, 1, 2, 3, 4)), ep_sharding))
                w2_weight_scale = jax.lax.with_sharding_constraint(
                    w2_weight_scale, Format(Layout((0, 1, 2, 3)), ep_sharding))

                w13_bias = jax.lax.with_sharding_constraint(
                    w13_bias, Format(Layout((0, 1, 2, 3)), ep_sharding))
                w2_bias = jax.lax.with_sharding_constraint(
                    w2_bias, Format(Layout((0, 1, 2)), ep_sharding))
            else:
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
                w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
                w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)

                w13_bias = jnp.expand_dims(w13_bias, 1)
                w2_bias = jnp.expand_dims(w2_bias, 1)

                if layer.use_ep:
                    ep_sharding = NamedSharding(self.mesh,
                                                P(ShardingAxisName.EXPERT))

                    w13_weight = jax.lax.with_sharding_constraint(
                        w13_weight, ep_sharding)
                    w2_weight = jax.lax.with_sharding_constraint(
                        w2_weight, ep_sharding)

                    w13_weight_scale = jax.lax.with_sharding_constraint(
                        w13_weight_scale, ep_sharding)
                    w2_weight_scale = jax.lax.with_sharding_constraint(
                        w2_weight_scale, ep_sharding)

                    w13_bias = jax.lax.with_sharding_constraint(
                        w13_bias, ep_sharding)
                    w2_bias = jax.lax.with_sharding_constraint(
                        w2_bias, ep_sharding)

                else:
                    output_sizes = [intermediate_size, intermediate_size]
                    n_shards = get_mesh_shape_product(
                        self.mesh, ShardingAxisName.MLP_TENSOR)
                    assert intermediate_size % n_shards == 0

                    # Reorder w13 weights so that splitting w1 and w3 output
                    # can happen locally without any collective operations.
                    w13_weight = reorder_concatenated_tensor_for_sharding(
                        w13_weight,
                        output_sizes,
                        n_shards,
                        dim=1,
                    )
                    w13_weight_scale = reorder_concatenated_tensor_for_sharding(
                        w13_weight_scale,
                        output_sizes,
                        n_shards,
                        dim=3,
                    )
                    w13_bias = reorder_concatenated_tensor_for_sharding(
                        w13_bias,
                        output_sizes,
                        n_shards,
                        dim=2,
                    )

                    w13_weight = jax.lax.with_sharding_constraint(
                        w13_weight,
                        NamedSharding(
                            self.mesh,
                            P(None, ShardingAxisName.MLP_TENSOR, None)))
                    w2_weight = jax.lax.with_sharding_constraint(
                        w2_weight,
                        NamedSharding(
                            self.mesh,
                            P(None, None, ShardingAxisName.MLP_TENSOR)))
                    w13_weight_scale = jax.lax.with_sharding_constraint(
                        w13_weight_scale,
                        NamedSharding(
                            self.mesh,
                            P(None, None, None, ShardingAxisName.MLP_TENSOR)))
                    w2_weight_scale = jax.lax.with_sharding_constraint(
                        w2_weight_scale,
                        NamedSharding(
                            self.mesh,
                            P(None, ShardingAxisName.MLP_TENSOR, None, None)))
                    w13_bias = jax.lax.with_sharding_constraint(
                        w13_bias,
                        NamedSharding(
                            self.mesh,
                            P(None, None, ShardingAxisName.MLP_TENSOR)))
                    w2_bias = jax.lax.with_sharding_constraint(
                        w2_bias, NamedSharding(self.mesh, P(None, None, None)))

            return w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias

        w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias = wrapper(
            w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale,
            w2_bias)

        layer.w13_weight = Parameter(torch_view(w13_weight),
                                     requires_grad=False)
        layer.w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)

        layer.w13_weight_scale = Parameter(torch_view(w13_weight_scale),
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(torch_view(w2_weight_scale),
                                          requires_grad=False)

        layer.w13_bias = Parameter(torch_view(w13_bias), requires_grad=False)
        layer.w2_bias = Parameter(torch_view(w2_bias), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if layer.scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        x = jax_view(x)
        w13_weight = jax_view(layer.w13_weight)
        w2_weight = jax_view(layer.w2_weight)
        w13_weight_scale = jax_view(layer.w13_weight_scale)
        w2_weight_scale = jax_view(layer.w2_weight_scale)
        w13_bias = jax_view(layer.w13_bias)
        w2_bias = jax_view(layer.w2_bias)
        gating_output = jax_view(router_logits)

        if self.use_kernel:
            actual_hidden_size = x.shape[-1]
            padding_size = w13_weight.shape[-2] - actual_hidden_size
            x = jnp.pad(x, ((0, 0), (0, padding_size)))
            output = fused_ep_moe(
                mesh=self.mesh,
                tokens=x,
                w1=w13_weight,
                w2=w2_weight,
                w1_scale=w13_weight_scale,
                w2_scale=w2_weight_scale,
                b1=w13_bias,
                b2=w2_bias,
                gating_output=gating_output,
                subc_quant_wsz=REQUANTIZED_BLOCK_SIZE,
                top_k=layer.top_k,
                ep_axis_name=self.ep_axis_name,
                renormalize_topk_logits=layer.renormalize,
                act_fn=layer.activation,
                **self.block_size,
            )[:, :actual_hidden_size]
        else:
            output = fused_moe_func(
                hidden_states=x,
                w1=w13_weight,
                w2=w2_weight,
                w1_scale=w13_weight_scale,
                w2_scale=w2_weight_scale,
                w1_bias=w13_bias,
                w2_bias=w2_bias,
                gating_output=gating_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
                mesh=self.mesh,
                use_ep=layer.use_ep,
                activation=layer.activation,
            )

        return torch_view(output)
