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

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEConfig, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.quant_methods import (UNQUANTIZED,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.fused_moe import fused_moe_func
from tpu_inference.layers.vllm.linear_common import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation, torch_to_jax_param)
from tpu_inference.layers.vllm.quantization.common import (
    JaxCommonConfig, JaxCommonLinearConfig)
from tpu_inference.utils import get_mesh_shape_product

P = PartitionSpec
logger = init_logger(__name__)


def align_to(a, b):
    return (a + b - 1) // b * b


@register_quantization_config(get_tpu_quant_method(UNQUANTIZED))
class VllmUnquantizedConfig(QuantizationConfig, JaxCommonConfig):

    @classmethod
    def get_name(cls) -> str:
        return UNQUANTIZED

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Always supported

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No extra configs required.

    @classmethod
    def from_config(cls, _: dict[str, Any]) -> "VllmUnquantizedConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            return VllmUnquantizedLinearMethod(linear_config)
        if isinstance(layer, FusedMoE):
            moe_config = self.get_moe_config(layer)
            return VllmUnquantizedFusedMoEMethod(moe_config, self.mesh)
        if isinstance(layer, Attention):
            return None
        return None


class VllmUnquantizedLinearMethod(UnquantizedLinearMethod):

    def __init__(self, jax_config: JaxCommonLinearConfig):
        self.jax_config = jax_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = torch_to_jax_param(
            layer.weight,
            NamedSharding(self.jax_config.mesh,
                          self.jax_config.weight_sharding),
            self.jax_config.output_sizes,
            self.jax_config.n_shards,
            self.jax_config.fuse_matmuls,
        )
        delattr(layer, "weight")
        layer.weight = weight

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")

            bias = torch_to_jax_param(
                layer.bias,
                NamedSharding(self.jax_config.mesh,
                              self.jax_config.bias_sharding),
                self.jax_config.output_sizes,
                self.jax_config.n_shards,
                self.jax_config.fuse_matmuls,
            )
            delattr(layer, "bias")
            layer.bias = bias

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer, LinearBase)

        with jax.named_scope(layer._get_name()):
            if in_sharding := self.jax_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.jax_config.mesh, in_sharding))

            if self.jax_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

            if out_sharding := self.jax_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.jax_config.mesh, out_sharding))

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)

        outs = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.jax_config.output_sizes, self.jax_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = x.jax()
        outs = []
        for i, weight in enumerate(layer.weight):
            weight_jax = jax_view(weight)

            out = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self,
                 moe: FusedMoEConfig,
                 mesh: Mesh,
                 ep_axis_name: str = 'model'):
        super().__init__(moe)
        self.mesh = mesh
        self.use_kernel = envs.USE_MOE_EP_KERNEL and moe.use_ep
        self.ep_axis_name = ep_axis_name
        # TODO: Use autotune table once we have it.
        self.block_size = {
            "bt": 64,
            "bf": 1024,
            "bd1": 1536,
            "bd2": 1536,
            "btc": 64,
            "bfc": 1024,
            "bd1c": 1536,
            "bd2c": 1536,
        }

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        moe: FusedMoEConfig,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        raise NotImplementedError(
            "Selecting gemm implementation is currently not supported.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)

        num_experts, hidden_size, intermediate_size = w2_weight.shape

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)

        if layer.activation == "swigluoai":
            # When using swigluoai, vLLM splits gmm output in a interleaved way.
            # However, interleaved split is not performant on TPU. Therefore,
            # we preprocess the weight so that splitting gmm output by middle
            # can still get the same result.
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = jnp.concat([w1_weight, w3_weight], axis=1)

            if self.moe.has_bias:
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
            num_experts = w13_weight.shape[0]
            intermediate_size = w13_weight.shape[1] // 2
            hidden_size = w13_weight.shape[2]

            padded_intermediate_size = align_to(intermediate_size, 256)
            padded_hidden_size = align_to(hidden_size, 256)

            # Transpose w2_weight to (num_experts, intermediate_size, hidden_size)
            w13_weight = w13_weight.reshape(num_experts, 2, intermediate_size,
                                            hidden_size)
            w13_weight = jnp.swapaxes(w13_weight, 3, 2)

            w2_weight = jnp.swapaxes(w2_weight, 2, 1)

            w13_weight = jnp.pad(
                w13_weight,
                ((0, 0), (0, 0), (0, padded_hidden_size - hidden_size),
                 (0, padded_intermediate_size - intermediate_size)),
                constant_values=0)

            w2_weight = jnp.pad(
                w2_weight,
                ((0, 0), (0, padded_intermediate_size - intermediate_size),
                 (0, padded_hidden_size - hidden_size)),
                constant_values=0)

            # Apply EP sharding
            ep_sharding = NamedSharding(self.mesh, P("model"))

            w13_weight = jax.device_put(
                w13_weight,
                Format(Layout((0, 1, 2, 3)),
                       NamedSharding(self.mesh, P("model", None, None, None))))
            w2_weight = jax.device_put(
                w2_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P("model", None, None))))

            if self.moe.has_bias:
                w13_bias = w13_bias.astype(jnp.float32).reshape(
                    num_experts, 2, 1, intermediate_size)
                w2_bias = w2_bias.astype(jnp.float32).reshape(
                    num_experts, 1, hidden_size)

                w13_bias = jnp.pad(
                    w13_bias,
                    ((0, 0), (0, 0), (0, 0),
                     (0, padded_intermediate_size - intermediate_size)),
                    constant_values=0)

                w2_bias = jnp.pad(w2_bias,
                                  ((0, 0), (0, 0),
                                   (0, padded_hidden_size - hidden_size)),
                                  constant_values=0)

                # Apply EP sharding
                w13_bias = jax.device_put(
                    w13_bias, Format(Layout((0, 1, 2, 3)), ep_sharding))
                w2_bias = jax.device_put(
                    w2_bias, Format(Layout((0, 1, 2)), ep_sharding))
        else:
            if self.moe.has_bias:
                w13_bias = jnp.expand_dims(w13_bias, 1)
                w2_bias = jnp.expand_dims(w2_bias, 1)

            if layer.use_ep:
                ep_sharding = NamedSharding(self.mesh,
                                            P(ShardingAxisName.EXPERT))
                w13_weight = jax.device_put(
                    w13_weight, Format(Layout((0, 1, 2)), ep_sharding))
                w2_weight = jax.device_put(
                    w2_weight, Format(Layout((0, 1, 2)), ep_sharding))

                if self.moe.has_bias:
                    w13_bias = jax.device_put(
                        w13_bias, Format(Layout((0, 1, 2)), ep_sharding))
                    w2_bias = jax.device_put(
                        w2_bias, Format(Layout((0, 1, 2)), ep_sharding))

            else:
                output_sizes = [intermediate_size, intermediate_size]
                n_shards = get_mesh_shape_product(self.mesh,
                                                  ShardingAxisName.MLP_TENSOR)
                assert intermediate_size % n_shards == 0

                w13_weight = reorder_concatenated_tensor_for_sharding(
                    w13_weight, output_sizes, n_shards, dim=1)
                w13_weight = jax.device_put(
                    w13_weight,
                    Format(
                        Layout((0, 1, 2)),
                        NamedSharding(
                            self.mesh,
                            P(None, ShardingAxisName.MLP_TENSOR, None))))
                w2_weight = jax.device_put(
                    w2_weight,
                    Format(
                        Layout((0, 1, 2)),
                        NamedSharding(
                            self.mesh,
                            P(None, None, ShardingAxisName.MLP_TENSOR))))

                if self.moe.has_bias:
                    w13_bias = reorder_concatenated_tensor_for_sharding(
                        w13_bias, output_sizes, n_shards, dim=2)

                    w13_bias = jax.device_put(
                        w13_bias,
                        Format(
                            Layout((0, 1, 2)),
                            NamedSharding(
                                self.mesh,
                                P(None, None, ShardingAxisName.MLP_TENSOR))))
                    w2_bias = jax.device_put(
                        w2_bias,
                        Format(Layout((0, 1, 2)),
                               NamedSharding(self.mesh, P(None, None, None))))

        layer.w13_weight = Parameter(torch_view(w13_weight),
                                     requires_grad=False)
        layer.w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(torch_view(w13_bias),
                                       requires_grad=False)
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
        w13_bias = w2_bias = None
        if self.moe.has_bias:
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
                b1=w13_bias,
                b2=w2_bias,
                gating_output=gating_output,
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
                w1_scale=None,
                w2_scale=None,
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
