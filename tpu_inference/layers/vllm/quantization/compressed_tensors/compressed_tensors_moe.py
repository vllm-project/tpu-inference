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

from typing import Union

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationArgs
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod, CompressedTensorsW8A8Fp8MoEMethod)

from tpu_inference.layers.vllm.fused_moe import fused_moe_func
from tpu_inference.layers.vllm.linear_common import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod

logger = init_logger(__name__)


class VllmCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):

    @staticmethod
    def get_moe_method(
        quant_config: "VllmCompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> CompressedTensorsMoEMethod:
        assert isinstance(layer, FusedMoE)

        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError("All MoE projections need to have same "
                             "quantization scheme but found multiple")

        if scheme_dict is None:
            return VllmUnquantizedFusedMoEMethod(layer.moe_config,
                                                 quant_config.mesh)

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        if quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return VllmCompressedTensorsW8A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config, quant_config.mesh)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod,
                                            JaxCommonConfig):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        mesh: Mesh,
    ):
        super().__init__(weight_quant, input_quant, moe)
        self.mesh = mesh

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Docstring for process_weights_after_loading

        :param self: Description
        :param layer: Description
        :type layer: torch.nn.Module

        Steps:
        1. Read weights from layer object and convert to jax arrays
        2. Interleave concat w13 weights
        3. Shard weights for tp (rowwise w13, colwise w2)
        4. Initialize Params as torch.nn.Parameter
            a. w13_weight - float8_e4m3fn shape: (num_experts, 2 x intermediate_size, input_size)
            b. w2_weight - float8_e4m3fn shape: (num_experts, output_size, intermediate_size)
            c. w13_weight_scale - FP32 shape: (num_experts, 2 x intermediate_size, 1)
            d. w2_weight_scale - FP32shape: (num_experts, output_size, 1)
        """
        assert isinstance(layer, FusedMoE)

        # Read weights from layer object
        w13_weight = t2j(
            layer.w13_weight, use_dlpack=False
        )  # float8_e4m3fn shape: (num_experts, 2 x intermediate_size, input_size)
        w13_weight_scale = t2j(
            layer.w13_weight_scale, use_dlpack=False
        )  # FP32 shape: (num_experts, 2 x intermediate_size, 1)
        w2_weight = t2j(
            layer.w2_weight, use_dlpack=False
        )  # float8_e4m3fn shape: (num_experts, output_size, intermediate_size)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w13_weight_scale = w13_weight_scale.astype(jnp.bfloat16)
        w2_weight_scale = w2_weight_scale.astype(jnp.bfloat16)
        intermediate_size = layer.w13_weight.shape[1] // 2
        assert intermediate_size == w2_weight.shape[-1]
        n_shards = self.mesh.shape["model"]
        assert intermediate_size % n_shards == 0
        num_experts, hidden_size, intermediate_size = w2_weight.shape
        assert w2_weight_scale.shape == (num_experts, hidden_size, 1)
        assert w13_weight.shape == (num_experts, 2 * intermediate_size,
                                    hidden_size)
        assert w13_weight_scale.shape == (num_experts, 2 * intermediate_size,
                                          1)

        if not layer.use_ep:
            # Interleave concat w13 weights
            w13_weight = reorder_concatenated_tensor_for_sharding(
                w13_weight,
                split_sizes=(intermediate_size, intermediate_size),
                dim=1,
                n_shards=n_shards,
            )
            # Interleave concat w13 weight scales
            w13_weight_scale = reorder_concatenated_tensor_for_sharding(
                w13_weight_scale,
                split_sizes=(intermediate_size, intermediate_size),
                dim=1,
                n_shards=n_shards,
            )

        # 160,5120,1 -> 160,1,5120
        w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        # 160,1,5120 -> 160, 1, 1, 5120   (num_experts, num_blocks, 1, outer_dim)
        w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
        w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
        w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)

        if layer.use_ep:
            # Apply EP sharding
            ep_sharding = NamedSharding(self.mesh, P("model"))

            w13_weight = jax.lax.with_sharding_constraint(
                w13_weight, ep_sharding)
            w2_weight = jax.lax.with_sharding_constraint(
                w2_weight, ep_sharding)

            w13_weight_scale = jax.lax.with_sharding_constraint(
                w13_weight_scale, ep_sharding)
            w2_weight_scale = jax.lax.with_sharding_constraint(
                w2_weight_scale, ep_sharding)

        else:
            # Shard weights for tp (rowwise w13, colwise w2)
            w13_format = Format(
                Layout((0, 1, 2)),  # expert, 2xintermed, input
                NamedSharding(self.mesh, P(None, "model", None)),
            )  # rowwise sharding on intermed dim

            w13_scale_format = Format(
                Layout(
                    (0, 1, 2, 3)),  #  (num_experts, num_blocks, 1, outer_dim)
                NamedSharding(self.mesh, P(None, None, None, "model")),
            )  # col wise GMM sharding on intermed dim

            # Local shard shape: (num_experts, 2 x (intermediate_size // n_shards), input_size)
            w13_weight = jax.lax.with_sharding_constraint(
                w13_weight, w13_format)
            # Local shard shape: (num_experts, (intermediate_size // n_shards), 1)
            w13_weight_scale = jax.lax.with_sharding_constraint(
                w13_weight_scale, w13_scale_format)

            # Shard weights for tp (colwise w2)
            w2_format = Format(
                Layout((0, 1, 2)),  # expert, intermed, hidden
                NamedSharding(self.mesh, P(None, None, "model")),
            )
            # Local shard shape: (num_experts, hidden, (intermediate_size // n_shards))
            # #  (num_experts, num_blocks, 1, outer_dim)
            w2_weight = jax.lax.with_sharding_constraint(w2_weight, w2_format)

            w2_scale_format = Format(
                Layout((0, 1, 2, 3)),  # expert, intermed, 1
                NamedSharding(self.mesh, P(None, None, None, None)),
            )
            # Local shard shape: (num_experts, intermediate_size // n_shards, 1)
            w2_weight_scale = jax.lax.with_sharding_constraint(
                w2_weight_scale, w2_scale_format)

        w13_weight = Parameter(torch_view(w13_weight), requires_grad=False)
        w13_weight_scale = Parameter(torch_view(w13_weight_scale),
                                     requires_grad=False)
        w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)
        w2_weight_scale = Parameter(torch_view(w2_weight_scale),
                                    requires_grad=False)

        layer.w13_weight = w13_weight
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight = w2_weight
        layer.w2_weight_scale = w2_weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if layer.activation != "silu":
            raise NotImplementedError(
                "Only silu is supported for activation function.")
        if layer.scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        # TODO: Use MoE kernel when it supports fp8
        x = jax_view(x)
        w13_weight = jax_view(layer.w13_weight)
        w2_weight = jax_view(layer.w2_weight)
        w13_weight_scale = jax_view(layer.w13_weight_scale)
        w2_weight_scale = jax_view(layer.w2_weight_scale)
        gating_output = jax_view(router_logits)
        out = torch_view(
            fused_moe_func(
                hidden_states=x,
                w1=w13_weight,
                w2=w2_weight,
                w1_scale=w13_weight_scale,
                w2_scale=w2_weight_scale,
                w1_bias=None,
                w2_bias=None,
                gating_output=gating_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
                mesh=self.mesh,
                use_ep=layer.use_ep,
                activation=layer.activation,
            ))

        return out
