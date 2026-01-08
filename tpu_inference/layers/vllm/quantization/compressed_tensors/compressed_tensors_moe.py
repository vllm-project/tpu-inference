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

import jax
import torch
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import Mesh
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod, CompressedTensorsW8A8Fp8MoEMethod)

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.fused_moe import (fused_moe_apply,
                                                 select_moe_backend)
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

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
                                            VllmQuantConfig):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        mesh: Mesh,
    ):
        super().__init__(weight_quant, input_quant, moe)

        self.mesh = mesh
        self.moe_backend = select_moe_backend(self.moe)

        self.extra_backend_kwargs = {}

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
        # who is the caller?: tpu_runner.load_model->vllm_model_wraper.load_weights->process_weights_after_loading.
        # what are the shape of the weights?
        print('xw32 process_weights_after_loading begins.')
        # xw32: Comment out this
        # Below didn't work
        # if not layer.layer_name.startswith('model.layers.0.') and not layer.layer_name.startswith('model.layers.1.'):
        #     return

        assert isinstance(layer, FusedMoE)

        # xw32 N.B (TODO: cleanup, remove the concrete number and leave the rest.):
        # layer.w13_weight: [160, 5120, 6144]=[num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [160, 5120, 1]=[num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [160, 6144, 2560]=[num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [160, 6144, 1]=[num_experts, hidden_size, 1]
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_fp8_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                weights=FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_scale_per_channel=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
                    w2_scale_per_channel=None,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_fp8_moe_weights(
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
        layer.w13_scale_per_channel = Parameter(weights.w13_scale_per_channel,
                                                requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)
        layer.w2_scale_per_channel = Parameter(weights.w2_scale_per_channel,
                                               requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)
        print('xw32 process_weights_after_loading ends.')

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
