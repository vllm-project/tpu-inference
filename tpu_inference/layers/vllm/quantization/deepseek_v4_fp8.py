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
"""TPU quant config for DeepSeek-V4's "deepseek_v4_fp8" checkpoint format.

DeepSeek-V4 uses FP8 block-quant for all linear/attention layers, but MoE
expert weights vary by checkpoint:
  - expert_dtype="fp4" (e.g. DeepSeek-V4-Flash): MXFP4 packed uint8 weights
    with ue8m0 (float8_e8m0fnu) exponent scales stored as "weight_scale" (no _inv).
  - expert_dtype="fp8" (e.g. DeepSeek-V4-Flash-Base): FP8 block-quant weights
    with float32 scales stored as "weight_scale_inv" (pre-inverted).

The naming distinction matters:
  weight_scale     — scale applied directly: output = dequant(w) * scale
  weight_scale_inv — pre-inverted scale:    output = dequant(w) * weight_scale_inv
                     (avoids a divide, used by VllmFp8MoEMethod)
"""

from typing import Optional, Union

import jax
import torch
import torch.nn as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config

from tpu_inference.layers.common.quant_methods import DSV4_FP8
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.fp8 import (
    VllmFp8LinearMethod, VllmFp8MoEMethod)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

logger = init_logger(__name__)


class VllmDeepseekV4Fp4MoEMethod(FusedMoEMethodBase):
    """Skeleton stub for DeepSeek-V4 FP4 (MXFP4) expert MoE on TPU.

    FP4 checkpoint parameter naming:
      checkpoint key:   experts.N.w{1,3}.scale  (ue8m0 exponent, not pre-inverted)
      mapped name:      experts.N.w{1,3}.weight_scale
      param name here:  w13_weight_scale (no _inv — scale is applied directly)

    TODO: implement real MXFP4 dequantization and MoE forward for JAX/TPU.
    """

    def __init__(self, moe_config, mesh: Mesh) -> None:
        super().__init__(moe_config)
        self.mesh = mesh
        # TODO (3d): add self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        # to use by `shard_moe_weights` in `process_weights_after_loading`.

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Allocates empty parameter buffers that vLLM's weight loader fills from
        # the checkpoint. Parameter names must match the mapped names produced by
        # _make_deepseek_v4_weights_mapper("fp4") in deepseek_v4.py.
        #
        # Weights are packed into uint8 in the checkpoint so the hidden dimension
        # is halved relative to the logical weight shape.
        
        mxfp4_block = 32 # each ue8m0 scale covers a block of 32 FP4 elements,

        # w13_weight: fused gate (w1) + up-projection (w3) weights.
        # Logical shape would be [E, 2*inter, hidden] but MXFP4 packs 2 values/byte
        # → [E, 2*inter, hidden//2] uint8.
        w13_weight = nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # w2_weight: down-projection weights.
        # Logical shape [E, hidden, inter] → packed [E, hidden, inter//2] uint8.
        w2_weight = nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # w13_weight_scale: one ue8m0 scale per 32 FP4 elements
        # along the hidden dimension. Stored as uint8.
        # Shape: [E, 2*inter, hidden//32].
        w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        # w2_weight_scale: same ue8m0 format for the down-projection.
        # Shape: [E, hidden, inter//32].
        w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = "block"
        set_weight_attrs(w13_weight_scale, scale_attrs)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        # FusedMoEQuantConfig is only needed for modular GPU kernels so 
        # set as a no-op for GPU."
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # TODO  replace this skeleton with the full pipeline:
        #   1. Dequantize: uint8 packed + ue8m0 scales → bfloat16
        #   2. quantize_moe_weights(bfloat16_weights, desired_dtype, block_size)
        #      requantizes to kernel dtype (e.g. fp8 e4m3) and pads to alignment.
        #   3. shard_moe_weights(weights, self.moe_backend, self.mesh)
        #      applies TP or EP sharding based on self.moe_backend
        #      This replaces the hardcoded EP temporary workaround.
        #
        # The skeleton bypasses shard_moe_weights because its Layout objects
        # assume FP8 4D weight shapes incompatible with our 3D MXFP4 tensors.
        # Direct jax.device_put with EP sharding prevents HBM OOM for now.
        ep_sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))
        for attr in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            tensor = getattr(layer, attr)
            jax_array = t2j(tensor, use_dlpack=False)
            jax_array = jax.device_put(jax_array, ep_sharding)
            setattr(layer, attr, Parameter(torch_view(jax_array), requires_grad=False))

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO: implement real MXFP4 MoE forward for TPU (currently a stub):
        #   1. Dequantize w13/w2 from MXFP4 using ue8m0 scales
        #   2. Run MoE routing and expert matmuls (using existing fused_moe_gmm.py)
        #
        # For step 2 above, pre-compute topk_ids from the lookup table
        # instead of softmax routing if hash-MoE layers are detected:
        #   hash_table = getattr(layer, 'hash_indices_table', None)
        #   precomputed_topk_ids = jax_view(hash_table[input_ids]) if (
        #       hash_table is not None and input_ids is not None) else None
        # Then pass precomputed_topk_ids to vllm_moe_apply.
        return torch.zeros(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)


@register_quantization_config(DSV4_FP8)
class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):
    """TPU quantization config for "deepseek_v4_fp8" format.
    Registered under "deepseek_v4_fp8".

    Dispatches MoE quant methods based on the checkpoint's expert_dtype:
      - "fp4" → VllmDeepseekV4Fp4MoEMethod (w13_weight_scale, no _inv)
      - "fp8" → VllmFp8MoEMethod            (w13_weight_scale_inv)
    Linear layers always use VllmFp8LinearMethod (FP8 block-quant).
    """

    @classmethod
    def get_name(cls) -> str:
        return DSV4_FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[Union[vllm_linear.LinearMethodBase, QuantizeMethodBase]]:
        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmFp8LinearMethod(self, linear_config)

            case FusedMoE():
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedFusedMoEMethod(layer.moe_config)

                if self.expert_dtype == "fp4":
                    logger.info_once(
                        "Using VllmDeepseekV4Fp4MoEMethod (MXFP4 stub) "
                        "for expert_dtype='fp4' MoE layer at %s", prefix)
                    return VllmDeepseekV4Fp4MoEMethod(layer.moe_config, self.mesh)
                else:
                    # expert_dtype == "fp8": use standard FP8 block-quant MoE
                    logger.info_once(
                        "Using VllmFp8MoEMethod for expert_dtype='fp8' "
                        "MoE layer at %s", prefix)
                    layer.moe_config = self.get_moe_config(layer)
                    return VllmFp8MoEMethod(self, layer, self.mesh)

            case Attention():
                logger.warning_once(
                    "FP8KVCacheMethod is not implemented for deepseek_v4_fp8. "
                    "Skipping quantization for attention layer.")
                return None

            case _:
                return None