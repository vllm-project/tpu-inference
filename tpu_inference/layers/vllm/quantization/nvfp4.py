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
"""NVFP4 (ModelOpt FP4) quantization support for TPU.

Loads NVFP4 checkpoints (uint8-packed FP4 weights with two-level scales),
dequantizes to float32, then re-quantizes to FP8 blockwise format to reuse
the existing TPU FP8 kernel path.
"""

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.common.quantization.fp8 import Fp8LinearMethod
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

P = PartitionSpec

logger = init_logger(__name__)

# NVFP4 block size (elements per scale group)
NVFP4_GROUP_SIZE = 16

# Re-quantize NVFP4 weights to FP8 with this block size for the TPU kernel.
REQUANT_BLOCK_SIZE = 128


@register_quantization_config(NVFP4)
class VllmNvfp4Config(QuantizationConfig, VllmQuantConfig):
    """NVFP4 quantization config for TPU.

    Standalone implementation that avoids importing from
    vllm.model_executor.layers.quantization.modelopt (which pulls in
    GPU-specific kernel modules that crash on TPU).
    """

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = True,
        kv_cache_quant_algo: Optional[str] = None,
        exclude_modules: Optional[list[str]] = None,
        group_size: int = 16,
    ):
        super().__init__()
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        self.group_size = group_size
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.ignored_layers = exclude_modules or []

    @classmethod
    def get_name(cls):
        return NVFP4

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def override_quantization_method(cls,
                                     hf_quant_cfg,
                                     user_quant,
                                     hf_config=None) -> Optional[str]:
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if quant_method != "modelopt":
            return None
        quant_algo = hf_quant_cfg.get("quant_algo", "")
        if "NVFP4" in quant_algo or "FP4" in quant_algo:
            return NVFP4
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VllmNvfp4Config":
        """Parse HF quantization_config dict into VllmNvfp4Config."""
        if "quantization" in config:
            quant_config = config["quantization"]
            quant_algo = quant_config.get("quant_algo", "")
            kv_cache_quant_method = quant_config.get("kv_cache_quant_algo")
            exclude_modules = quant_config.get("exclude_modules", [])
            group_size = quant_config.get("group_size")
        else:
            quant_algo = config.get("quant_algo", "")
            kv_cache_scheme = config.get("kv_cache_scheme")
            if (isinstance(kv_cache_scheme, dict)
                    and kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8):
                kv_cache_quant_method = "FP8"
            else:
                kv_cache_quant_method = None
            exclude_modules = config.get("ignore", [])
            group_size = config.get("group_size")

        if isinstance(group_size, str):
            group_size = int(group_size)

        # Also check config_groups for group_size
        if group_size is None:
            config_groups = config.get("config_groups", {})
            for group in config_groups.values():
                weights = group.get("weights", {})
                if "group_size" in weights:
                    group_size = weights["group_size"]
                    break

        return cls(
            is_checkpoint_nvfp4_serialized="NVFP4" in str(quant_algo).upper(),
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=exclude_modules or [],
            group_size=group_size or NVFP4_GROUP_SIZE,
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[Union[QuantizeMethodBase]]:
        match layer:
            case LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmNvfp4LinearMethod(self, linear_config)
            case FusedMoE():
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedFusedMoEMethod(layer.moe_config)
                layer.moe_config = self.get_moe_config(layer)
                return VllmNvfp4MoEMethod(self, layer, self.mesh)
            case Attention():
                logger.warning_once(
                    "NVFP4 attention quantization is not implemented. "
                    "Skipping quantization for this layer.")
                return None
            case _:
                return None


def _dequantize_nvfp4_weights(
    weight_packed: jax.Array,
    weight_scale: jax.Array,
    weight_global_scale: jax.Array,
) -> jax.Array:
    """Dequantize NVFP4 packed weights to float32.

    Args:
        weight_packed: uint8 [out, in/2] — 2 FP4 values per byte
        weight_scale: float8_e4m3fn [out, in/group_size] — per-block scales
        weight_global_scale: float32 scalar — per-tensor global scale

    Returns:
        float32 [out, in] — dequantized weights
    """
    # Unpack uint8 → float4_e2m1fn [out, in]
    fp4_weights = u8_unpack_e2m1(weight_packed)

    # Fold global scale into block scale: effective_scale = block_scale * global_scale
    effective_scale = weight_scale.astype(jnp.float32) * weight_global_scale

    # Dequantize using block scales
    return dequantize_tensor(
        fp4_weights,
        effective_scale,
        axis=(0, 1),
        out_dtype=jnp.float32,
    )


class VllmNvfp4LinearMethod(Fp8LinearMethod):
    """NVFP4 linear method for TPU.

    Loads NVFP4 weights, dequantizes, re-quantizes to FP8 blockwise,
    then uses the existing FP8 kernel path for inference.
    """

    def __init__(
        self,
        quant_config: VllmNvfp4Config,
        linear_config: VllmQuantLinearConfig,
    ):
        Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register NVFP4 weight parameters for checkpoint loading."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        group_size = getattr(self.quant_config, 'group_size', NVFP4_GROUP_SIZE)

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                f"Input size {input_size_per_partition} must be divisible by "
                f"NVFP4 group size {group_size}")

        # Packed FP4 weight: 2 values per uint8 byte
        from vllm.model_executor.parameter import (ModelWeightParameter,
                                                   PerTensorScaleParameter)

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Per-tensor input activation scale
        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

        # Per-tensor weight global scale
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per-block weight scale (E4M3)
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequant NVFP4 → requant to FP8 blockwise for TPU kernel."""
        assert isinstance(layer, LinearBase)

        # Convert torch tensors to JAX
        weight_packed = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)

        # Collapse per-partition global scales to single scalar via max()
        weight_global_scale = t2j(layer.weight_scale_2, use_dlpack=False)
        weight_global_scale = jnp.max(weight_global_scale)

        # Clean up original attributes
        for attr in ('weight', 'weight_scale', 'weight_scale_2',
                     'input_scale'):
            if hasattr(layer, attr):
                delattr(layer, attr)

        # Handle bias
        if hasattr(
                layer,
                'bias') and layer.bias is not None and not layer.skip_bias_add:
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, 'bias')
        else:
            bias = None

        # Dequantize NVFP4 to float32
        dequantized = _dequantize_nvfp4_weights(weight_packed, weight_scale,
                                                weight_global_scale)

        # Re-quantize to FP8 blockwise
        requant_dtype = jnp.float8_e4m3fn
        requant_block_size = REQUANT_BLOCK_SIZE

        # Process per output partition (for QKV parallel etc.)
        output_sizes = tuple(self.linear_config.output_sizes)
        weights_list = []
        scales_list = []
        start = 0
        for output_size in output_sizes:
            end = start + output_size
            w_slice = dequantized[start:end]
            w_q, w_s = quantize_tensor(requant_dtype,
                                       w_slice,
                                       block_size=requant_block_size)
            weights_list.append(w_q)
            scales_list.append(w_s)
            start = end

        fp8_weight = jnp.concatenate(weights_list, axis=0)
        fp8_scale = jnp.concatenate(scales_list, axis=0)

        # Process for TP sharding and reordering
        weights = process_linear_weights(
            LinearWeights(
                weight=fp8_weight,
                weight_scale=fp8_scale,
                zero_point=None,
                bias=bias,
            ),
            fused=self.linear_config.fuse_matmuls,
            output_sizes=output_sizes,
            reorder_size=self.linear_config.n_shards,
        )

        # Reshape scale for blockwise kernel: [out, in/block] → [in/block, 1, out]
        if self.linear_config.enable_quantized_matmul_kernel:
            weights.weight_scale = jnp.expand_dims(jnp.transpose(
                weights.weight_scale),
                                                   axis=1)

        # Shard to TPU mesh
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            layer.weight_scale = Parameter(weights.weight_scale,
                                           requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            layer.weight_scale = to_parameter_list(weights.weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using FP8 kernel (weights already converted to FP8)."""
        with jax.named_scope(layer._get_name()):
            x_jax = jax_view(x)
            bias_jax = jax_view(
                bias) if bias is not None and not layer.skip_bias_add else None
            if self.linear_config.fuse_matmuls:
                weight_jax = jax_view(layer.weight)
                weight_scale_jax = jax_view(layer.weight_scale)
                out = self._apply_fused(x_jax, weight_jax, weight_scale_jax,
                                        bias_jax)
            else:
                assert isinstance(layer.weight, torch.nn.ParameterList)
                assert isinstance(layer.weight_scale, torch.nn.ParameterList)
                weight_and_scale = [
                    (jax_view(w), jax_view(s))
                    for w, s in zip(layer.weight, layer.weight_scale)
                ]
                if bias is not None and not layer.skip_bias_add:
                    assert isinstance(bias, torch.nn.ParameterList)
                    bias_jax = [jax_view(b) for b in bias]
                out = self._apply_split(x_jax,
                                        weight_and_scale,
                                        bias_jax,
                                        mesh=self.linear_config.mesh)
            return torch_view(out)


class VllmNvfp4MoEMethod(FusedMoEMethodBase):
    """NVFP4 MoE method for TPU.

    Dequantizes NVFP4 MoE weights to float32, then re-quantizes to FP4
    with larger block size using the existing MoE quantization pipeline.
    """

    def __init__(
        self,
        quant_config: VllmNvfp4Config,
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

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register NVFP4 MoE weight parameters."""
        from vllm.model_executor.utils import set_weight_attrs

        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        group_size = getattr(self.quant_config, 'group_size', NVFP4_GROUP_SIZE)

        # Packed FP4 weights: 2 values per uint8 byte
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_size // 2,
                        dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        hidden_size,
                        intermediate_size_per_partition // 2,
                        dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-block weight scales (E4M3)
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_size // group_size,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts,
                        hidden_size,
                        intermediate_size_per_partition // group_size,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Per-tensor global scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Bias (optional)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            2 * intermediate_size_per_partition,
                            dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequant NVFP4 MoE weights → requant for TPU MoE kernel."""
        assert isinstance(layer, FusedMoE)

        # Load packed weights and block scales
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        # Load global scales — shape [E, 2] for w13, [E] for w2
        w13_global_scale = t2j(layer.w13_weight_scale_2, use_dlpack=False)
        w2_global_scale = t2j(layer.w2_weight_scale_2, use_dlpack=False)

        # Handle bias if present
        w13_bias = t2j(layer.w13_bias,
                       use_dlpack=False) if self.moe.has_bias else None
        w2_bias = t2j(layer.w2_bias,
                      use_dlpack=False) if self.moe.has_bias else None

        @jax.jit
        def process_nvfp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_global_scale,
            w2_weight,
            w2_weight_scale,
            w2_global_scale,
            w13_bias,
            w2_bias,
        ):
            # Unpack uint8 → float4_e2m1fn
            w13_fp4 = u8_unpack_e2m1(w13_weight)
            w2_fp4 = u8_unpack_e2m1(w2_weight)

            # Fold global scales into block scales and dequantize
            # w13_global_scale shape: [E, 2] or [E] — broadcast to match block scale
            if w13_global_scale.ndim == 2:
                # [E, 2] → [E, 2, 1] for broadcasting with [E, 2*I, H/16]
                # But block scale is [E, N, M] where N = fused dim
                # We need to split, apply separately, then concat
                half_scale_n = w13_weight_scale.shape[1] // 2

                w1_scale_eff = w13_weight_scale[:, :half_scale_n].astype(
                    jnp.float32) * w13_global_scale[:, 0:1].reshape(-1, 1)
                w3_scale_eff = w13_weight_scale[:, half_scale_n:].astype(
                    jnp.float32) * w13_global_scale[:, 1:2].reshape(-1, 1)
                w13_scale_eff = jnp.concatenate([w1_scale_eff, w3_scale_eff],
                                                axis=1)

                w13_dequant = dequantize_tensor(w13_fp4,
                                                w13_scale_eff,
                                                axis=(1, 2),
                                                out_dtype=jnp.float32)
            else:
                w13_scale_eff = w13_weight_scale.astype(
                    jnp.float32) * w13_global_scale.reshape(-1, 1, 1)
                w13_dequant = dequantize_tensor(w13_fp4,
                                                w13_scale_eff,
                                                axis=(1, 2),
                                                out_dtype=jnp.float32)

            if w2_global_scale.ndim == 1:
                w2_scale_eff = w2_weight_scale.astype(
                    jnp.float32) * w2_global_scale.reshape(-1, 1, 1)
            else:
                w2_scale_eff = w2_weight_scale.astype(
                    jnp.float32) * w2_global_scale
            w2_dequant = dequantize_tensor(w2_fp4,
                                           w2_scale_eff,
                                           axis=(1, 2),
                                           out_dtype=jnp.float32)

            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            # Re-quantize to FP4 with larger block size for kernel efficiency
            requant_block_size = 512
            weights = quantize_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_dequant,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_dequant,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                jnp.float4_e2m1fn,
                requant_block_size,
            )

            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_nvfp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_global_scale,
            w2_weight,
            w2_weight_scale,
            w2_global_scale,
            w13_bias,
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
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(
            layer=layer,
            weights=weights,
            quant_method_instance=self,
            x=x,
            router_logits=router_logits,
        )
