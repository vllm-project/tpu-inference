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

from typing import Callable

import jax
import jax.numpy as jnp
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.linear import \
    register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization import modelopt as vllm_modelopt
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.parameter import (ModelWeightParameter,
                                           PerTensorScaleParameter)

from tpu_inference import envs
from tpu_inference.layers.common import quant_methods
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, shard_linear_weights)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import MODELOPT_NVFP4
from tpu_inference.layers.common.quantization import dequantize_nvfp4
from tpu_inference.layers.common.quantization import \
    unquantized as common_unquantized
from tpu_inference.layers.common.sharding import ShardingAxisName
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


@register_quantization_config(MODELOPT_NVFP4)
class VllmModelOptNvFp4Config(vllm_modelopt.ModelOptNvFp4Config,
                              VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return quant_methods.MODELOPT_NVFP4

    def get_quant_method(self, layer, prefix):
        if self.is_layer_excluded(prefix):
            if isinstance(layer, vllm_linear.LinearBase):
                return VllmUnquantizedLinearMethod(
                    self.get_linear_config(layer))
            return None
        if isinstance(layer, vllm_linear.LinearBase):
            return VllmModelOptNvFp4LinearMethod(self,
                                                 self.get_linear_config(layer))
        elif isinstance(layer, FusedMoE):
            return VllmModelOptNvFp4FusedMoE(self, layer, self.mesh)
        return super().get_quant_method(layer, prefix)


@register_weight_loader_v2_supported_method
class VllmModelOptNvFp4LinearMethod(vllm_modelopt.ModelOptNvFp4LinearMethod,
                                    VllmUnquantizedLinearMethod):

    def __init__(self, quant_config: VllmModelOptNvFp4Config,
                 linear_config: VllmQuantLinearConfig):
        self.quant_config = quant_config
        self.marlin_input_dtype = None
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # NVFP4 weight processing for TPU.

        weight = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        weight_global_scale_tensor = t2j(layer.weight_scale_2,
                                         use_dlpack=False)
        weight_global_scale = weight_global_scale_tensor.max()

        if envs.DISABLE_WEIGHT_REQUANTIZATION:
            # When weight requantization is disabled, we keep the weights in
            # their original quantized format and dequantize them during matmul.
            mesh = self.linear_config.mesh
            # weights is [out, in/2], weight_scale is [out, in/group_size]
            weights = LinearWeights(weight=weight,
                                    weight_scale=weight_scale,
                                    zero_point=None,
                                    bias=None)
            # shard_linear_weights assumes transposed=True by default,
            # which means it shards the first dimension as 'out'.
            sharded_weights = shard_linear_weights(
                weights,
                mesh,
                self.linear_config.weight_sharding,
                self.linear_config.bias_sharding,
            )

            # Replace parameters
            delattr(layer, "weight")
            delattr(layer, "weight_scale")
            delattr(layer, "input_scale")
            delattr(layer, "weight_scale_2")

            layer.weight = Parameter(torch_view(sharded_weights.weight),
                                     requires_grad=False)
            layer.weight_scale = Parameter(torch_view(
                sharded_weights.weight_scale),
                                           requires_grad=False)
            layer.weight_scale_2 = Parameter(
                torch_view(weight_global_scale_tensor), requires_grad=False)
            return

        @jax.jit
        def dequantize_and_process(weight, weight_scale, weight_global_scale):
            return dequantize_nvfp4(
                weight,
                weight_scale,
                weight_global_scale,
                group_size=self.quant_config.group_size,
                out_dtype=jnp.bfloat16,
            )

        dequantized_weight = dequantize_and_process(weight, weight_scale,
                                                    weight_global_scale)

        # Now we have BF16 weights, we can shard them.
        mesh = self.linear_config.mesh
        # TPU Inference usually expects transposed weight [in, out] for Linear.
        # vLLM's NVFP4 weight is [out, in].
        dequantized_weight = dequantized_weight.T

        weights = LinearWeights(weight=dequantized_weight,
                                weight_scale=None,
                                zero_point=None,
                                bias=None)
        sharded_weights = shard_linear_weights(
            weights,
            mesh,
            self.linear_config.weight_sharding,
            self.linear_config.bias_sharding,
        )

        # Replace parameters
        delattr(layer, "weight")
        delattr(layer, "weight_scale")
        delattr(layer, "input_scale")
        delattr(layer, "weight_scale_2")

        layer.weight = Parameter(torch_view(sharded_weights.weight),
                                 requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        if not envs.DISABLE_WEIGHT_REQUANTIZATION:
            return super().apply(layer, x, bias)

        with jax.named_scope(layer._get_name()):
            if in_sharding := self.linear_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.linear_config.mesh, in_sharding))

            x_jax = jax_view(x)
            weight = jax_view(layer.weight)
            weight_scale = jax_view(layer.weight_scale)
            weight_global_scale = jax_view(layer.weight_scale_2).max()

            # Dequantize during matmul
            # layer.weight is [out, in/2]
            # layer.weight_scale is [out, in/group_size]
            dequantized_weight = dequantize_nvfp4(
                weight,
                weight_scale,
                weight_global_scale,
                group_size=self.quant_config.group_size,
                out_dtype=x_jax.dtype,
            )
            # Matmul expects [in, out]
            dequantized_weight = dequantized_weight.T

            out_jax = jnp.matmul(x_jax, dequantized_weight)
            if bias is not None:
                out_jax += jax_view(bias)

            out = torch_view(out_jax)
            if out_sharding := self.linear_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.linear_config.mesh,
                                         out_sharding))
            return out


class VllmModelOptNvFp4FusedMoE(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: VllmModelOptNvFp4Config,
        layer: torch.nn.Module,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)
        # TODO?
        # self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
        #     self.nvfp4_backend
        # )
        self.use_global_sf = False

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self) -> Callable:
        return self.apply_monolithic

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called.")

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        FP4 variants use 'weight_scale_2' pattern for per-tensor weight scales.
        """
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

        assert self.quant_config.is_checkpoint_nvfp4_serialized

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        global_num_experts = extra_weight_attrs.get("global_num_experts")
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition //
                self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

        global_sf_num_experts = (global_num_experts
                                 if self.use_global_sf else num_experts)
        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(
                global_sf_num_experts,
                w13_num_shards,
                dtype=torch.float32,
            ),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(global_sf_num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w13_weight_scale_2 = t2j(layer.w13_weight_scale_2, use_dlpack=False)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w2_weight_scale_2 = t2j(layer.w2_weight_scale_2, use_dlpack=False)

        if envs.DISABLE_WEIGHT_REQUANTIZATION:
            tp_size = get_mesh_shape_product(self.mesh,
                                             ShardingAxisName.ATTN_HEAD)

            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=w13_weight_scale,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=w2_weight_scale,
                w2_bias=None,
            )
            # process_moe_weights handles transposes and padding needed for
            # the MoE kernels.
            weights = process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=tp_size,
            )
            weights = torch_view(
                shard_moe_weights(weights, self.moe_backend, self.mesh))

            # Replace parameters
            delattr(layer, "w13_weight")
            delattr(layer, "w13_weight_scale")
            delattr(layer, "w13_weight_scale_2")
            delattr(layer, "w13_input_scale")
            delattr(layer, "w2_weight")
            delattr(layer, "w2_weight_scale")
            delattr(layer, "w2_weight_scale_2")
            delattr(layer, "w2_input_scale")

            layer.w13_weight = Parameter(weights.w13_weight,
                                         requires_grad=False)
            layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                               requires_grad=False)
            layer.w13_weight_scale_2 = Parameter(
                torch_view(w13_weight_scale_2), requires_grad=False)
            layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
            layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                              requires_grad=False)
            layer.w2_weight_scale_2 = Parameter(torch_view(w2_weight_scale_2),
                                                requires_grad=False)
            return

        @jax.jit
        def dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        ):
            # w13_weight: [E, I*S, H/2]
            # w13_weight_scale: [E, I*S, H/group_size]
            # w13_weight_scale_2: [E, S]

            e, intermediate_shards, half_h = w13_weight.shape
            h = half_h * 2
            group_size = self.quant_config.group_size

            # Handle w13
            w13_flat = w13_weight.reshape(-1, half_h)
            w13_scale_flat = w13_weight_scale.reshape(-1, h // group_size)
            s = w13_weight_scale_2.shape[1]
            w13_gscale_flat = jnp.repeat(w13_weight_scale_2,
                                         intermediate_shards // s,
                                         axis=1).reshape(-1, 1)

            w13_dq = dequantize_nvfp4(w13_flat, w13_scale_flat,
                                      w13_gscale_flat, group_size,
                                      jnp.bfloat16)
            w13_dq = w13_dq.reshape(e, intermediate_shards, h)

            # Handle w2
            e, h, half_i = w2_weight.shape
            i = half_i * 2
            w2_flat = w2_weight.reshape(-1, half_i)
            w2_scale_flat = w2_weight_scale.reshape(-1, i // group_size)
            w2_gscale_flat = w2_weight_scale_2.reshape(-1, 1)

            w2_dq = dequantize_nvfp4(w2_flat, w2_scale_flat, w2_gscale_flat,
                                     group_size, jnp.bfloat16)
            w2_dq = w2_dq.reshape(e, h, i)

            return w13_dq, w2_dq

        w13_dq, w2_dq = dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        )

        weights = common_unquantized.process_unquantized_moe_weights(
            mesh=self.mesh,
            moe_backend=self.moe_backend,
            activation=layer.activation,
            w13_weight=w13_dq,
            w13_bias=None,
            w2_weight=w2_dq,
            w2_bias=None)

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        # Replace parameters
        delattr(layer, "w13_weight")
        delattr(layer, "w13_weight_scale")
        delattr(layer, "w13_weight_scale_2")
        delattr(layer, "w13_input_scale")
        delattr(layer, "w2_weight")
        delattr(layer, "w2_weight_scale")
        delattr(layer, "w2_weight_scale_2")
        delattr(layer, "w2_input_scale")

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_monolithic

        if not envs.DISABLE_WEIGHT_REQUANTIZATION:
            # Current implementation only supports dequantized weights in apply_monolithic
            # if they were dequantized at load time.
            weights = FusedMoEWeights(
                w13_weight=jax_view(layer.w13_weight),
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=jax_view(layer.w2_weight),
                w2_weight_scale=None,
                w2_bias=None,
            )
            return vllm_moe_apply(layer=layer,
                                  weights=weights,
                                  quant_method_instance=self,
                                  x=x,
                                  router_logits=router_logits)

        # DISABLE_WEIGHT_REQUANTIZATION=1 path: dequantize at runtime
        w13_weight = jax_view(layer.w13_weight)
        w13_weight_scale = jax_view(layer.w13_weight_scale)
        w13_weight_scale_2 = jax_view(layer.w13_weight_scale_2)
        w2_weight = jax_view(layer.w2_weight)
        w2_weight_scale = jax_view(layer.w2_weight_scale)
        w2_weight_scale_2 = jax_view(layer.w2_weight_scale_2)

        group_size = self.quant_config.group_size

        @jax.jit
        def dequantize_runtime(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        ):
            # w13_weight is [E, H/2, 2*I] (sharded and transposed by process_moe_weights)
            # w13_weight_scale is [E, H/group_size, 1, 2*I]
            # w2_weight is [E, 2*I/2, H] -> actually w2 is [E, I/2, H] in process_moe_weights?
            # Wait, w2_weight in process_moe_weights:
            # w2_weight = jnp.swapaxes(w2_weight, 1, 2)
            # Original w2 was [E, H, I/2]. swapaxes(1, 2) -> [E, I/2, H]

            # w13 dequant
            # dequantize_nvfp4 expects [N, K/2].
            # Here N=2*I, K=H.
            # w13_weight is [E, K/2, N].
            w13_dq = jax.vmap(lambda w, s, s2: dequantize_nvfp4(
                w.T,
                s.transpose(2, 1, 0).reshape(-1, w.shape[0] * 2 // group_size),
                s2.max(),
                group_size,
                jnp.bfloat16,
            ).T)(w13_weight, w13_weight_scale, w13_weight_scale_2)

            # w2 dequant
            # dequantize_nvfp4 expects [N, K/2].
            # Here N=H, K=I.
            # w2_weight is [E, K/2, N].
            w2_dq = jax.vmap(lambda w, s, s2: dequantize_nvfp4(
                w.T,
                s.transpose(2, 1, 0).reshape(-1, w.shape[0] * 2 // group_size),
                s2.max(),
                group_size,
                jnp.bfloat16,
            ).T)(w2_weight, w2_weight_scale, w2_weight_scale_2)

            return w13_dq, w2_dq

        w13_dq, w2_dq = dequantize_runtime(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        )

        weights = FusedMoEWeights(
            w13_weight=w13_dq,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_dq,
            w2_weight_scale=None,
            w2_bias=None,
        )

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return None
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    @property
    def supports_eplb(self) -> bool:
        return True
