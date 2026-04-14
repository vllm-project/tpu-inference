import torch
import jax
from jax.sharding import NamedSharding, PartitionSpec
from typing import Optional

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import CompressedTensorsW4A16Fp4
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
from torch.nn.parameter import Parameter

from tpu_inference.layers.vllm.quantization.configs import VllmQuantLinearConfig
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.quantization.unquantized import _load_weight_for_layer, VllmUnquantizedLinearMethod as TPUUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from torchax.interop import torch_view

P = PartitionSpec
logger = init_logger(__name__)

class VllmCompressedTensorsW4A16Fp4(CompressedTensorsW4A16Fp4):
    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__()
        self.linear_config = linear_config
        self.unquantized_fallback = TPUUnquantizedLinearMethod(linear_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: callable,
        **kwargs,
    ):
        super().create_weights(layer, output_partition_sizes, input_size_per_partition, params_dtype, weight_loader, **kwargs)
        layer.params_dtype = params_dtype
        from vllm.model_executor.parameter import PerTensorScaleParameter
        # Some W4A4 checkpoints mistakenly have input_global_scale in the checkpoint 
        # despite us forcing it into a W4A16 fallback. We add a dummy parameter here so 
        # the checkpoint loader doesn't crash on an unknown key.
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "weight_packed"):
            return

        # Grab packed weights from CPU
        weight_packed = layer.weight_packed.data
        weight_scale = layer.weight_scale.data
        weight_global_scale = layer.weight_global_scale.data

        # Dequantize NVFP4 back to bf16
        # Note: dequantize_to_dtype returns float32
        weight_f32 = dequantize_to_dtype(
            tensor_fp4=weight_packed,
            tensor_sf=weight_scale,
            global_scale=weight_global_scale.max().to(torch.float32),
            dtype=torch.float32,
            block_size=self.group_size,
            swizzle=False, # We assume packed weights from compressed_tensors are not yet swizzled
        )
        
        dtype = layer.params_dtype if hasattr(layer, "params_dtype") else torch.bfloat16
        weight_bf16 = weight_f32.to(dtype)

        # Remove packed parameters to free CPU memory
        layer.weight_packed.untyped_storage().resize_(0)
        layer.weight_scale.untyped_storage().resize_(0)
        layer.weight_global_scale.untyped_storage().resize_(0)
        if hasattr(layer, "input_global_scale"):
            layer.input_global_scale.untyped_storage().resize_(0)
            del layer.input_global_scale
        del layer.weight_packed
        del layer.weight_scale
        del layer.weight_global_scale

        # Re-register as a standard weight param
        layer.weight = Parameter(weight_bf16, requires_grad=False)

        # Now let the unquantized method process it (which will load to TPU mesh)
        self.unquantized_fallback.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # standard linear method on TPU
        return self.unquantized_fallback.apply(layer, x, bias)

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from tpu_inference.layers.vllm.quantization.unquantized import VllmUnquantizedFusedMoEMethod

class VllmCompressedTensorsW4A16Fp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, weight_quant, input_quant, moe, mesh):
        super().__init__(moe)
        self.group_size = 16
        self.mesh = mesh
        self.unquantized_fallback = VllmUnquantizedFusedMoEMethod(moe, mesh)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.utils import set_weight_attrs
        from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
        
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards * intermediate_size_per_partition, hidden_size // 2, requires_grad=False, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards * intermediate_size_per_partition, hidden_size // self.group_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition // self.group_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_weight_scale_2 = torch.nn.Parameter(torch.empty(num_experts, w13_num_shards, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(torch.empty(num_experts, w13_num_shards, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self):
        return self.unquantized_fallback.apply_monolithic

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return self.unquantized_fallback.get_fused_moe_quant_config(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_weight_packed"):
            return

        dtype = layer.params_dtype if hasattr(layer, "params_dtype") else torch.bfloat16

        def unpack_and_set(layer_attr_prefix):
            packed = getattr(layer, f"{layer_attr_prefix}_packed").data
            scale = getattr(layer, f"{layer_attr_prefix}_scale").data
            g_scale = getattr(layer, f"{layer_attr_prefix}_global_scale").data

            # The underlying dequantize_to_dtype expects a 2D tensor (M, packed_K).
            # MoE packed weights are 3D (E, M, packed_K). We flatten E and M.
            num_experts = packed.shape[0]
            m_dim = packed.shape[1]
            packed_k_dim = packed.shape[2]

            packed_2d = packed.reshape(num_experts * m_dim, packed_k_dim)
            scale_2d = scale.reshape(num_experts * m_dim, scale.shape[-1])

            f32_weight_2d = dequantize_to_dtype(
                tensor_fp4=packed_2d,
                tensor_sf=scale_2d,
                global_scale=g_scale.max().to(torch.float32),
                dtype=torch.float32,
                block_size=self.group_size,
                swizzle=False
            )
            
            # Reshape back to 3D (E, M, K)
            f32_weight = f32_weight_2d.reshape(num_experts, m_dim, f32_weight_2d.shape[-1])
            
            getattr(layer, f"{layer_attr_prefix}_packed").untyped_storage().resize_(0)
            getattr(layer, f"{layer_attr_prefix}_scale").untyped_storage().resize_(0)
            getattr(layer, f"{layer_attr_prefix}_global_scale").untyped_storage().resize_(0)
            delattr(layer, f"{layer_attr_prefix}_packed")
            delattr(layer, f"{layer_attr_prefix}_scale")
            delattr(layer, f"{layer_attr_prefix}_global_scale")

            setattr(layer, layer_attr_prefix, Parameter(f32_weight.to(dtype), requires_grad=False))
            
        unpack_and_set("w13_weight")
        unpack_and_set("w2_weight")

        self.unquantized_fallback.process_weights_after_loading(layer)
        
    def apply(self, layer, x, router_logits):
        return self.unquantized_fallback.apply(layer, x, router_logits)

