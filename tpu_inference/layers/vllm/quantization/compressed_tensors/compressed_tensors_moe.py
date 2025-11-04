from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationStrategy
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import call_jax, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, fp8_w8a8_moe_quant_config)
from vllm.model_executor.layers.quantization.compressed_tensors import \
    compressed_tensors_moe as vllm_ct_moe
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS, WNA16_SUPPORTED_TYPES_MAP)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW8A8Fp8MoEMethod(vllm_ct_moe.CompressedTensorsMoEMethod
                                        ):

    def __init__(self, quant_config: "CompressedTensorsConfig",
                 moe: FusedMoEConfig, mesh: Mesh):
        super().__init__(moe)
        self.mesh = mesh
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        per_tensor = (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                      and self.input_quant.strategy
                      == QuantizationStrategy.TENSOR)
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not (per_tensor or per_channel):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization.")

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = False
        self.rocm_aiter_moe_enabled = False  # is_rocm_aiter_moe_enabled()

        # cutlass path
        self.is_fp8_w8a8_sm100 = False
        self.use_cutlass = False
        self.disable_expert_map = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        intermediate_size = layer.w13_weight.shape[1] // 2
        w1_weight = layer.w13_weight[:, :intermediate_size]
        w3_weight = layer.w13_weight[:, intermediate_size:]
        w1_weight_scale = layer.w13_weight_scale[:, :intermediate_size]
        w3_weight_scale = layer.w13_weight_scale[:, intermediate_size:]

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        w1_weight = t2j(w1_weight, use_dlpack=False)
        w1_weight_scale = t2j(w1_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        w3_weight = t2j(w3_weight, use_dlpack=False)
        w3_weight_scale = t2j(w3_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)

        if layer.use_ep:
            format = Format(Layout((0, 1, 2)),
                            NamedSharding(self.mesh, P("model", None, None)))
            w1_weight = jax.device_put(w1_weight, format)
            w1_weight_scale = jax.device_put(w1_weight_scale, format)
            w3_weight = jax.device_put(w3_weight, format)
            w3_weight_scale = jax.device_put(w3_weight_scale, format)
            w2_weight = jax.device_put(w2_weight, format)
            w2_weight_scale = jax.device_put(w2_weight_scale, format)
        else:
            assert intermediate_size == w2_weight.shape[-1]
            n_shards = self.mesh.shape["model"]
            assert intermediate_size % n_shards == 0

            # TODO: enable this if using fused weights
            # output_sizes = [intermediate_size, intermediate_size]
            # w13_weight = reorder_concatenated_tensor_for_sharding(
            #    w13_weight, output_sizes, n_shards, dim=1
            # )

            w13_format = Format(
                Layout((0, 1, 2)),
                NamedSharding(self.mesh, P(None, "model", None)))
            w1_weight = jax.device_put(w1_weight, w13_format)
            w1_weight_scale = jax.device_put(w1_weight_scale, w13_format)
            w3_weight = jax.device_put(w3_weight, w13_format)
            w3_weight_scale = jax.device_put(w3_weight_scale, w13_format)
            w2_weight = jax.device_put(
                w2_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P(None, None, "model"))),
            )
            w2_weight_scale = jax.device_put(
                w2_weight_scale,
                Format(Layout((0, 1, 2)), NamedSharding(self.mesh, P())),
            )  # replicate

        w1_weight = Parameter(torch_view(w1_weight), requires_grad=False)
        w1_weight_scale = Parameter(torch_view(w1_weight_scale),
                                    requires_grad=False)
        w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)
        w2_weight_scale = Parameter(torch_view(w2_weight_scale),
                                    requires_grad=False)
        w3_weight = Parameter(torch_view(w3_weight), requires_grad=False)
        w3_weight_scale = Parameter(torch_view(w3_weight_scale),
                                    requires_grad=False)

        # TODO dont reuse variable
        layer.w13_weight = w1_weight
        layer.w13_weight_scale = w1_weight_scale
        layer.w2_weight = w2_weight
        layer.w2_weight_scale = w2_weight_scale
        layer.w3_weight = w3_weight
        layer.w3_weight_scale = w3_weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if activation != "silu":
            raise NotImplementedError(
                "Only silu is supported for activation function.")
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        # TODO: Use MoE kernel when it supports fp8

        seqlen = x.shape[0]

        expert_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights,
                                                    top_k,
                                                    dim=-1)
        if renormalize:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        # cond ffn
        # e = total num of exp = 160
        # t = seqlen
        # o = config.imtermediate size
        # i = config.dim
        #torch.einsum("ti, eoi -> teo", x, layer.w13_weight) * self.w13_weight_scale)
        ux1 = call_jax(jax.lax.dot,
                       x,
                       layer.w13_weight,
                       dimension_numbers=(((1, ), (2, )), ((), ())),
                       preferred_element_type=jnp.bfloat16.dtype)
        x1 = F.silu(ux1 * layer.w13_weight_scale.squeeze(2))

        #x3 = torch.einsum("ti, eoi -> teo", x, layer.w3_weight) * self.w3_weight_scale
        x3 = call_jax(jax.lax.dot,
                      x,
                      layer.w3_weight,
                      dimension_numbers=(((1, ), (2, )), ((), ())),
                      preferred_element_type=jnp.bfloat16.dtype
                      ) * layer.w3_weight_scale.squeeze(2)

        #expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), self.w2_weight) * self.w2_weight_scale
        expert_outs = call_jax(
            jax.lax.dot,
            x1 * x3,
            layer.w2_weight,
            dimension_numbers=(((2, ), (2, )), ((1, ), (0, ))),
            preferred_element_type=jnp.bfloat16.dtype).transpose(
                0, 1) * layer.w2_weight_scale.squeeze(2)

        seq_indexes = torch.arange(seqlen, device='jax').unsqueeze(1)
        expert_outs = expert_outs[seq_indexes, expert_indices]

        # out = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        out = call_jax(jax.lax.dot,
                       expert_outs,
                       expert_weights,
                       dimension_numbers=(((1, ), (1, )), ((0, ), (0, ))),
                       preferred_element_type=jnp.bfloat16.dtype)

        return out

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            raise AssertionError('Blockwise quant for MoE not supported yet')

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        if self.use_marlin:
            return None

        per_act_token = self.input_quant.strategy == QuantizationStrategy.TOKEN
        per_channel_quant = self.weight_quant.strategy == QuantizationStrategy.CHANNEL

        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=per_act_token,
            per_out_ch_quant=per_channel_quant,
            block_shape=layer.weight_block_size,
        )
