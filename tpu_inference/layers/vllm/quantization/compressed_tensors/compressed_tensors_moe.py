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
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationArgs)
from jax.sharding import Mesh
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import \
    FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod, CompressedTensorsW8A8Fp8MoEMethod)

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quantization import ct_u32_unpack_u4
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

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
        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            return VllmCompressedTensorsWNA16MoEMethod(
                weight_quant=weight_quant,
                moe=layer.moe_config,
                mesh=quant_config.mesh,
            )
        raise RuntimeError(
            f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod,
                                            VllmQuantConfig):

    def __init__(self,
                 weight_quant: QuantizationArgs,
                 input_quant: QuantizationArgs,
                 moe: FusedMoEConfig,
                 mesh: Mesh,
                 ep_axis_name: str = "model"):
        super().__init__(weight_quant, input_quant, moe)

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    @property
    def is_monolithic(self) -> bool:
        return True

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

        # N.B
        # layer.w13_weight: [num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [num_experts, hidden_size, 1]
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
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                weights=FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
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
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)


class VllmCompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod,
                                          VllmQuantConfig):
    """Compressed-tensors WNA16 (weight-only int4/int8) MoE for TPU.

    Eagerly dequantizes packed int4 weights to bfloat16 during loading,
    handling g_idx activation reordering for grouped quantization.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.num_bits = weight_quant.num_bits
        self.pack_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size if weight_quant.group_size else -1
        self.symmetric = weight_quant.symmetric
        self.has_g_idx = (
            weight_quant.actorder == ActivationOrdering.GROUP)

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(moe)
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> None:
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.pop("intermediate_size_full", None)
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy,
        })
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight_packed = Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.pack_factor,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight_packed)
        set_weight_attrs(w13_weight_packed, extra_weight_attrs)

        w2_weight_packed = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight_packed)
        set_weight_attrs(w2_weight_packed, extra_weight_attrs)

        if self.group_size == -1:
            num_groups_w13 = 1
            num_groups_w2 = 1
        else:
            num_groups_w13 = hidden_size // self.group_size
            num_groups_w2 = intermediate_size_per_partition // self.group_size

        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, {"load_full_w2": False})

        w13_weight_shape = Parameter(
            torch.empty(num_experts, 2), requires_grad=False)
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w2_weight_shape = Parameter(
            torch.empty(num_experts, 2), requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

        w13_g_idx = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        if self.moe.has_bias:
            w13_bias = Parameter(
                torch.empty(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = Parameter(
                torch.empty(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_packed = t2j(layer.w13_weight_packed, use_dlpack=False)
        w2_packed = t2j(layer.w2_weight_packed, use_dlpack=False)
        w13_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.has_g_idx:
            w13_g_idx_raw = t2j(layer.w13_weight_g_idx, use_dlpack=False)
            w2_g_idx_raw = t2j(layer.w2_weight_g_idx, use_dlpack=False)
        else:
            w13_g_idx_raw = w2_g_idx_raw = None

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        for attr in [
            "w13_weight_packed", "w2_weight_packed",
            "w13_weight_scale", "w2_weight_scale",
            "w13_weight_g_idx", "w2_weight_g_idx",
            "w13_g_idx_sort_indices", "w2_g_idx_sort_indices",
            "w13_weight_shape", "w2_weight_shape",
        ]:
            if hasattr(layer, attr):
                delattr(layer, attr)

        if self.has_g_idx and w13_g_idx_raw is not None:
            w13_perm = jnp.argsort(w13_g_idx_raw, axis=-1)
            w13_inv_perm = jnp.argsort(w13_perm, axis=-1)
            w2_perm = jnp.argsort(w2_g_idx_raw, axis=-1)
            w2_inv_perm = jnp.argsort(w2_perm, axis=-1)
        else:
            num_w13_cols = w13_packed.shape[1]
            num_w2_cols = w2_packed.shape[1]
            w13_perm = jnp.broadcast_to(
                jnp.arange(num_w13_cols, dtype=jnp.int32)[jnp.newaxis, :],
                (w13_packed.shape[0], num_w13_cols))
            w13_inv_perm = w13_perm
            w2_perm = jnp.broadcast_to(
                jnp.arange(num_w2_cols, dtype=jnp.int32)[jnp.newaxis, :],
                (w2_packed.shape[0], num_w2_cols))
            w2_inv_perm = w2_perm

        group_size = self.group_size
        num_bits = self.num_bits
        symmetric = self.symmetric
        do_w13_perm = self.has_g_idx
        do_w2_perm = self.has_g_idx

        @jax.jit
        def process_wna16_moe_weights(
            w13_packed: jax.Array,
            w13_scale: jax.Array,
            w2_packed: jax.Array,
            w2_scale: jax.Array,
            w13_perm: jax.Array,
            w13_inv_perm: jax.Array,
            w2_perm: jax.Array,
            w2_inv_perm: jax.Array,
            w13_bias: jax.Array | None,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            # Swap axes so packed dim is last, then unpack
            # w13_packed: (E, H//pack, 2*I) -> (E, 2*I, H//pack)
            w13 = jnp.swapaxes(w13_packed, 1, 2)
            w13 = ct_u32_unpack_u4(w13)  # (E, 2*I, H)

            w2 = jnp.swapaxes(w2_packed, 1, 2)
            w2 = ct_u32_unpack_u4(w2)  # (E, H, I)

            # Scales: (E, num_groups, out_size) -> (E, out_size, num_groups)
            w13_s = jnp.swapaxes(w13_scale, 1, 2)
            w2_s = jnp.swapaxes(w2_scale, 1, 2)

            # Apply g_idx column permutation on input dim (last dim)
            if do_w13_perm:
                w13 = jax.vmap(lambda w, p: w[:, p])(w13, w13_perm)
            if do_w2_perm:
                w2 = jax.vmap(lambda w, p: w[:, p])(w2, w2_perm)

            # Dequantize: unpack groups, apply offset+scale
            def _dequant(weight, scales, input_size):
                E, O, I = weight.shape
                eff_gs = I if group_size == -1 else group_size
                n_groups = I // eff_gs
                w = weight.reshape(E, O, n_groups, eff_gs)
                w_f = w.astype(jnp.bfloat16)
                if symmetric:
                    off = jnp.array(
                        1 << (num_bits - 1), dtype=jnp.bfloat16)
                    w_deq = (w_f - off) * scales[:, :, :, jnp.newaxis]
                else:
                    w_deq = w_f * scales[:, :, :, jnp.newaxis]
                return w_deq.reshape(E, O, I)

            w13_deq = _dequant(w13, w13_s, w13.shape[-1])
            w2_deq = _dequant(w2, w2_s, w2.shape[-1])

            # Apply inverse permutation to restore original column order
            if do_w13_perm:
                w13_deq = jax.vmap(
                    lambda w, p: w[:, p])(w13_deq, w13_inv_perm)
            if do_w2_perm:
                w2_deq = jax.vmap(
                    lambda w, p: w[:, p])(w2_deq, w2_inv_perm)

            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                weights=FusedMoEWeights(
                    w13_weight=w13_deq,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_deq,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_wna16_moe_weights(
            w13_packed, w13_scale, w2_packed, w2_scale,
            w13_perm, w13_inv_perm, w2_perm, w2_inv_perm,
            w13_bias, w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(
            weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(
            weights.w2_weight, requires_grad=False)

        if weights.w13_weight_scale is not None:
            layer.w13_weight_scale = Parameter(
                weights.w13_weight_scale, requires_grad=False)
        else:
            layer.w13_weight_scale = None
        if weights.w2_weight_scale is not None:
            layer.w2_weight_scale = Parameter(
                weights.w2_weight_scale, requires_grad=False)
        else:
            layer.w2_weight_scale = None

        if self.moe.has_bias:
            layer.w13_bias = Parameter(
                weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(
                weights.w2_bias, requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=(
                jax_view(layer.w13_weight_scale)
                if layer.w13_weight_scale is not None else None),
            w13_bias=(
                jax_view(layer.w13_bias)
                if self.moe.has_bias else None),
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=(
                jax_view(layer.w2_weight_scale)
                if layer.w2_weight_scale is not None else None),
            w2_bias=(
                jax_view(layer.w2_bias)
                if self.moe.has_bias else None),
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
