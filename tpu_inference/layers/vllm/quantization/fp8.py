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
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.quantization import fp8 as vllm_fp8
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    shard_linear_weights, to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights, process_fp8_moe_weights_direct,
    shard_moe_weights)
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.common.quant_methods import FP8
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


def _get_local_expert_ids(layer: FusedMoE) -> tuple[int, ...] | None:
    local_expert_ids = getattr(layer, '_tpu_ep_local_expert_ids', None)
    if not local_expert_ids:
        return None
    return tuple(int(expert_id) for expert_id in local_expert_ids)


def _take_local_experts(array, local_expert_ids: tuple[int, ...], axis: int):
    if isinstance(array, torch.Tensor):
        expert_indices = torch.as_tensor(local_expert_ids,
                                         device=array.device,
                                         dtype=torch.long)
        return torch.index_select(array, dim=axis, index=expert_indices)

    expert_indices = jnp.asarray(local_expert_ids, dtype=jnp.int32)
    return jnp.take(array, expert_indices, axis=axis)


def _free_cpu_parameter_storage(layer: torch.nn.Module, param_name: str) -> None:
    tensor = getattr(layer, param_name, None)
    if tensor is None:
        return

    try:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
            tensor.untyped_storage().resize_(0)
    except Exception:
        logger.debug("Could not shrink CPU storage for %s.%s",
                     type(layer).__name__, param_name)

    delattr(layer, param_name)


@register_quantization_config(FP8)
class VllmFp8Config(vllm_fp8.Fp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return FP8

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
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
                if self.is_checkpoint_fp8_serialized:
                    layer.moe_config = self.get_moe_config(layer)
                    return VllmFp8MoEMethod(self, layer, self.mesh)
                else:
                    raise NotImplementedError(
                        "FP8OnelineMoEMethod is not supported.")
            case Attention():
                logger.warning_once("FP8KVCacheMethod is not implemented. "
                                    "Skipping quantization for this layer.")
                return None
            case _:
                return None


class VllmFp8LinearMethod(vllm_fp8.Fp8LinearMethod,
                          common_fp8.Fp8LinearMethod):

    def __init__(
        self,
        quant_config: VllmFp8Config,
        linear_config: VllmQuantLinearConfig,
    ):

        # Per https://github.com/vllm-project/vllm/pull/32929,
        # init_fp8_linear_kernel is now called by super().__init__
        # but does not support TPU backends as expected.
        # use_marlin was also changed to be determined via isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel).
        # We need to monkeypatch init_fp8_linear_kernel and explicitly set use_marlin = True
        # in order to bypass using native vLLM's vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py:scaled_quantize.
        vllm_fp8.init_fp8_linear_kernel = lambda *args, **kwargs: None
        super().__init__(quant_config)
        self.use_marlin = True

        self.linear_config = linear_config
        if self.linear_config.enable_quantized_matmul_kernel and not self.linear_config.requant_block_size:
            raise ValueError(
                "You should set REQUANTIZE_BLOCK_SIZE to enable quantized matmul kernel. Please set the value or disable the quantized matmul kernel."
            )
        if not self.linear_config.enable_quantized_matmul_kernel and self.linear_config.requant_block_size:
            raise ValueError(
                "Blockwise quantization is supported by quantized matmul kernel. Please enable quantized_matmul_kernel or unset the quantize block size to trigger XLA per-channel quantization."
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, vllm_linear.LinearBase)

        assert self.block_quant
        weight = t2j(layer.weight, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "weight")

        weight_scale = t2j(layer.weight_scale_inv, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "weight_scale_inv")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            _free_cpu_parameter_storage(layer, "bias")
        else:
            bias = None

        # TP-selective loader pre-slices params by coarse_tp (8) in build_tp_plan,
        # but linear_config.output_sizes was captured at model-init time with
        # TP=1 (full output size). For single-output (non-fused) params the
        # actual weight dim after slicing won't match; use the real dim instead.
        # Fused params (gate+up, QKV) are in _skip_names so they are never
        # pre-sliced — their config sizes remain correct.
        _cfg_sizes = tuple(self.linear_config.output_sizes)
        _actual_dim0 = weight.shape[0]
        if len(_cfg_sizes) == 1 and _cfg_sizes[0] != _actual_dim0:
            _output_sizes = (_actual_dim0,)
        else:
            _output_sizes = _cfg_sizes

        weights = common_fp8.process_blockwise_fp8_linear_weights(
            weight,
            weight_scale,
            bias=bias,
            weight_block_size=tuple(self.weight_block_size),
            requant_block_size=self.linear_config.requant_block_size,
            output_sizes=_output_sizes,
            requant_weight_dtype=self.linear_config.requant_weight_dtype,
            fuse_matmuls=self.linear_config.fuse_matmuls,
            n_shards=self.linear_config.n_shards)
        if self.linear_config.enable_quantized_matmul_kernel:
            # The quantized_matmul_kernel expects weight scales shaped (n_out_features, 1, n_blocks) for blockwisze quantization.
            weights.weight_scale = jnp.expand_dims(
                jnp.transpose(weights.weight_scale),
                axis=1,
            )
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
                # jax_view cannot handle ParameterList directly, so we explicitly
                # convert them to list of jax.Array.
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


class VllmFp8MoEMethod(vllm_fp8.Fp8MoEMethod):

    def __init__(self,
                 quant_config: vllm_fp8.Fp8Config,
                 layer: torch.nn.Module,
                 mesh: Mesh,
                 ep_axis_name: str = "model"):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")
        self.fp8_backend = None

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        assert self.block_quant
        assert not self.moe.has_bias

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "w13_weight")
        w13_weight_scale = t2j(layer.w13_weight_scale_inv, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "w13_weight_scale_inv")

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "w2_weight")
        w2_weight_scale = t2j(layer.w2_weight_scale_inv, use_dlpack=False)
        _free_cpu_parameter_storage(layer, "w2_weight_scale_inv")

        # In multi-host EP, the fused tensor may still have the global expert
        # axis even though only the local expert shard was loaded on this host.
        # Slice by the actual global expert ids owned by this worker so the
        # local weight rows stay aligned with the router logits.
        # Slice weights to local experts. The pre-allocated tensor has shape
        # (num_experts_global, ...) with only local experts filled; extract them.
        local_expert_ids = _get_local_expert_ids(layer)
        if local_expert_ids is not None and len(local_expert_ids) < w13_weight.shape[0]:
            w13_weight = _take_local_experts(w13_weight, local_expert_ids, axis=0)
            w13_weight_scale = _take_local_experts(w13_weight_scale,
                                                   local_expert_ids,
                                                   axis=0)
            w2_weight = _take_local_experts(w2_weight, local_expert_ids, axis=0)
            w2_weight_scale = _take_local_experts(w2_weight_scale,
                                                  local_expert_ids,
                                                  axis=0)
        else:
            _local_count = getattr(layer, '_tpu_ep_local_count', None)
            if _local_count is not None and _local_count < w13_weight.shape[0]:
                w13_weight = w13_weight[:_local_count]
                w13_weight_scale = w13_weight_scale[:_local_count]
                w2_weight = w2_weight[:_local_count]
                w2_weight_scale = w2_weight_scale[:_local_count]

        # TODO: do we need to support bias?
        input_weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        )

        weight_block_size = None
        if self.weight_block_size is not None:
            weight_block_size = tuple(self.weight_block_size)

        if (weight_block_size is not None
                and envs.MOE_SKIP_REQUANTIZATION):
            # Fast path: skip dequant/requant, direct FP8 shape transform.
            # Same `cpu_mesh_context()` as the legacy branch so the
            # expand_dims / reshape ops stay on the CPU mesh and don't
            # allocate temporaries on the TPU HBM during load.
            logger.info_once(
                "[MoE] Skipping requantization — direct FP8 path")
            with cpu_mesh_context():
                weights = process_fp8_moe_weights_direct(
                    input_weights,
                    moe_backend=self.moe_backend,
                    mesh=self.mesh,
                    activation=layer.activation.value,
                    weight_block_size=weight_block_size,
                )
        else:
            with cpu_mesh_context():
                weights = process_fp8_moe_weights(
                    input_weights,
                    moe_backend=self.moe_backend,
                    mesh=self.mesh,
                    activation=layer.activation.value,
                    # Convert to tuple so jax jit can hash it
                    weight_block_size=weight_block_size,
                )
        # layer.num_experts in vLLM's FusedMoE resolves to local_num_experts
        # (e.g., 32 for EP=8 with 256 experts). shard_moe_weights needs the
        # GLOBAL count so make_array_from_process_local_data reconstructs
        # the correct distributed array in multi-host mode.
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh,
                              num_experts_global=layer.global_num_experts))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale_inv = Parameter(weights.w13_weight_scale,
                                               requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(weights.w2_weight_scale,
                                              requires_grad=False)

        # shard_moe_weights calls jax.block_until_ready on its output before
        # returning, so host→device DMAs are complete by this point and
        # Python's normal refcount drop (end of function scope) reliably
        # frees the t2j intermediates. No manual gc needed.

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        # Keep router logits aligned with the locally-loaded expert rows.
        # After EP rank propagation each worker owns a distinct global expert
        # subset, so slicing the first N experts is incorrect.
        if not envs.NEW_MODEL_DESIGN:
            local_expert_ids = _get_local_expert_ids(layer)
            if local_expert_ids is not None and router_logits.shape[-1] > len(local_expert_ids):
                router_logits = _take_local_experts(router_logits,
                                                    local_expert_ids,
                                                    axis=1)
            else:
                _local_count = getattr(layer, '_tpu_ep_local_count', None)
                if _local_count is not None and router_logits.shape[-1] > _local_count:
                    router_logits = router_logits[:, :_local_count]

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale_inv),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale_inv),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        result = vllm_moe_apply(layer=layer,
                               weights=weights,
                               quant_method_instance=self,
                               x=x,
                               router_logits=router_logits)

        return result
