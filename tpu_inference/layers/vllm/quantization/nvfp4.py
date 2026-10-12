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

This module provides the TPU-specific implementation for loading and executing
NVIDIA's ModelOpt FP4 (NVFP4) quantized checkpoints in vLLM.

Important notes:
 - Weight Processing: During `process_weights_after_loading`, we:
   - Unpack the 8-bit packed into unpacked FP4 weights
   - Pre-fuse the FP8 block scales and the FP32 global scale into a single,
     unified FP32 1D-Blockwise scale.

 - Execution:
   - Linear (via sharded_quantized_matmul) and MoE (via GMM_v2) will dequantize weights
     on-the-fly.
"""

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ACT, WEIGHT, CkptCtx, KNvfp4Dynamic, KNvfp4Static, ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE, Shapes)
from vllm.model_executor.utils import set_weight_attrs

import tpu_inference.envs as envs
from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _get_expert_shard_axis, process_quantized_moe_weights,
    shard_moe_weights_to_tpu)
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.common.quantization import u8_unpack_e2m1
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _tensor_is_in_cpu
from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.fp8 import (_free_torch_storage,
                                                        _release_host_memory)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod,
    _load_weight_for_layer)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j, to_jax_dtype

logger = init_logger(__name__)


@register_quantization_config(NVFP4)
class VllmNvfp4Config(ModelOptNvFp4Config, VllmQuantConfig):
    """NVFP4 config for TPU. Inherits config parsing from upstream."""

    @classmethod
    def get_name(cls):
        return NVFP4

    @classmethod
    def override_quantization_method(
            cls,
            hf_quant_cfg,
            user_quant,
            hf_config=None) -> Optional[QuantizationMethods]:
        if hf_quant_cfg is not None:
            quant_library = str(hf_quant_cfg.get("quant_library", "")).upper()
            quant_algo = str(hf_quant_cfg.get("quant_algo", "")).upper()
            if quant_library == "MODELOPT" and quant_algo == "NVFP4":
                return NVFP4
        return super().override_quantization_method(hf_quant_cfg, user_quant,
                                                    hf_config)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[Union[QuantizeMethodBase]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.is_layer_excluded(prefix):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmNvfp4LinearMethod(self, linear_config)
        elif isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)
            layer.moe_config = self.get_moe_config(layer)
            return VllmNvfp4MoEMethod(self, layer, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once(
                "NVFP4 attention quantization is not implemented. "
                "Skipping quantization for this layer.")
        return None


class VllmNvfp4LinearMethod(VllmUnquantizedLinearMethod):
    """NVFP4 linear for TPU.

    Reuses upstream KNvfp4Static and KNvfp4Dynamic for parameter registration.
    process_weights_after_loading unpacks FP4 and computes block scales.
    apply routes to vllm_linear_apply for OTF dequantization.
    """

    def __init__(self, quant_config: 'VllmNvfp4Config',
                 linear_config: VllmQuantLinearConfig):
        VllmUnquantizedLinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        del input_size, output_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.output_partition_sizes = output_partition_sizes
        shapes = Shapes(output_partition_sizes, input_size_per_partition,
                        params_dtype)
        group_size = getattr(self.quant_config, "group_size", 16)
        ctx = CkptCtx(group_size=group_size)

        KNvfp4Static().create_weights(layer, WEIGHT, ctx, shapes,
                                      weight_loader)
        KNvfp4Dynamic().create_weights(layer, ACT, ctx, shapes, weight_loader)

        def scalar_weight_loader(param, loaded_weight, *args, **kwargs):
            assert loaded_weight.numel() == 1
            value = loaded_weight.item()
            param.data.fill_(value)

        for name in ("input_scale", "weight_scale_2"):
            if hasattr(layer, name):
                getattr(layer, name).weight_loader = scalar_weight_loader

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, LinearBase)
        weight_packed = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        # vLLM allocates this based on the output sizes, so it's the same scalar broadcasted (potentially) multiple times,
        # so we can just grab one, see here:
        # https://github.com/vllm-project/vllm/blob/10ebb40d62e024116c1a4473f8e357a3e72761ed/vllm/model_executor/layers/quantization/modelopt.py#L1149
        weight_global_scale = t2j(layer.weight_scale_2, use_dlpack=False)
        # assert that all elements are the same
        assert jnp.all(weight_global_scale == weight_global_scale[0])
        weight_global_scale = weight_global_scale[0]

        for attr in ('weight', 'weight_scale', 'weight_scale_2',
                     'input_scale'):
            if hasattr(layer, attr):
                delattr(layer, attr)

        if hasattr(
                layer,
                'bias') and layer.bias is not None and not layer.skip_bias_add:
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, 'bias')
        else:
            bias = None

        @jax.jit
        def _unpack_and_scale(weight_packed, weight_scale,
                              weight_global_scale):
            fp4 = u8_unpack_e2m1(weight_packed)  # (O, I) float4_e2m1fn
            fp4 = jnp.transpose(fp4)

            # Combine FP8 block scale & FP32 global scale
            block_scale = weight_scale.astype(
                jnp.float32) * weight_global_scale  # (O, I/group)
            block_scale = jnp.transpose(block_scale)

            return fp4, block_scale

        weight_fp4, block_scale = _unpack_and_scale(weight_packed,
                                                    weight_scale,
                                                    weight_global_scale)

        weights = process_linear_weights(
            LinearWeights(weight=weight_fp4,
                          weight_scale=block_scale,
                          zero_point=None,
                          bias=bias),
            fused=self.linear_config.fuse_matmuls,
            output_sizes=self.linear_config.output_sizes,
            reorder_size=self.linear_config.n_shards,
            enable_kernel=self.linear_config.enable_quantized_matmul_kernel,
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
                out = self._apply_split(x_jax, weight_and_scale, bias_jax)
            return torch_view(out)

    def _apply_fused(self,
                     x: jax.Array,
                     weight_jax: jax.Array,
                     weight_scale_jax: jax.Array,
                     bias: Optional[jax.Array] = None) -> jax.Array:

        outs = sharded_quantized_matmul(
            x,
            weight_jax,
            weight_scale_jax,
            self.linear_config.weight_sharding,
            mesh=self.linear_config.mesh,
            defer_all_reduce=self.linear_config.defer_all_reduce)

        if bias is not None:
            outs += bias
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return jnp.concatenate(outs, axis=-1)

    def _apply_split(self,
                     x: jax.Array,
                     weight_and_scale: Sequence[tuple[jax.Array, jax.Array]],
                     bias: Optional[Sequence[jax.Array]] = None,
                     mesh: Optional[Mesh] = None) -> jax.Array:

        outs = []
        for i, (weight, weight_scale) in enumerate(weight_and_scale):

            out = sharded_quantized_matmul(
                x,
                weight,
                weight_scale,
                self.linear_config.weight_sharding,
                mesh=mesh,
                defer_all_reduce=self.linear_config.defer_all_reduce)

            if bias is not None:
                out += bias[i]
            outs.append(out)
        return jnp.concatenate(outs, axis=-1)


def _local_experts_loader(loader, local_mask: list[bool]):
    """Wrap a MoE weight loader to skip experts outside `local_mask`."""

    def weight_loader(param, loaded_weight, *args, **kwargs):
        # The MoE weight loader's signature is
        # (param, loaded_weight, weight_name, shard_id, expert_id,
        #  return_success).
        expert_id = kwargs.get("expert_id", args[2] if len(args) > 2 else None)
        if expert_id is not None and not local_mask[expert_id]:
            return_success = kwargs.get("return_success",
                                        args[3] if len(args) > 3 else False)
            # Handled: another host holds this expert.
            return True if return_success else None
        return loader(param, loaded_weight, *args, **kwargs)

    return weight_loader


# Module level, so it is traced and compiled once rather than once per MoE
# layer (a jit defined inside the processing method is a new function on
# every call).
@jax.jit
def _unpack_and_fold_scales(
    weights: FusedMoEWeights,
    w13_global_scale: jax.Array,
    w2_global_scale: jax.Array,
) -> FusedMoEWeights:
    # Unpack packed uint8 -> native float4_e2m1fn (no value change).
    w13_fp4 = u8_unpack_e2m1(weights.w13_weight)
    w2_fp4 = u8_unpack_e2m1(weights.w2_weight)

    # Combine FP32 global × FP8 block -> single FP32 block scale.
    if w13_global_scale.ndim == 2:  # (E, 2) — separate w1/w3 globals.
        half = weights.w13_weight_scale.shape[1] // 2
        w1_eff = weights.w13_weight_scale[:, :half].astype(
            jnp.float32) * w13_global_scale[:, 0:1].reshape(-1, 1, 1)
        w3_eff = weights.w13_weight_scale[:, half:].astype(
            jnp.float32) * w13_global_scale[:, 1:2].reshape(-1, 1, 1)
        w13_scale_eff = jnp.concatenate([w1_eff, w3_eff], axis=1)
    else:
        w13_scale_eff = weights.w13_weight_scale.astype(
            jnp.float32) * w13_global_scale.reshape(-1, 1, 1)
    w2_scale_eff = weights.w2_weight_scale.astype(
        jnp.float32) * w2_global_scale.reshape(-1, 1, 1)

    return FusedMoEWeights(
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale_eff,
        w13_bias=weights.w13_bias,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale_eff,
        w2_bias=weights.w2_bias,
    )


def _ignore_late_weight(param, loaded_weight, *args, **kwargs):
    """Weight loader of a parameter that is no longer used: accept and drop."""
    return_success = kwargs.get("return_success",
                                args[3] if len(args) > 3 else False)
    return True if return_success else None


class _InOrderProcessing:
    """Process MoE layers as they finish loading, in the order they were made.

    On a multi-host slice every host must issue the layers' processing
    (sharded JAX computations) in the same order, but each host finishes
    loading layers in the order its files arrive. A loaded layer waits until
    every layer created before it has been processed.
    """

    def __init__(self):
        self._entries = []  # (layer, quant_method), in creation order
        self._loaded = set()
        self._next = 0

    def register(self, layer, quant_method) -> None:
        self._entries.append((layer, quant_method))

    def mark_loaded(self, layer) -> None:
        self._loaded.add(id(layer))
        logger.info("[nvfp4-inorder] loaded %s; %d/%d loaded, %d processed",
                    getattr(layer, "layer_name", "?"), len(self._loaded),
                    len(self._entries), self._next)
        while (self._next < len(self._entries)
               and id(self._entries[self._next][0]) in self._loaded):
            ready, quant_method = self._entries[self._next]
            logger.info("[nvfp4-inorder] processing %s", ready.layer_name)
            quant_method._process_loaded_weights(ready)
            self._next += 1


class VllmNvfp4MoEMethod(FusedMoEMethodBase, VllmQuantizationMethod):
    """NVFP4 MoE for TPU.

    Reuses upstream ModelOptNvFp4FusedMoE.create_weights for parameter
    registration; adds bias parameters upstream does not register.
    Weights stay packed E2M1 in HBM; the FP32 global × FP8 block scales are
    combined into a single FP32 blockwise scale of size 16 and the matmul
    runs through gmm_v2's dequant-in-VMEM path.

    Large checkpoints: with VLLM_INCREMENTAL_FP8_LOADING the layer is
    processed as soon as all of its experts are loaded (freeing its host
    copy), and with MOE_STAGE_WEIGHTS_ON_HOST each weight is placed straight
    at its expert sharding instead of passing through one device. Both match
    the FP8 MoE method.
    """

    # Per-expert parameters the layer waits for before incremental
    # processing, with the shard ids the MoE weight loader passes for them.
    # Activation scales are not used on TPU, so they are not waited for.
    _W13_PARAMS = ("w13_weight", "w13_weight_scale", "w13_weight_scale_2")
    _W2_PARAMS = ("w2_weight", "w2_weight_scale", "w2_weight_scale_2")

    def __init__(self, quant_config, layer, mesh, ep_axis_name="model"):
        # Skip ModelOptNvFp4FusedMoE.__init__ (selects GPU experts backend).
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        # Upstream's create_weights references self.use_global_sf to size the
        # input_scale params; we don't use them so any value works — keep
        # False to match a CPU/TPU "no global SF" path.
        self.use_global_sf = False
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

    @property
    def is_monolithic(self):
        return True

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.params_dtype = params_dtype
        ModelOptNvFp4FusedMoE.create_weights(self, layer, num_experts,
                                             hidden_size,
                                             intermediate_size_per_partition,
                                             params_dtype,
                                             **extra_weight_attrs)
        # Upstream's create_weights only updates `extra_weight_attrs` after
        # the scale params are registered, so vLLM's MoE weight loader does
        # not see `quant_method` on them and rejects the load. Set them
        # explicitly here.
        from vllm.model_executor.layers.fused_moe import \
            FusedMoeWeightScaleSupported
        for name in ("w13_weight_scale", "w2_weight_scale"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # Upstream NVFP4 MoE does not register bias; some models (e.g.
        # gpt-oss) need it. Keep our own bias registration.
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        if jax.process_count() > 1:
            self._keep_local_experts_only(layer,
                                          self._local_expert_mask(num_experts))

    def _expert_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P(_get_expert_shard_axis(self.mesh)))

    def _local_expert_mask(self, num_experts: int) -> list[bool]:
        """Which experts this host's devices hold under the expert sharding."""
        mask = [False] * num_experts
        index_map = self._expert_sharding().addressable_devices_indices_map(
            (num_experts, ))
        for index in index_map.values():
            for e in range(num_experts)[index[0]]:
                mask[e] = True
        return mask

    def _keep_local_experts_only(self, layer: torch.nn.Module,
                                 local_mask: list[bool]) -> None:
        """Drop the checkpoint tensors of experts this host does not hold.

        Every host loads the whole checkpoint, but the expert-sharded weights
        are built with make_array_from_callback, which reads only this host's
        shards. Not copying the other hosts' experts leaves their pages
        untouched, so host memory holds 1/num_hosts of the experts. This
        matters when a checkpoint scatters each layer's experts across many
        files (as ModelOpt exports do): no layer is complete, and so none can
        be processed and freed, until nearly the whole checkpoint is loaded.
        """
        if all(local_mask):
            return
        # Incremental processing still frees each layer once it is loaded,
        # but in model order (see _InOrderProcessing): hosts finish layers in
        # whatever order their files arrive. With a checkpoint that stores
        # each layer's experts together, host memory stays at about a layer.
        order = getattr(self.quant_config, "_moe_process_order", None)
        if order is None:
            order = _InOrderProcessing()
            self.quant_config._moe_process_order = order
        order.register(layer, self)
        self._process_order = order
        for name in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
                     "w13_input_scale", "w2_weight", "w2_weight_scale",
                     "w2_weight_scale_2", "w2_input_scale", "w13_bias",
                     "w2_bias"):
            param = getattr(layer, name, None)
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                param.weight_loader = _local_experts_loader(loader, local_mask)

    def get_fused_moe_quant_config(self, layer):
        return None

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Process the layer once every expert's weights are loaded.

        Called by the incremental loader after each weight load. Reuses the
        VLLM_INCREMENTAL_FP8_LOADING switch, which turns on incremental
        processing for the quantized MoE layers.
        """
        if not envs.VLLM_INCREMENTAL_FP8_LOADING or self.moe.has_bias:
            # Bias layers process after the full load.
            return

        w13_shards = ("w1", "w3") if self.moe.is_act_and_mul else ("w1", )
        if param_name in self._W13_PARAMS:
            shard_ids = w13_shards
        elif param_name in self._W2_PARAMS:
            shard_ids = ("w2", )
        else:
            return
        # The MoE weight loader's signature is
        # (param, loaded_weight, weight_name, shard_id, expert_id, ...).
        shard_id = kwargs.get("shard_id", args[1] if len(args) > 1 else None)
        expert_id = kwargs.get("expert_id", args[2] if len(args) > 2 else None)
        if expert_id is None or shard_id not in shard_ids:
            return

        if not hasattr(layer, "_loaded_weights"):
            layer._loaded_weights = set()
        layer._loaded_weights.add((param_name, expert_id, shard_id))

        per_expert = (len(self._W13_PARAMS) * len(w13_shards) +
                      len(self._W2_PARAMS))
        if len(layer._loaded_weights) >= per_expert * layer.global_num_experts:
            order = getattr(self, "_process_order", None)
            if order is not None:
                # Hosts keeping only their own experts process in model order.
                order.mark_loaded(layer)
                return
            logger.debug("[nvfp4-incremental] Processing MoE layer %s",
                         layer.layer_name)
            self._process_loaded_weights(layer)

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, RoutedExperts)
        # Unless already processed during incremental loading.
        if hasattr(layer, "w13_weight") and _tensor_is_in_cpu(
                layer.w13_weight):
            self._process_loaded_weights(layer)
        # The load is over, so no activation scale can still arrive.
        for attr in ('w13_input_scale', 'w2_input_scale'):
            if hasattr(layer, attr):
                delattr(layer, attr)

    def _process_loaded_weights(self, layer):
        # Activation-quant scales registered by upstream are unused on TPU.
        # Free them, but keep the parameters until the load is over: with
        # incremental processing a layer can be processed before all of its
        # input_scale tensors have streamed in, and RoutedExperts.load_weights
        # fails on a missing parameter.
        for attr in ('w13_input_scale', 'w2_input_scale'):
            param = getattr(layer, attr, None)
            if param is not None:
                _free_torch_storage(param)
                param.weight_loader = _ignore_late_weight

        # With MOE_STAGE_WEIGHTS_ON_HOST each weight goes straight to the
        # expert sharding that shard_moe_weights_to_tpu below would give it,
        # without a full-size stop on one device.
        expert_sharding = NamedSharding(self.mesh,
                                        P(_get_expert_shard_axis(self.mesh)))
        on_host = envs.MOE_STAGE_WEIGHTS_ON_HOST
        param_names = [
            "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale",
            "w13_weight_scale_2", "w2_weight_scale_2"
        ]
        if self.moe.has_bias:
            param_names += ["w13_bias", "w2_bias"]
        loaded = {
            name:
            _load_weight_for_layer(layer,
                                   name,
                                   expert_sharding,
                                   stage_on_host=on_host)
            for name in param_names
        }
        # The host-to-device copies are asynchronous and read the torch
        # storage until they finish; freeing it before then leaves its pages
        # resident (several GiB per layer on a multi-host load).
        jax.block_until_ready(list(loaded.values()))
        # Free the host copies now that the weights are on the TPU.
        for name in param_names:
            _free_torch_storage(getattr(layer, name))
            delattr(layer, name)

        w13_weight = loaded["w13_weight"]
        w13_weight_scale = loaded["w13_weight_scale"]
        w2_weight = loaded["w2_weight"]
        w2_weight_scale = loaded["w2_weight_scale"]
        w13_global_scale = loaded["w13_weight_scale_2"]
        w2_global_scale = loaded["w2_weight_scale_2"]
        w13_bias = loaded.get("w13_bias")
        w2_bias = loaded.get("w2_bias")
        del loaded

        # Create FusedMoEWeights with original weights
        weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=w13_bias,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=w2_bias,
        )

        # Shard weights to TPU before processing to avoid OOM
        weights = shard_moe_weights_to_tpu(weights, self.mesh)

        desired_quant_dtype = to_jax_dtype(
            envs.MOE_REQUANTIZE_WEIGHT_DTYPE
        ) if envs.MOE_REQUANTIZE_WEIGHT_DTYPE else None
        requant_block_size = int(envs.MOE_REQUANTIZE_BLOCK_SIZE
                                 ) if envs.MOE_REQUANTIZE_BLOCK_SIZE else None

        weights = _unpack_and_fold_scales(weights, w13_global_scale,
                                          w2_global_scale)
        # Outside jit, as in the FP8 MoE method: on multi-host slices
        # process_quantized_moe_weights places weights with
        # general_device_put, which needs concrete arrays.
        weights = process_quantized_moe_weights(
            weights=weights,
            moe_backend=self.moe_backend,
            mesh=self.mesh,
            activation=layer.activation,
            weight_block_size=(1, 16),  # NVFP4 block size is 16
            desired_quant_dtype=desired_quant_dtype,
            requant_block_size=requant_block_size,
        )

        weights = torch_view(weights)

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)
        _release_host_memory()

    def apply_monolithic(self, layer, x, router_logits, **kwargs):
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
                              router_logits=router_logits,
                              input_ids=kwargs.get("input_ids", None))
