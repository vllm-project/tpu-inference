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
"""JAX-native ``compressed-tensors`` quantization config (issue #2261).

Composes (does not subclass) the upstream vLLM ``CompressedTensorsConfig`` to
reuse its config-group parsing and scheme detection, then dispatches each layer
to the existing JAX fp8 quant methods.
"""

from types import MappingProxyType
from typing import Optional

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig as VllmUpstreamCTConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match, should_ignore_layer)

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (QuantizationConfig,
                                                           QuantLinearConfig)
from tpu_inference.layers.jax.quantization.fp8 import (
    Fp8BlockwiseLinearMethod, Fp8FusedMoEMethod, Fp8TensorwiseLinearMethod,
    Fp8TensorwiseMergedLinearMethod)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)


class _Fp8BlockConfigShim:
    """Stand-in for Fp8Config passed to Fp8BlockwiseLinearMethod.

    That method only reads ``quant_config.weight_block_size``, so exposing that
    single attribute avoids fabricating a full Fp8Config.
    """

    def __init__(self, weight_block_size):
        self.weight_block_size = weight_block_size


def _weight_block_size(weight_quant) -> Optional[list[int]]:
    """Return [block_n, block_k], or None if the weights are not block-quantized."""
    block = getattr(weight_quant, "block_structure", None)
    return list(block) if block is not None else None


class CompressedTensorsConfig(QuantizationConfig):
    """JAX-native ``compressed-tensors`` config; registered in the quant map."""

    def __init__(self, hf_quant_config: dict):
        # Reuse upstream parsing of config_groups -> target_scheme_map + ignore.
        self._ct = VllmUpstreamCTConfig.from_config(hf_quant_config)
        self._target_scheme_map = self._ct.target_scheme_map
        self._ignore = self._ct.ignore
        # packed_modules_mapping drives fused-layer (gate_up/qkv) ignore
        # semantics; not yet wired for the JAX path, so match on plain names.
        self._fused_mapping = getattr(self._ct, "packed_modules_mapping",
                                      MappingProxyType({}))

    def _match_target(self, layer: JaxModule, prefix: str) -> Optional[dict]:
        """Return the config-group scheme for ``layer``, or None if unmatched.

        Match priority mirrors upstream ``find_matched_target``: the layer path
        first, then the module class name.
        """
        for target in self._target_scheme_map:
            if check_equal_or_regex_match(prefix, [target]):
                return self._target_scheme_map[target]
        # compressed-tensors also targets layers by module class name (e.g.
        # "Linear"). Upstream matches on module.__class__.__name__; our JAX
        # layers are named differently, so map JaxEinsum and MoE fused-expert
        # layers onto "Linear" (expert sub-layers are linear modules in CT).
        if isinstance(layer, (JaxEinsum, JaxRoutedExperts,
                              JaxMoE)) and "Linear" in self._target_scheme_map:
            return self._target_scheme_map["Linear"]
        return None

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, (JaxRoutedExperts, JaxMoE)):
            if should_ignore_layer(prefix,
                                   ignore=self._ignore,
                                   fused_mapping=self._fused_mapping):
                return UnquantizedFusedMoEMethod(layer)
            scheme = self._match_target(layer, prefix)
            if scheme is None:
                return UnquantizedFusedMoEMethod(layer)
            weight_quant = scheme.get("weights")
            input_quant = scheme.get("input_activations")
            if self._ct._is_fp8_w8a8(weight_quant, input_quant):
                return Fp8FusedMoEMethod(_weight_block_size(weight_quant))
            return UnquantizedFusedMoEMethod(layer)
        if not isinstance(layer, JaxEinsum):
            return None

        linear_config = QuantLinearConfig(layer, enable_sp=False)

        if should_ignore_layer(prefix,
                               ignore=self._ignore,
                               fused_mapping=self._fused_mapping):
            return UnquantizedLinearMethod(linear_config)

        scheme = self._match_target(layer, prefix)
        if scheme is None:
            return UnquantizedLinearMethod(linear_config)

        weight_quant = scheme.get("weights")
        input_quant = scheme.get("input_activations")
        # _is_fp8_w8a8 is a private upstream helper; reused deliberately to keep
        # scheme detection identical to vLLM (accepted coupling risk).
        if self._ct._is_fp8_w8a8(weight_quant, input_quant):
            block = _weight_block_size(weight_quant)
            if block is not None:
                if isinstance(layer, JaxMergedColumnParallelLinear):
                    # TODO(#2261): need to implement blockwise fp8 for JaxMergedColumnParallelLinear
                    raise NotImplementedError(
                        "compressed-tensors blockwise fp8 is not yet supported "
                        "for JaxMergedColumnParallelLinear layers.")
                # compressed-tensors serializes the dequant scale as
                # "weight_scale" (DeepSeek-style checkpoints, the method's
                # default, use "weight_scale_inv"), so create the param under
                # the name the checkpoint will look up.
                return Fp8BlockwiseLinearMethod(
                    _Fp8BlockConfigShim(block),
                    layer,
                    linear_config,
                    weight_scale_name="weight_scale")
            if isinstance(layer, JaxMergedColumnParallelLinear):
                return Fp8TensorwiseMergedLinearMethod(layer, linear_config)
            return Fp8TensorwiseLinearMethod(layer, linear_config)

        # TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported).
        raise NotImplementedError(
            f"compressed-tensors scheme for layer '{prefix}' is not yet "
            "supported in the JAX path.")
