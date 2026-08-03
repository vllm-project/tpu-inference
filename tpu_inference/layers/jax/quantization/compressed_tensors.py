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

import json
import os
from types import MappingProxyType
from typing import Iterator, Optional

from safetensors import safe_open
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
from tpu_inference.layers.jax.quantization.mxfp4 import (
    MXFP4_PACK_QUANTIZED_FORMAT, CompressedTensorsMxfp4MoEMethod)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


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


# A ``*ForConditionalGeneration`` checkpoint nests the whole text stack under
# this prefix (Kimi-K3, Llama-4, Gemma-4-MM all do), while the model's own
# module paths -- what ``get_quant_method`` is asked about -- do not carry it.
_WRAPPER_TEXT_PREFIX = "language_model."

_SAFETENSORS_INDEX = "model.safetensors.index.json"


def _checkpoint_tensor_names(model_name_or_path: str,
                             download_dir: Optional[str]) -> Iterator[str]:
    """Every tensor name in the checkpoint.

    Prefers the safetensors index, for two reasons. It is one small JSON
    instead of every shard's header (the alternative opens 96+ files for a
    multi-TB checkpoint), and it is present locally even when the weights
    themselves never are: loading from object storage downloads the config
    files -- the index among them -- and streams only the shards, so reading
    tensor names off the shards fails with "cannot find any *.safetensors
    files" and takes the checkpoint's own layout out of the decision.

    Falls back to the shard headers for single-file checkpoints, which have no
    index.
    """
    index = None
    if os.path.isdir(model_name_or_path):
        local_index = os.path.join(model_name_or_path, _SAFETENSORS_INDEX)
        if os.path.isfile(local_index):
            index = local_index
    else:
        # An HF repo id: fetch only the index (one small JSON) rather than
        # materializing every weight shard just to read tensor names off it.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            index = hf_hub_download(repo_id=model_name_or_path,
                                    filename=_SAFETENSORS_INDEX,
                                    cache_dir=download_dir)
        except EntryNotFoundError:
            # Single-file repos ship no index; read the shard header below.
            index = None
    if index is not None:
        with open(index) as f:
            yield from json.load(f)["weight_map"]
        return
    # Imported here: weight_utils pulls in the model stack, and this module is
    # imported from the quantization package that stack itself imports.
    from tpu_inference.models.jax.utils.weight_utils import \
        get_model_weights_files
    for path in get_model_weights_files(model_name_or_path, download_dir):
        with safe_open(path, framework="numpy") as f:
            yield from f.keys()


# A module is stored compressed when the checkpoint gives it one of these
# companion tensors. Pack-quantized formats (MXFP4) rename the weight itself
# to `weight_packed`; float-quantized formats (FP8) keep a plain `weight` and
# mark compression with `weight_scale` alone. Matching only `weight_packed`
# classified EVERY module of an FP8 checkpoint as uncompressed and silently
# built the whole model unquantized (caught by gemma-4 FP8-Dynamic in CI).
_COMPRESSED_COMPANION_SUFFIXES = (".weight_packed", ".weight_scale")


def _compressed_module_paths(
        model_name_or_path: Optional[str],
        download_dir: Optional[str]) -> Optional[set[str]]:
    """Module paths the checkpoint stores compressed.

    The returned set also holds every ancestor of such a path, so a caller
    asking about a parent module (an MoE block owns a subtree of per-expert
    modules) is a plain set lookup rather than a scan over a checkpoint that
    can have hundreds of thousands of tensors.

    Returns ``None`` when the checkpoint cannot be inspected, which callers
    read as "no information, trust the config".
    """
    if not model_name_or_path:
        return None
    try:
        packed = set()
        for name in _checkpoint_tensor_names(model_name_or_path, download_dir):
            for suffix in _COMPRESSED_COMPANION_SUFFIXES:
                if name.endswith(suffix):
                    path = name[:-len(suffix)]
                    break
            else:
                continue
            if path.startswith(_WRAPPER_TEXT_PREFIX):
                path = path[len(_WRAPPER_TEXT_PREFIX):]
            while path and path not in packed:
                packed.add(path)
                path = path.rpartition(".")[0]
    except Exception as e:
        logger.warning(
            "[compressed-tensors] could not read the checkpoint tensor names "
            "for %s (%s: %s); falling back to the config's target/ignore lists "
            "to decide which layers are compressed. If the checkpoint leaves "
            "some targeted modules uncompressed, loading them will fail.",
            model_name_or_path,
            type(e).__name__, e)
        return None
    logger.info(
        "[compressed-tensors] checkpoint %s stores %d module paths compressed "
        "(counting the parents of each); using that, not the config ignore "
        "list, to select quant methods.", model_name_or_path, len(packed))
    return packed


class CompressedTensorsConfig(QuantizationConfig):
    """JAX-native ``compressed-tensors`` config; registered in the quant map."""

    def __init__(self,
                 hf_quant_config: dict,
                 model_name_or_path: str | None = None,
                 download_dir: str | None = None):
        # Reuse upstream parsing of config_groups -> target_scheme_map + ignore.
        self._ct = VllmUpstreamCTConfig.from_config(hf_quant_config)
        self._target_scheme_map = self._ct.target_scheme_map
        self._ignore = self._ct.ignore
        # packed_modules_mapping drives fused-layer (gate_up/qkv) ignore
        # semantics; not yet wired for the JAX path, so match on plain names.
        self._fused_mapping = getattr(self._ct, "packed_modules_mapping",
                                      MappingProxyType({}))
        self._format = hf_quant_config.get("format")
        # The config's ignore list is not a reliable statement of what is
        # actually compressed: a checkpoint can leave a target module
        # uncompressed without listing it (Kimi-K3 stores the MoE latent
        # down/up projections and the attention-residual projections as plain
        # bf16 while its ignore list only covers self_attn / shared_experts /
        # mlp / lm_head). The checkpoint's own tensor names are authoritative,
        # so record which module paths carry a compression companion tensor.
        self._compressed_modules = _compressed_module_paths(
            model_name_or_path, download_dir)

    def _is_compressed_in_checkpoint(self, prefix: str) -> bool:
        """Whether the checkpoint actually stores ``prefix`` compressed.

        True for a compressed module and for any ancestor of one -- an MoE
        block is asked about by the path that owns the per-expert subtree
        (`<prefix>.<expert_id>.<proj>`), never by a leaf path.

        Falls back to trusting the config when the checkpoint could not be
        inspected (``None``), so behaviour is unchanged for callers that do not
        pass a model path.
        """
        if self._compressed_modules is None:
            return True
        return prefix in self._compressed_modules

    def _stored_plain(self, prefix: str) -> bool:
        """True when a config-targeted ``prefix`` is stored uncompressed.

        Every such demotion is logged: the config says the module should be
        quantized, so building it unquantized must be visible in the load log
        rather than silent.
        """
        if self._is_compressed_in_checkpoint(prefix):
            return False
        logger.info(
            "[compressed-tensors] %s targeted by quant config but stored "
            "plain in checkpoint; using unquantized.", prefix)
        return True

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
            if scheme is None or self._stored_plain(prefix):
                return UnquantizedFusedMoEMethod(layer)
            weight_quant = scheme.get("weights")
            input_quant = scheme.get("input_activations")
            if self._format == MXFP4_PACK_QUANTIZED_FORMAT:
                return CompressedTensorsMxfp4MoEMethod(layer, weight_quant)
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

        if self._stored_plain(prefix):
            # Targeted by the config groups, but stored uncompressed.
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
