from typing import Optional

import torch
from jax.sharding import PartitionSpec
from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase  # noqa: E501
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig, CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod, CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)

from tpu_commons.models.vllm.quantization.common import JaxCommonConfig
from tpu_commons.models.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    JaxCompressedTensorsW8A8Fp8
from tpu_commons.models.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import \
    JaxCompressedTensorsW8A8Int8
from tpu_commons.models.vllm.quantization.unquantized import \
    JaxUnquantizedConfig

P = PartitionSpec
logger = init_logger(__name__)


@register_quantization_config("jax-compressed-tensors")
class JaxCompressedTensorsConfig(CompressedTensorsConfig, JaxCommonConfig):

    @classmethod
    def get_name(cls) -> str:
        return "jax-compressed-tensors"

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping)

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            format = scheme_dict.get("format")

        if weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        # TODO(kyuyeunk): Add support for different act_quant_format
        act_quant_format = is_activation_quantization_format(  # noqa: F841
            format
        ) if format is not None else is_activation_quantization_format(
            self.quant_format)

        linear_config = self.get_linear_config(layer)
        if self._is_fp8_w8a8(weight_quant, input_quant):
            is_static_input_scheme = input_quant and not input_quant.dynamic
            return JaxCompressedTensorsW8A8Fp8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=is_static_input_scheme,
                jax_config=linear_config,
            )
        if self._is_dynamic_token_w8a8(weight_quant, input_quant):
            return JaxCompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric,
                jax_config=linear_config,
            )
        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return JaxUnquantizedConfig.get_quant_method(self, layer, prefix)
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return JaxUnquantizedConfig.get_quant_method(
                    self, layer, prefix)
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, FusedMoE):
            raise NotImplementedError(
                "FusedMoE quantization is currently not supported.")
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        return None
