import copy

from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

from tpu_inference.layers.vllm.quantization.awq import VllmAWQConfig
from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig  # noqa: E501
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedConfig


def get_tpu_quantization_config(vllm_config: VllmConfig,
                                mesh: Mesh) -> QuantizationConfig:
    model_config = copy.deepcopy(vllm_config.model_config)
    # TODO(kyuyeunk): Add support for "tpu_int8".
    method_to_config: dict[str, str] = {
        None: VllmUnquantizedConfig,
        "compressed-tensors": VllmCompressedTensorsConfig,
        "awq": VllmAWQConfig,
    }
    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported."
            f" Supported methods are {method_to_config.keys()}")
    quant_config = method_to_config[model_config.quantization]
    assert issubclass(quant_config, JaxCommonConfig)
    quant_config.set_configs(vllm_config, mesh)

    model_config.quantization = quant_config.get_name()
    return VllmConfig.get_quantization_config(model_config,
                                              vllm_config.load_config)
