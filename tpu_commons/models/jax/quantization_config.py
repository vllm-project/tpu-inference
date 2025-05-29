import dataclasses
import enum
from typing import Any, Dict, List, Optional, Type, Union

import jax.numpy as jnp

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.config import ModelConfig

logger = init_logger(__name__)


class QuantizationMethod(enum.Enum):
    BITS_AND_BYTES = "bitsandbytes"
    AWQ = "awq"


@dataclasses.dataclass(init=True, repr=True)
class QuantizationConfig:
    quant_method: QuantizationMethod

    def get_quant_method(self) -> QuantizationMethod:
        return self.quant_method


@dataclasses.dataclass(init=True, repr=True)
class AWQConfig(QuantizationConfig):
    bits: int
    group_size: int
    zero_point: bool

    def __init__(self, bits: int, group_size: int, zero_point: bool, **kwargs):
        self.quant_method = QuantizationMethod.AWQ
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitsAndBytesConfig":
        return cls(**config)


@dataclasses.dataclass(init=True, repr=True)
class BitsAndBytesConfig(QuantizationConfig):
    load_in_8bit: bool
    load_in_4bit: bool
    llm_int8_threshold: float
    llm_int8_skip_modules: List[str]
    llm_int8_enable_fp32_cpu_offload: bool
    llm_int8_has_fp16_weight: bool
    bnb_4bit_compute_dtype: Union[jnp.dtype, str]
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_storage: Union[jnp.dtype, str]

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        llm_int8_threshold: float = 6.0,
        llm_int8_skip_modules: Optional[List[str]] = None,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        bnb_4bit_compute_dtype: Optional[Union[jnp.dtype, str]] = None,
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        bnb_4bit_quant_storage: Optional[Union[jnp.dtype, str]] = None,
        **kwargs,
    ) -> None:
        self.quant_method = QuantizationMethod.BITS_AND_BYTES
        unsupported_params = [
            "load_in_4bit",
            "llm_int8_threshold",
            "llm_int8_skip_modules",
            "llm_int8_enable_fp32_cpu_offload",
            "llm_int8_has_fp16_weight",
            "bnb_4bit_compute_dtype",
            "bnb_4bit_quant_type",
            "bnb_4bit_use_double_quant",
            "bnb_4bit_quant_storage",
        ]
        logger.info("Initializing BitsAndBytesConfig.")
        logger.info(f"Unsupported parameters: {unsupported_params}.")

        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage

        if load_in_4bit:
            raise ValueError("load_in_4bit=True is not supported.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitsAndBytesConfig":
        return cls(**config)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json", "quant_config.json"]


_QUANTIZATION_CONFIG_REGISTRY = {
    QuantizationMethod.BITS_AND_BYTES.value: BitsAndBytesConfig,
    QuantizationMethod.AWQ.value: AWQConfig,
}


def get_quantization_config_class(
    quantization: Optional[str], ) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(
            f"Quantization method {quantization} is not supported.")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


def get_quantization_config(
        model_config: ModelConfig) -> Optional[QuantizationConfig]:
    supported_quantization = [m.value for m in QuantizationMethod]
    quantization = None

    hf_quant_config = getattr(model_config.hf_config, "quantization_config",
                              None)
    if hf_quant_config is not None:
        quantization = str(hf_quant_config["quant_method"]).lower()

    if quantization is not None:
        if quantization not in supported_quantization:
            raise ValueError(
                f"Quantization method {quantization} is not supported. The list of supported methods is {supported_quantization}."
            )
        quant_cls = get_quantization_config_class(quantization)
        return quant_cls.from_config(hf_quant_config)
    else:
        return None
