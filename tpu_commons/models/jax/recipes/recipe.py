from dataclasses import dataclass


from tpu_commons.models.jax.common.quantization import QuantizationConfig
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.tpu_commons.models.jax.common.model import ModelConfig


@dataclass
class Recipe():
    model: ModelConfig
    sharding: ShardingConfig
    quant: QuantizationConfig
    # serving: Serving #TODO:
