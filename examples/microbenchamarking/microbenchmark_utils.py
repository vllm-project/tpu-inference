# pytype: disable=import-error
# pytype: disable=module-attr
# pytype: disable=attribute-error
# pytype: disable=wrong-arg-types
import bisect
import os
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np


from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.sharding import build_mesh
from tpu_commons.utils import make_optimized_mesh

logger = init_logger(__name__)
power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens


@dataclass
class Sampler:
    type: str
    std: float = None

    def generate_samples(self, shape: Tuple[int], fill_val: Any) -> np.array:
        if self.type.lower() == "fixed":
            return np.full(shape, fill_val)
        elif self.type.lower() == "normal":
            return np.random.normal(loc=0.0, scale=self.std, size=shape)


def init_mesh(vllm_config, devices) -> None:
    try:
        sharding_strategy = \
            vllm_config.additional_config["sharding"]["sharding_strategy"]
    except KeyError:
        sharding_strategy = {"tensor_parallelism": len(devices)}

    if os.getenv("NEW_MODEL_DESIGN", False):
        mesh = build_mesh(devices, sharding_strategy)
    else:
        try:
            dp = sharding_strategy["data_parallelism"]
        except KeyError:
            dp = 1
        try:
            tp = sharding_strategy["tensor_parallelism"]
        except KeyError:
            tp = len(devices)

        axis_names = ("data", "model")
        mesh_shape = (dp, tp)
        # for deepseekv3
        # axis_names = ('data', 'expert', 'model')
        # mesh_shape = (dp, 1, tp)

        mesh = make_optimized_mesh(mesh_shape, axis_names, devices=devices)
    return mesh


def nearest_power_of_two(val: int) -> int:
    index = bisect.bisect_left(power_of_two, val)
    assert index < len(power_of_two)
    return power_of_two[index]
