# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import jax
import jax.numpy as jnp
import yaml

from tpu_commons.logger import init_logger

logger = init_logger(__name__)

MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)

QUANTIZATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE = 2000


def quantize(x: jax.Array, quant_dtype: jnp.dtype, clip_to_dtype: bool = True):
    """Quantizes uses a per-tensor approach.
      TODO (jacobplatin): support a per-token approach

    Args:
        x: the value to quantize
        quant_dtype: the dtype to quantize to
        clip_to_dtype: whether to clip the value to

    Returns:
         x (jax.Array): the quantized value
         scale (jax.Array): the scale factor (of shape (1,))
          NOTE: this should really be a float, but static types don't play
          nicely with JAX tracing
    """
    # Would be nicer to do this as a dictionary, but indexing with
    # a jnp.dtype didn't work for some reason
    if quant_dtype == jnp.int8:
        dtype_max = MAX_INT8
    elif quant_dtype == jnp.int4:
        dtype_max = MAX_INT4
    elif quant_dtype == jnp.float8_e4m3fn:
        dtype_max = E4M3_MAX
    else:
        raise ValueError(f"Unsupported quant dtype: {quant_dtype}")

    scale = jnp.max(jnp.abs(x)) / dtype_max
    # Ensure scales are not zero to avoid division by zero errors.
    scale = jnp.maximum(scale, 1e-6)
    # TODO (jacobplatin): is this cast to FP32 something we want?
    x = x.astype(jnp.float32) / scale

    if clip_to_dtype:
        dtype_info = jnp.finfo(quant_dtype)
        x = jnp.clip(x, a_min=dtype_info.min, a_max=dtype_info.max)

    x = (x).astype(quant_dtype)

    # Upcast to float32 to avoid a SMEM Mosaic error with bfloat16
    # NOTE: the scales are really floats but static types don't play
    # nicely with JAX tracing
    scale = scale.reshape(-1).astype(jnp.float32)

    return x, scale


def quantization_config_file_path_to_dict(
        quantization_config_file_path: str) -> dict:
    """
    Converts a quantization config YAML file path to a dictionary.

    Args:
        quantization_config_file_path: the path to the quantization config YAML file

    Returns:
        a dictionary containing the quantization config
    """
    all_entries = os.listdir(QUANTIZATION_CONFIG_PATH)
    for filename in all_entries:
        if filename == quantization_config_file_path:
            path = os.path.join(QUANTIZATION_CONFIG_PATH, filename)
            with open(path, "r") as f:
                return yaml.safe_load(f)
    raise ValueError(
        f"Could not find quantization config file with name '{quantization_config_file_path}' in 'tpu_commons/models/jax/utils/quantization/configs."
    )
