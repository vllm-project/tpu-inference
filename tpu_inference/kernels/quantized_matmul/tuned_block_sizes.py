# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for quantized matmul kernel."""

import json
import os
import pathlib
from typing import NamedTuple

from tpu_inference.logger import init_logger
from tpu_inference.utils import get_tpu_generation, get_tpu_name_slug

logger = init_logger(__name__)


class TunedKey(NamedTuple):
    tpu_version: int
    n_batch: int
    n_out: int
    n_in: int
    x_q_dtype: str
    w_q_dtype: str


class TunedValue(NamedTuple):
    batch_block_size: int
    out_block_size: int
    in_block_size: int
    n_lane_multiplier: int = 1


_TUNING_DATA_CACHE = {}


def _get_tuning_file_path(tpu_slug: str) -> str:
    """Maps tpu_slug (str) to the corresponding JSON filename."""
    fname = f"{tpu_slug}.json"

    # Assume data is in tpu_inference/kernels/tuned_data/matmul/
    # file: tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py
    # target: tpu_inference/kernels/tuned_data/quantized_matmul/{fname}
    base_path = pathlib.Path(__file__).parent.resolve()
    # Go up from quantized_matmul -> kernels -> tuned_data -> quantized_matmul
    data_dir = base_path.parent / "tuned_data" / "quantized_matmul"

    return str(data_dir / fname)


def _load_tuning_data(tpu_slug: str) -> dict:
    """Loads tuning data for the given TPU slug from JSON."""
    if tpu_slug in _TUNING_DATA_CACHE:
        return _TUNING_DATA_CACHE[tpu_slug]

    file_path = _get_tuning_file_path(tpu_slug)
    if not os.path.exists(file_path):
        _TUNING_DATA_CACHE[tpu_slug] = None
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            _TUNING_DATA_CACHE[tpu_slug] = data
            return data
    except Exception as e:
        logger.error(f"Failed to load tuning data from {file_path}: {e}")
        _TUNING_DATA_CACHE[tpu_slug] = None
        return None


DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: 48 * 1024 * 1024}


def get_device_vmem_limit() -> int:
    tpu_version = get_tpu_generation()
    if tpu_version not in DEVICE_VMEM_LIMIT:
        logger.warning_once(
            'VMEM limit for TPU version %d not found. Using default VMEM limit '
            'of 96MiB', tpu_version)
        return 96 * 1024 * 1024
    return DEVICE_VMEM_LIMIT[tpu_version]


def get_tuned_block_sizes(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
) -> TunedValue:
    """Retrieve the tuned block sizes for the given parameters.

  Args:
      n_batch: The batch size.
      n_out: The number of output features.
      n_in: The number of input features.
      x_q_dtype: The data type of the activation ('int8' or 'float8_e4m3fn').
      w_q_dtype: The data type of the weight ('int8' or 'float8_e4m3fn').

  Returns:
      tuple: A tuple containing the batch_block_size, out_block_size, and
      in_block_size.
  """
    # Use slug for JSON lookup
    tpu_slug = get_tpu_name_slug()

    # Construct comma-separated key string for JSON lookup
    # Exclude tpu_version from the string key as it distinguishes the file
    json_key = f"{n_batch},{n_out},{n_in},{x_q_dtype},{w_q_dtype}"

    data = _load_tuning_data(tpu_slug)
    if data and json_key in data:
        # JSON validation: value should be list [batch, out, in]
        val = data[json_key]

        if isinstance(val, dict) and "config" in val:
            # TODO: Remove this check once we confirm all JSONs are flat lists
            cfg = val["config"]
            return TunedValue(cfg["batch_block_size"], cfg["out_block_size"],
                              cfg["in_block_size"])
        return TunedValue(*val)

    tpu_generation = get_tpu_generation()
    keys = (tpu_generation, n_batch, n_out, n_in, x_q_dtype, w_q_dtype)
    logger.warning_once(
        'Couldn`t find tuned sizes for the quantized matmul kernel with %s',
        keys)
    return TunedValue(128, 128, 128)
