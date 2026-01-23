# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for quantized matmul kernel."""

import json
import os
import pathlib
from typing import NamedTuple

import jax

from tpu_inference.logger import init_logger
from tpu_inference.utils import get_tpu_generation, get_tpu_name_slug

logger = init_logger(__name__)

_TUNING_DATA_CACHE = {}


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


def _get_tuning_file_path(device_name: str) -> str:
    """Maps device name to the corresponding JSON filename."""
    # Use standardized slug from utils
    slug = get_tpu_name_slug(device_name)

    # Assume data is in tpu_inference/kernels/tuned_data/quantized_matmul/
    base_path = pathlib.Path(__file__).parent.resolve()
    # Go up from quantized_matmul -> kernels -> tuned_data -> quantized_matmul
    data_dir = base_path.parent / "tuned_data" / "quantized_matmul"

    return str(data_dir / f"{slug}.json")


def _load_tuning_data(device_name: str) -> dict:
    """Loads tuning data for the given device from JSON."""
    if device_name in _TUNING_DATA_CACHE:
        return _TUNING_DATA_CACHE[device_name]

    file_path = _get_tuning_file_path(device_name)
    if not os.path.exists(file_path):
        _TUNING_DATA_CACHE[device_name] = None
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            _TUNING_DATA_CACHE[device_name] = data
            return data
    except Exception as e:
        logger.error(f"Failed to load tuning data from {file_path}: {e}")
        _TUNING_DATA_CACHE[device_name] = None
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
    # Determine the TPU device kind (e.g. "TPU v5 Lite") to resolve the correct JSON file.
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        # Fallback for environments without TPU devices (e.g. CI/CD)
        kind = "TPU v4"

    data = _load_tuning_data(kind)

    key_str = f"{n_batch},{n_out},{n_in},{x_q_dtype},{w_q_dtype}"

    tuned_values = None
    if data:
        tuned_values = data.get(key_str)

    if tuned_values is None:
        logger.warning_once(
            'Couldn`t find tuned sizes for the quantized matmul kernel with %s',
            key_str)
        return TunedValue(128, 128, 128)
    else:
        return TunedValue(*tuned_values)
