# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for quantized matmul kernel."""

import re
from typing import NamedTuple

import jax

from tpu_inference.logger import init_logger

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


TUNED_BLOCK_SIZES_RAW = {
    # go/keep-sorted start
    (7, 1024, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'):
    (1024, 2048, 4096),
    (7, 1024, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'):
    (1024, 1280, 5120),
    (7, 1024, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'):
    (1024, 1792, 4096),
    (7, 1024, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'):
    (1024, 3072, 1024),
    (7, 1024, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              2048),
    (7, 1024, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              3584),
    (7, 1024, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                               2560),
    (7, 1024, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                               2560),
    (7, 1024, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1024,
                                                               2560),
    (7, 1024, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1024,
                                                              4096),
    (7, 1024, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1280,
                                                              5120),
    (7, 1024, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560, 4096),
    (7, 1024, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                               3584),
    (7, 1024, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2048, 4096),
    (7, 128, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 3584, 4096),
    (7, 128, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 2560, 5120),
    (7, 128, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 14336,
                                                              1024),
    (7, 128, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1536, 4096),
    (7, 128, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 2048, 2048),
    (7, 128, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1024, 7168),
    (7, 128, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1024,
                                                              12800),
    (7, 128, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 512, 25600),
    (7, 128, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1280, 4096),
    (7, 128, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1280, 5120),
    (7, 128, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 5120, 2048),
    (7, 128, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 1024,
                                                              14336),
    (7, 128, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (128, 2048, 4096),
    (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 3584, 4096),
    (7, 16, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 6400, 2560),
    (7, 16, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 2048, 8192),
    (7, 16, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 768, 4096),
    (7, 16, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 2048, 2048),
    (7, 16, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 4096, 1792),
    (7, 16, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 1024, 12800),
    (7, 16, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 640, 25600),
    (7, 16, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 1280, 4096),
    (7, 16, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 1280, 5120),
    (7, 16, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 5120, 2048),
    (7, 16, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 8192, 1792),
    (7, 16, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (16, 4096, 2048),
    (7, 2048, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1792,
                                                               4096),
    (7, 2048, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560,
                                                               5120),
    (7, 2048, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1024,
                                                               8192),
    (7, 2048, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 3072,
                                                              2048),
    (7, 2048, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 4096, 2048),
    (7, 2048, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              1792),
    (7, 2048, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1024,
                                                               2560),
    (7, 2048, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                               6400),
    (7, 2048, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1280,
                                                              4096),
    (7, 2048, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                              2560),
    (7, 2048, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                              2048),
    (7, 2048, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                               2048),
    (7, 2048, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              2048),
    (7, 256, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 3584, 4096),
    (7, 256, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 2560, 5120),
    (7, 256, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 1792, 8192),
    (7, 256, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 3072, 2048),
    (7, 256, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 2048, 2048),
    (7, 256, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 4096, 1792),
    (7, 256, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 5120, 3200),
    (7, 256, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 5120, 2560),
    (7, 256, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 5120, 1024),
    (7, 256, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 1280, 5120),
    (7, 256, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 1280, 8192),
    (7, 256, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 1024,
                                                              14336),
    (7, 256, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 8192, 1024),
    (7, 32, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 14336, 1024),
    (7, 32, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 12800, 1280),
    (7, 32, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 14336, 1024),
    (7, 32, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 1536, 4096),
    (7, 32, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 2048, 2048),
    (7, 32, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 2048, 7168),
    (7, 32, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 1280, 12800),
    (7, 32, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 640, 25600),
    (7, 32, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 1280, 4096),
    (7, 32, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 1280, 5120),
    (7, 32, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 1280, 8192),
    (7, 32, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 8192, 1792),
    (7, 32, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (32, 2048, 4096),
    (7, 4096, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1792,
                                                               4096),
    (7, 4096, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560,
                                                               5120),
    (7, 4096, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1792,
                                                               4096),
    (7, 4096, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (256, 3072, 4096),
    (7, 4096, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 4096, 2048),
    (7, 4096, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              3584),
    (7, 4096, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (2048, 1024,
                                                               2560),
    (7, 4096, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                               2560),
    (7, 4096, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                              2048),
    (7, 4096, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                              2560),
    (7, 4096, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 5120, 2048),
    (7, 4096, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                               2048),
    (7, 4096, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              4096),
    (7, 512, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1792, 4096),
    (7, 512, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560, 5120),
    (7, 512, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1024, 8192),
    (7, 512, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1024, 4096),
    (7, 512, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2048, 2048),
    (7, 512, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 4096, 1024),
    (7, 512, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 5120, 1280),
    (7, 512, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560, 2560),
    (7, 512, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1280, 4096),
    (7, 512, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 1280, 5120),
    (7, 512, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 5120, 1024),
    (7, 512, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 4096, 2048),
    (7, 512, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2048, 4096),
    (7, 64, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 14336, 1024),
    (7, 64, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 3200, 5120),
    (7, 64, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 2048, 8192),
    (7, 64, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 1536, 4096),
    (7, 64, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 2048, 2048),
    (7, 64, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 1024, 7168),
    (7, 64, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 1024, 12800),
    (7, 64, 5120, 25600, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 640, 25600),
    (7, 64, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 1280, 4096),
    (7, 64, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 1280, 5120),
    (7, 64, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 2560, 4096),
    (7, 64, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 8192, 1792),
    (7, 64, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (64, 2048, 4096),
    (7, 8192, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1792,
                                                               4096),
    (7, 8192, 25600, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 2560,
                                                               5120),
    (7, 8192, 28672, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1792,
                                                               4096),
    (7, 8192, 3072, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 3072, 4096),
    (7, 8192, 4096, 2048, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 4096,
                                                              2048),
    (7, 8192, 4096, 7168, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              3584),
    (7, 8192, 5120, 12800, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2560,
                                                               1280),
    (7, 8192, 5120, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1280,
                                                              4096),
    (7, 8192, 5120, 5120, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 1280,
                                                              5120),
    (7, 8192, 5120, 8192, 'float8_e4m3fn', 'float8_e4m3fn'): (512, 5120, 2048),
    (7, 8192, 8192, 14336, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                               3584),
    (7, 8192, 8192, 4096, 'float8_e4m3fn', 'float8_e4m3fn'): (1024, 2048,
                                                              4096),
    # go/keep-sorted end
}

TUNED_BLOCK_SIZES: dict[TunedKey, TunedValue] = {
    TunedKey(*key): TunedValue(*value)
    for key, value in TUNED_BLOCK_SIZES_RAW.items()
}

DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: 48 * 1024 * 1024}


def get_device_vmem_limit() -> int:
    tpu_version = get_tpu_version()
    if tpu_version not in DEVICE_VMEM_LIMIT:
        logger.warning_once(
            'VMEM limit for TPU version %d not found. Using default VMEM limit '
            'of 96MiB', tpu_version)
        return 96 * 1024 * 1024
    return DEVICE_VMEM_LIMIT[tpu_version]


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    match = re.match(r'^TPU[^\d]*(\d+)', kind)
    if match is None:
        return -1
    return int(match.group(1))


def get_key(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
) -> TunedKey:
    """Returns the key for the given parameters."""
    return TunedKey(
        get_tpu_version(),
        n_batch,
        n_out,
        n_in,
        x_q_dtype,
        w_q_dtype,
    )


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
    key = get_key(
        n_batch,
        n_out,
        n_in,
        x_q_dtype,
        w_q_dtype,
    )
    tuned_value = TUNED_BLOCK_SIZES.get(key)
    if tuned_value is None:
        logger.warning_once(
            'Couldn`t find tuned sizes for the quantized matmul kernel with %s',
            key)
        return TunedValue(128, 128, 128)
    else:
        return tuned_value
