# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
from typing import Optional

import time

from vllm.logger import init_logger

MIN_NUM_SEQS = 8

logger = init_logger(__name__)

def determine_do_sampling(top_k: int, temperature: float) -> bool:
    """
  Determine whether sampling should be done for the next tokens in the model forward pass.

  Args:
    top_k: The top_k value (from SamplingParams).
    temperature: The temperature value (from SamplingParams).

  Returns:
    True if sampling should be done, False otherwise.
  """
    return top_k != 1 and temperature != 0.0


def pad_to_multiple(x: int,
                    multiple: int = 8,
                    max_limit: Optional[int] = None,
                    keep_one: bool = False) -> int:
    assert x > 0
    if keep_one and x == 1:
        return x
    x = x + (-x % multiple)
    if max_limit is not None:
        x = min(x, max_limit)
    return x


def get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)

class LatencyTracker:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        logger.info(f"Latency for '{self.name}': {elapsed_time:.3f} seconds")