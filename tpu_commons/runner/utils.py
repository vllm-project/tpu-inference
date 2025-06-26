# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
from typing import Optional


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
