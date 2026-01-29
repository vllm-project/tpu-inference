# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for ragged paged attention."""
from jax._src import dtypes


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return (dtypes.bit_width(dtype)
            if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()
