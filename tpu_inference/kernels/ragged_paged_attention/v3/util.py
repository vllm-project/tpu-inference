"""Utility functions for ragged paged attention."""
import jax
from jax._src import dtypes


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return dtypes.bit_width(dtype)
    return dtypes.itemsize_bits(dtype)


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


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        return -1
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]
    if kind.endswith('p') or kind.endswith('e'):
        kind = kind[:-1]
    if kind == 'TPU7x':
        return 7
    assert kind[:-1] == 'TPU v', kind
    return int(kind[-1])
