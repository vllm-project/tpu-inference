# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from tpu_inference import envs


def is_disagg_enabled() -> bool:
    # We triggrer our code path as long as prefill slices are set. This
    # allows us to test interleave mode effectively with the code path
    # for comparison purposes.
    return bool(envs.PREFILL_SLICES)


def _parse_slices(slices_str: str) -> Tuple[int, ...]:
    """Parse slices environment variable and return the a list of integers, each the size of a slice.

    For example, if slices_str is set to `2x2,2x1,2x4`, we should return `(4, 2, 8)`.

    Throws exception if the slice str is malformed.
    """
    if not slices_str:
        return ()

    try:
        slice_sizes = []
        for s in slices_str.split(','):
            dims = s.split('x')
            if len(dims) == 1:
                slice_sizes.append(int(dims[0]))
            elif len(dims) == 2:
                slice_sizes.append((int(dims[0]), int(dims[1])))
            else:
                raise ValueError("Each slice must be in 'N' or 'NxM' format.")
        return tuple(slice_sizes)
    except ValueError as e:
        raise ValueError(f"Malformed slice string: '{slices_str}'") from e


def get_prefill_slices() -> Tuple[int, ...]:
    if not envs.PREFILL_SLICES:
        return ()
    return _parse_slices(envs.PREFILL_SLICES)


def get_decode_slices() -> Tuple[int, ...]:
    if not envs.DECODE_SLICES:
        return ()
    return _parse_slices(envs.DECODE_SLICES)
