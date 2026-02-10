# SPDX-License-Identifier: Apache-2.0
"""All-gather matmul kernel's tuned block sizes."""

import re

import jax

# key:
#    - tpu_version
#    - m
#    - n
#    - k
#    - dtype
#    - tp_size
# value:
#    - bn
#    - bk
TUNED_BLOCK_SIZES = {
    # go/keep-sorted start
    (6, 1024, 51200, 5120, 'bfloat16', 8): (6400, 2560),
    (6, 1024, 57344, 8192, 'bfloat16', 8): (7168, 8192),
    (6, 2048, 51200, 5120, 'bfloat16', 8): (1280, 5120),
    (6, 2048, 57344, 8192, 'bfloat16', 8): (1024, 8192),
    (6, 4096, 51200, 5120, 'bfloat16', 8): (3200, 5120),
    (6, 8192, 51200, 5120, 'bfloat16', 8): (1280, 5120),
    # go/keep-sorted end
}


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        return -1
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]

    # v6: "TPU v6"
    # v7: "TPU7x"
    assert kind[:3] == 'TPU', kind
    return int(re.search(r'\d+', kind).group())


def get_key(
    m,
    n,
    k,
    dtype,
    tp_size,
):
    """Returns the key for the given parameters."""
    return (
        get_tpu_version(),
        m,
        n,
        k,
        dtype,
        tp_size,
    )


def get_tuned_block_sizes(m, n, k, dtype_name, tp_size):
    """Returns the tuned block sizes for the given parameters."""
    key = get_key(m, n, k, dtype_name, tp_size)
    return TUNED_BLOCK_SIZES.get(key, (None, None))
