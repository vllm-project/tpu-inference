"""Auto-tuned block sizes for ragged paged attention."""

import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, get_dtype_packing, get_tpu_version, next_power_of_2)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_device_name

logger = init_logger(__name__)

# key
#   - device_name
#     - page_size
#       - q_{q_dtype_name}_kv_{kv_dtype_name}
#         - q_head-{num_q_heads}_kv_head-{num_kv_heads}-_head-{head_dim}
#           - max_model_len
# value:
#   - (num_kv_pages_per_block, num_queries_per_block)
TUNED_BLOCK_SIZES = {
    'TPU v5e': {
        128: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-8_kv_head-2_head-64': {
                    4096: (16, 32),
                    8192: (32, 128),
                    128: (1, 16),
                    256: (1, 64),
                    512: (4, 128),
                    1024: (4, 16),
                    2048: (16, 64),
                },
                'q_head-64_kv_head-8_head-64': {
                    128: (1, 16),
                    4096: (16, 16),
                    1024: (8, 8),
                    256: (2, 16),
                    8192: (16, 32),
                    2048: (8, 16),
                    512: (4, 8),
                },
                'q_head-32_kv_head-4_head-64': {
                    256: (2, 8),
                    512: (4, 32),
                    1024: (8, 8),
                    2048: (16, 8),
                    4096: (32, 32),
                    8192: (16, 32),
                    128: (1, 8),
                },
                'q_head-16_kv_head-2_head-64': {
                    128: (1, 128),
                    256: (2, 128),
                    512: (4, 32),
                    1024: (8, 16),
                    2048: (8, 32),
                    4096: (16, 32),
                    8192: (16, 32),
                },
            }
        },
        256: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-16_kv_head-2_head-64': {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 32),
                    8192: (16, 16),
                    256: (1, 128),
                    512: (2, 128),
                },
                'q_head-64_kv_head-8_head-64': {
                    256: (1, 8),
                    512: (2, 32),
                    1024: (4, 16),
                    2048: (8, 8),
                    4096: (8, 32),
                    8192: (8, 32),
                },
                'q_head-8_kv_head-2_head-64': {
                    256: (1, 8),
                    512: (1, 32),
                    1024: (4, 32),
                    2048: (8, 64),
                    4096: (8, 16),
                    8192: (16, 32),
                },
                'q_head-32_kv_head-4_head-64': {
                    256: (1, 16),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (8, 16),
                    8192: (8, 32),
                },
            }
        },
    },
    'TPU v6e': {
        128: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-8_kv_head-2_head-64': {
                    4096: (32, 32),
                    8192: (32, 128),
                    128: (1, 64),
                    256: (2, 128),
                    512: (4, 256),
                    1024: (8, 16),
                    2048: (16, 32),
                },
                'q_head-64_kv_head-8_head-64': {
                    128: (1, 32),
                    4096: (32, 16),
                    1024: (8, 32),
                    256: (2, 16),
                    8192: (32, 8),
                    2048: (16, 32),
                    512: (4, 32),
                },
                'q_head-32_kv_head-4_head-64': {
                    256: (2, 16),
                    512: (4, 128),
                    1024: (8, 64),
                    2048: (16, 32),
                    4096: (16, 16),
                    8192: (32, 32),
                    128: (1, 64),
                },
                'q_head-16_kv_head-2_head-64': {
                    128: (1, 128),
                    256: (2, 128),
                    512: (4, 128),
                    1024: (8, 64),
                    2048: (8, 32),
                    4096: (32, 32),
                    8192: (32, 32),
                },
            }
        },
        256: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-16_kv_head-2_head-64': {
                    1024: (4, 128),
                    2048: (8, 32),
                    4096: (16, 16),
                    8192: (16, 16),
                    256: (1, 64),
                    512: (2, 32),
                },
                'q_head-64_kv_head-8_head-64': {
                    256: (1, 32),
                    512: (2, 32),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                'q_head-8_kv_head-2_head-64': {
                    256: (1, 8),
                    512: (2, 128),
                    1024: (4, 64),
                    2048: (8, 32),
                    4096: (8, 32),
                    8192: (16, 128),
                },
                'q_head-32_kv_head-4_head-64': {
                    256: (1, 32),
                    512: (2, 8),
                    1024: (4, 8),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
            }
        },
    },
    'TPU v7': {
        128: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-8_kv_head-2_head-64': {
                    4096: (32, 16),
                    8192: (32, 64),
                    128: (1, 16),
                    256: (2, 64),
                    512: (4, 16),
                    1024: (8, 32),
                    2048: (16, 32),
                },
                'q_head-64_kv_head-8_head-64': {
                    128: (1, 16),
                    4096: (32, 8),
                    1024: (8, 16),
                    256: (2, 16),
                    8192: (32, 16),
                    2048: (16, 16),
                    512: (4, 16),
                },
                'q_head-32_kv_head-4_head-64': {
                    128: (4, 16),
                    256: (2, 8),
                    512: (4, 16),
                    1024: (8, 16),
                    2048: (16, 32),
                    4096: (32, 64),
                    8192: (32, 24),
                    16384: (32, 24),
                },
                'q_head-16_kv_head-2_head-64': {
                    128: (1, 64),
                    256: (2, 8),
                    512: (4, 8),
                    1024: (8, 16),
                    2048: (16, 16),
                    4096: (32, 32),
                    8192: (32, 32),
                },
            }
        },
        256: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-16_kv_head-2_head-64': {
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 8),
                    8192: (16, 16),
                    256: (1, 64),
                    512: (2, 32),
                },
                'q_head-64_kv_head-8_head-64': {
                    256: (1, 8),
                    512: (2, 16),
                    1024: (4, 32),
                    2048: (8, 16),
                    4096: (16, 16),
                    8192: (16, 16),
                },
                'q_head-8_kv_head-2_head-64': {
                    256: (1, 256),
                    512: (2, 16),
                    1024: (4, 16),
                    2048: (8, 16),
                    4096: (16, 32),
                    8192: (16, 16),
                },
                'q_head-32_kv_head-4_head-64': {
                    256: (1, 64),
                    512: (2, 32),
                    1024: (4, 8),
                    2048: (8, 8),
                    4096: (16, 32),
                    8192: (16, 32),
                },
            }
        },
    },
}


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
) -> tuple[int, int]:
    """Search tuned values for (num_kv_pages_per_blk, num_queries_per_blk)."""

    keys = get_lookup_keys(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
    )
    device, page_size, dtypes, head_dims, max_model_len = keys

    try:
        bkv_p, bq = TUNED_BLOCK_SIZES[device][page_size][dtypes][head_dims][
            max_model_len]
        logger.info_once('RPA v3 kernel: Found tuned sizes for %s', keys)
    except KeyError:
        logger.warning_once('RPA v3 kernel: Couldn`t find tuned sizes for %s',
                            keys)
        # When not available use a sensible default based on TPU version
        # Set default block sizes for each tpu_version.
        tpu_version = get_tpu_version()
        if tpu_version < 4:
            raise NotImplementedError('TPU version must be 4 or higher.')
        match tpu_version:
            case 4:
                # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
                bkv_p, bq = (512 // page_size, 32)
            case 7:
                bkv_p, bq = (4096 // page_size, 32)
            case _:
                bkv_p, bq = (2048 // page_size, 32)

        bkv_p, bq = (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))

    logger.info_once("RPA v3 kernel tuned block sizes: bkv_p=%s, bq=%s", bkv_p,
                     bq)

    return bkv_p, bq


def get_lookup_keys(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_model_len,
):
    """Get the lookup keys for tuned block sizes."""
    (
        page_size,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    ) = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f'q_{q_dtype_name}_kv_{kv_dtype_name}',
        f'q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}',
        next_power_of_2(max_model_len),
    )


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
):
    """Get the simplified key."""
    assert head_dim == 64
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads),
        head_dim,
        next_power_of_2(max_model_len),
    )
