"""Auto-tuned block sizes for ragged paged attention."""

import jax.numpy as jnp

from tpu_commons.kernels.ragged_paged_attention.v3.util import (
    align_to, get_device_name, get_dtype_packing, get_tpu_version,
    next_power_of_2)

# key[device_name]:
#     - page_size
#     - q_dtype_name
#     - kv_dtype_name
#     - actual_num_q_heads
#     - actual_num_kv_heads
#     - head_dim
#     - max_num_tokens
#     - max_model_len = page_size * pages_per_seq
# value:
#     - num_kv_pages_per_block
#     - num_queries_per_block
TUNED_BLOCK_SIZES = {}


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
    """Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table."""
    tpu_version = get_tpu_version()
    if tpu_version < 4:
        raise NotImplementedError('TPU version must be 4 or higher.')
    key = get_simplified_key(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
    )
    device_name = get_device_name()

    # Default block sizes.
    bkv_p, bq = (2048 // page_size, 32)
    if tpu_version == 4:
        # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
        bkv_p, bq = (512 // page_size, 32)
    elif device_name in TUNED_BLOCK_SIZES:
        if key in TUNED_BLOCK_SIZES[device_name]:
            bkv_p, bq = TUNED_BLOCK_SIZES[device_name][key]
    return (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))


def get_simplified_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
):
    """Get the simplified key to reduce the number of combinations."""
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    assert num_kv_heads_x2 % 2 == 0

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads_x2 // 2),
        align_to(head_dim, 128),
        next_power_of_2(max_model_len),
    )
