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
"""Auto-tuned block sizes for ragged paged attention."""

import json
import os
import pathlib

import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, get_dtype_packing, next_power_of_2)
from tpu_inference.logger import init_logger
from tpu_inference.utils import (get_device_name, get_tpu_generation,
                                 get_tpu_name_slug)

logger = init_logger(__name__)

_TUNING_DATA_CACHE = {}


def _get_tuning_file_path(device_name: str) -> str:
    """Maps device name to the corresponding JSON filename."""
    # Use standardized slug from utils
    # e.g., "tpu_v6e", "tpu_v7"
    slug = get_tpu_name_slug(device_name)

    # Assume data is in tpu_inference/kernels/tuned_data/ragged_paged_attention/v3/
    # file: tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py
    # target: tpu_inference/kernels/tuned_data/ragged_paged_attention/v3/{slug}.json
    base_path = pathlib.Path(__file__).parent.resolve()
    # Go up from v3 -> rag_paged_att -> kernels -> tuned_data -> rpa -> v3
    data_dir = base_path.parent.parent / "tuned_data" / "ragged_paged_attention" / "v3"

    return str(data_dir / f"{slug}.json")


def _load_tuning_data(device_name: str) -> dict:
    """Loads tuning data for the given device from JSON."""
    if device_name in _TUNING_DATA_CACHE:
        return _TUNING_DATA_CACHE[device_name]

    file_path = _get_tuning_file_path(device_name)
    if not os.path.exists(file_path):
        # Cache None to avoid repeated filesystem checks
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


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    sliding_window=None,
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
        sliding_window,
    )
    device, page_size_key, dtypes, head_dims, extra = keys

    bkv_p, bq = None, None

    data = _load_tuning_data(device)
    if data:
        try:
            # Keys in JSON are strings
            entry = data[str(page_size_key)][dtypes][head_dims][extra]

            # Support both list format [bkv_p, bq] and dict format
            if isinstance(entry, dict) and "config" in entry:
                # Future proofing for object-based registry
                cfg = entry["config"]
                bkv_p = cfg.get("num_kv_pages_per_block")
                bq = cfg.get("num_q_per_block")
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                bkv_p, bq = entry
        except (KeyError, TypeError):
            # Not found in JSON
            pass

    if bkv_p is None:
        logger.warning_once(
            'Couldn`t find tuned sizes for the RPA v3 kernel with %s', keys)
        # When not available use a sensible default based on TPU version
        tpu_version = get_tpu_generation()
        if tpu_version < 4 and tpu_version != -1:
            # -1 means not on TPU / unknown, we typically don't raise here to allow import/testing off-device
            # But if on older TPU, it's not supported.
            pass

        match tpu_version:
            case 4:
                # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
                bkv_p, bq = (512 // page_size, 32)
            case 7:
                bkv_p, bq = (4096 // page_size, 32)
            case _:
                # Default for v5, v6, etc.
                bkv_p, bq = (2048 // page_size, 32)

    # We should consider the actual page_per_seq and max_num_tokens.
    # If page_per_seq < bkv_p or max_num_tokens < bq, using the bkv_p or bq may
    # waste computation. So we need the min here.
    bkv_p, bq = (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))

    logger.debug_once(
        'RPA v3 kernel tuned block sizes for %s: bkv_p=%s, bq=%s', keys, bkv_p,
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
    sliding_window,
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
        sliding_window,
    ) = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        sliding_window,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f'q_{q_dtype_name}_kv_{kv_dtype_name}',
        f'q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}',
        f'max_model_len-{next_power_of_2(max_model_len)}-sw-{sliding_window}',
    )


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
    sliding_window,
):
    """Get the simplified key."""
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
        next_power_of_2(num_kv_heads_x2) // 2,
        align_to(head_dim, 64),  # Align to 64 to support both 64 and 128
        next_power_of_2(max_model_len),
        sliding_window,
    )
