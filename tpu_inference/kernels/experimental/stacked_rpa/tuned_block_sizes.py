# Copyright 2026 Google LLC
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
"""Auto-tuned block sizes for the stacked RPA kernel.

Consulted by ``wrapper.ragged_paged_attention`` (which falls back to the
heuristic ``calculate_block_sizes`` when no tuned entry matches).

Lookup key (all components are static at trace time):
    (device_name, page_size, dtype_key, head_key, max_model_len, mode_symbol,
     kv_layout, batched_tokens_bucket, sliding_window_key)
where
    device_name   = jax.devices()[0].device_kind            e.g. "TPU7x"
    page_size     = next_power_of_2(page_size)               e.g. 256
    dtype_key     = "q_{q_dtype}_kv_{kv_dtype}"              e.g. "q_bfloat16_kv_float8_e4m3fn"
    head_key      = "q{Hq}_kv{Hkv}_d{D}" (per-chip heads, pow2)  e.g. "q8_kv1_d64"
    max_model_len = next_power_of_2(max_model_len)           e.g. 262144
    mode_symbol   = RpaCase.symbol                           "d" (decode) | "m" (mixed)
    kv_layout     = ServingConfigs.kv_layout.value            e.g. "seq_along_lane"
    batched_tokens_bucket = bucket_batched_tokens(total_q_tokens)  e.g. 256
    sliding_window_key    = sliding_window or "none"               e.g. 2048 | "none"

Value: (bq_sz, bq_c_sz, bkv_sz, batch_size, n_buffer) -> configs.BlockSizes.

The last three components are optional positional suffixes, so older entries that
omit them still resolve via a truncating fallback ladder (see
``get_tuned_block_sizes``):
  1. full key            (layout + batched-tokens bucket + sliding_window)
  2. drop sliding_window (reuse the window-agnostic "none" entry, same bucket)
  3. drop batched-tokens (layout-only key -- the original seq_along_lane entries)
  4. drop layout         (legacy non-layout key, head_along_sublane only)

The ``batched_tokens_bucket`` is derived from the *static, 128-padded*
``serve_cfgs.total_q_tokens`` (= ``queries.shape[0]``), not the ragged
``cu_q_lens[-1]`` (which is traced and cannot key a dict). The autotuner buckets
the same padded quantity so emitted keys match the reader.
"""

import jax
import jax.numpy as jnp

from tpu_inference.kernels.experimental.stacked_rpa import configs


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= n (n>=1)."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


# Batched-token buckets used as a lookup-key dimension. A server step's padded
# query-token count is snapped *up* to one of these so a spec-decode step
# (small total_q) selects a different tuned tile than a large chunked-prefill
# step on the same shape.
_BATCHED_TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)


def bucket_batched_tokens(total_q_tokens: int) -> int:
    """Snap a (static, padded) total query-token count up to a tuning bucket.

    Anything above the largest bucket clamps to it. The reader and the autotune
    writer must bucket the *same* padded quantity so emitted keys match.
    """
    n = max(int(total_q_tokens), 1)
    for b in _BATCHED_TOKEN_BUCKETS:
        if n <= b:
            return b
    return _BATCHED_TOKEN_BUCKETS[-1]


def get_device_name() -> str:
    """Normalized device kind, e.g. 'TPU7x' / 'TPU v6e'."""
    return jax.devices()[0].device_kind


def compute_head_key(num_q_heads: int, num_kv_heads: int,
                     head_dim: int) -> str:
    """Per-chip head config key (head counts rounded up to powers of two)."""
    return (f"q{next_power_of_2(num_q_heads)}"
            f"_kv{next_power_of_2(num_kv_heads)}"
            f"_d{head_dim}")


def compute_dtype_key(q_dtype, kv_dtype) -> str:
    return f"q_{jnp.dtype(q_dtype).name}_kv_{jnp.dtype(kv_dtype).name}"


def make_lookup_key(
    *,
    device_name: str,
    page_size: int,
    q_dtype,
    kv_dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    mode_symbol: str,
    kv_layout: configs.KVLayout | str | None = None,
    batched_tokens_bucket: int | None = None,
    sliding_window: int | None = None,
) -> tuple:
    """Build the tuned-table key shared by the writer and the reader.

    ``batched_tokens_bucket`` and ``sliding_window`` are optional positional
    suffixes appended only after ``kv_layout``. ``sliding_window`` is appended
    only when ``batched_tokens_bucket`` is given (the two new dims always travel
    together in a full key), so truncating the tuple yields the fallback ladder.
    """
    key = (
        device_name,
        next_power_of_2(page_size),
        compute_dtype_key(q_dtype, kv_dtype),
        compute_head_key(num_q_heads, num_kv_heads, head_dim),
        next_power_of_2(max_model_len),
        mode_symbol,
    )
    if kv_layout is None:
        return key
    layout_key = (kv_layout.value if isinstance(kv_layout, configs.KVLayout)
                  else str(kv_layout))
    key = (*key, layout_key)
    if batched_tokens_bucket is None:
        return key
    sw_key = "none" if sliding_window is None else int(sliding_window)
    return (*key, int(batched_tokens_bucket), sw_key)


# Tuned entries may be added for validated, model-agnostic workloads.
TUNED_BLOCK_SIZES: dict[tuple, tuple[int, int, int, int, int]] = {}


def get_tuned_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    max_model_len: int,
    mode: configs.RpaCase,
) -> configs.BlockSizes | None:
    """Return tuned ``BlockSizes`` for this config, or ``None`` if absent.

    All key components derive from static (trace-time) values so this is safe
    to call from inside the jitted wrapper.
    """
    base = dict(
        device_name=get_device_name(),
        page_size=serve_cfgs.page_size,
        q_dtype=serve_cfgs.dtype_q,
        kv_dtype=serve_cfgs.dtype_kv,
        num_q_heads=model_cfgs.num_q_heads,
        num_kv_heads=model_cfgs.num_kv_heads,
        head_dim=model_cfgs.head_dim,
        max_model_len=max_model_len,
        mode_symbol=mode.symbol,
    )
    bucket = bucket_batched_tokens(serve_cfgs.total_q_tokens)
    sw = model_cfgs.sliding_window

    entry = None
    if serve_cfgs.kv_layout is not None:
        # Level 1: full key (layout + batched-tokens bucket + sliding_window).
        entry = TUNED_BLOCK_SIZES.get(
            make_lookup_key(
                **base,
                kv_layout=serve_cfgs.kv_layout,
                batched_tokens_bucket=bucket,
                sliding_window=sw,
            ))
        # Level 2: a windowed step may reuse the window-agnostic ("none") entry
        # at the same bucket (a no-window tile is a safe superset).
        if entry is None and sw is not None:
            entry = TUNED_BLOCK_SIZES.get(
                make_lookup_key(
                    **base,
                    kv_layout=serve_cfgs.kv_layout,
                    batched_tokens_bucket=bucket,
                    sliding_window=None,
                ))
        # Level 3: drop the batched-tokens bucket -> layout-only key. This is
        # where the original seq_along_lane entries (no bucket/sw) resolve.
        if entry is None:
            entry = TUNED_BLOCK_SIZES.get(
                make_lookup_key(**base, kv_layout=serve_cfgs.kv_layout))
    if entry is None:
        return None
    bq_sz, bq_c_sz, bkv_sz, batch_size, n_buffer = entry
    block_sizes = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_c_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=n_buffer,
    )
    if mode == configs.RpaCase.MIXED:
        block_sizes = block_sizes.cap_bq_to_total_q(serve_cfgs.total_q_tokens)
    return block_sizes
