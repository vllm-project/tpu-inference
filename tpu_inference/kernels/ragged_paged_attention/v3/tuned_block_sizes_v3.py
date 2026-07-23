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
"""Tuned 4-tuple block sizes for the RPA v3 (head_dim>=128) kernel.

This table is the v3-native complement to the legacy 2-tuple
``tuned_block_sizes.TUNED_BLOCK_SIZES`` (which stores ``(num_kv_pages_per_blk,
num_queries_per_blk)`` and is consumed by the v2 / hd64 kernels). The main v3
kernel does not size blocks in *pages*; it sizes four independent block dims:

    (bq_sz, bkv_sz, bq_csz, bkv_csz)

where ``*_sz`` is the *fetch* (DMA / prefetch) block and ``*_csz`` is the
*compute* block. Decoupling fetch from compute is exactly what lets the decode
case use a deep KV prefetch (double-buffering) with a small compute block --
the lever the ``get_default_block_sizes`` heuristic cannot express because it
sets ``bkv_sz == bkv_csz`` for the v7 DECODE case.

Lookup precedence in ``kernel.py``:

    caller override (d/p/m_block_sizes)  ->  THIS table  ->  heuristic default

A miss falls through to ``get_default_block_sizes`` (returns ``None`` here), so
adding or omitting an entry can never regress an untuned shape: it only ever
*replaces* the heuristic for a shape we have measured on real silicon.

Key format mirrors ``tuned_block_sizes.get_lookup_keys`` so the two tables stay
searchable with the same helper. The leaf value is the 4-tuple above (in
*tokens*, aligned to ``page_size`` for the kv dims by the caller's validator).
"""

from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import (
    get_lookup_keys)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# key (same nesting as the legacy table):
#   device_name
#     page_size
#       q_{q_dtype}_kv_{kv_dtype}
#         q_head-{Hq}_kv_head-{Hkv}_head-{D}
#           max_model_len-{L}-sw-{sw}
#             case ("decode" | "prefill" | "mixed")
# value: (bq_sz, bkv_sz, bq_csz, bkv_csz)   -- fetch/compute split, in tokens.
#
# Provenance for every entry lives in the comment next to it: hardware, model,
# mesh, and the measured throughput delta vs. the heuristic default. Entries
# without an on-silicon measurement do NOT belong here -- a miss is free.
TUNED_BLOCK_SIZES_V3 = {
    'TPU v7': {
        128: {
            'q_bfloat16_kv_bfloat16': {
                # Qwen3-0.6B, DP16xTP4 -> per-shard q_head=4, kv_head=2, D=128.
                'q_head-4_kv_head-2_head-128': {
                    # tpu7x 4x4x4, DAPO GRPO decode (ignore_eos), mds=64/128.
                    # The v7 DECODE heuristic sets bkv_sz==bkv_csz==16384 (it
                    # clamps min_bkv_sz_to_peak=16384 to max_kv), which starves
                    # Pallas double-buffering. A deep 16384 fetch with a 4096
                    # compute block gives fetch/compute overlap:
                    #   64,929 -> 96,292 tok/s (+48%), reproduced 4x (+/-0.5%).
                    # Inverted-U sweep: 2048:85.3k 4096:90.3k 8192(split):95.4k
                    #   16384(split):96.3k(PEAK) 32768:VMEM-OOM.
                    #
                    # Keyed at the 16384 max_model_len bucket: the entry needs
                    # max_kv >= bkv_sz(16384), and the heuristic default of
                    # 16384 already implies max_kv >= 16384 for this run. Larger
                    # context buckets (24576/32768) are expected to behave the
                    # same (decode is bkv-fetch-bound, not context-bound) but
                    # have not been separately measured -- they miss and fall
                    # back to the heuristic until measured.
                    'max_model_len-16384-sw-None': {
                        'decode': (1, 16384, 1, 4096),
                    },
                },
            },
        },
    },
}


def get_tuned_block_sizes_v3(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    case_name,
    sliding_window=None,
):
    """Look up the tuned 4-tuple ``(bq_sz, bkv_sz, bq_csz, bkv_csz)``.

    Returns ``None`` on any miss so the caller falls back to the heuristic
    ``get_default_block_sizes`` -- an untuned shape is never regressed.

    Args:
      case_name: one of ``"decode"``, ``"prefill"``, ``"mixed"``.
      (all other args mirror ``get_tuned_block_sizes`` / ``get_lookup_keys``.)
    """
    device, page_size_key, dtypes, head_dims, extra = get_lookup_keys(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
        sliding_window,
    )
    try:
        value = TUNED_BLOCK_SIZES_V3[device][page_size_key][dtypes][head_dims][
            extra][case_name]
    except KeyError:
        return None

    logger.info_once(
        'RPA v3 kernel tuned 4-tuple block sizes for %s case=%s: %s',
        (device, page_size_key, dtypes, head_dims, extra), case_name, value)
    return value
