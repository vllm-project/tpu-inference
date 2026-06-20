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
"""Heuristic family tagger for verified diffs.

Maps a unified diff onto the tpu-inference-perf family letters (A–J).
Used to auto-populate ``PositiveExample.family_tags`` so cross-kernel
knowledge transfer (``ExamplePool.for_family``) works without manual
classification of every win.

The mapping is intentionally conservative — false positives in tagging
would pollute cross-kernel suggestions. Each rule looks for concrete
syntactic signals (specific function names, dtype mentions, identifiable
patterns from the perf-skill casebook).
"""

from __future__ import annotations

import re

# Family signatures. Each: (family_letter, list of regex patterns that
# indicate this family).
_FAMILY_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    # A — memory hierarchy & fusion
    ("A", [
        re.compile(r"donate_argnames\b"),
        re.compile(r"input_output_aliases\b"),
        re.compile(r"\.at\[.*\]\.set\("),
        re.compile(r"emit_pipeline\b"),
    ]),
    # B — pipelining & overlap
    ("B", [
        re.compile(r"async_copy\b"),
        re.compile(r"DMA_BUFFERS|n_buffer|num_buffer", re.IGNORECASE),
        re.compile(r"prev_p|prev_alpha|prev_q_slice"),
    ]),
    # C — MXU vs memory-bound
    ("C", [
        re.compile(r"one_hot\b"),
        re.compile(r"bitcast_convert_type\b"),
        re.compile(r"ragged_gather|ragged_scatter"),
    ]),
    # D — regime specialization
    ("D", [
        re.compile(r"pallas_call\b"),
        re.compile(r"static_q_len|static_argname"),
        re.compile(r"is_decode_only|decode_path|prefill_path|RpaCase"),
    ]),
    # E — tiling, layout, block-size
    ("E", [
        re.compile(r"BLOCK_M|BLOCK_N|BLOCK_K"),
        re.compile(r"bkv_p|bkv_sz|bq_sz|bq_csz|block_sizes"),
        re.compile(r"with_layout_constraint|major_to_minor"),
        re.compile(r"VMEM|vmem_capacity|VMEM_LIMIT", re.IGNORECASE),
    ]),
    # F — collectives & sharding
    ("F", [
        re.compile(r"psum\b"),
        re.compile(r"psum_scatter|all_gather|reduce_scatter|all_to_all"),
        re.compile(r"PartitionSpec|with_sharding_constraint|shard_map"),
        re.compile(r"replication_factor|tile_kv"),
    ]),
    # G — DP attention & scheduling
    ("G", [
        re.compile(r"dp_attention|attn_dp|ATTN_DATA"),
        re.compile(r"_pending_new_requests|DP_SCHED_BATCH_PREFILL"),
        re.compile(r"local_slots|per_rank_pool"),
    ]),
    # H — quantization & dtype-for-bandwidth
    ("H", [
        re.compile(r"\bf?p?[48]_e[245]m[12]fn\b"),
        re.compile(r"out_dtype\b"),
        re.compile(r"\.astype\(\s*(?:out_dtype|jnp\.bfloat16|"
                   r"jnp\.float32|jnp\.float8)"),
        re.compile(r"preferred_element_type"),
        re.compile(r"dtype\s*=\s*(?:out_dtype|jnp\.bfloat16|jnp\.float32)"),
        re.compile(r"nvfp4|NVFP4|float4_e2m1"),
        re.compile(r"dequant"),
    ]),
    # I — host & dispatch overhead
    ("I", [
        re.compile(r"tree_unflatten|tree_flatten|tree_structure"),
        re.compile(r"nnx\.merge|nnx\.split"),
        re.compile(r"state_leaves|_treedef"),
        re.compile(r"jax\.clear_caches"),
    ]),
    # J — do less work / algebraic identities
    ("J", [
        re.compile(r"end_bkv_idx|causal_skip"),
        re.compile(r"size-gated|size_gated|vmem_capacity\s*\*\s*0\.[0-9]+"),
        re.compile(r"zero_initialize\s*=\s*False"),
        re.compile(r"distributive law|distributive_law"),
    ]),
]


def tag_diff(diff_text: str) -> list[str]:
    """Return the family letters this diff most likely instantiates.

    Letters are returned in alphabetical order; duplicates removed. An
    empty list means no signal was strong enough to tag.
    """
    found: set[str] = set()
    for fam, patterns in _FAMILY_PATTERNS:
        for pat in patterns:
            if pat.search(diff_text):
                found.add(fam)
                break
    return sorted(found)


def describe_tags(tags: list[str]) -> str:
    """Render a one-line family description from a tag list."""
    names = {
        "A": "memory hierarchy/fusion",
        "B": "pipelining/overlap",
        "C": "MXU-vs-memory-bound",
        "D": "regime specialization",
        "E": "tiling/layout/block-size",
        "F": "collectives/sharding",
        "G": "DP attention/scheduling",
        "H": "quantization/dtype-for-bandwidth",
        "I": "host/dispatch overhead",
        "J": "work elimination/algebraic identity",
    }
    parts = []
    for t in tags:
        parts.append(f"{t}={names.get(t, '?')}")
    return "; ".join(parts) if parts else "(no family detected)"
