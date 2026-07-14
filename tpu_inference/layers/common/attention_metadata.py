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

import functools
from dataclasses import dataclass

import jax


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "pcp_q_pos_offsets",
        "pcp_cu_q_lens",
        "pcp_kv_cache_lens",
        "pcp_kv_lens",
        "pcp_distribution",
    ],
    meta_fields=["padded_num_reqs"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    # None for pooling models that using no KV cache
    block_tables: jax.Array | None = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 — physical slot id (∈ [0, _mamba_num_blocks))
    # in the mamba kv-cache for the request currently in each persistent-
    # batch position. Used by mamba/GDN ops to read/write recurrent state
    # without going through `block_tables`, since the mamba pool is
    # smaller than the attention pool under compact-mamba sizing.
    # None for models without mamba layers; pure-mamba models would also
    # use this field, only hybrid models exercise it today.
    mamba_state_indices: jax.Array | None = None

    # (pcp, max_num_seqs) int32 — PCP only, sharded on `pcp`. Per rank, the
    # within-current start of its head and tail chunk: [rank*C, (2*pcp-1-rank)*C].
    # The two chunks are fused into ONE ragged current-phase launch as two
    # "sequences", so these are that launch's per-seq q_pos_offsets.
    pcp_q_pos_offsets: jax.Array | None = None

    # (pcp, max_num_seqs + 1) int32 — PCP only, sharded on `pcp`. Per rank, the
    # fused current-phase cu_q_lens: [0, C, C + tail_real]. The head chunk is
    # always fully real (C); the tail is clamped so padding past the real current
    # length is excluded (may be 0 when the tail is wholly padding).
    pcp_cu_q_lens: jax.Array | None = None

    # (3,) int32 — PCP only: [0, 0, 2]. The fused current phase presents the
    # head+tail chunks as TWO prefill sequences. (The cache phase is a single
    # sequence and uses `request_distribution`.)
    pcp_distribution: jax.Array | None = None

    # (max_num_seqs,) int32 — PCP only: [T, T, 0...] where T = num_computed +
    # num_current. Both fused current-phase seqs are the SAME request, so both
    # carry the request's real total kv length.
    pcp_kv_lens: jax.Array | None = None

    pcp_kv_cache_lens: jax.Array | None = None

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1
