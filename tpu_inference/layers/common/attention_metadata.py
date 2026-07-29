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
        "has_initial_state",
        "pcp_q_pos_offsets",
        "pcp_kv_cache_lens",
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

    # (max_num_seqs,) int32, 0/1 — 1 when the request occupying this
    # persistent-batch position already has recurrent/conv state in its mamba
    # slot from an earlier step, i.e. this step must resume from that state
    # rather than zero-initialize it. Runner-computed and authoritative;
    # linear-attention ops must consume it and must NOT re-derive it.
    #
    # Why published rather than derived: the natural derivation
    # `(seq_lens - query_lens) > 0` needs `query_lens`, and
    # `query_start_loc` is laid out as a per-DP-rank concatenation of
    # `max_num_reqs_per_dp_rank + 1` entries — total `max_num_seqs +
    # dp_size`, not `max_num_seqs + 1`. Differencing it globally yields
    # `max_num_seqs + dp_size - 1` values (one bogus entry per rank seam)
    # instead of `max_num_seqs`, so the derivation is only valid at
    # dp_size == 1, or inside a `shard_map` whose in_spec slices both arrays
    # per rank. Ops that run outside such a shard_map (the JAX-native model
    # path) silently get the wrong shape/values under attention DP.
    #
    # int32 rather than bool because the runner packs it through the int32
    # `DeviceBuffer` blob; consumers should treat any nonzero as True.
    # A boolean flag rather than a `num_computed_tokens` count: with
    # speculative decoding the host-side count is optimistic (it assumes every
    # proposed token was accepted, corrected on device afterwards), while the
    # "> 0" predicate is exact — a request cannot have had tokens rejected
    # before it has been prefilled at all.
    # None for models without mamba layers.
    has_initial_state: jax.Array | None = None

    # (max_num_seqs, ) int32 — PCP only. For a single request, it is [rank*C, (2*pcp-1-rank)*C].
    pcp_q_pos_offsets: jax.Array | None = None

    # (max_num_seqs,) int32 — PCP only: [P, P, 0...] where P = num_computed.
    # The kernel derives the new KV length as
    # `seq_lens - pcp_kv_cache_lens`, so only real tokens are attended/written.
    pcp_kv_cache_lens: jax.Array | None = None

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "has_initial_state",
    ],
    meta_fields=["padded_num_reqs"],
)
@dataclass
class SharedAttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
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

    # (max_num_seqs,) int32, 0/1 — see `AttentionMetadata.has_initial_state`.
    has_initial_state: jax.Array | None = None

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1