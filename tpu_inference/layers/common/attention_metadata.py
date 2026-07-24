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
        "mamba_slot_read_offsets",
        "mamba_request_distribution",
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
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from
    # `base_slot + offset` (the checkpoint of the last accepted token) and
    # writes fresh checkpoints starting at `base_slot`. Indexed by physical
    # slot (not batch position) so the value survives requests being
    # rescheduled, condensed, or skipped for several steps. Updated on
    # device after each rejection-sampling step; None unless the model has
    # mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | None = None
    # (3 * dp_size,) int32 — GDN-specific request distribution, same
    # 3-counters-per-rank format as `request_distribution` but with the
    # first segment covering all *windowed* sequences (plain decodes and
    # speculative verify windows of up to num_spec + 1 tokens) instead of
    # only 1-token decodes. The persistent batch is ordered
    # [decode][verify][prefill/mixed] so both segmentations hold at once:
    # ragged paged attention keeps its 1-token decode front segment while
    # the GDN kernel runs its windowed mode over the first two groups.
    # None unless the model has mamba layers and spec decoding is enabled.
    mamba_request_distribution: jax.Array | None = None

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

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1