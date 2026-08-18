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
import math
from dataclasses import dataclass

import jax
from vllm.utils.math_utils import cdiv


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["query_start_loc", "kv_cache_lens", "q_pos_offsets"],
    meta_fields=["cache_pages"],
)
@dataclass
class PCPMetadata:
    """Prefill Context Parallelism metadata, passed via AttentionMetadata.pcp."""
    # (pcp_size, max_num_reqs+1) int32 — per-rank cumulative query lengths.
    # Sharded as P('pcp', None); each rank slice is its own cu_q_lens.
    query_start_loc: jax.Array
    # (max_num_reqs,) int32 — num_computed tokens per virtual seq (cache
    # boundary). Replicated (P()). The kernel derives new KV length as
    # seq_lens - kv_cache_lens so only real tokens are attended/written.
    kv_cache_lens: jax.Array
    # (pcp_size, max_num_reqs) int32 — per-rank, per-seq Q position offsets.
    # Sharded as P('pcp', None).
    q_pos_offsets: jax.Array
    # STATIC (meta field): a rung of `pcp_cache_page_buckets` giving an UPPER
    # bound on the KV pages this request's cached tokens occupy, used to bound
    # the gather-KV cache all-gather.  0 means nothing is cached, in which case
    # the cache phase is elided entirely.  REQUIRED: a default would silently
    # elide the cache phase for any caller that forgot to set it.
    cache_pages: int


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "pcp",
    ],
    meta_fields=["padded_num_reqs", "pcp_cache_pages"],
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

    # PCP-specific metadata. None when not running prefill context parallelism.
    pcp: PCPMetadata | None = None

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1

    # PCP gather-KV only. Number of kv pages occupied by the current request.
    pcp_cache_pages: int | None = None


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


PCP_CACHE_PAGE_BUCKET_COUNT = 5


def pcp_cache_page_buckets(max_num_blocks_per_req: int) -> list[int]:
    """The buckets for the `pcp_cache_pages` value, including 0.
    """
    buckets = {0, max_num_blocks_per_req}
    n = PCP_CACHE_PAGE_BUCKET_COUNT - len(buckets)
    if n > 0 and max_num_blocks_per_req > 1:
        step = math.log(max_num_blocks_per_req) / (n + 1)
        for i in range(1, n + 1):
            v = 1 << max(0, round(math.exp(step * i)).bit_length() - 1)
            buckets.add(min(max(v, 1), max_num_blocks_per_req))
    return sorted(buckets)


def round_up_pcp_cache_pages(num_computed_tokens: int, block_size: int,
                             max_num_blocks_per_req: int) -> int:
    """Round a request's number of kv pages up to the nearest bucket.
    """
    if num_computed_tokens <= 0:
        return 0
    live_pages = cdiv(num_computed_tokens, block_size)
    for b in pcp_cache_page_buckets(max_num_blocks_per_req):
        if b >= live_pages:
            return b
    return max_num_blocks_per_req
