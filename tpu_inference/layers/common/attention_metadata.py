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
import numpy as np
from vllm.utils.math_utils import cdiv


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "query_start_loc", "kv_cache_lens", "q_pos_offsets", "kv_new_starts",
        "kv_token_order"
    ],
    meta_fields=["has_cached_kv", "num_reqs"],
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
    # (max_num_seqs,) int32 — base offset of each fused seq's current-KV block
    # inside the all-gathered new-KV buffer (zeros for a single request).
    # Replicated (P()).
    kv_new_starts: jax.Array
    # (padded_num_tokens,) int32 — permutation taking the all-gathered current
    # K/V from rank order to request-major token order.  Replicated (P()).
    kv_token_order: jax.Array
    # STATIC (meta field): whether any request in the batch has cached KV.
    # False elides the cache phase entirely.  REQUIRED: a default would
    # silently elide the cache phase for any caller that forgot to set it.
    has_cached_kv: bool
    # STATIC (meta field): number of requests fused into this launch, padded
    # to `runner.pcp_num_reqs_paddings`.
    num_reqs: int = 1


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

    # PCP only. Number of kv pages occupied by the current request.
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


class GroupedAttentionMetadata(dict):
    """``{layer_name: AttentionMetadata}`` that flattens once per KV-cache group.

    Every layer in a KV-cache group shares one ``block_tables`` array,  it
    flattens to the unique per-group entries. After unflattening inside the
    trace, every layer of a group holds the *same* ``AttentionMetadata`` object,
    so anything derived from the block tables gets computed once per group instead
    of once per layer.
    """

    def __init__(
        self,
        groups: "tuple[AttentionMetadata, ...]",
        layer_names_per_group: "tuple[tuple[str, ...], ...]",
    ):
        self.groups = tuple(groups)
        self.layer_names_per_group = tuple(
            tuple(names) for names in layer_names_per_group)
        assert len(self.groups) == len(self.layer_names_per_group)
        super().__init__({
            name: self.groups[gid]
            for gid, names in enumerate(self.layer_names_per_group)
            for name in names
        })


jax.tree_util.register_pytree_node(
    GroupedAttentionMetadata,
    lambda m: (m.groups, m.layer_names_per_group),
    lambda layer_names_per_group, groups: GroupedAttentionMetadata(
        groups, layer_names_per_group),
)


def pcp_token_layout(num_scheduled_tokens: list[int],
                     pcp_size: int) -> tuple[list[int], list[int], int]:
    """Per-request zigzag chunking for multi-request PCP.

    Each request is split into its own 2*pcp_size chunks, and its head+tail
    pair occupies a fixed-width slot in every rank's region of the token
    buffer. Returns (C, off, S):

      C[i]   chunk size of request i, ceil(n_i / 2P)
      off[i] start of request i's slot within one rank's region
      S      live tokens per rank; the global buffer needs pcp_size * S rows
    """
    two_p = 2 * pcp_size
    off, acc, C = [], 0, []
    for n in num_scheduled_tokens:
        c = cdiv(n, two_p)
        C.append(c)
        off.append(acc)
        acc += 2 * c
    return C, off, acc


def pcp_seq_arrays(chunk: list[int], off: list[int], pcp_size: int,
                   n_off: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-seq attention metadata for a `pcp_token_layout` result.

    Request i is presented to the kernel as seqs 2i (head chunk) and 2i+1
    (tail chunk). Returns int32 arrays (cu_row, q_pos, kv_new_starts):

      cu_row[n_off + 1]          cumulative query rows per seq. Rank-invariant
                                 (both halves are full length). Entries past
                                 2R repeat the last value: they are never
                                 iterated, and a repeat inside the iterated
                                 range would be a zero-length seq, which
                                 hangs the kernel.
      q_pos[pcp_size, n_off]     each seq's query position offset on each
                                 rank: rank r holds chunk r (head) and chunk
                                 2P-1-r (tail) of every request.
      kv_new_starts[n_off]       base of request i's block in the
                                 request-major current K/V buffer,
                                 2P * sum(chunk[:i]) == pcp_size * off[i].
    """
    n_reqs = len(chunk)
    assert 2 * n_reqs <= n_off, (n_reqs, n_off)
    c = np.asarray(chunk, np.int64)
    o = np.asarray(off, np.int64)
    ranks = np.arange(pcp_size)
    cu_row = np.zeros(n_off + 1, np.int32)
    cu_row[1:2 * n_reqs + 1:2] = o + c
    cu_row[2:2 * n_reqs + 2:2] = o + 2 * c
    cu_row[2 * n_reqs + 1:] = cu_row[2 * n_reqs]
    q_pos = np.zeros((pcp_size, n_off), np.int32)
    q_pos[:, 0:2 * n_reqs:2] = ranks[:, None] * c
    q_pos[:, 1:2 * n_reqs:2] = (2 * pcp_size - 1 - ranks)[:, None] * c
    kv_new_starts = np.zeros(n_off, np.int32)
    kv_new_starts[:2 * n_reqs] = np.repeat(pcp_size * o, 2)
    return cu_row, q_pos, kv_new_starts
