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
"""Host-side preprocessing for prefill context parallelism (PCP).

PCP splits each prefill request into 2 * pcp_size zigzag chunks; rank r
holds chunks r (head) and 2P-1-r (tail) of every request, and the kernel
sees request i as seqs 2i and 2i+1. The layout helpers here are pure numpy;
`PCPPreprocessor.prepare_inputs` applies them to the runner's host buffers
once per step.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.utils.math_utils import cdiv

from tpu_inference.layers.common.attention_metadata import PCPMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner import utils as runner_utils
from tpu_inference.utils import device_array


def pcp_token_layout(num_scheduled_tokens: list[int],
                     pcp_size: int,
                     align: int = 1) -> tuple[list[int], list[int], int]:
    """Returns (chunk, off, s_live): per-request chunk size ceil(n_i / 2P)
    rounded up to a multiple of `align`, the start of each request's
    head+tail slot within a rank's region, and the live rows per rank.

    `align` is the KV-cache page size in production: page-multiple chunks
    keep every token-order page of the new-KV buffer contiguous in the
    rank-order all_gather result, which is what lets the kernel unshuffle
    K/V through the `pcp_page_order` map during its mandatory HBM->VMEM
    copy instead of a separate gather pass. The rounding costs at most
    2P*(align-1) padding rows per request."""
    two_p = 2 * pcp_size
    off, acc, C = [], 0, []
    for n in num_scheduled_tokens:
        c = cdiv(cdiv(n, two_p), align) * align
        C.append(c)
        off.append(acc)
        acc += 2 * c
    return C, off, acc


def pcp_buffer_tokens(num_scheduled_tokens: list[int],
                      pcp_size: int,
                      align: int = 1) -> int:
    """Rows the token buffer must hold for this batch (can exceed the raw
    token count, since every chunk rounds up independently)."""
    _, _, s_live = pcp_token_layout(num_scheduled_tokens, pcp_size, align)
    return pcp_size * s_live


def pcp_max_buffer_tokens(max_num_batched_tokens: int, max_num_seqs: int,
                          pcp_size: int, align: int = 1) -> int:
    """Upper bound of `pcp_buffer_tokens` over any batch the scheduler
    admits: each request rounds up by less than 2P * align rows."""
    return max_num_batched_tokens + 2 * pcp_size * align * max_num_seqs


def pcp_batch_layout(num_scheduled_tokens: list[int],
                     t_pad: int,
                     pcp_size: int,
                     align: int = 1) -> tuple[list[int], list[int]]:
    """Chunk sizes and slot offsets of a batch inside a `t_pad`-row buffer.

    One layout for any request count: a single request is simply R = 1 of
    the general page-aligned zigzag pack.
    """
    chunk, off, s_live = pcp_token_layout(num_scheduled_tokens, pcp_size,
                                          align)
    assert t_pad % pcp_size == 0 and t_pad >= pcp_size * s_live, (
        f"PCP token bucket {t_pad} cannot hold {pcp_size * s_live} tokens "
        f"({len(num_scheduled_tokens)} reqs, chunks {chunk})")
    return chunk, off


def pcp_seq_arrays(chunk: list[int], off: list[int], pcp_size: int,
                   n_slots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-seq metadata for `n_slots` seq slots: cu_row[n_slots + 1] cumulative
    query rows (rank-invariant), q_pos[pcp_size, n_slots] per-rank query
    position offsets, kv_new_starts[n_slots] base of each request's block in
    the request-major current-K/V buffer."""
    n_reqs = len(chunk)
    assert 2 * n_reqs <= n_slots, (n_reqs, n_slots)
    c = np.asarray(chunk, np.int64)
    o = np.asarray(off, np.int64)
    ranks = np.arange(pcp_size)
    cu_row = np.zeros(n_slots + 1, np.int32)
    cu_row[1:2 * n_reqs + 1:2] = o + c
    cu_row[2:2 * n_reqs + 2:2] = o + 2 * c
    cu_row[2 * n_reqs + 1:] = cu_row[2 * n_reqs]
    q_pos = np.zeros((pcp_size, n_slots), np.int32)
    q_pos[:, 0:2 * n_reqs:2] = ranks[:, None] * c
    q_pos[:, 1:2 * n_reqs:2] = (2 * pcp_size - 1 - ranks)[:, None] * c
    kv_new_starts = np.zeros(n_slots, np.int32)
    kv_new_starts[:2 * n_reqs] = np.repeat(pcp_size * o, 2)
    return cu_row, q_pos, kv_new_starts


def pcp_page_order(chunk: list[int], off: list[int], pcp_size: int,
                   s_pad: int, t_pad: int, page_size: int) -> np.ndarray:
    """Per-page unshuffle map for the all-gathered new-KV buffer.

    Entry j is the page of the rank-order buffer holding token-order page j
    (request-major, the coordinate space `kv_new_starts` indexes). The
    kernel's current-phase fetch walks this map one page-sized DMA at a
    time, exactly like the paged cache side of a mixed fetch, so the
    rank-order buffer never needs a separate gather pass.

    Valid because every chunk is a whole number of pages (`pcp_token_layout`
    with align=page_size): a token-order page then falls wholly inside one
    chunk and is contiguous in the rank-order buffer. Uncovered entries
    (dead region tails) stay 0 and are never fetched.
    """
    two_p = 2 * pcp_size
    assert s_pad % page_size == 0 and t_pad == pcp_size * s_pad, (s_pad,
                                                                  t_pad)
    order = np.zeros(t_pad // page_size, np.int32)
    ranks = np.arange(pcp_size)
    for c_i, off_i in zip(chunk, off):
        assert c_i % page_size == 0 and off_i % page_size == 0, (c_i, off_i)
        cp = c_i // page_size  # pages per chunk
        kv_base_p = pcp_size * off_i // page_size
        j = np.arange(cp)
        for h in (0, 1):
            chunk_idx = ranks if h == 0 else two_p - 1 - ranks
            buf_p = (ranks[:, None] * (s_pad // page_size) +
                     (off_i + h * c_i) // page_size + j[None, :])
            tok_p = chunk_idx[:, None] * cp + j[None, :]
            order[kv_base_p + tok_p.ravel()] = buf_p.ravel()
    return order


def pcp_token_permutation(num_scheduled_tokens: list[int], chunk: list[int],
                          off: list[int], t_pad: int,
                          pcp_size: int) -> np.ndarray:
    """Returns perm: perm[g] is the natural-order source of rank-order
    slot g (-1 for padding).  The K/V side needs no per-token map -- the
    kernel unshuffles the rank-order buffer through `pcp_page_order`."""
    two_p = 2 * pcp_size
    s_pad = t_pad // pcp_size
    src_off = np.cumsum([0] + list(num_scheduled_tokens))[:-1]
    perm = np.full(t_pad, -1, np.int64)
    ranks = np.arange(pcp_size)
    for i, n_i in enumerate(num_scheduled_tokens):
        c_i = chunk[i]
        j = np.arange(c_i)
        for h in (0, 1):
            chunk_idx = ranks if h == 0 else two_p - 1 - ranks
            dst = (ranks[:, None] * s_pad + off[i] + h * c_i + j[None, :])
            tok = chunk_idx[:, None] * c_i + j[None, :]
            real = tok < n_i
            perm[dst[real]] = src_off[i] + tok[real]
    return perm


class PCPPreprocessor:
    """Per-step host preprocessing for a PCP-enabled runner."""

    def __init__(self, pcp_size: int, mesh: Mesh,
                 num_reqs_paddings: list[int], page_size: int):
        assert pcp_size > 1, pcp_size
        self.pcp_size = pcp_size
        self.mesh = mesh
        self.num_reqs_paddings = num_reqs_paddings
        # KV-cache page size as the KERNEL sees it (cache_config.block_size);
        # chunks are rounded to page multiples so the kernel can unshuffle
        # the all-gathered current K/V through `pcp_page_order`.
        self.page_size = page_size
        self._pcp_spec = NamedSharding(
            mesh, PartitionSpec(ShardingAxisName.PREFILL_CONTEXT, None))
        self._repl_spec = NamedSharding(mesh, PartitionSpec())

    def metadata_to_device(self, cu_row: np.ndarray, q_pos: np.ndarray,
                           kv_cache_lens: np.ndarray,
                           kv_new_starts: np.ndarray,
                           kv_page_order: np.ndarray, *, has_cached_kv: bool,
                           num_reqs: int) -> PCPMetadata:
        """Place host arrays as a `PCPMetadata`; also used by the compilation
        manager so precompiled and runtime metadata share one sharding."""
        query_start_loc, q_pos_offsets = device_array(
            self.mesh, (np.tile(cu_row, (self.pcp_size, 1)), q_pos),
            sharding=self._pcp_spec)
        kv_cache_lens, kv_new_starts, kv_page_order = device_array(
            self.mesh, (kv_cache_lens, kv_new_starts, kv_page_order),
            sharding=self._repl_spec)
        return PCPMetadata(
            query_start_loc=query_start_loc,
            kv_cache_lens=kv_cache_lens,
            q_pos_offsets=q_pos_offsets,
            kv_new_starts=kv_new_starts,
            kv_page_order=kv_page_order,
            has_cached_kv=has_cached_kv,
            num_reqs=num_reqs,
        )

    def prepare_inputs(
        self,
        num_scheduled_tokens: list[int],
        num_computed_tokens: list[int],
        t_pad: int,
        positions: np.ndarray,
        input_ids: np.ndarray,
        seq_lens: np.ndarray,
        request_distribution: np.ndarray,
        logits_indices: np.ndarray,
    ) -> PCPMetadata:
        """Permute `positions`/`input_ids` into rank order in place, overwrite
        `seq_lens`, `request_distribution` and `logits_indices` with their
        PCP values, and return the attention metadata."""
        pcp_size = self.pcp_size
        counts = num_scheduled_tokens
        computed = num_computed_tokens
        for n_i, l_i in zip(counts, computed):
            if n_i == 1 and l_i > 0:
                raise NotImplementedError(
                    "PCP supports prefill-only batches; got a decode "
                    f"request (num_scheduled=1, num_computed={l_i}).")

        num_pcp_reqs = len(counts)
        chunk, off = pcp_batch_layout(counts, t_pad, pcp_size,
                                      align=self.page_size)
        perm = pcp_token_permutation(counts, chunk, off, t_pad, pcp_size)
        # Per-page unshuffle map for the all-gathered current K/V; the kernel
        # walks it during its KV fetch, so the rank-order buffer needs no
        # gather pass.  One layout for any request count: a single request is
        # simply R = 1 here.
        kv_page_order = pcp_page_order(chunk, off, pcp_size,
                                       t_pad // pcp_size, t_pad,
                                       self.page_size)
        valid = perm >= 0
        src_idx = perm[valid]
        for buf in (positions, input_ids):
            src = buf.copy()
            buf[:] = 0
            buf[valid] = src[src_idx]

        n_seqs = 2 * num_pcp_reqs
        n_slots = len(seq_lens)
        assert n_seqs <= n_slots, (
            f"PCP needs {n_seqs} attention seq slots, have {n_slots}")

        def per_seq(xs):
            return np.repeat(np.asarray(xs, np.int32), 2)

        seq_lens[:n_seqs] = per_seq(
            [l_i + n_i for n_i, l_i in zip(counts, computed)])
        seq_lens[n_seqs:] = 0
        request_distribution[:] = (0, 0, n_seqs)
        kv_cache_lens_np = np.zeros(n_slots, np.int32)
        kv_cache_lens_np[:n_seqs] = per_seq(computed)

        cu_row, q_pos_np, kv_new_starts_np = pcp_seq_arrays(
            chunk, off, pcp_size, n_slots)
        # A zero-length seq hangs the kernel.
        assert np.all(np.diff(cu_row[:n_seqs + 1]) > 0), (
            f"zero-length PCP seq in cu_q_lens: {cu_row[:n_seqs + 1]}")

        # The global slot of each request's last token: token n_i - 1 sits
        # in zigzag chunk c (rank c heads, rank 2P-1-c tails), at row
        # rank * s_pad + off_i + half * C_i + (n_i - 1) % C_i.
        s_pad = t_pad // pcp_size
        last = np.asarray(counts) - 1
        c_arr = np.asarray(chunk)
        ci = last // c_arr
        half = (ci >= pcp_size).astype(np.int64)
        rank = np.where(half == 0, ci, 2 * pcp_size - 1 - ci)
        logits_indices[:] = -1
        logits_indices[:num_pcp_reqs] = (rank * s_pad + np.asarray(off) +
                                         half * c_arr + last % c_arr)

        return self.metadata_to_device(
            cu_row,
            q_pos_np,
            kv_cache_lens_np,
            kv_new_starts_np,
            kv_page_order,
            has_cached_kv=any(l_i > 0 for l_i in computed),
            num_reqs=runner_utils.get_padded_token_len(self.num_reqs_paddings,
                                                       num_pcp_reqs),
        )
