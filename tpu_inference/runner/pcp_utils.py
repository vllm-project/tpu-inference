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
                     pcp_size: int) -> tuple[list[int], list[int], int]:
    """Returns (chunk, off, s_live): per-request chunk size ceil(n_i / 2P),
    the start of each request's head+tail slot within a rank's region, and
    the live rows per rank."""
    two_p = 2 * pcp_size
    off, acc, C = [], 0, []
    for n in num_scheduled_tokens:
        c = cdiv(n, two_p)
        C.append(c)
        off.append(acc)
        acc += 2 * c
    return C, off, acc


def pcp_buffer_tokens(num_scheduled_tokens: list[int], pcp_size: int) -> int:
    """Rows the token buffer must hold for this batch (can exceed the raw
    token count, since every chunk rounds up independently)."""
    _, _, s_live = pcp_token_layout(num_scheduled_tokens, pcp_size)
    return pcp_size * s_live


def pcp_max_buffer_tokens(max_num_batched_tokens: int, max_num_seqs: int,
                          pcp_size: int) -> int:
    """Upper bound of `pcp_buffer_tokens` over any batch the scheduler
    admits: each request rounds up by less than 2P rows."""
    return max_num_batched_tokens + 2 * pcp_size * max_num_seqs


def pcp_batch_layout(num_scheduled_tokens: list[int], t_pad: int,
                     pcp_size: int) -> tuple[list[int], list[int]]:
    """Chunk sizes and slot offsets of a batch inside a `t_pad`-row buffer.

    A single request fills the buffer (chunk = t_pad / 2P) because
    pcp_forward's kernel-side K/V remap derives the chunk from the buffer
    width.
    """
    two_p = 2 * pcp_size
    chunk, off, s_live = pcp_token_layout(num_scheduled_tokens, pcp_size)
    assert t_pad % pcp_size == 0 and t_pad >= pcp_size * s_live, (
        f"PCP token bucket {t_pad} cannot hold {pcp_size * s_live} tokens "
        f"({len(num_scheduled_tokens)} reqs, chunks {chunk})")
    if len(num_scheduled_tokens) == 1:
        assert t_pad % two_p == 0, (t_pad, two_p)
        chunk = [t_pad // two_p]
        off = [0]
    return chunk, off


def pcp_seq_arrays(chunk: list[int], off: list[int], pcp_size: int,
                   n_off: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-seq metadata for `n_off` seq slots: cu_row[n_off + 1] cumulative
    query rows (rank-invariant), q_pos[pcp_size, n_off] per-rank query
    position offsets, kv_new_starts[n_off] base of each request's block in
    the request-major current-K/V buffer."""
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


def pcp_token_permutation(num_scheduled_tokens: list[int], chunk: list[int],
                          off: list[int], t_pad: int,
                          pcp_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (perm, kv_order): perm[g] is the natural-order source of
    rank-order slot g (-1 for padding); kv_order maps the all-gathered
    current K/V from rank order to request-major token order."""
    two_p = 2 * pcp_size
    s_pad = t_pad // pcp_size
    src_off = np.cumsum([0] + list(num_scheduled_tokens))[:-1]
    perm = np.full(t_pad, -1, np.int64)
    kv_order = np.zeros(t_pad, np.int32)
    ranks = np.arange(pcp_size)
    for i, n_i in enumerate(num_scheduled_tokens):
        c_i = chunk[i]
        j = np.arange(c_i)
        kv_base = pcp_size * off[i]
        for h in (0, 1):
            chunk_idx = ranks if h == 0 else two_p - 1 - ranks
            dst = (ranks[:, None] * s_pad + off[i] + h * c_i + j[None, :])
            tok = chunk_idx[:, None] * c_i + j[None, :]
            real = tok < n_i
            perm[dst[real]] = src_off[i] + tok[real]
            kv_order[kv_base + tok.ravel()] = dst.ravel()
    return perm, kv_order


class PCPPreprocessor:
    """Per-step host preprocessing for a PCP-enabled runner."""

    def __init__(self, pcp_size: int, mesh: Mesh,
                 num_reqs_paddings: list[int]):
        assert pcp_size > 1, pcp_size
        self.pcp_size = pcp_size
        self.mesh = mesh
        self.num_reqs_paddings = num_reqs_paddings
        self._pcp_spec = NamedSharding(
            mesh, PartitionSpec(ShardingAxisName.PREFILL_CONTEXT, None))
        self._repl_spec = NamedSharding(mesh, PartitionSpec())

    def metadata_to_device(self, cu_row: np.ndarray, q_pos: np.ndarray,
                           kv_cache_lens: np.ndarray,
                           kv_new_starts: np.ndarray,
                           kv_token_order: np.ndarray, *, has_cached_kv: bool,
                           num_reqs: int) -> PCPMetadata:
        """Place host arrays as a `PCPMetadata`; also used by the compilation
        manager so precompiled and runtime metadata share one sharding."""
        query_start_loc, q_pos_offsets = device_array(
            self.mesh, (np.tile(cu_row, (self.pcp_size, 1)), q_pos),
            sharding=self._pcp_spec)
        kv_cache_lens, kv_new_starts, kv_token_order = device_array(
            self.mesh, (kv_cache_lens, kv_new_starts, kv_token_order),
            sharding=self._repl_spec)
        return PCPMetadata(
            query_start_loc=query_start_loc,
            kv_cache_lens=kv_cache_lens,
            q_pos_offsets=q_pos_offsets,
            kv_new_starts=kv_new_starts,
            kv_token_order=kv_token_order,
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
        chunk, off = pcp_batch_layout(counts, t_pad, pcp_size)
        perm, kv_order = pcp_token_permutation(counts, chunk, off, t_pad,
                                               pcp_size)
        valid = perm >= 0
        src_idx = perm[valid]
        for buf in (positions, input_ids):
            src = buf.copy()
            buf[:] = 0
            buf[valid] = src[src_idx]

        n_seqs = 2 * num_pcp_reqs
        n_off = len(seq_lens)
        assert n_seqs <= n_off, (
            f"PCP needs {n_seqs} attention seq slots, have {n_off}")

        def per_seq(xs):
            return np.repeat(np.asarray(xs, np.int32), 2)

        seq_lens[:n_seqs] = per_seq(
            [l_i + n_i for n_i, l_i in zip(counts, computed)])
        seq_lens[n_seqs:] = 0
        request_distribution[:] = (0, 0, n_seqs)
        kv_cache_lens_np = np.zeros(n_off, np.int32)
        kv_cache_lens_np[:n_seqs] = per_seq(computed)

        cu_row, q_pos_np, kv_new_starts_np = pcp_seq_arrays(
            chunk, off, pcp_size, n_off)
        # A zero-length seq hangs the kernel.
        assert np.all(np.diff(cu_row[:n_seqs + 1]) > 0), (
            f"zero-length PCP seq in cu_q_lens: {cu_row[:n_seqs + 1]}")

        # The global slot of each request's last token.
        logits_indices[:] = -1
        logits_indices[:num_pcp_reqs] = kv_order[pcp_size * np.asarray(off) +
                                                 np.asarray(counts) - 1]

        return self.metadata_to_device(
            cu_row,
            q_pos_np,
            kv_cache_lens_np,
            kv_new_starts_np,
            kv_order,
            has_cached_kv=any(l_i > 0 for l_i in computed),
            num_reqs=runner_utils.get_padded_token_len(self.num_reqs_paddings,
                                                       num_pcp_reqs),
        )
