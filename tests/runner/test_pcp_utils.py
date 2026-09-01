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
"""Tests for runner/pcp_utils.py; `test_prepare_inputs` needs pcp_size devices."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm.utils.math_utils import cdiv, next_power_of_2

from tpu_inference.runner.pcp_utils import (PCPPreprocessor, pcp_batch_layout,
                                            pcp_buffer_tokens,
                                            pcp_max_buffer_tokens,
                                            pcp_seq_arrays, pcp_token_layout,
                                            pcp_token_permutation)

# (pcp_size, scheduled tokens per request)
LAYOUTS = [
    (2, [5]),
    (2, [4096]),
    (2, [22061, 3000]),
    (4, [7, 1, 300]),
    (2, [1, 1]),
    (8, [2539, 1200, 7, 3000, 64, 1, 500, 999]),
]


def _t_pad(counts, pcp):
    """A power-of-two bucket that holds the layout and is divisible by 2P."""
    return next_power_of_2(max(pcp_buffer_tokens(counts, pcp), 2 * pcp))


def _layout(counts, pcp):
    """(t_pad, chunk, off) of a batch in the bucket `_t_pad` picks."""
    t_pad = _t_pad(counts, pcp)
    chunk, off = pcp_batch_layout(counts, t_pad, pcp)
    return t_pad, chunk, off


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_token_layout(pcp, counts):
    chunk, off, s_live = pcp_token_layout(counts, pcp)
    assert chunk == [cdiv(n, 2 * pcp) for n in counts]
    assert off == list(np.cumsum([0] + [2 * c for c in chunk])[:-1])
    assert s_live == sum(2 * c for c in chunk)
    assert pcp_buffer_tokens(counts, pcp) == pcp * s_live
    assert pcp_buffer_tokens(counts, pcp) <= pcp_max_buffer_tokens(
        sum(counts), len(counts), pcp)


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_batch_layout(pcp, counts):
    t_pad, chunk, off = _layout(counts, pcp)
    if len(counts) == 1:
        # Single request: the chunk comes from the buffer width.
        assert chunk == [t_pad // (2 * pcp)] and off == [0]
    else:
        assert (chunk, off) == pcp_token_layout(counts, pcp)[:2]


def test_batch_layout_rejects_short_buffer():
    with pytest.raises(AssertionError):
        pcp_batch_layout([100, 100], 64, 2)


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_token_permutation(pcp, counts):
    t_pad, chunk, off = _layout(counts, pcp)
    s_pad = t_pad // pcp
    perm, kv_order = pcp_token_permutation(counts, chunk, off, t_pad, pcp)
    total = sum(counts)
    # Every real token lands in exactly one slot; everything else is padding.
    assert sorted(perm[perm >= 0].tolist()) == list(range(total))
    src_off = np.cumsum([0] + counts)[:-1]
    for i, n_i in enumerate(counts):
        c_i = chunk[i]
        for tok in range(n_i):
            slot = kv_order[pcp * off[i] + tok]
            # kv_order undoes perm on the live rows.
            assert perm[slot] == src_off[i] + tok
            # Zigzag: chunk k sits on rank k (head) or 2P-1-k (tail).
            k = tok // c_i
            rank = k if k < pcp else 2 * pcp - 1 - k
            assert slot // s_pad == rank
    # Slots reserved for a request are distinct across the request.
    for i in range(len(counts)):
        lo, hi = pcp * off[i], pcp * off[i] + 2 * pcp * chunk[i]
        assert len(set(kv_order[lo:hi].tolist())) == hi - lo


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_seq_arrays(pcp, counts):
    _, chunk, off = _layout(counts, pcp)
    n_off = 2 * len(counts) + 3
    cu_row, q_pos, kv_new_starts = pcp_seq_arrays(chunk, off, pcp, n_off)
    n_seqs = 2 * len(counts)
    assert cu_row.shape == (n_off + 1, )
    assert np.all(np.diff(cu_row[:n_seqs + 1]) > 0)
    assert np.all(cu_row[n_seqs:] == cu_row[n_seqs])
    for i, c_i in enumerate(chunk):
        assert cu_row[2 * i] == off[i]
        assert cu_row[2 * i + 1] - cu_row[2 * i] == c_i
        assert cu_row[2 * i + 2] - cu_row[2 * i + 1] == c_i
        for r in range(pcp):
            assert q_pos[r, 2 * i] == r * c_i
            assert q_pos[r, 2 * i + 1] == (2 * pcp - 1 - r) * c_i
        assert kv_new_starts[2 * i] == kv_new_starts[2 * i + 1] == pcp * off[i]
    assert np.all(q_pos[:, n_seqs:] == 0)


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_prepare_inputs(pcp, counts):
    if len(jax.devices()) < pcp:
        pytest.skip(f"needs {pcp} devices")
    mesh = Mesh(np.array(jax.devices()[:pcp]), ("pcp", ))
    pre = PCPPreprocessor(pcp, mesh, [1, 8])

    t_pad, chunk, off = _layout(counts, pcp)
    n_reqs = len(counts)
    n_off = 2 * 8
    # Cached prefixes on the longer requests (a cached 1-token request is a
    # decode, rejected below).
    computed = [(3 * i) % 40 if n > 1 else 0 for i, n in enumerate(counts)]
    total = sum(counts)
    # Natural-order buffers: token g carries id 1000 + g and position g.
    input_ids = np.zeros(t_pad, np.int32)
    input_ids[:total] = 1000 + np.arange(total)
    positions = np.zeros(t_pad, np.int32)
    positions[:total] = np.arange(total)
    seq_lens = np.full(n_off, 7, np.int32)
    request_distribution = np.array([n_reqs, 0, 0], np.int32)
    logits_indices = np.full(8, 5, np.int32)

    md = pre.prepare_inputs(counts, computed, t_pad, positions, input_ids,
                            seq_lens, request_distribution, logits_indices)

    perm, kv_order = pcp_token_permutation(counts, chunk, off, t_pad, pcp)
    live = perm >= 0
    assert np.array_equal(input_ids[live], 1000 + perm[live])
    assert np.array_equal(positions[live], perm[live])
    assert np.all(input_ids[~live] == 0) and np.all(positions[~live] == 0)

    n_seqs = 2 * n_reqs
    assert np.array_equal(
        seq_lens[:n_seqs],
        np.repeat([n + c for n, c in zip(counts, computed)], 2))
    assert np.all(seq_lens[n_seqs:] == 0)
    assert request_distribution.tolist() == [0, 0, n_seqs]
    # Each request's logits slot holds its last real token.
    src_off = np.cumsum([0] + counts)[:-1]
    assert np.array_equal(perm[logits_indices[:n_reqs]],
                          src_off + np.asarray(counts) - 1)
    assert np.all(logits_indices[n_reqs:] == -1)

    assert md.query_start_loc.shape == (pcp, n_off + 1)
    assert md.q_pos_offsets.shape == (pcp, n_off)
    assert np.array_equal(
        np.asarray(md.kv_cache_lens)[:n_seqs], np.repeat(computed, 2))
    assert md.has_cached_kv == (max(computed) > 0)
    assert md.num_reqs == (1 if n_reqs == 1 else 8)
    assert np.array_equal(np.asarray(md.kv_token_order), kv_order)
    assert np.array_equal(
        np.asarray(md.kv_new_starts)[:n_seqs],
        np.repeat([pcp * o for o in off], 2))


def test_prepare_inputs_rejects_decode():
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    mesh = Mesh(np.array(jax.devices()[:2]), ("pcp", ))
    pre = PCPPreprocessor(2, mesh, [1, 8])
    # Rejected before any buffer is touched, so shapes do not matter.
    buf = np.zeros(16, np.int32)
    with pytest.raises(NotImplementedError):
        pre.prepare_inputs([1], [5], 16, buf, buf, buf, buf, buf)
