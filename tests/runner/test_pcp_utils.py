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
from vllm.utils.math_utils import cdiv, next_power_of_2, round_up

from tpu_inference.runner.pcp_utils import (PCPPreprocessor, pcp_batch_layout,
                                            pcp_buffer_tokens,
                                            pcp_max_buffer_tokens,
                                            pcp_page_order, pcp_seq_arrays,
                                            pcp_token_layout,
                                            pcp_token_permutation)

# KV page size the production layout aligns chunks to (cache block_size).
PAGE = 16

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
    return next_power_of_2(
        max(pcp_buffer_tokens(counts, pcp, align=PAGE), 2 * pcp))


def _layout(counts, pcp):
    """(t_pad, chunk, off) of a batch in the bucket `_t_pad` picks."""
    t_pad = _t_pad(counts, pcp)
    chunk, off = pcp_batch_layout(counts, t_pad, pcp, align=PAGE)
    return t_pad, chunk, off


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_token_layout(pcp, counts):
    chunk, off, s_live = pcp_token_layout(counts, pcp, align=1)
    assert chunk == [cdiv(n, 2 * pcp) for n in counts]
    assert off == list(np.cumsum([0] + [2 * c for c in chunk])[:-1])
    assert s_live == sum(2 * c for c in chunk)
    assert pcp_buffer_tokens(counts, pcp, align=1) == pcp * s_live
    assert pcp_buffer_tokens(counts, pcp,
                             align=1) <= pcp_max_buffer_tokens(sum(counts),
                                                               len(counts),
                                                               pcp,
                                                               align=1)


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_batch_layout(pcp, counts):
    t_pad, chunk, off = _layout(counts, pcp)
    # One layout for any request count: R = 1 is not a special case.
    assert (chunk, off) == pcp_token_layout(counts, pcp, align=PAGE)[:2]


def test_batch_layout_rejects_short_buffer():
    with pytest.raises(AssertionError):
        pcp_batch_layout([100, 100], 64, 2, align=1)


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_token_permutation(pcp, counts):
    t_pad, chunk, off = _layout(counts, pcp)
    s_pad = t_pad // pcp
    perm = pcp_token_permutation(counts, chunk, off, t_pad, pcp)
    total = sum(counts)
    # Every real token lands in exactly one slot; everything else is padding.
    assert sorted(perm[perm >= 0].tolist()) == list(range(total))
    src_off = np.cumsum([0] + counts)[:-1]
    for i, n_i in enumerate(counts):
        c_i = chunk[i]
        for tok in range(n_i):
            # Zigzag: chunk k sits on rank k (head) or 2P-1-k (tail), at row
            # rank * s_pad + off_i + half * C_i + tok % C_i.
            k = tok // c_i
            rank = k if k < pcp else 2 * pcp - 1 - k
            half = 0 if k < pcp else 1
            slot = rank * s_pad + off[i] + half * c_i + tok % c_i
            assert perm[slot] == src_off[i] + tok


@pytest.mark.parametrize("pcp,counts", LAYOUTS)
def test_seq_arrays(pcp, counts):
    _, chunk, off = _layout(counts, pcp)
    n_slots = 2 * len(counts) + 3
    cu_row, q_pos, kv_new_starts = pcp_seq_arrays(chunk, off, pcp, n_slots)
    n_seqs = 2 * len(counts)
    assert cu_row.shape == (n_slots + 1, )
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
    pre = PCPPreprocessor(pcp, mesh, [1, 8], PAGE)

    t_pad, chunk, off = _layout(counts, pcp)
    n_reqs = len(counts)
    n_slots = 2 * 8
    # Cached prefixes on the longer requests (a cached 1-token request is a
    # decode, rejected below).
    computed = [(3 * i) % 40 if n > 1 else 0 for i, n in enumerate(counts)]
    total = sum(counts)
    # Natural-order buffers: token g carries id 1000 + g and position g.
    input_ids = np.zeros(t_pad, np.int32)
    input_ids[:total] = 1000 + np.arange(total)
    positions = np.zeros(t_pad, np.int32)
    positions[:total] = np.arange(total)
    seq_lens = np.full(n_slots, 7, np.int32)
    request_distribution = np.array([n_reqs, 0, 0], np.int32)
    logits_indices = np.full(8, 5, np.int32)

    md = pre.prepare_inputs(counts, computed, t_pad, positions, input_ids,
                            seq_lens, request_distribution, logits_indices)

    perm = pcp_token_permutation(counts, chunk, off, t_pad, pcp)
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

    assert md.query_start_loc.shape == (pcp, n_slots + 1)
    assert md.q_pos_offsets.shape == (pcp, n_slots)
    assert np.array_equal(
        np.asarray(md.kv_cache_lens)[:n_seqs], np.repeat(computed, 2))
    assert md.has_cached_kv == (max(computed) > 0)
    assert md.num_reqs == (1 if n_reqs == 1 else 8)
    assert np.array_equal(
        np.asarray(md.kv_page_order),
        pcp_page_order(chunk, off, pcp, t_pad // pcp, t_pad, PAGE))
    assert np.array_equal(
        np.asarray(md.kv_new_starts)[:n_seqs],
        np.repeat([pcp * o for o in off], 2))


def test_prepare_inputs_rejects_decode():
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    mesh = Mesh(np.array(jax.devices()[:2]), ("pcp", ))
    pre = PCPPreprocessor(2, mesh, [1, 8], PAGE)
    # Rejected before any buffer is touched, so shapes do not matter.
    buf = np.zeros(16, np.int32)
    with pytest.raises(NotImplementedError):
        pre.prepare_inputs([1], [5], 16, buf, buf, buf, buf, buf)


# ---------------- page-aligned layout and the kv_page_order map --------------
#
# The kernel's kv_page_order fetch is only correct if every token-order page
# of the new-KV buffer is CONTIGUOUS in the rank-order all_gather result and
# the map points at its first row. These tests pin that invariant against a
# brute-force per-token expansion of the zigzag layout.


def _per_token_order(chunk, off, pcp, s_pad):
    """Brute force: token-order index -> rank-order row, for every slot."""
    two_p = 2 * pcp
    total = two_p * sum(chunk)
    order = np.zeros(total, np.int64)
    ranks = np.arange(pcp)
    base = 0
    for c_i, o_i in zip(chunk, off):
        j = np.arange(c_i)
        for h in (0, 1):
            chunk_idx = ranks if h == 0 else two_p - 1 - ranks
            dst = ranks[:, None] * s_pad + o_i + h * c_i + j[None, :]
            tok = chunk_idx[:, None] * c_i + j[None, :]
            order[base + tok.ravel()] = dst.ravel()
        base += two_p * c_i
    return order


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("align", [16, 128])
def test_aligned_chunks_and_offsets(pcp, align):
    ns = [1, 130, 22061, 3000, align, 2 * pcp * align]
    C, off, s_live = pcp_token_layout(ns, pcp, align=align)
    for n_i, c_i in zip(ns, C):
        assert c_i % align == 0
        assert 2 * pcp * c_i >= n_i
        # Tightest aligned chunk: one align-quantum less no longer covers.
        assert 2 * pcp * (c_i - align) < n_i
    for o_i in off:
        assert o_i % align == 0
    assert s_live == sum(2 * c for c in C)


def test_padding_bound_per_request():
    pcp, align = 2, 128
    for n in [1, 127, 128, 129, 511, 512, 513, 22061]:
        C, _, _ = pcp_token_layout([n], pcp, align=align)
        waste = 2 * pcp * C[0] - n
        assert waste < 2 * pcp * align


def _check_page_order(ns, pcp, page):
    C, off, s_live = pcp_token_layout(ns, pcp, align=page)
    t_pad = round_up(pcp * s_live, 2 * pcp * page)
    s_pad = t_pad // pcp
    got = pcp_page_order(C, off, pcp, s_pad, t_pad, page)
    assert got.shape == (t_pad // page, )
    assert got.dtype == np.int32
    want = _per_token_order(C, off, pcp, s_pad)
    live_pages = 2 * pcp * sum(C) // page
    for p in range(live_pages):
        seg = want[p * page:(p + 1) * page]
        # The invariant the kernel fetch relies on: one contiguous run...
        assert np.all(np.diff(seg) == 1), (p, seg[:4])
        # ...starting exactly where the map says.
        assert got[p] * page == seg[0], p
    # In-range even for dead entries (the kernel clamps but still reads).
    assert np.all((got >= 0) & (got < t_pad // page))


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("page", [16, 128])
def test_page_order_matches_per_token_map(pcp, page):
    _check_page_order([22061, 3000], pcp, page)
    _check_page_order([1, 130, 4 * pcp * page + 3], pcp, page)


@pytest.mark.parametrize("pcp", [2, 4])
def test_page_order_single_request_is_r1_of_general(pcp):
    # A single request is simply R = 1 of the general layout; the full
    # contiguity/map checker must pass on it unchanged.
    _check_page_order([1000], pcp, 16)


def test_page_order_rejects_unaligned_chunk():
    with pytest.raises(AssertionError):
        pcp_page_order([24], [0], 2, 48, 96, 16)


def test_page_order_rejects_unaligned_region():
    with pytest.raises(AssertionError):
        pcp_page_order([16], [0], 2, 40, 80, 16)
