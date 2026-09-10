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
"""KV-cache writeback correctness for the stacked RPA kernel.

The kernel appends the step's new K/V into the paged cache as part of the same
launch that computes attention, so the returned cache is half its contract --
but the rest of the suite only ever looks at the attention output. These tests
pin both halves: the new token has to land on the right page and lane, nothing
else may be disturbed, and a following step has to be able to read it back.

The writeback is 128-lane granular, so the ``kv_lens`` below are chosen to put
the new token in several different 128-lane chunks of a page, including the
first and last, and at both aligned and unaligned offsets.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.stacked_rpa import wrapper

NUM_KV_HEADS = 1
NUM_Q_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 512
# New token at kv_len-1, i.e. lane (kv_len-1) % PAGE_SIZE: chunk 0 aligned,
# chunk 1 aligned, chunk 0 unaligned, chunk 2, last lane of the page, and a
# second page.
KV_LENS = (1, 129, 70, 300, 512, 600)


def _cdiv(a, b):
    return -(-a // b)


def _build(kv_lens, seed=0):
    """One single-token decode step over ``len(kv_lens)`` sequences."""
    rng = np.random.default_rng(seed)
    num_seqs = len(kv_lens)
    pages_per_seq = _cdiv(max(kv_lens), PAGE_SIZE)
    kv_dtype = jnp.float8_e4m3fn

    q = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_Q_HEADS, HEAD_DIM)) * 0.5,
        jnp.bfloat16)
    k = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)) * 0.5,
        kv_dtype)
    v = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)) * 0.5,
        kv_dtype)

    shape = wrapper.get_kv_cache_shape(
        total_num_pages=pages_per_seq * num_seqs + 4,
        page_size=PAGE_SIZE,
        actual_num_kv_heads=NUM_KV_HEADS,
        actual_head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
    )
    kv_cache = jnp.asarray(rng.standard_normal(shape) * 0.3, kv_dtype)

    # Shuffle the page table so a correct writeback cannot be mistaken for
    # "wrote to the page whose index happens to equal the slot".
    page_indices = rng.permutation(pages_per_seq * num_seqs +
                                   4)[:pages_per_seq * num_seqs]
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(np.asarray(kv_lens, np.int32)),
        page_indices=jnp.asarray(page_indices.astype(np.int32)),
        cu_q_lens=jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32)),
        distribution=jnp.asarray(np.array([num_seqs] * 3, np.int32)),
        pages_per_seq=pages_per_seq,
    )


def _run(b):
    return wrapper.ragged_paged_attention(
        queries=b["q"],
        keys=b["k"],
        values=b["v"],
        kv_cache=b["kv_cache"],
        kv_lens=b["kv_lens"],
        page_indices=b["page_indices"],
        cu_q_lens=b["cu_q_lens"],
        distribution=b["distribution"],
        dispatch_hint="decode_only",
    )


def _slot(b, s):
    """(physical page, lane) the new token of sequence ``s`` must land on."""
    pos = int(b["kv_lens"][s]) - 1
    page = int(b["page_indices"][s * b["pages_per_seq"] + pos // PAGE_SIZE])
    return page, pos % PAGE_SIZE


@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_new_kv_lands_on_the_right_page_and_lane(kv_lens):
    b = _build(kv_lens)
    _, out_cache = jax.block_until_ready(_run(b))
    out = np.asarray(out_cache.astype(jnp.float32))

    for s in range(len(kv_lens)):
        page, lane = _slot(b, s)
        k_exp = np.asarray(b["k"][s, 0].astype(jnp.float32))
        v_exp = np.asarray(b["v"][s, 0].astype(jnp.float32))
        np.testing.assert_array_equal(
            out[page, 0, :, lane],
            k_exp,
            err_msg=f"seq {s}: K not written at page {page} lane {lane}")
        np.testing.assert_array_equal(
            out[page, 1, :, lane],
            v_exp,
            err_msg=f"seq {s}: V not written at page {page} lane {lane}")


@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_live_tokens_are_preserved(kv_lens):
    """Every token already in the cache must survive the writeback.

    This is the safety property the kernel actually owes: the writeback copies
    a VMEM window back over the page, so if it ever covered a lane holding a
    live token that it had not fetched, that token would be replaced by stale
    VMEM. Lanes at or past ``kv_len`` are scratch -- a later append overwrites
    them before anything reads them -- so they are excluded here and pinned by
    ``test_writeback_touches_only_the_new_token_chunk`` instead.
    """
    b = _build(kv_lens)
    before = np.array(b["kv_cache"].astype(jnp.float32))
    live = [(s, _slot(b, s)) for s in range(len(kv_lens))]

    _, out_cache = jax.block_until_ready(_run(b))
    after = np.asarray(out_cache.astype(jnp.float32))

    for s, (page, lane) in live:
        # Tokens [0, kv_len-1) of this sequence, i.e. everything before the
        # one this step appends.
        for pos in range(int(b["kv_lens"][s]) - 1):
            p = int(b["page_indices"][s * b["pages_per_seq"] +
                                      pos // PAGE_SIZE])
            ln = pos % PAGE_SIZE
            np.testing.assert_array_equal(
                after[p, :, :, ln],
                before[p, :, :, ln],
                err_msg=f"seq {s}: live token {pos} (page {p} lane {ln}) was "
                f"clobbered by the writeback")


@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_writeback_touches_only_the_new_token_chunk(kv_lens):
    """The writeback must stay inside the new token's 128-lane chunk.

    A page-granular writeback rewrites the whole page from a VMEM window that
    was only ever filled up to the sequence length, so it scribbles stale VMEM
    across the rest of the page. Trimming it to the 128-lane window the new
    token lands in bounds that blast radius (and moves page_size/128 fewer
    bytes). Anything outside the chunk must be untouched.
    """
    b = _build(kv_lens)
    before = np.array(b["kv_cache"].astype(jnp.float32))
    chunks = [_slot(b, s) for s in range(len(kv_lens))]

    _, out_cache = jax.block_until_ready(_run(b))
    after = np.asarray(out_cache.astype(jnp.float32))

    allowed = np.zeros(before.shape[0::3], dtype=bool)  # [pages, lanes]
    for page, lane in chunks:
        lo = (lane // 128) * 128
        allowed[page, lo:lo + 128] = True

    changed = np.any(after != before, axis=(1, 2))  # [pages, lanes]
    stray = np.argwhere(changed & ~allowed)
    assert stray.size == 0, (
        f"{len(stray)} (page, lane) positions changed outside the new token's "
        f"128-lane chunk; first at {tuple(stray[0])}")


def _build_multi(q_lens, prev_lens, seed=3):
    """One step where each sequence contributes ``q_lens[s]`` new tokens."""
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    kv_lens = [p + q for p, q in zip(prev_lens, q_lens)]
    pages_per_seq = _cdiv(max(kv_lens), PAGE_SIZE)
    total_q = sum(q_lens)
    kv_dtype = jnp.float8_e4m3fn

    shape = wrapper.get_kv_cache_shape(
        total_num_pages=pages_per_seq * num_seqs + 4,
        page_size=PAGE_SIZE,
        actual_num_kv_heads=NUM_KV_HEADS,
        actual_head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
    )
    cu = np.zeros(num_seqs + 1, dtype=np.int32)
    np.cumsum(q_lens, out=cu[1:])
    return dict(
        q=jnp.asarray(
            rng.standard_normal((total_q, NUM_Q_HEADS, HEAD_DIM)) * 0.5,
            jnp.bfloat16),
        k=jnp.asarray(
            rng.standard_normal((total_q, NUM_KV_HEADS, HEAD_DIM)) * 0.5,
            kv_dtype),
        v=jnp.asarray(
            rng.standard_normal((total_q, NUM_KV_HEADS, HEAD_DIM)) * 0.5,
            kv_dtype),
        kv_cache=jnp.asarray(rng.standard_normal(shape) * 0.3, kv_dtype),
        kv_lens=jnp.asarray(np.asarray(kv_lens, np.int32)),
        page_indices=jnp.asarray(
            rng.permutation(pages_per_seq * num_seqs +
                            4)[:pages_per_seq * num_seqs].astype(np.int32)),
        cu_q_lens=jnp.asarray(cu),
        pages_per_seq=pages_per_seq,
        q_lens=q_lens,
        prev_lens=prev_lens,
    )


def _assert_all_new_tokens_written(b, out_cache, label):
    after = np.asarray(out_cache.astype(jnp.float32))
    k = np.asarray(b["k"].astype(jnp.float32))
    v = np.asarray(b["v"].astype(jnp.float32))
    cu = np.asarray(b["cu_q_lens"])
    for s, (prev, qn) in enumerate(zip(b["prev_lens"], b["q_lens"])):
        for j in range(qn):
            pos = prev + j
            page = int(b["page_indices"][s * b["pages_per_seq"] +
                                         pos // PAGE_SIZE])
            lane = pos % PAGE_SIZE
            t = int(cu[s]) + j
            np.testing.assert_array_equal(
                after[page, 0, :, lane],
                k[t, 0],
                err_msg=f"{label}: seq {s} token {pos} K missing at page "
                f"{page} lane {lane}")
            np.testing.assert_array_equal(
                after[page, 1, :, lane],
                v[t, 0],
                err_msg=f"{label}: seq {s} token {pos} V missing at page "
                f"{page} lane {lane}")


def test_mixed_writes_every_new_token():
    """Chunked prefill: many new tokens per sequence, spanning 128-lane chunks.

    The MIXED writeback loops over pages rather than persisting a single
    boundary token, so it exercises a different arm of the trimmed window --
    including runs that start mid-chunk and cross a page boundary.
    """
    q_lens = [1, 5, 130, 200]
    prev_lens = [0, 60, 380, 700]
    b = _build_multi(q_lens, prev_lens)
    n = len(q_lens)
    _, out_cache = jax.block_until_ready(
        wrapper.ragged_paged_attention(
            queries=b["q"],
            keys=b["k"],
            values=b["v"],
            kv_cache=b["kv_cache"],
            kv_lens=b["kv_lens"],
            page_indices=b["page_indices"],
            cu_q_lens=b["cu_q_lens"],
            distribution=jnp.asarray(np.array([0, 0, n], np.int32)),
        ))
    _assert_all_new_tokens_written(b, out_cache, "mixed")


def test_spec_decode_writes_every_draft_token():
    """Multi-token decode (decode_q_len > 1) must persist the whole draft."""
    draft = 4
    n = 4
    b = _build_multi([draft] * n, [70, 200, 508, 1000], seed=5)
    _, out_cache = jax.block_until_ready(
        wrapper.ragged_paged_attention(
            queries=b["q"],
            keys=b["k"],
            values=b["v"],
            kv_cache=b["kv_cache"],
            kv_lens=b["kv_lens"],
            page_indices=b["page_indices"],
            cu_q_lens=b["cu_q_lens"],
            distribution=jnp.asarray(np.array([n, n, n], np.int32)),
            decode_q_len=draft,
            dispatch_hint="decode_only",
        ))
    _assert_all_new_tokens_written(b, out_cache, "spec-decode")


@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_appended_token_is_visible_to_the_next_step(kv_lens):
    """End-to-end: step 2 must attend over the token step 1 appended.

    Reads the cache back through the kernel rather than by indexing, so a
    writeback that lands on the wrong lane shows up as a wrong answer even if
    it happens to be self-consistent.
    """
    b = _build(kv_lens)
    _, cache1 = jax.block_until_ready(_run(b))

    # Same sequences, one token longer, and this time ask for a query that is
    # a copy of the appended key: the largest logit must be the new token.
    kv_lens2 = tuple(n + 1 for n in kv_lens)
    b2 = _build(kv_lens2, seed=1)
    b2["kv_cache"] = cache1
    b2["page_indices"] = b["page_indices"]
    b2["pages_per_seq"] = b["pages_per_seq"]
    b2["q"] = jnp.asarray(
        np.repeat(np.asarray(b["k"].astype(jnp.float32)), NUM_Q_HEADS, axis=1),
        jnp.bfloat16)

    out2, _ = jax.block_until_ready(_run(b2))
    out2 = np.asarray(out2.astype(jnp.float32))
    assert np.isfinite(out2).all(), "non-finite attention output"

    # The step-1 token is the argmax key for this query, so the output has to
    # sit closer to its value than to the mean of the sequence's other values.
    for s in range(len(kv_lens)):
        v_new = np.asarray(b["v"][s, 0].astype(jnp.float32))
        got = out2[s, 0]
        assert np.dot(got, v_new) > 0, (
            f"seq {s}: output does not correlate with the value appended by "
            f"the previous step -- writeback likely landed on the wrong lane")
