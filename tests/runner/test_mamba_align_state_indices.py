# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba prefix caching: read/write state slot derivation.

`build_align_state_indices` turns the mamba group's block table into the pair
of slots a step reads its recurrent state from and writes its checkpoint to.
It touches no device state, so it is checked here against hand-worked block
tables rather than on a TPU.

The invariant it encodes comes from vLLM's
`Scheduler._mamba_block_aligned_split`: block row `p` holds the state after
exactly `(p + 1) * block_size` tokens.
"""
import numpy as np
import pytest
# torch and vllm must be imported before tpu_inference, as in the other
# runner tests: tpu_inference preloads an XLA copy whose static initializers
# collide with jaxlib's if the two load in the other order.
import torch  # noqa: F401
from vllm.sampling_params import SamplingParams  # noqa: F401

from tpu_inference.runner.mamba_prefix_caching import \
    build_align_state_indices

BLOCK_SIZE = 16
MAX_NUM_REQS = 4


def _build(block_table: np.ndarray, num_computed: list[int],
           num_scheduled: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Run the derivation over `block_table` for the given token counts."""
    num_reqs = len(num_computed)
    computed = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    computed[:num_reqs] = num_computed

    return build_align_state_indices(
        block_table=block_table,
        num_computed_tokens=computed,
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        block_size=BLOCK_SIZE,
        max_num_reqs=MAX_NUM_REQS,
        req_indices=np.arange(num_reqs),
    )


def test_first_chunk_writes_first_row():
    """A fresh request has no state to resume from.

    Its read row clamps to row 0 — the request's own first block. That slot
    is never actually read: the kernel gates the load on `has_initial_state`
    (`seq_lens - query_lens > 0`), which is false here. The clamp only has to
    keep the index in range.
    """
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    block_table[0, 0] = 7

    write, read = _build(block_table, num_computed=[0], num_scheduled=[16])

    assert write[0] == 7, "state after 16 tokens belongs in row 0"
    assert read[0] == 7


def test_step_inside_a_block_reads_and_writes_the_same_slot():
    """Not every step crosses a boundary; those keep one slot."""
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    block_table[0, 0] = 7

    # Tokens 4..12 of a 16-token block: both endpoints sit in row 0.
    write, read = _build(block_table, num_computed=[4], num_scheduled=[8])

    assert write[0] == 7
    assert read[0] == 7


def test_boundary_crossing_reads_previous_row_and_writes_the_new_one():
    """The case the whole design exists for.

    Resuming at a block boundary must read the checkpoint the previous step
    left behind and write into the freshly allocated block, so the cached
    state survives for another request to hit.
    """
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    block_table[0, 0] = 7  # holds the state after 16 tokens
    block_table[0, 1] = 9  # allocated for the state after 32 tokens

    write, read = _build(block_table, num_computed=[16], num_scheduled=[16])

    assert read[0] == 7
    assert write[0] == 9


def test_prefix_cache_hit_resumes_from_the_cached_block():
    """A cache hit admits a request with num_computed_tokens > 0.

    Its read slot must be the block the *earlier* request populated, which
    is what makes the shared prefix free rather than recomputed.
    """
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    # Rows 0-1 came from the prefix cache; row 2 is newly allocated.
    block_table[0, 0] = 21
    block_table[0, 1] = 22
    block_table[0, 2] = 30

    write, read = _build(block_table, num_computed=[32], num_scheduled=[10])

    assert read[0] == 22, "must resume from the last cached checkpoint"
    assert write[0] == 30


def test_decode_step_stays_on_its_block():
    """A single decode token rarely crosses a boundary."""
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    block_table[0, 2] = 30

    write, read = _build(block_table, num_computed=[40], num_scheduled=[1])

    assert read[0] == 30
    assert write[0] == 30


def test_decode_step_that_crosses_a_boundary():
    """...but when it does, it must still move to the new block."""
    block_table = np.zeros((MAX_NUM_REQS, 8), dtype=np.int32)
    block_table[0, 2] = 30
    block_table[0, 3] = 31

    # 48 computed tokens = exactly 3 blocks, so the next token opens row 3.
    write, read = _build(block_table, num_computed=[48], num_scheduled=[1])

    assert read[0] == 30
    assert write[0] == 31


def test_padded_requests_point_at_the_null_block():
    """Unused persistent-batch slots must not alias a live request's state.

    Positions past `num_reqs` are padding the kernel still walks over, so
    they have to land on block id 0 rather than on whatever the block table
    happens to hold.
    """
    block_table = np.full((MAX_NUM_REQS, 8), 99, dtype=np.int32)
    block_table[0, 0] = 7
    block_table[1, 0] = 8

    write, read = _build(block_table,
                         num_computed=[0, 0],
                         num_scheduled=[16, 16])

    assert list(write) == [7, 8, 0, 0]
    assert list(read) == [7, 8, 0, 0]


def test_no_requests_yields_all_null_blocks():
    block_table = np.full((MAX_NUM_REQS, 8), 5, dtype=np.int32)

    write, read = _build(block_table, num_computed=[], num_scheduled=[])

    assert not write.any()
    assert not read.any()


@pytest.mark.parametrize("num_computed", [16, 32, 48, 64])
def test_read_row_matches_vllm_preprocess_mamba(num_computed: int):
    """Cross-check against the formula GPU uses to pick its copy source.

    `vllm/v1/worker/mamba_utils.py::preprocess_mamba` computes
    `prev = (num_computed_tokens - 1) // block_size` and
    `curr = cdiv(num_computed + num_scheduled, block_size) - 1`, then copies
    prev into curr. The TPU path must select the same two rows.
    """
    num_scheduled = 16
    block_table = np.arange(MAX_NUM_REQS * 8, dtype=np.int32).reshape(
        MAX_NUM_REQS, 8) + 1

    write, read = _build(block_table,
                         num_computed=[num_computed],
                         num_scheduled=[num_scheduled])

    expected_read_row = (num_computed - 1) // BLOCK_SIZE
    expected_write_row = -(-(num_computed + num_scheduled) // BLOCK_SIZE) - 1

    assert read[0] == block_table[0, expected_read_row]
    assert write[0] == block_table[0, expected_write_row]
