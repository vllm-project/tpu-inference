# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Read-suppression for mamba slots no forward pass has checkpointed.

The GDN kernel writes one checkpoint per pass, at
`(seq_len - 1) // block_size`, but vLLM lets a request resume at any boundary
its retention mask kept. `_resolve_mamba_prior_state` tracks what the runner
actually wrote and clears `has_initial_state` for the rest, so those requests
start from zeros instead of reading a previous owner's state.
"""

import numpy as np
import pytest

from tpu_inference.runner.tpu_runner import TPUModelRunner

BLOCK = 256
MAX_REQS = 8


class FakeBatch:

    def __init__(self, table, num_computed, num_tokens):
        self.block_table = {1: self}
        self._table = np.asarray(table, dtype=np.int32)
        self.num_computed_tokens_cpu = np.asarray(num_computed, dtype=np.int32)
        self.num_tokens = np.asarray(num_tokens, dtype=np.int32)

    def get_cpu_tensor(self):
        return self._table


class FakePBM:

    def __init__(self):
        self.mamba_kv_cache_group_id = 1
        self.mamba_block_size = BLOCK
        self.new_mamba_blocks = {}


def _runner(table, num_computed, num_tokens):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.max_num_reqs = MAX_REQS
    r._written_mamba_slots = set()
    r.persistent_batch_manager = FakePBM()
    r.input_batch = FakeBatch(table, num_computed, num_tokens)
    return r


def _resolve(r, nreq=1, scheduled=None):
    sched = [np.asarray(scheduled if scheduled is not None else [1] * nreq)]
    return r._resolve_mamba_prior_state([np.arange(nreq)], sched, 1, MAX_REQS)


def test_no_mamba_group_returns_none():
    r = _runner([[0] * 4], [0], [4])
    r.persistent_batch_manager.mamba_kv_cache_group_id = None
    assert _resolve(r) is None


def test_fresh_request_is_not_suppressed():
    """num_computed == 0 reads nothing anyway (the kernel's DMA is
    length-zero), so it must not be flagged."""
    r = _runner([[11, 12, 0, 0]], num_computed=[0], num_tokens=[BLOCK])
    assert _resolve(r, scheduled=[BLOCK])[0] == 1


def test_resume_on_an_unwritten_slot_is_suppressed():
    # 2048 computed -> read_col 7; nothing has ever been written.
    table = [[0] * 7 + [77, 78]]
    r = _runner(table, num_computed=[2048], num_tokens=[2049])
    assert _resolve(r)[0] == 0


def test_resume_on_a_slot_this_runner_wrote_is_allowed():
    table = [[0] * 7 + [77, 78]]
    r = _runner(table, num_computed=[2048], num_tokens=[2049])
    # Step 1: prefill ending at 2048 writes write_col 7 -> slot 77.
    r.input_batch.num_computed_tokens_cpu = np.array([0], dtype=np.int32)
    r.input_batch.num_tokens = np.array([2048], dtype=np.int32)
    _resolve(r, scheduled=[2048])
    assert 77 in {s & 0xFFFFFF for s in r._written_mamba_slots}
    # Step 2: decode resumes at 2048 -> read_col 7 -> slot 77, now written.
    r.input_batch.num_computed_tokens_cpu = np.array([2048], dtype=np.int32)
    r.input_batch.num_tokens = np.array([2049], dtype=np.int32)
    assert _resolve(r)[0] == 1


def test_reallocating_a_slot_revokes_it():
    table = [[0] * 7 + [77, 78]]
    r = _runner(table, num_computed=[0], num_tokens=[2048])
    _resolve(r, scheduled=[2048])  # writes slot 77
    # The scheduler hands slot 77 to someone else; its contents are stale now.
    r.persistent_batch_manager.new_mamba_blocks = {"other": [77]}
    r.input_batch.num_computed_tokens_cpu = np.array([2048], dtype=np.int32)
    r.input_batch.num_tokens = np.array([2049], dtype=np.int32)
    assert _resolve(r)[0] == 0


def test_ranks_do_not_alias_each_others_slots():
    """Slot ids are rank-local, so rank 0's slot 77 must not satisfy a read
    of rank 1's slot 77."""
    table = np.zeros((MAX_REQS, 9), dtype=np.int32)
    table[0, 7] = 77   # rank 0's request
    table[4, 7] = 77   # rank 1's request, same local id
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.max_num_reqs = MAX_REQS
    r._written_mamba_slots = set()
    r.persistent_batch_manager = FakePBM()
    r.input_batch = FakeBatch(table, [0] * MAX_REQS, [2048] * MAX_REQS)

    # Rank 0 alone runs a pass that writes its slot 77.
    r._resolve_mamba_prior_state([np.array([0]), np.array([])],
                                 [np.array([2048]), np.array([])], 2, 4)

    # Now both ranks try to resume from their own slot 77.
    r.input_batch.num_computed_tokens_cpu = np.array([2048] * MAX_REQS,
                                                     dtype=np.int32)
    r.input_batch.num_tokens = np.array([2049] * MAX_REQS, dtype=np.int32)
    mask = r._resolve_mamba_prior_state([np.array([0]), np.array([4])],
                                        [np.array([1]), np.array([1])], 2, 4)
    assert mask[0] == 1, "rank 0 wrote this slot"
    assert mask[4] == 0, "rank 1 never wrote its own slot 77"


def test_mask_is_laid_out_per_dp_rank():
    table = np.zeros((MAX_REQS, 9), dtype=np.int32)
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.max_num_reqs = MAX_REQS
    r._written_mamba_slots = set()
    r.persistent_batch_manager = FakePBM()
    r.input_batch = FakeBatch(table, [2048] * MAX_REQS, [2049] * MAX_REQS)
    mask = r._resolve_mamba_prior_state([np.array([0]), np.array([4])],
                                        [np.array([1]), np.array([1])], 2, 4)
    # Rank 1's request lands at offset 1 * max_num_reqs_per_dp_rank.
    assert mask[0] == 0 and mask[4] == 0
    assert mask.shape == (MAX_REQS, )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
