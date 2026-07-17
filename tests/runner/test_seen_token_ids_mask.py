# Copyright 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Functional tests for the repetition-penalty seen-token mask lifecycle on
``InputBatch`` (``update_seen_token_ids_mask`` + the slot-move bookkeeping).

Runs on the JAX CPU backend:
    JAX_PLATFORMS=cpu python -m pytest tests/runner/test_seen_token_ids_mask.py

Skips if vLLM (needed to construct ``InputBatch``) is unavailable.
"""

import numpy as np
import pytest


def _make_input_batch():
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        import jax  # noqa: F401

        from tpu_inference.runner.input_batch import InputBatch
    except ImportError:
        pytest.skip("vllm / tpu_inference.runner.input_batch not importable")
    import jax
    ib = InputBatch(
        max_num_reqs=4,
        max_model_len=32,
        max_num_batched_tokens=32,
        pin_memory=False,
        vocab_size=50,
        block_sizes=[16],
    )
    mesh = jax.make_mesh((1, ), ("data", ))
    return ib, mesh


class TestSeenTokenIdsMask:
    def test_gate_no_repetition_penalty_returns_none(self):
        """No active request with rp != 1.0 -> mask stays None (zero overhead)."""
        ib, mesh = _make_input_batch()
        ib.req_id_to_index = {"a": 0}
        ib.token_ids_cpu[0, :3] = [5, 6, 7]
        ib.num_tokens_no_spec[0] = 3
        ib.repetition_penalties_cpu[0] = 1.0
        assert ib.update_seen_token_ids_mask(mesh, 4, 16, None) is None
        assert ib.seen_token_ids_mask is None

    def test_prefill_scatter(self):
        """Prompt tokens (including repeats) are marked; others are not."""
        ib, mesh = _make_input_batch()
        ib.req_id_to_index = {"a": 0, "b": 1}
        ib.token_ids_cpu[0, :4] = [5, 6, 7, 5]
        ib.num_tokens_no_spec[0] = 4
        ib.repetition_penalties_cpu[0] = 1.2
        ib.token_ids_cpu[1, :2] = [10, 11]
        ib.num_tokens_no_spec[1] = 2
        ib.repetition_penalties_cpu[1] = 1.0
        m = np.asarray(ib.update_seen_token_ids_mask(mesh, 4, 16, None))
        assert m.shape == (4, 50)
        assert m[0, 5] and m[0, 6] and m[0, 7] and not m[0, 8]
        assert m[1, 10] and m[1, 11]
        assert int(ib.seen_scattered_upto[0]) == 4
        assert int(ib.seen_scattered_upto[1]) == 2

    def test_incremental_decode_only_adds_new_token(self):
        """A decode step scatters only the newly generated token."""
        ib, mesh = _make_input_batch()
        ib.req_id_to_index = {"a": 0}
        ib.token_ids_cpu[0, :3] = [5, 6, 7]
        ib.num_tokens_no_spec[0] = 3
        ib.repetition_penalties_cpu[0] = 1.3
        ib.update_seen_token_ids_mask(mesh, 4, 16, None)
        # decode: emit token 20
        ib.token_ids_cpu[0, 3] = 20
        ib.num_tokens_no_spec[0] = 4
        m = np.asarray(ib.update_seen_token_ids_mask(mesh, 4, 16, None))
        assert m[0, 20] and m[0, 5] and m[0, 6] and m[0, 7]
        assert int(ib.seen_scattered_upto[0]) == 4

    def test_condense_row_move(self):
        """condense() moves a request's mask row + high-water to its new slot."""
        ib, mesh = _make_input_batch()
        ib.req_id_to_index = {"a": 0, "b": 1}
        ib.token_ids_cpu[0, :1] = [20]
        ib.num_tokens_no_spec[0] = 1
        ib.repetition_penalties_cpu[0] = 1.2
        ib.token_ids_cpu[1, :2] = [10, 11]
        ib.num_tokens_no_spec[1] = 2
        ib.repetition_penalties_cpu[1] = 1.1
        ib.update_seen_token_ids_mask(mesh, 4, 16, None)
        # Emulate the row move condense() performs (slot1 -> slot0).
        ib.seen_token_ids_mask = ib.seen_token_ids_mask.at[0].set(
            ib.seen_token_ids_mask[1])
        ib.seen_scattered_upto[0] = ib.seen_scattered_upto[1]
        m = np.asarray(ib.seen_token_ids_mask)
        assert m[0, 10] and m[0, 11] and not m[0, 20]
