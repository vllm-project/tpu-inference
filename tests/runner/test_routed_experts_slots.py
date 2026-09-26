# Copyright 2026 Google LLC
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
"""Unit tests for routed-experts physical slot reconstruction.

The routed experts of a step are stored into the worker-side slot buffer
(``TPUAuxOutputWorker``) keyed by physical KV-cache slot, and read back
block-relative from position 0. So each step entry must place a request's
tokens at their true absolute positions. ``reconstruct_slots`` takes an
explicit ``start_pos`` (the two call sites have different token contracts);
``_routed_experts_step_entries`` sources it from ``scheduler_output`` (the
pre-step computed-token count), which is correct on both the sync and async
output paths -- unlike ``req_state.num_computed_tokens``, whose advancement
timing differs between them.

The slots must also be keyed by the full-attention KV-cache group
(``get_routed_experts_attn_gid``). That is not always group 0: KV-cache groups
follow model layer order, so a hybrid model whose first layer is linear
attention (Qwen3.5, ``full_attention_interval=4``) gets
``[linear, linear, linear, full]``.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata, AuxOutputSchedulerConnector)

from tpu_inference.runner.routed_experts import (RoutedExpertsStepEntry,
                                                 TPUAuxOutputWorker,
                                                 reconstruct_slots)
from tpu_inference.runner.tpu_runner import (
    _routed_experts_step_entries, _snapshot_block_ids_for_routed_experts)

# Block ID owned by the non-full-attention groups in these tests.
_WRONG_GROUP_BLOCK_ID = 7


def _req(num_computed_tokens, block_ids, num_groups=1, attn_gid=0):
    # Only num_computed_tokens and block_ids are read. CachedRequestState
    # stores one list of block IDs per KV-cache group; `block_ids` here are the
    # full-attention group's, placed at `attn_gid`. The other groups get a
    # distinct, deliberately wrong block ID so that reading the wrong group is
    # visible in the resulting slots rather than silently plausible.
    groups = [[_WRONG_GROUP_BLOCK_ID] for _ in range(num_groups)]
    groups[attn_gid] = block_ids
    return SimpleNamespace(num_computed_tokens=num_computed_tokens,
                           block_ids=groups)


class TestReconstructSlots:
    """Slot math given an explicit absolute start position."""

    def test_slots_are_block_relative_from_start_pos(self):
        # 5 tokens starting at absolute position 0 in block 1 -> slots 16..20.
        slots = reconstruct_slots([1],
                                  num_tokens=5,
                                  block_size=16,
                                  start_pos=0)
        np.testing.assert_array_equal(
            slots, np.array([16, 17, 18, 19, 20], dtype=np.int32))

    def test_single_token_at_offset(self):
        # One token at absolute position 5 in block 1 -> slot 21.
        slots = reconstruct_slots([1],
                                  num_tokens=1,
                                  block_size=16,
                                  start_pos=5)
        np.testing.assert_array_equal(slots, np.array([21], dtype=np.int32))

    def test_spanning_multiple_blocks(self):
        # 20 tokens from position 0 across blocks [1, 2], block_size 16.
        slots = reconstruct_slots([1, 2],
                                  num_tokens=20,
                                  block_size=16,
                                  start_pos=0)
        expected = np.concatenate(
            [1 * 16 + np.arange(16), 2 * 16 + np.arange(4)]).astype(np.int32)
        np.testing.assert_array_equal(slots, expected)

    def test_zero_tokens_returns_empty(self):
        slots = reconstruct_slots([1],
                                  num_tokens=0,
                                  block_size=16,
                                  start_pos=0)
        assert slots.size == 0

    def test_negative_start_pos_raises(self):
        # A negative start_pos would silently produce wrong slots via numpy
        # negative indexing; the helper must reject it loudly instead.
        with pytest.raises(AssertionError):
            reconstruct_slots([1], num_tokens=5, block_size=16, start_pos=-2)


class TestSnapshotBlockIds:
    """The snapshot both selects the KV-cache group and decouples slot
    reconstruction from the lifetime of the request."""

    def test_snapshot_copies_the_block_list(self):
        req_state = _req(5, [1, 2])
        runner = SimpleNamespace(requests={"r0": req_state})

        snapshot = _snapshot_block_ids_for_routed_experts(runner, ["r0"], 0)
        # Mutating the live state afterwards must not change the snapshot.
        req_state.block_ids[0].append(99)

        assert snapshot == {"r0": [1, 2]}

    def test_absent_request_yields_empty_list(self):
        runner = SimpleNamespace(requests={})
        assert _snapshot_block_ids_for_routed_experts(runner, ["gone"], 0) == {
            "gone": []
        }

    def test_hybrid_model_snapshots_full_attention_group(self):
        # Qwen3.5-shaped hybrid layout: 4 KV-cache groups, full attention last
        # (layer 0 is linear attention, so the groups are
        # [linear, linear, linear, full]). Keying the slots by group 0 wrote
        # the prompt's routing to the linear-attention blocks, where the
        # scheduler -- which reads back through the full-attention group --
        # never looks, so every prompt row came back zero-filled.
        req_state = _req(5, [1], num_groups=4, attn_gid=3)
        runner = SimpleNamespace(requests={"r0": req_state})

        snapshot = _snapshot_block_ids_for_routed_experts(runner, ["r0"], 3)

        assert snapshot == {"r0": [1]}  # group 0 would have given [7]

    def test_out_of_range_group_raises(self):
        # Silently falling back to another group would reintroduce the bug in
        # its hardest-to-spot form, so this must fail loudly.
        runner = SimpleNamespace(requests={"r0": _req(5, [1])})
        with pytest.raises(AssertionError):
            _snapshot_block_ids_for_routed_experts(runner, ["r0"], 3)


def _new_req_sched(req_id, num_computed_tokens):
    return SimpleNamespace(req_id=req_id,
                           num_computed_tokens=num_computed_tokens)


def _empty_cached():
    return SimpleNamespace(req_ids=[], num_computed_tokens=[])


def _entry_slots(entry, block_size):
    return reconstruct_slots(entry.block_ids, len(entry.rows), block_size,
                             entry.token_start)


class TestRoutedExpertsStepEntries:
    """End-to-end slot derivation for the per-step routed-experts entries
    (the caller that had the slot-start bug). The chunk start is sourced from
    scheduler_output (pre-step), NOT from req_state.num_computed_tokens, whose
    advancement timing differs between the sync and async output paths."""

    def test_new_request_prefill_slots_block_relative(self):
        req_id = "req0"
        n = 5  # 5-token prompt prefill
        block_size = 16
        num_layers, top_k = 2, 4
        # Set req_state.num_computed_tokens to a wrong sentinel: the result must
        # be derived from scheduler_output (chunk start 0), not this field.
        req_state = _req(num_computed_tokens=99999, block_ids=[1])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 routed_experts_attn_gid=0,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[_new_req_sched(req_id, 0)],
            scheduled_cached_reqs=_empty_cached(),
        )
        # Distinct per-(layer, token) values to detect any mis-mapping.
        expert_indices_cpu = np.arange(num_layers * n * top_k,
                                       dtype=np.int32).reshape(
                                           num_layers, n, top_k)

        [entry] = _routed_experts_step_entries(
            runner=runner,
            scheduler_output=scheduler_output,
            expert_indices_cpu=expert_indices_cpu,
            req_ids=[req_id],
            sampled_token_ids=[[0]],
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=n,
            block_ids_by_req=_snapshot_block_ids_for_routed_experts(
                runner, [req_id], 0),
        )

        # Prompt tokens at positions 0..4 in block 1 -> slots 16..20.
        # (The buggy num_computed_tokens-based code underflowed to -5 on the
        # sync path and produced 21..25 on the async path; sourcing the chunk
        # start from scheduler_output is correct for both.)
        np.testing.assert_array_equal(
            _entry_slots(entry, block_size),
            np.array([16, 17, 18, 19, 20], dtype=np.int32))
        # Rows are the model output transposed to (tokens, layers, top_k).
        np.testing.assert_array_equal(entry.rows,
                                      expert_indices_cpu.transpose(1, 0, 2))
        assert entry.num_accepted == n

    def test_cached_request_chunk_slots_use_pre_step_start(self):
        # A cached/running request whose chunk starts at absolute position 20.
        req_id = "reqC"
        n = 1
        block_size = 16
        num_layers, top_k = 2, 4
        req_state = _req(num_computed_tokens=99999, block_ids=[1, 2])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 routed_experts_attn_gid=0,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[req_id],
                                                  num_computed_tokens=[20]),
        )
        expert_indices_cpu = np.arange(num_layers * n * top_k,
                                       dtype=np.int32).reshape(
                                           num_layers, n, top_k)

        [entry] = _routed_experts_step_entries(
            runner=runner,
            scheduler_output=scheduler_output,
            expert_indices_cpu=expert_indices_cpu,
            req_ids=[req_id],
            sampled_token_ids=[[0]],
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=n,
            block_ids_by_req=_snapshot_block_ids_for_routed_experts(
                runner, [req_id], 0),
        )

        # Position 20 -> block_ids[20 // 16 = 1] = block 2 -> 2*16 + 20%16 = 36.
        np.testing.assert_array_equal(_entry_slots(entry, block_size),
                                      np.array([36], dtype=np.int32))

    def test_hybrid_prefill_slots_use_full_attention_group(self):
        """Regression: on a hybrid model the prompt's slots must be keyed by
        the full-attention KV-cache group, not by group 0.

        Qwen3.5 has 3 linear-attention groups ahead of its full-attention
        group, so the hardcoded group 0 sent every prompt row to the
        linear-attention blocks. The scheduler reads the completed prefill
        back through the full-attention blocks, so `routed_experts[:P]` came
        back all zeros -- or, once block IDs were recycled across groups,
        holding another request's routing.
        """
        req_id = "req0"
        n = 5
        block_size = 16
        num_layers, top_k = 2, 4
        attn_gid = 3
        req_state = _req(num_computed_tokens=0,
                         block_ids=[1],
                         num_groups=4,
                         attn_gid=attn_gid)
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 routed_experts_attn_gid=attn_gid,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[_new_req_sched(req_id, 0)],
            scheduled_cached_reqs=_empty_cached(),
        )
        expert_indices_cpu = np.arange(num_layers * n * top_k,
                                       dtype=np.int32).reshape(
                                           num_layers, n, top_k)

        [entry] = _routed_experts_step_entries(
            runner=runner,
            scheduler_output=scheduler_output,
            expert_indices_cpu=expert_indices_cpu,
            req_ids=[req_id],
            sampled_token_ids=[[0]],
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=n,
            block_ids_by_req=_snapshot_block_ids_for_routed_experts(
                runner, [req_id], attn_gid),
        )

        # Full-attention block 1 -> slots 16..20. Keying by group 0 would have
        # produced block 7 -> slots 112..116.
        np.testing.assert_array_equal(
            _entry_slots(entry, block_size),
            np.array([16, 17, 18, 19, 20], dtype=np.int32))

    def test_request_evicted_before_output_resolves(self):
        """Regression: the reconstruction must not depend on the request still
        being in `runner.requests`.

        With the engine's batch queue, `execute_model` for the next step runs
        `_update_states`, which pops finished requests, *before* the previous
        step's async output is resolved. Reading `runner.requests[req_id]` at
        resolution time therefore raised
        `KeyError: '<req_id>'` and killed EngineCore mid-generation -- seen with
        two concurrent requests where one finished first.
        """
        req_id = "reqGone"
        n = 1
        block_size = 16
        num_layers, top_k = 2, 4
        req_state = _req(num_computed_tokens=0, block_ids=[1, 2])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 routed_experts_attn_gid=0,
                                 requests={req_id: req_state})
        # Snapshot at dispatch, while the request is still live...
        snapshot = _snapshot_block_ids_for_routed_experts(runner, [req_id], 0)
        # ...then the request finishes and `_update_states` evicts it.
        runner.requests.pop(req_id)

        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[req_id],
                                                  num_computed_tokens=[20]),
        )
        expert_indices_cpu = np.arange(num_layers * n * top_k,
                                       dtype=np.int32).reshape(
                                           num_layers, n, top_k)

        [entry] = _routed_experts_step_entries(
            runner=runner,
            scheduler_output=scheduler_output,
            expert_indices_cpu=expert_indices_cpu,
            req_ids=[req_id],
            sampled_token_ids=[[0]],
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=n,
            block_ids_by_req=snapshot,
        )

        # Same slots as if the request were still live.
        np.testing.assert_array_equal(_entry_slots(entry, block_size),
                                      np.array([36], dtype=np.int32))


_BLOCK_SIZE = 4
_NUM_LAYERS, _TOP_K = 2, 3


def _rows(start, end):
    # Row value encodes its absolute position so misplaced rows are visible.
    return np.broadcast_to(
        np.arange(start, end, dtype=np.int32)[:, None, None],
        (end - start, _NUM_LAYERS, _TOP_K)).copy()


def _worker(num_blocks=16, dp_size=1):
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(
        get_num_experts=lambda: 8))
    return TPUAuxOutputWorker(vllm_config,
                              SimpleNamespace(num_blocks=num_blocks),
                              block_size=_BLOCK_SIZE,
                              dp_size=dp_size)


class _Request:
    """The request fields the upstream scheduler connector reads."""

    def __init__(self, req_id, num_prompt_tokens, prompt_start=0):
        self.request_id = req_id
        self.num_tokens = num_prompt_tokens
        self.num_output_tokens = 0
        self.block_hashes = []
        self.sampling_params = SimpleNamespace(
            routed_experts_prompt_start=prompt_start)
        self.finished = False

    def is_finished(self):
        return self.finished

    def append_token(self):
        self.num_tokens += 1
        self.num_output_tokens += 1


def _schedule(connector, requests, num_scheduled):
    return connector.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens=num_scheduled),
        {r.request_id: r
         for r in requests})


def _entry(req_id, start, end, num_accepted, block_ids, dp_rank=0):
    return RoutedExpertsStepEntry(req_id=req_id,
                                  rows=_rows(start, end),
                                  token_start=start,
                                  num_accepted=num_accepted,
                                  block_ids=block_ids,
                                  dp_rank=dp_rank)


class TestTPUAuxOutputWorker:
    """The worker output must satisfy the upstream scheduler connector's
    ``take_output``: rows cover [emit start, num_tokens - 1)."""

    def test_chunked_prefill_then_decode(self):
        connector = AuxOutputSchedulerConnector()
        worker = _worker()
        req = _Request("r0", num_prompt_tokens=6)
        blocks = [1, 2]

        # Chunk 1: positions 0..3, no token sampled -> nothing emitted.
        worker.begin_step(_schedule(connector, [req], {"r0": 4}))
        assert worker.process_step(["r0"],
                                   [_entry("r0", 0, 4, 0, blocks)]) == {}

        # Chunk 2 completes the prompt and samples a token.
        worker.begin_step(_schedule(connector, [req], {"r0": 2}))
        out = worker.process_step(["r0"], [_entry("r0", 4, 6, 2, blocks)])
        req.append_token()
        np.testing.assert_array_equal(connector.take_output(req, out),
                                      _rows(0, 6))

        # Decode position 6 emits only its own row.
        worker.begin_step(_schedule(connector, [req], {"r0": 1}))
        out = worker.process_step(["r0"], [_entry("r0", 6, 7, 1, blocks)])
        req.append_token()
        assert out["r0"].token_start == 6
        np.testing.assert_array_equal(connector.take_output(req, out),
                                      _rows(6, 7))

    def test_prompt_start_skips_leading_rows(self):
        connector = AuxOutputSchedulerConnector()
        worker = _worker()
        req = _Request("r0", num_prompt_tokens=5, prompt_start=3)

        worker.begin_step(_schedule(connector, [req], {"r0": 5}))
        out = worker.process_step(["r0"], [_entry("r0", 0, 5, 5, [1, 2])])
        req.append_token()
        np.testing.assert_array_equal(connector.take_output(req, out),
                                      _rows(3, 5))

    def test_prefix_hit_reads_rows_computed_by_earlier_request(self):
        connector = AuxOutputSchedulerConnector()
        worker = _worker()
        a = _Request("a", num_prompt_tokens=4)
        worker.begin_step(_schedule(connector, [a], {"a": 4}))
        worker.process_step(["a"], [_entry("a", 0, 4, 4, [3])])

        # b hits a's cached block 3 and only executes position 4.
        b = _Request("b", num_prompt_tokens=5)
        worker.begin_step(_schedule(connector, [a, b], {"b": 1}))
        out = worker.process_step(["b"], [_entry("b", 4, 5, 1, [3, 5])])
        b.append_token()
        np.testing.assert_array_equal(connector.take_output(b, out),
                                      _rows(0, 5))

    def test_spec_decode_emits_only_accepted_rows(self):
        worker = _worker()
        worker.begin_step(AuxOutputConnectorMetadata(0, {"r0": 4}, {}, ()))
        # Four scheduled positions, two accepted.
        out = worker.process_step(["r0"], [_entry("r0", 4, 8, 2, [1, 2])])
        assert out["r0"].token_start == 4
        np.testing.assert_array_equal(out["r0"].rows, _rows(4, 6))

    def test_finish_waits_for_in_flight_output(self):
        # Async scheduling: step N+1's metadata finishes the request before
        # step N's output is processed.
        worker = _worker()
        worker.begin_step(AuxOutputConnectorMetadata(0, {"r0": 0}, {}, ()))
        worker.begin_step(AuxOutputConnectorMetadata(0, {}, {}, ("r0", )))

        out = worker.process_step(["r0"], [_entry("r0", 0, 3, 3, [1])])
        np.testing.assert_array_equal(out["r0"].rows, _rows(0, 3))
        assert "r0" not in worker._requests

    def test_dp_ranks_keep_separate_slots(self):
        # Block IDs are per-rank local, so the same ID on two ranks must not
        # collide.
        worker = _worker(dp_size=2)
        worker.begin_step(
            AuxOutputConnectorMetadata(0, {
                "x": 0,
                "y": 0
            }, {}, ()))
        out = worker.process_step(["x", "y"], [
            _entry("x", 0, 2, 2, [1], dp_rank=0),
            RoutedExpertsStepEntry("y", _rows(10, 12), 0, 2, [1], dp_rank=1),
        ])
        np.testing.assert_array_equal(out["x"].rows, _rows(0, 2))
        np.testing.assert_array_equal(out["y"].rows, _rows(10, 12))

    def test_generation_change_resets_state(self):
        worker = _worker()
        worker.begin_step(AuxOutputConnectorMetadata(0, {"r0": 0}, {}, ()))
        worker.process_step(["r0"], [_entry("r0", 0, 2, 0, [1])])

        worker.begin_step(AuxOutputConnectorMetadata(1, {"r1": 0}, {}, ()))
        assert set(worker._requests) == {"r1"}
