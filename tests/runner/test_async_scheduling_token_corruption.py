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
"""Unit tests for async scheduling placeholder reconciliation in TPUModelRunner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager
from tpu_inference.runner.tpu_runner import AsyncPreResults, TPUModelRunner


def _make_cached_req(
    req_id: str,
    prompt_ids: list[int],
    output_ids: list[int],
    block_ids: list[int] | None = None,
) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt_ids),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=[list(block_ids or [1])],
        num_computed_tokens=(len(prompt_ids) + len(output_ids) -
                             1 if output_ids else len(prompt_ids)),
        prefill_token_ids=list(prompt_ids) + list(output_ids),
        lora_request=None,
        output_token_ids=list(output_ids),
    )


def _make_minimal_runner(
    requests: dict[str, CachedRequestState],
    *,
    async_scheduling: bool = True,
    max_num_reqs: int = 8,
    max_model_len: int = 64,
) -> TPUModelRunner:
    """Constructs a lightweight TPUModelRunner without TPU device init."""
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.requests = requests
    runner.max_model_len = max_model_len
    runner.input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len * max_num_reqs,
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
    )
    for req_state in requests.values():
        runner.input_batch.add_request(req_state)
    runner.encoder_cache = {}
    runner.uses_mrope = False
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace())
    runner.is_last_rank = True
    runner.scheduler_config = SimpleNamespace(
        async_scheduling=async_scheduling)
    runner.persistent_batch_manager = PersistentBatchManager(
        runner.requests,
        runner.input_batch,
        runner.encoder_cache,
        runner.uses_mrope,
        runner.model_config,
        runner.is_last_rank,
    )
    runner.aux_output_worker = None
    runner.get_mrope_input_positions_fn = None
    runner.speculative_config = None
    runner.eos_token_id = 2
    runner._pre_async_results = None
    runner.vllm_config = SimpleNamespace()
    return runner


def _make_sched_output(
    *,
    cached_req_ids: list[str] = (),
    num_computed_tokens: list[int] = (),
    num_output_tokens: list[int] = (),
    new_reqs: list | None = None,
    new_block_ids: list | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
    preempted_req_ids: set[str] | None = None,
    scheduled_spec_decode_tokens: dict[str, list[int]] | None = None,
) -> SimpleNamespace:
    cached_ids = list(cached_req_ids)
    n = len(cached_ids)
    cached_reqs = CachedRequestData(
        req_ids=cached_ids,
        new_token_ids=[[] for _ in range(n)],
        new_block_ids=(list(new_block_ids)
                       if new_block_ids is not None else [None] * n),
        num_computed_tokens=list(num_computed_tokens),
        num_output_tokens=list(num_output_tokens),
    )
    sched_map = dict(num_scheduled_tokens or {})
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs or ()),
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=sched_map,
        total_num_scheduled_tokens=sum(sched_map.values()),
        scheduled_spec_decode_tokens=dict(scheduled_spec_decode_tokens or {}),
        finished_req_ids=set(finished_req_ids or ()),
        preempted_req_ids=set(preempted_req_ids or ()),
        free_encoder_mm_hashes=[],
        assigned_dp_rank={},
        aux_output_connector_metadata=None,
    )


def _attach_prev_step_with_placeholders(
    runner: TPUModelRunner,
    req_ids: list[str],
    sampled_tokens: list[int],
    *,
    discard_indices: list[int] | None = None,
    is_continue_decode: bool = False,
) -> None:
    """Simulates step T-1 having appended placeholder 0 tokens for non-discarded reqs."""
    discard_set = set(discard_indices or ())
    pre_sched = _make_sched_output(
        num_scheduled_tokens={rid: 1
                              for rid in req_ids})
    request_seq_lens = []
    placeholder_map = {}
    for idx, rid in enumerate(req_ids):
        req_state = runner.requests[rid]
        req_idx = runner.input_batch.req_id_to_index[rid]
        start_idx = int(runner.input_batch.num_tokens_no_spec[req_idx])
        request_seq_lens.append((idx, req_state, start_idx))
        if idx not in discard_set:
            req_state.output_token_ids.append(0)
            runner.input_batch.token_ids_cpu[req_idx, start_idx] = 0
            runner.input_batch.num_tokens_no_spec[req_idx] = start_idx + 1
            runner.input_batch.num_tokens[req_idx] = start_idx + 1
            placeholder_map[rid] = idx

    runner._pre_async_results = AsyncPreResults(
        req_ids=list(req_ids),
        next_tokens=np.array(sampled_tokens, dtype=np.int32),
        request_seq_lens=request_seq_lens,
        discard_sampled_tokens_req_indices=list(discard_indices or ()),
        placeholder_req_id_to_index=placeholder_map,
        logits_indices_selector=None,
        scheduler_output=pre_sched,
        is_continue_decode=is_continue_decode,
    )


def _fake_host_extract(runner, spec_meta, next_tokens, selector, discard,
                       num_reqs):
    discard_set = set(discard)
    arr = np.asarray(next_tokens)
    return [[] if i in discard_set else [int(arr[i])] for i in range(num_reqs)]


def test_unscheduled_eviction_and_readd_preserves_sampled_token():
    """Unscheduled live req evicted from persistent batch must not retain placeholder 0."""
    r1 = _make_cached_req("r1", [1, 2, 3], [10])
    r2 = _make_cached_req("r2", [4, 5, 6], [20])
    runner = _make_minimal_runner({"r1": r1, "r2": r2})
    _attach_prev_step_with_placeholders(runner, ["r1", "r2"], [42, 99])
    assert r1.output_token_ids == [10, 0]

    # Step T: r1 is temporarily unscheduled (not finished, not preempted); r2 continues.
    step_t = _make_sched_output(
        cached_req_ids=["r2"],
        num_computed_tokens=[5],
        num_output_tokens=[2],
        num_scheduled_tokens={"r2": 1},
    )
    assert runner._should_reconcile_prev_results_before_update(step_t) is True

    class _StopAfterUpdate(Exception):
        pass

    with patch(
            "tpu_inference.runner.tpu_runner.runner_utils.host_extract_sampled_tokens",
            side_effect=_fake_host_extract,
    ), patch.object(
            TPUModelRunner,
            "_can_use_continue_decode",
            side_effect=_StopAfterUpdate,
    ):
        with pytest.raises(_StopAfterUpdate):
            TPUModelRunner._execute_model(runner, step_t)

    assert runner._pre_async_results is None
    assert "r1" not in runner.input_batch.req_id_to_index
    assert r1.output_token_ids == [10, 42]
    assert 0 not in r1.output_token_ids
    assert r2.output_token_ids == [20, 99]

    # Step T+1: r1 is re-added as a cached request; token_ids_cpu must have 42, never 0.
    step_t1 = _make_sched_output(
        cached_req_ids=["r1", "r2"],
        num_computed_tokens=[4, 5],
        num_output_tokens=[2, 2],
        num_scheduled_tokens={
            "r1": 1,
            "r2": 1
        },
    )
    runner.persistent_batch_manager.update_states(step_t1, None)
    r1_idx = runner.input_batch.req_id_to_index["r1"]
    assert runner.input_batch.token_ids_cpu[r1_idx, :5].tolist() == [
        1, 2, 3, 10, 42
    ]


def test_zero_token_paused_all_step_reconciles_before_eviction():
    """0-token PAUSED_ALL step must reconcile async results BEFORE update_states evicts reqs."""
    r1 = _make_cached_req("r1", [1, 2, 3], [10])
    r2 = _make_cached_req("r2", [4, 5, 6], [20])
    runner = _make_minimal_runner({"r1": r1, "r2": r2})
    _attach_prev_step_with_placeholders(runner, ["r1", "r2"], [42, 99])

    step_paused = _make_sched_output(
        cached_req_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
        num_scheduled_tokens={},
    )
    with patch(
            "tpu_inference.runner.tpu_runner.runner_utils.host_extract_sampled_tokens",
            side_effect=_fake_host_extract,
    ), patch(
            "tpu_inference.runner.tpu_runner.has_kv_transfer_group",
            return_value=False,
    ):
        out = TPUModelRunner._execute_model(runner, step_paused)

    assert out is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._pre_async_results is None
    assert runner.input_batch.req_id_to_index == {}
    assert r1.output_token_ids == [10, 42]
    assert r2.output_token_ids == [20, 99]


def test_drop_stale_output_rollback_preserves_last_real_token():
    """When num_output_tokens rolls back, early reconcile prevents late pop of last real token."""
    r1 = _make_cached_req("r1", [1, 2, 3], [10, 11])
    runner = _make_minimal_runner({"r1": r1})
    _attach_prev_step_with_placeholders(runner, ["r1"], [42])
    assert r1.output_token_ids == [10, 11, 0]

    # Scheduler drops step T-1's in-flight token (num_output_tokens rolls back 3 -> 2).
    step_t = _make_sched_output(
        cached_req_ids=["r1"],
        num_computed_tokens=[4],
        num_output_tokens=[2],
        num_scheduled_tokens={"r1": 1},
    )
    assert runner._should_reconcile_prev_results_before_update(step_t) is True

    class _StopAfterUpdate(Exception):
        pass

    with patch(
            "tpu_inference.runner.tpu_runner.runner_utils.host_extract_sampled_tokens",
            side_effect=_fake_host_extract,
    ), patch.object(
            TPUModelRunner,
            "_can_use_continue_decode",
            side_effect=_StopAfterUpdate,
    ):
        with pytest.raises(_StopAfterUpdate):
            TPUModelRunner._execute_model(runner, step_t)

    # Last real token 11 is preserved; dropped token 42 is trimmed cleanly;
    # _pre_async_results is cleared so sample_tokens() cannot pop 11 later.
    assert runner._pre_async_results is None
    assert r1.output_token_ids == [10, 11]
    idx = runner.input_batch.req_id_to_index["r1"]
    assert int(runner.input_batch.num_tokens[idx]) == 5
    assert int(runner.input_batch.num_tokens_no_spec[idx]) == 5
    assert runner.input_batch.token_ids_cpu[idx, :5].tolist() == [
        1, 2, 3, 10, 11
    ]


def test_steady_state_decode_skips_early_reconcile():
    """Steady-state decode must keep async reconciliation deferred (0% overlap loss)."""
    r1 = _make_cached_req("r1", [1, 2, 3], [10])
    r2 = _make_cached_req("r2", [4, 5, 6], [20])
    r3_preempt = _make_cached_req("r3_preempt", [7, 8, 9], [30])
    r4_prefill = _make_cached_req("r4_prefill", [11, 12, 13], [])
    runner = _make_minimal_runner({
        "r1": r1,
        "r2": r2,
        "r3_preempt": r3_preempt,
        "r4_prefill": r4_prefill,
    })
    # r4_prefill is chunked prefill (index 3 discarded -> no placeholder).
    _attach_prev_step_with_placeholders(
        runner,
        ["r1", "r2", "r3_preempt", "r4_prefill"],
        [42, 99, 77, 0],
        discard_indices=[3],
    )

    # Step T: r1 continues, r2 finishes, r3_preempt is preempted (state dropped),
    # r4_prefill is paused (never had a placeholder).
    step_steady = _make_sched_output(
        cached_req_ids=["r1"],
        num_computed_tokens=[5],
        num_output_tokens=[2],
        num_scheduled_tokens={"r1": 1},
        finished_req_ids={"r2"},
        preempted_req_ids={"r3_preempt"},
    )
    assert runner._should_reconcile_prev_results_before_update(
        step_steady) is False


def test_is_continue_decode_reconciles_once_without_double_pop():
    """is_continue_decode must reconcile once and propagate updated cached_reqs counts."""
    r1 = _make_cached_req("r1", [1, 2, 3], [10])
    runner = _make_minimal_runner({"r1": r1})
    _attach_prev_step_with_placeholders(runner, ["r1"], [42],
                                        is_continue_decode=True)
    runner._pre_async_results.continue_decode_actual_steps = np.array(1)

    step_t = _make_sched_output(
        cached_req_ids=["r1"],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        num_scheduled_tokens={"r1": 1},
    )
    call_count = 0

    def _fake_modify():
        nonlocal call_count
        call_count += 1
        r1.num_computed_tokens = 7
        r1.output_token_ids = [10, 42, 43, 44, 45]

    runner._modify_prev_results = _fake_modify

    class _AfterUpdateStates(Exception):
        pass

    with patch.object(
            TPUModelRunner,
            "_can_use_continue_decode",
            side_effect=_AfterUpdateStates,
    ):
        with pytest.raises(_AfterUpdateStates):
            TPUModelRunner._execute_model(runner, step_t)

    assert call_count == 1
    assert runner._pre_async_results is None
    assert step_t.scheduled_cached_reqs.num_computed_tokens == [7]
    assert step_t.scheduled_cached_reqs.num_output_tokens == [5]


def test_modify_prev_results_reconciles_alive_evicted_and_skips_dropped():
    """_modify_prev_results reconciles alive evicted reqs, skips finished/replaced reqs."""
    r_evicted = _make_cached_req("r_evicted", [1, 2, 3], [10])
    r_finished = _make_cached_req("r_finished", [4, 5, 6], [20])
    r_replaced_old = _make_cached_req("r_replaced", [7, 8, 9], [30])
    runner = _make_minimal_runner({
        "r_evicted": r_evicted,
        "r_finished": r_finished,
        "r_replaced": r_replaced_old,
    })
    _attach_prev_step_with_placeholders(
        runner,
        ["r_evicted", "r_finished", "r_replaced"],
        [42, 99, 77],
    )

    # Evict r_evicted (still live in requests), finish r_finished, replace r_replaced.
    runner.input_batch.remove_request("r_evicted")
    runner.input_batch.remove_request("r_finished")
    del runner.requests["r_finished"]
    r_replaced_new = _make_cached_req("r_replaced", [7, 8, 9], [30])
    runner.requests["r_replaced"] = r_replaced_new

    with patch(
            "tpu_inference.runner.tpu_runner.runner_utils.host_extract_sampled_tokens",
            side_effect=_fake_host_extract,
    ):
        runner._modify_prev_results()

    assert r_evicted.output_token_ids == [10, 42]
    assert r_finished.output_token_ids == [20, 0]
    assert r_replaced_new.output_token_ids == [30]
