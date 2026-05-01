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

import numpy as np
import pytest
import torch
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from tpu_inference.runner.input_batch import CachedRequestState, InputBatch

# Default parameters for creating InputBatch instances in tests
MAX_NUM_REQS = 8
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = 2048
VOCAB_SIZE = 32000
BLOCK_SIZES = [16]


@pytest.fixture
def input_batch():
    """Provides a clean InputBatch instance for each test."""
    return InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        pin_memory=False,
        vocab_size=VOCAB_SIZE,
        block_sizes=BLOCK_SIZES,
        is_spec_decode=True,
    )


def create_dummy_request(req_id: str,
                         prompt_len: int = 10,
                         output_len: int = 5,
                         sampling_params: SamplingParams = None,
                         pooling_params: PoolingParams = None,
                         block_ids=None) -> CachedRequestState:
    """Helper function to create a CachedRequestState instance."""
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.9, top_k=50)

    prompt_token_ids = list(range(prompt_len))
    output_token_ids = list(range(prompt_len, prompt_len + output_len))

    if block_ids is None:
        # Create dummy block ids based on length
        num_blocks = (prompt_len + output_len + BLOCK_SIZES[0] -
                      1) // BLOCK_SIZES[0]
        block_ids = [[i] for i in range(1, num_blocks + 1)]

    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
        output_token_ids=output_token_ids,
    )


def test_initialization(input_batch: InputBatch):
    """Tests if the InputBatch is initialized with correct default values."""
    assert input_batch.max_num_reqs == MAX_NUM_REQS
    assert input_batch.num_reqs == 0
    assert len(input_batch.req_ids) == 0
    assert not input_batch.req_id_to_index
    assert input_batch.all_greedy
    assert input_batch.is_spec_decode


def test_add_request(input_batch: InputBatch):
    """Tests adding a single request to the batch."""
    req = create_dummy_request("req-1", prompt_len=20, output_len=4)
    input_batch.add_request(req)

    assert input_batch.num_reqs == 1
    assert "req-1" in input_batch.req_id_to_index
    assert input_batch.req_id_to_index["req-1"] == 0
    assert input_batch.req_ids == ["req-1"]
    assert len(input_batch.spec_decode_unsupported_reqs) == 0

    # Verify token data
    assert input_batch.num_prompt_tokens[0] == 20
    assert input_batch.num_tokens[0] == 24
    assert input_batch.num_tokens_no_spec[0] == 24
    expected_tokens = np.array(req.prompt_token_ids + req.output_token_ids)
    np.testing.assert_array_equal(input_batch.token_ids_cpu[0, :24],
                                  expected_tokens)

    # Verify sampling params
    assert input_batch.temperature_cpu[0] == 0.8
    assert input_batch.top_p_cpu[0] == 0.9
    assert input_batch.top_k_cpu[0] == 50


def test_add_multiple_requests(input_batch: InputBatch):
    """Tests adding multiple requests and checks their indices."""
    req1 = create_dummy_request("req-1")
    req2 = create_dummy_request("req-2")

    input_batch.add_request(req1)
    input_batch.add_request(req2)

    assert input_batch.num_reqs == 2
    assert input_batch.req_ids == ["req-1", "req-2"]
    assert input_batch.req_id_to_index["req-1"] == 0
    assert input_batch.req_id_to_index["req-2"] == 1
    assert input_batch.num_tokens[1] == len(req2.prompt_token_ids) + len(
        req2.output_token_ids)
    assert input_batch.num_tokens_no_spec[1] == len(
        req2.prompt_token_ids) + len(req2.output_token_ids)


def test_remove_request(input_batch: InputBatch):
    """Tests removing a request, which leaves a gap in the batch."""
    req1 = create_dummy_request("req-1")
    req2 = create_dummy_request("req-2")
    input_batch.add_request(req1)
    input_batch.add_request(req2)

    removed_index = input_batch.remove_request("req-1")

    assert removed_index == 0
    assert input_batch.num_reqs == 1
    assert "req-1" not in input_batch.req_id_to_index
    assert input_batch._req_ids[0] is None  # Slot is now empty
    assert input_batch._req_ids[1] == "req-2"
    assert "req-1" not in input_batch.greedy_reqs


def test_condense(input_batch: InputBatch):
    """Tests condensing the batch after removing requests."""
    reqs = [create_dummy_request(f"req-{i}") for i in range(4)]
    for req in reqs:
        input_batch.add_request(req)

    # Remove requests from the middle and start
    input_batch.remove_request("req-1")
    input_batch.remove_request("req-0")

    # Before condense: [None, None, "req-2", "req-3"]
    assert input_batch._req_ids[0] is None
    assert input_batch._req_ids[1] is None
    assert input_batch.num_reqs == 2

    # Condense should move req-2 and req-3 to the front
    empty_indices = sorted([0, 1], reverse=True)
    input_batch.condense(empty_indices)

    assert input_batch.num_reqs == 2
    assert len(input_batch.req_ids) == 2
    assert input_batch.req_ids == ["req-3", "req-2"]
    assert input_batch.req_id_to_index["req-2"] == 1
    assert input_batch.req_id_to_index["req-3"] == 0

    # Check if a property was moved correctly
    assert input_batch.num_tokens[0] == len(reqs[2].prompt_token_ids) + len(
        reqs[2].output_token_ids)
    assert input_batch.num_tokens_no_spec[0] == len(
        reqs[2].prompt_token_ids) + len(reqs[2].output_token_ids)


def test_swap_states(input_batch: InputBatch):
    """Tests swapping the states of two requests."""
    req1 = create_dummy_request("req-1", prompt_len=10, output_len=1)
    req2 = create_dummy_request("req-2",
                                prompt_len=20,
                                output_len=2,
                                sampling_params=SamplingParams(top_p=0.5))

    input_batch.add_request(req1)
    input_batch.add_request(req2)

    # Capture states before swap
    req1_tokens_before = input_batch.token_ids_cpu[0].copy()
    req2_tokens_before = input_batch.token_ids_cpu[1].copy()
    req1_top_p_before = input_batch.top_p_cpu[0]
    req2_top_p_before = input_batch.top_p_cpu[1]

    input_batch.swap_states(0, 1)

    # Check IDs and mappings
    assert input_batch.req_ids == ["req-2", "req-1"]
    assert input_batch.req_id_to_index["req-1"] == 1
    assert input_batch.req_id_to_index["req-2"] == 0

    # Check swapped data
    assert input_batch.top_p_cpu[0] == req2_top_p_before
    assert input_batch.top_p_cpu[1] == req1_top_p_before
    np.testing.assert_array_equal(input_batch.token_ids_cpu[0],
                                  req2_tokens_before)
    np.testing.assert_array_equal(input_batch.token_ids_cpu[1],
                                  req1_tokens_before)


def test_swap_states_only_swaps_active_tokens(input_batch: InputBatch):
    """Tests that swap_states only swaps up to max_active_token_count columns,
    leaving data beyond that range untouched (optimization correctness)."""
    # Create requests with very different lengths.
    req_short = create_dummy_request("req-short", prompt_len=5, output_len=1)
    req_long = create_dummy_request("req-long", prompt_len=50, output_len=10)

    input_batch.add_request(req_short)
    input_batch.add_request(req_long)

    # max_active_token_count = max(6, 60) = 60
    max_active = max(
        int(input_batch.num_tokens[0]),
        int(input_batch.num_tokens[1]),
    )
    assert max_active == 60

    # Capture only the active region before swap.
    short_active_before = input_batch.token_ids_cpu[0, :max_active].copy()
    long_active_before = input_batch.token_ids_cpu[1, :max_active].copy()

    input_batch.swap_states(0, 1)

    # Active region should be swapped.
    np.testing.assert_array_equal(input_batch.token_ids_cpu[0, :max_active],
                                  long_active_before)
    np.testing.assert_array_equal(input_batch.token_ids_cpu[1, :max_active],
                                  short_active_before)

    # num_tokens should also be swapped correctly.
    assert input_batch.num_tokens[0] == 60
    assert input_batch.num_tokens[1] == 6
    assert input_batch.num_tokens_no_spec[0] == 60
    assert input_batch.num_tokens_no_spec[1] == 6


def test_mamba_state_indices_unique_per_request(input_batch: InputBatch):
    """Two concurrent requests must never receive the same mamba slot id —
    if they did, the GDN op would write both requests' recurrent state
    into the same physical cache slot and corrupt one of them. Also
    verifies slot 0 is reserved (vLLM's null block convention)."""
    for i in range(4):
        input_batch.add_request(create_dummy_request(f"req-{i}"))

    slots = input_batch.mamba_state_indices_cpu[:input_batch.num_reqs].tolist()
    assert len(set(slots)) == len(slots), \
        f"Slots not unique across concurrent requests: {slots}"
    assert all(s >= 1 for s in slots), \
        f"Slot 0 is the null block and must not be assigned: {slots}"


def test_mamba_state_indices_freed_on_remove(input_batch: InputBatch):
    """Removing a request must return its slot id to the free pool so the
    next add reuses it (rather than running the pool dry)."""
    req = create_dummy_request("req-0")
    input_batch.add_request(req)
    slot_first = int(input_batch.mamba_state_indices_cpu[0])

    input_batch.remove_request("req-0")
    # condense() with empty_indices=[0] would early-return because num_reqs==0,
    # so we just verify the slot is back in the pool.
    assert slot_first in input_batch._free_mamba_slots

    new_req = create_dummy_request("req-new")
    input_batch.add_request(new_req)
    # The most-recently-freed slot is at the end of the free list — pop()
    # returns it, so the new request gets the same id back.
    assert int(input_batch.mamba_state_indices_cpu[0]) == slot_first


def test_mamba_state_indices_follow_condense(input_batch: InputBatch):
    """When condense moves a request to a different persistent-batch slot,
    its mamba state id must follow it — otherwise the GDN op reads stale
    state from the recurrent_state cache. This is the bug that broke gsm8k
    accuracy on Qwen3.5-397B (only triggered when requests finish out of
    order, which `--ignore-eos` benchmarks suppress)."""
    reqs = [create_dummy_request(f"req-{i}") for i in range(4)]
    for req in reqs:
        input_batch.add_request(req)

    # Snapshot the slot id assigned to each request before any churn.
    slot_for_req = {
        rid: int(input_batch.mamba_state_indices_cpu[idx])
        for rid, idx in input_batch.req_id_to_index.items()
    }

    # Remove the lower-indexed requests so condense has to move the higher
    # ones down: [req-0, req-1, req-2, req-3] → [req-3, req-2, _, _].
    input_batch.remove_request("req-0")
    input_batch.remove_request("req-1")
    input_batch.condense(sorted([0, 1], reverse=True))

    # After condense, indexing-by-persistent-slot must yield the same slot
    # id each request had before the move.
    assert input_batch.req_id_to_index["req-3"] == 0
    assert input_batch.req_id_to_index["req-2"] == 1
    assert int(input_batch.mamba_state_indices_cpu[0]) == slot_for_req["req-3"]
    assert int(input_batch.mamba_state_indices_cpu[1]) == slot_for_req["req-2"]


def test_mamba_state_indices_no_duplicate_in_padded_tail(
        input_batch: InputBatch):
    """The padded tail `mamba_state_indices_cpu[num_reqs:]` must not contain
    any slot id that is also used by an active request.

    If it did, the GDN op's `recurrent_state.at[state_indices].set(...)`
    scatter would have duplicate destination indices: the active position
    writes the new state, the padded position writes back the stale
    pre-update state, and JAX's scatter-with-duplicates is undefined on
    XLA. The active request's freshly computed state silently loses the
    race and the request decodes from corrupted recurrent state.

    Reproduces the production symptom: outputs look fine on a fresh batch,
    then turn to garbage as soon as the persistent batch starts churning
    (out-of-order completions → `condense` → stale slot ids in the tail).
    """
    # Fill, then remove a mix of low and high indices so condense actually
    # has to move requests downward (the case that left stale ids before
    # the fix).
    for i in range(MAX_NUM_REQS):
        input_batch.add_request(create_dummy_request(f"req-{i}"))
    for req_id in ("req-0", "req-2", "req-5"):
        input_batch.remove_request(req_id)
    input_batch.condense(sorted([0, 2, 5], reverse=True))

    num_reqs = input_batch.num_reqs
    active = set(input_batch.mamba_state_indices_cpu[:num_reqs].tolist())
    tail = set(input_batch.mamba_state_indices_cpu[num_reqs:].tolist())
    assert active.isdisjoint(tail), (
        "Padded tail still references active slot ids "
        f"(overlap={active & tail}); the GDN scatter will alias.")
    # Tail should be the null slot (0) so the scatter folds harmlessly into
    # the reserved null block.
    assert tail <= {0}, f"Padded tail contains non-null slot ids: {tail}"


def test_mamba_state_indices_remove_clears_position(input_batch: InputBatch):
    """remove_request must clear the slot id at the vacated position so it
    does not survive into the padded tail (the source of the scatter-alias
    bug fixed alongside the trailing-tail invariant)."""
    input_batch.add_request(create_dummy_request("req-0"))
    input_batch.add_request(create_dummy_request("req-1"))

    input_batch.remove_request("req-1")
    # Without condense the position is still in the active range from
    # `mamba_state_indices_cpu`'s perspective, but `req_id_to_index` no
    # longer references it; the field at that index must read as null
    # so the next add doesn't observe a leftover from a prior occupant.
    assert int(input_batch.mamba_state_indices_cpu[1]) == 0


def test_mamba_state_indices_swap(input_batch: InputBatch):
    """swap_states swaps the request mappings, so the slot ids must swap too."""
    input_batch.add_request(create_dummy_request("req-0"))
    input_batch.add_request(create_dummy_request("req-1"))
    slot0 = int(input_batch.mamba_state_indices_cpu[0])
    slot1 = int(input_batch.mamba_state_indices_cpu[1])

    input_batch.swap_states(0, 1)

    assert int(input_batch.mamba_state_indices_cpu[0]) == slot1
    assert int(input_batch.mamba_state_indices_cpu[1]) == slot0


def test_all_greedy_property(input_batch: InputBatch):
    """Tests the `all_greedy` property."""
    # Initially true
    assert input_batch.all_greedy

    # Add a greedy request, still true
    req_greedy = create_dummy_request(
        "req-g", sampling_params=SamplingParams(temperature=0.0))
    input_batch.add_request(req_greedy)
    assert input_batch.all_greedy

    # Manually add a random request for testing purposes
    input_batch.random_reqs.add("req-r")
    assert not input_batch.all_greedy

    # Remove it, should be true again
    input_batch.random_reqs.remove("req-r")
    assert input_batch.all_greedy


def test_get_pooling_metadata(input_batch: InputBatch):
    """Tests the get_pooling_metadata interface"""

    def states_eq(a: PoolingStates, b: PoolingStates):
        checks = [
            len(a.hidden_states_cache) == len(b.hidden_states_cache),
            all(
                torch.equal(x, y)
                for x, y in zip(a.hidden_states_cache, b.hidden_states_cache)),
        ]
        return all(checks)

    def meta_eq(a: PoolingMetadata, b: PoolingMetadata):
        assert a.prompt_token_ids is None and b.prompt_token_ids is None
        checks = [
            torch.equal(a.prompt_lens, b.prompt_lens),
            len(a.pooling_params) == len(b.pooling_params),
            len(a.pooling_states) == len(b.pooling_states),
            all(x == y for x, y in zip(a.pooling_params, b.pooling_params)),
            all(
                states_eq(x, y)
                for x, y in zip(a.pooling_states, b.pooling_states)),
            # ignore pooling cursor
        ]
        return all(checks)

    assert meta_eq(
        input_batch.get_pooling_metadata(),
        PoolingMetadata(
            prompt_lens=torch.tensor([], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[],
            pooling_states=[],
        ),
    ), "Initial value should be all empty"

    # Just some task value to pass assertion in PoolingMetadata.__post_init__
    pooling_param = PoolingParams(task="embed")
    pooling_state = PoolingStates()

    req_0 = create_dummy_request(
        "req-0",
        prompt_len=10,
        pooling_params=pooling_param,
    )
    input_batch.add_request(req_0)
    assert meta_eq(
        input_batch.get_pooling_metadata(),
        PoolingMetadata(
            prompt_lens=torch.tensor([10], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[pooling_param],
            pooling_states=[pooling_state],
        ),
    ), "Pooling states is populated by InputBatch object it self"

    input_batch.remove_request("req-0")
    assert meta_eq(
        input_batch.get_pooling_metadata(),
        PoolingMetadata(
            prompt_lens=torch.tensor([], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[],
            pooling_states=[],
        ),
    ), "After remove, back to empty state."
