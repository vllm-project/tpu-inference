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
from vllm.sampling_params import SamplingParams

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
        pooling_params=None,
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
