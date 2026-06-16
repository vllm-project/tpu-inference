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
"""Unit tests for the JAX DFlash speculative decoding proposer."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tpu_inference.spec_decode.jax.dflash import DFlashProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    m = jax.sharding.Mesh(devices, axis_names=("model", ))
    return m


# ----- Mock Classes for Proposer Initialization -----
class MockHFLikeConfig:

    def __init__(self):
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.block_size = 5
        self.dflash_config = {"mask_token_id": 0}


class MockDraftModelConfig:

    def __init__(self):
        self.hf_config = MockHFLikeConfig()


class MockSpeculativeConfig:

    def __init__(self):
        self.draft_model_config = MockDraftModelConfig()
        self.method = "dflash"
        self.num_speculative_tokens = 4


class MockModelConfig:

    def __init__(self):
        self.seed = 42


class MockVllmConfig:

    def __init__(self):
        self.speculative_config = MockSpeculativeConfig()
        self.model_config = MockModelConfig()


class MockBlockTableEntry:

    def get_cpu_tensor(self):
        return np.array([1, 2, 3, 4], dtype=np.int32)


class MockInputBatch:

    def __init__(self):
        self.req_ids = ["req_1"]
        self.block_table = {0: MockBlockTableEntry(), 1: MockBlockTableEntry()}


class MockKVCacheConfig:

    def __init__(self):
        self.kv_cache_groups = [object(), object()]  # Length 2


class MockRunner:

    def __init__(self, mesh):
        self.mesh = mesh
        self.max_num_tokens = 64
        self.max_model_len = 128
        self.input_batch = MockInputBatch()
        self.kv_cache_config = MockKVCacheConfig()


class MockDraftModelInterface:

    def __init__(self):
        self.model_fn = MagicMock()
        self.compute_logits_fn = MagicMock()
        self.combine_hidden_states_fn = MagicMock()
        self.state = nnx.State({})


# ----- Existing Minimal Tests -----
def test_sample_block_draft_tokens_uses_target_model_logits():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    call_record = {}
    target_state = jnp.array(0)

    def fake_compute_logits_fn(state, hidden_states, lora_metadata):
        call_record["shape"] = hidden_states.shape
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]], dtype=jnp.float32)

    proposer.compute_logits_fn = fake_compute_logits_fn

    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    draft_token_ids = proposer._sample_block_draft_tokens(
        target_state, hidden_states)

    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([1, 0], dtype=np.int32))
    assert call_record["shape"] == (2, 8)


def test_sample_block_draft_tokens_returns_1d_int_ids():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    proposer.compute_logits_fn = lambda _state, _hidden, _lora: jnp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    hidden_states = jnp.ones((3, 4), dtype=jnp.bfloat16)
    draft_token_ids = proposer._sample_block_draft_tokens(
        jnp.array(0), hidden_states)

    assert draft_token_ids.ndim == 1
    assert draft_token_ids.shape == (2, )
    assert jnp.issubdtype(draft_token_ids.dtype, jnp.integer)


# ----- New Comprehensive Tests -----
def test_next_padded_size():
    """Asserts power-of-2 on-device padding calculations."""
    assert DFlashProposer._next_padded_size(5) == 16
    assert DFlashProposer._next_padded_size(16) == 16
    assert DFlashProposer._next_padded_size(17) == 32
    assert DFlashProposer._next_padded_size(33) == 64


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device for testing."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    m = _make_single_device_mesh()
    with jax.set_mesh(m):
        yield m


def test_build_noise_block(mesh):
    """Validates the JIT-compiled noise blocks and RoPE position generation."""
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = mesh

    seq_len_arr = jnp.array([10], dtype=jnp.int32)
    next_token_ids = jnp.array([42], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        noise_ids, noise_positions = proposer._build_noise_block(
            seq_len_arr,
            next_token_ids,
            mask_token_id=0,
            block_size=3,
        )

    assert noise_ids.shape == (3, )
    assert noise_positions.shape == (3, )
    np.testing.assert_array_equal(np.asarray(noise_ids),
                                  np.array([42, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(noise_positions),
                                  np.array([10, 11, 12], dtype=np.int32))


@patch("tpu_inference.spec_decode.jax.dflash.get_model")
def test_dflash_proposer_multi_iteration_state_flow(mock_get_model, mesh):
    """Verifies DFlashProposer's entire state-tracking, cache cropping, and padding flow across multiple speculative iterations."""
    vllm_config = MockVllmConfig()
    runner = MockRunner(mesh)

    mock_mi = MockDraftModelInterface()
    # combine_hidden_states_fn acts as identity for simplicity
    mock_mi.combine_hidden_states_fn = lambda state, raw: raw
    mock_get_model.return_value = mock_mi

    proposer = DFlashProposer(vllm_config, runner)

    with jax.set_mesh(mesh):
        proposer.load_model(target_model=None)

    # Initial proposer state
    assert proposer._ctx_len == 0
    assert proposer._cache_len == 0
    assert proposer._prev_seq_len == 0

    # ----------------- ITERATION 1: Initial prefix accepted up to length 10 -----------------
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    attn_metadata_1 = AttentionMetadata(
        input_positions=jnp.array([0]),
        block_tables=jnp.array([0]),
        seq_lens=jnp.array([10], dtype=jnp.int32),
        query_start_loc=jnp.array([0]),
        request_distribution=jnp.array([0]),
    )

    input_ids = jnp.array([0])
    aux_hidden_states_1 = (jnp.ones((10, 32), dtype=jnp.bfloat16), )
    next_token_ids_1 = jnp.array([42], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        target_hidden_1, noise_ids_1, _, draft_metadata_1 = proposer.prepare_inputs(
            attn_metadata_1,
            input_ids,
            aux_hidden_states_1,
            next_token_ids_1,
        )

    # Check outputs of Iteration 1
    new_ctx_1, cache_len_arr_1, actual_ctx_count_arr_1 = target_hidden_1
    # Padding size for 10 is 16
    assert new_ctx_1.shape == (16, 32)
    assert int(cache_len_arr_1[0]) == 0
    assert int(actual_ctx_count_arr_1[0]) == 10

    # Check proposer state updates
    assert proposer._ctx_len == 10
    assert proposer._prev_seq_len == 10

    # Check draft proposer attention metadata sharding positions
    assert noise_ids_1.shape == (proposer.block_size, )
    np.testing.assert_array_equal(
        np.asarray(draft_metadata_1.query_start_loc),
        np.array([0, proposer.block_size], dtype=np.int32),
    )

    # Mock proposer model forward pass and logits sampler
    dummy_kv = [
        jnp.zeros((1, 4, 128, 16), dtype=jnp.bfloat16)
        for _ in range(proposer.num_layers * 2)
    ]
    proposer._draft_kv_caches = dummy_kv

    # model_fn returns draft_kv_caches, hidden_states (5 tokens, 32 hidden_size), and None
    mock_mi.model_fn.return_value = (
        dummy_kv,
        jnp.ones((5, 32), dtype=jnp.bfloat16),
        None,
    )
    # compute_logits_fn returns logits for 4 speculative tokens, vocabulary size 1000
    mock_mi.compute_logits_fn.return_value = jnp.ones((4, 1000),
                                                      dtype=jnp.float32)

    # Call propose for Iteration 1
    with jax.set_mesh(mesh):
        _, draft_token_ids_1 = proposer.propose(
            kv_caches=[],
            input_ids=input_ids,
            attn_metadata=draft_metadata_1,
            last_token_indices=jnp.zeros(1),
            target_hidden_states=target_hidden_1,
        )

    assert draft_token_ids_1.shape == (1, 4)
    # Cache length is updated to prefix (10) + block size (5) = 15
    assert proposer._cache_len == 15

    # ----------------- ITERATION 2: Speculative Rejection & Cache Cropping -----------------
    # Suppose out of 4 proposed tokens, the target model accepts 3.
    # Therefore, new accepted sequence length is 10 prefix + 3 accepted = 13.
    attn_metadata_2 = AttentionMetadata(
        input_positions=jnp.array([0]),
        block_tables=jnp.array([0]),
        seq_lens=jnp.array([13], dtype=jnp.int32),
        query_start_loc=jnp.array([0]),
        request_distribution=jnp.array([0]),
    )

    # Target model passes the accumulated auxiliary hidden states (now length 13)
    aux_hidden_states_2 = (jnp.ones((13, 32), dtype=jnp.bfloat16), )
    next_token_ids_2 = jnp.array([99], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        target_hidden_2, noise_ids_2, _, _ = proposer.prepare_inputs(
            attn_metadata_2,
            input_ids,
            aux_hidden_states_2,
            next_token_ids_2,
        )

    # Check outputs of Iteration 2
    new_ctx_2, cache_len_arr_2, actual_ctx_count_arr_2 = target_hidden_2

    # CRITICAL: Verify cache cropping logic
    # The cache length must be cropped back to the PREVIOUS sequence length (10),
    # preserving the accepted prefix and discarding the rejected noise entries in [10, 15)
    assert proposer._cache_len == 10
    assert int(cache_len_arr_2[0]) == 10

    # The proposer correctly identifies that 3 new context tokens need to be sharded
    assert int(actual_ctx_count_arr_2[0]) == 3
    # Padding size for 3 is 16
    assert new_ctx_2.shape == (16, 32)

    # Check proposer state updates
    assert proposer._ctx_len == 13
    assert proposer._prev_seq_len == 13
