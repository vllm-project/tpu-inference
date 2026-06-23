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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tpu_inference.spec_decode.jax.dflash import DFlashProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    m = jax.sharding.Mesh(
        device_mesh,
        axis_names=('data', 'attn_dp', 'expert', 'model'),
    )
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
def test_propose_uses_target_model_logits():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2
    proposer.block_size = proposer.num_speculative_tokens + 1  # 3

    # mock model_fn to return a dummy hidden_states tensor
    # shape: (num_reqs * block_size, hidden_size) = (1 * 3, 8) = (3, 8)
    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    proposer.model_fn = lambda state, kv_caches, input_ids, target_hidden_states, attn_metadata: (
        kv_caches, hidden_states, None, None)

    call_record = {}

    def fake_compute_logits_fn(state, hidden_states, lora_metadata):
        call_record["shape"] = hidden_states.shape
        # return logits of shape (2, 3) (matching flattened 2 draft tokens, 3 vocab size)
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]], dtype=jnp.float32)

    proposer.compute_logits_fn = fake_compute_logits_fn

    # Call JITted _propose
    _, draft_token_ids = proposer._propose(
        state_leaves=None,
        kv_caches=[],
        input_ids=None,
        attn_metadata=None,
        target_hidden_states=None,
    )

    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([[1, 0]], dtype=np.int32))
    assert call_record["shape"] == (2, 8)


def test_propose_returns_2d_int_ids():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2
    proposer.block_size = proposer.num_speculative_tokens + 1  # 3

    hidden_states = jnp.ones((3, 4), dtype=jnp.bfloat16)
    proposer.model_fn = lambda state, kv_caches, input_ids, target_hidden_states, attn_metadata: (
        kv_caches, hidden_states, None, None)

    proposer.compute_logits_fn = lambda _state, _hidden, _lora: jnp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    # Call _propose
    _, draft_token_ids = proposer._propose(
        state_leaves=None,
        kv_caches=[],
        input_ids=None,
        attn_metadata=None,
        target_hidden_states=None,
    )

    assert draft_token_ids.ndim == 2
    assert draft_token_ids.shape == (1, 2)
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


def test_build_noise_block_batched():
    proposer = object.__new__(DFlashProposer)

    # 2 requests in batch
    seq_lens = jnp.array([10, 20], dtype=jnp.int32)
    next_token_ids = jnp.array([100, 200], dtype=jnp.int32)
    mask_token_id = 0
    block_size = 3

    noise_ids, noise_positions = proposer._build_noise_block(
        seq_lens, next_token_ids, mask_token_id, block_size)

    # The output is expected to be flattened across the batch.
    assert noise_ids.shape == (6, )
    assert noise_positions.shape == (6, )

    # Check input ids are padded with mask_token_id correctly
    np.testing.assert_array_equal(
        np.asarray(noise_ids), np.array([100, 0, 0, 200, 0, 0],
                                        dtype=np.int32))

    # Check absolute position assignments per request
    np.testing.assert_array_equal(
        np.asarray(noise_positions),
        np.array([10, 11, 12, 20, 21, 22], dtype=np.int32))
