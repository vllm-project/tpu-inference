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
"""Unit tests for the Torchax DFlash speculative decoding proposer."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.spec_decode.torchax.dflash import DFlashTorchaxProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    m = jax.sharding.Mesh(devices, axis_names=("model", ))
    return m


# ----- Mock Classes for Proposer Initialization -----
class MockHFLikeConfig:

    def __init__(self):
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.block_size = 3
        self.dflash_config = {
            "mask_token_id": 0,
            "target_layer_ids": [0, 1],
        }


class MockDraftModelConfig:

    def __init__(self):
        self.hf_config = MockHFLikeConfig()
        self.model = "mock-draft-model"


class MockSpeculativeConfig:

    def __init__(self):
        self.draft_model_config = MockDraftModelConfig()
        self.method = "dflash"
        self.num_speculative_tokens = 2


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


# ----- Existing Isolated Tests -----
def _make_proposer(num_speculative_tokens=2, block_size=3):
    proposer = object.__new__(DFlashTorchaxProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.block_size = block_size
    proposer.mask_token_id = 0
    proposer.max_model_len = 128
    proposer._raw_hidden_dim = 16
    proposer._ctx_len = 0
    proposer._prev_seq_len = 0
    proposer._ctx_buf = jnp.zeros((128, 16), dtype=jnp.bfloat16)
    return proposer


def test_build_noise_block_shape_and_first_token():
    proposer = _make_proposer(num_speculative_tokens=2, block_size=3)
    seq_len_arr = jnp.array([10], dtype=jnp.int32)
    next_token_ids = jnp.array([42], dtype=jnp.int32)

    noise_ids, noise_positions = proposer._build_noise_block(
        seq_len_arr, next_token_ids, 0, 3)

    assert noise_ids.shape == (3, )
    assert noise_positions.shape == (3, )
    np.testing.assert_array_equal(np.asarray(noise_ids),
                                  np.array([42, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(noise_positions),
                                  np.array([10, 11, 12], dtype=np.int32))


def test_sample_block_draft_tokens_shape_and_dtype():
    proposer = _make_proposer(num_speculative_tokens=2, block_size=3)

    def fake_compute_logits(params, hidden_states, embed_weight):
        del params, embed_weight
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]], dtype=jnp.float32)

    proposer._compute_logits_fn = fake_compute_logits

    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    draft_ids = proposer._sample_block_draft_tokens({}, hidden_states,
                                                    jnp.zeros((3, 8)))

    assert draft_ids.ndim == 1
    assert draft_ids.shape == (2, )
    assert jnp.issubdtype(draft_ids.dtype, jnp.integer)
    np.testing.assert_array_equal(np.asarray(draft_ids),
                                  np.array([1, 0], dtype=np.int32))


def test_context_buffer_incremental_update():
    proposer = _make_proposer()
    assert proposer._ctx_len == 0

    raw = jnp.ones((5, 16), dtype=jnp.bfloat16) * 0.5
    proposer._prev_seq_len = 0
    seq_len = 5
    num_new = seq_len - proposer._ctx_len
    assert num_new == 5
    end = min(proposer._ctx_len + num_new, proposer.max_model_len)
    n_copy = end - proposer._ctx_len
    from jax import lax
    new_raw = raw[:n_copy].astype(jnp.bfloat16)
    proposer._ctx_buf = lax.dynamic_update_slice(proposer._ctx_buf, new_raw,
                                                 (proposer._ctx_len, 0))
    proposer._ctx_len = end

    assert proposer._ctx_len == 5
    np.testing.assert_allclose(np.asarray(proposer._ctx_buf[0, 0]),
                               0.5,
                               atol=0.01)
    np.testing.assert_allclose(np.asarray(proposer._ctx_buf[5, 0]),
                               0.0,
                               atol=0.01)


def test_context_crop_on_rejection():
    proposer = _make_proposer()
    proposer._ctx_len = 10
    proposer._prev_seq_len = 10

    seq_len = 7
    if proposer._prev_seq_len > 0 and seq_len < proposer._ctx_len:
        proposer._ctx_len = seq_len
    proposer._prev_seq_len = seq_len

    assert proposer._ctx_len == 7
    assert proposer._prev_seq_len == 7


# ----- New Comprehensive Lifecycle & Integration Tests -----
@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device for testing."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    m = _make_single_device_mesh()
    with jax.set_mesh(m):
        yield m


@patch("tpu_inference.models.torchax.dflash.DFlashTorchaxWrapper")
def test_torchax_proposer_lifecycle_flow(mock_wrapper_cls, mesh):
    """Verifies the entire lifecycle (load, precompile, prepare_inputs, propose, request-reset) of DFlashTorchaxProposer."""
    vllm_config = MockVllmConfig()
    runner = MockRunner(mesh)

    # Mock the DFlashTorchaxWrapper instance and methods
    mock_wrapper = MagicMock()
    mock_wrapper_cls.return_value = mock_wrapper

    # Mock parameters and functions returned by the wrapper
    mock_wrapper.params = {"weight": jnp.ones((2, 2))}
    mock_wrapper.embed_weight_jax = jnp.ones((10, 16), dtype=jnp.bfloat16)

    mock_draft_forward = MagicMock()
    mock_draft_forward.return_value = jnp.ones(
        (3, 16), dtype=jnp.bfloat16)  # block_size = 3
    mock_wrapper.get_draft_forward_fn.return_value = mock_draft_forward

    mock_compute_logits = MagicMock()
    mock_compute_logits.return_value = jnp.ones(
        (2, 10), dtype=jnp.float32)  # num_speculative_tokens = 2, vocab = 10
    mock_wrapper.get_compute_logits_fn.return_value = mock_compute_logits

    proposer = DFlashTorchaxProposer(vllm_config, runner)

    # 1. Test load_model
    with jax.set_mesh(mesh):
        proposer.load_model(target_model=None)

    assert proposer._wrapper is mock_wrapper
    assert proposer._params is mock_wrapper.params
    assert proposer._embed_weight is mock_wrapper.embed_weight_jax
    # Max model len is 128, next padded size is 128
    assert proposer._ctx_buf.shape == (128, 32)
    assert proposer._ctx_len == 0
    assert proposer._prev_seq_len == 0

    # 2. Test precompile
    proposer.precompile()
    # Padded shapes: 16, 32, 64, 128 (4 shapes)
    assert mock_draft_forward.call_count == 4
    # compute_logits is called within _sample_block_draft_tokens. Since _sample_block_draft_tokens
    # is JIT-compiled and the input shape is always the same, JAX caches the trace and only
    # executes the Python body (which calls the mock) once.
    assert mock_compute_logits.call_count == 1

    # Reset call counts for execution tests
    mock_draft_forward.reset_mock()
    mock_compute_logits.reset_mock()

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
    ctx_padded_1, position_ids_1, attention_mask_1 = target_hidden_1
    # Padding size for 10 is 16
    assert ctx_padded_1.shape == (16, 32)
    # Positions length = padded_ctx (16) + block_size (3) = 19
    assert position_ids_1.shape == (19, )
    assert attention_mask_1.shape == (19, )

    # Verify positions layout: prefix [0..9, 0..0] followed by noise [10..12]
    np.testing.assert_array_equal(
        np.asarray(position_ids_1[14:19]),
        np.array([0, 0, 10, 11, 12], dtype=np.int32),
    )
    # Verify attention mask layout: prefix [1 x 10, 0 x 6] followed by noise [1 x 3]
    np.testing.assert_array_equal(
        np.asarray(attention_mask_1[8:19]),
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
    )

    # Check proposer state updates
    assert proposer._ctx_len == 10
    assert proposer._prev_seq_len == 10

    # Test propose for Iteration 1
    with jax.set_mesh(mesh):
        _, draft_token_ids_1 = proposer.propose(
            kv_caches=[],
            input_ids=input_ids,
            attn_metadata=draft_metadata_1,
            last_token_indices=jnp.zeros(1),
            target_hidden_states=target_hidden_1,
        )

    assert draft_token_ids_1.shape == (1, 2)
    mock_draft_forward.assert_called_once()
    # Since precompile() has already warmed up the JIT cache for _sample_block_draft_tokens
    # with this shape, JAX uses the compiled HLO directly and does not re-execute the
    # Python body. Thus, the mock is not called again during propose.
    mock_compute_logits.assert_not_called()

    # Reset call counts
    mock_draft_forward.reset_mock()
    mock_compute_logits.reset_mock()

    # ----------------- ITERATION 2: Speculative Rejection & Cache Cropping -----------------
    # Out of 2 proposed tokens, the target model accepts 1.
    # Therefore, new accepted sequence length is 10 prefix + 1 accepted = 11.
    attn_metadata_2 = AttentionMetadata(
        input_positions=jnp.array([0]),
        block_tables=jnp.array([0]),
        seq_lens=jnp.array([11], dtype=jnp.int32),
        query_start_loc=jnp.array([0]),
        request_distribution=jnp.array([0]),
    )

    # Target model passes the accumulated auxiliary hidden states (now length 11)
    aux_hidden_states_2 = (jnp.ones((11, 32), dtype=jnp.bfloat16), )
    next_token_ids_2 = jnp.array([99], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        target_hidden_2, _, _, _ = proposer.prepare_inputs(
            attn_metadata_2,
            input_ids,
            aux_hidden_states_2,
            next_token_ids_2,
        )

    ctx_padded_2, position_ids_2, attention_mask_2 = target_hidden_2

    # Proposer updates: seq_len is 11. Since 11 > ctx_len (10), no cropping of prefix.
    # num_new = 11 - 10 = 1. Copy 1 new token to position 10 in buffer.
    assert proposer._ctx_len == 11
    assert proposer._prev_seq_len == 11
    # Padded size for 11 is 16
    assert ctx_padded_2.shape == (16, 32)

    # ----------------- ITERATION 3: Request Reset (Slot request changes) -----------------
    # Suppose a new request occupies the slot.
    runner.input_batch.req_ids = ["req_2"]

    attn_metadata_3 = AttentionMetadata(
        input_positions=jnp.array([0]),
        block_tables=jnp.array([0]),
        seq_lens=jnp.array([5],
                           dtype=jnp.int32),  # New request starts at length 5
        query_start_loc=jnp.array([0]),
        request_distribution=jnp.array([0]),
    )

    aux_hidden_states_3 = (jnp.ones((5, 32), dtype=jnp.bfloat16), )
    next_token_ids_3 = jnp.array([77], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        target_hidden_3, _, _, _ = proposer.prepare_inputs(
            attn_metadata_3,
            input_ids,
            aux_hidden_states_3,
            next_token_ids_3,
        )

    # CRITICAL: Verify that proposer state is fully reset when slot request changes
    assert proposer._ctx_len == 5
    assert proposer._prev_seq_len == 5
    assert proposer._last_req_id == "req_2"
