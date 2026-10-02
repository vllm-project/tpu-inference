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

from tpu_inference.models.common.interface import ModelInterface
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


def test_propose_passes_kv_cache_mapping_to_vllm_draft():
    """The vllm (torchax) draft step also takes the layer -> KV cache map."""
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2
    proposer.block_size = 3
    proposer.state_leaves = None
    proposer._is_vllm_draft = True
    layer_map = {"model.layers.32.self_attn.attn": 1}
    proposer.runner = MagicMock(layer_name_to_kvcache_index=layer_map)

    hidden_states = jnp.ones((3, 4), dtype=jnp.bfloat16)
    calls = {}

    def fake_vllm_draft_step(state, kv_caches, input_ids, target_hidden_states,
                             attn_metadata, layer_name_to_kvcache_index):
        calls["mapping"] = layer_name_to_kvcache_index
        return kv_caches, hidden_states, [], None

    proposer.model_fn = fake_vllm_draft_step
    proposer.compute_logits_fn = lambda _state, _hidden, _lora: jnp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    _, draft_token_ids = proposer.propose(
        kv_caches=[],
        input_ids=None,
        attn_metadata=None,
        last_token_indices=None,
        target_hidden_states=None,
    )

    assert calls["mapping"] == tuple(layer_map.items())
    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([[0, 1]], dtype=np.int32))


def test_get_vllm_shared_params_shares_target_embed_and_lm_head():
    proposer = object.__new__(DFlashProposer)
    embed, lm_head = jnp.zeros((4, 2)), jnp.ones((4, 2))
    proposer.runner = MagicMock(
        state={
            "vllm_model.model.embed_tokens.weight": embed,
            "vllm_model.lm_head.weight": lm_head,
            "vllm_model.model.norm.weight": jnp.ones((2, )),
        })

    shared = proposer._get_vllm_shared_params()

    assert set(shared) == {
        "vllm_model.model.embed_tokens.weight", "vllm_model.lm_head.weight"
    }
    assert shared["vllm_model.lm_head.weight"] is lm_head


def test_get_vllm_shared_params_tied_target_uses_embedding_as_lm_head():
    proposer = object.__new__(DFlashProposer)
    embed = jnp.zeros((4, 2))
    proposer.runner = MagicMock(
        state={"vllm_model.model.embed_tokens.weight": embed})

    shared = proposer._get_vllm_shared_params()

    assert shared["vllm_model.lm_head.weight"] is embed


def test_get_vllm_shared_params_ignores_non_dict_target_state():
    proposer = object.__new__(DFlashProposer)
    proposer.runner = MagicMock(state=None)

    assert proposer._get_vllm_shared_params() == {}


def _model_interface(model, state):
    return ModelInterface(model_fn=MagicMock(),
                          compute_logits_fn=MagicMock(),
                          pooler_fn=None,
                          combine_hidden_states_fn=MagicMock(),
                          multimodal_fns=None,
                          state=state,
                          state_leaves=state,
                          lora_manager=None,
                          model=model)


def _vllm_model():
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
    return object.__new__(VllmModelWrapper)


def _load_draft_configured_as_flax(monkeypatch, target_model, draft_model):
    """Runs load_model with both impls configured as flax_nnx while get_model
    returns ``draft_model``, as when the draft falls back to vLLM."""
    from tpu_inference.spec_decode.jax import dflash as dflash_module

    runner = MockRunner(_make_single_device_mesh())
    runner.model = target_model
    embed = jnp.zeros((4, 2))
    runner.state = {"vllm_model.model.embed_tokens.weight": embed}
    proposer = DFlashProposer(MockVllmConfig(), runner)

    monkeypatch.setattr(
        "tpu_inference.models.common.model_loader.resolve_model_impl_type",
        lambda *args, **kwargs: "flax_nnx")
    captured = {}

    def fake_get_model(*args, shared_params=None, **kwargs):
        captured["shared_params"] = shared_params
        return _model_interface(draft_model, {"w": jnp.zeros((2, ))})

    monkeypatch.setattr(dflash_module, "get_model", fake_get_model)
    proposer.load_model(runner.state)
    return proposer, captured["shared_params"], embed


def test_load_model_detects_vllm_fallback_draft(monkeypatch):
    proposer, shared_params, embed = _load_draft_configured_as_flax(
        monkeypatch, target_model=_vllm_model(), draft_model=_vllm_model())

    assert proposer._is_vllm_draft
    assert shared_params["vllm_model.model.embed_tokens.weight"] is embed
    assert shared_params["vllm_model.lm_head.weight"] is embed


def test_load_model_rejects_vllm_draft_with_flax_target(monkeypatch):
    with pytest.raises(ValueError, match="must match target"):
        _load_draft_configured_as_flax(monkeypatch,
                                       target_model=MagicMock(),
                                       draft_model=_vllm_model())


# ----- New Comprehensive Tests -----


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
