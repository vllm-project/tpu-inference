# SPDX-License-Identifier: Apache-2.0
# Test for LoRA weight loading API

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_commons.models.jax.utils.weight_utils import (
    load_hf_weights, load_hf_weights_on_thread, transfer_state_with_mappings)

# ----- nnx.Module Wrappers -----


class SourceLayer(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs(), (4, 4)))
        self.bias = nnx.Param(jax.random.normal(rngs(), (4, )))


class SourceModel(nnx.Module):

    def __init__(self, rngs):
        self.src_lm_head = nnx.Param(jax.random.normal(rngs(), (2, 4)))
        self.layers = {0: SourceLayer(rngs)}


class TargetLinear(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jnp.zeros((4, 4)))
        self.bias = nnx.Param(jnp.zeros((4, )))


class TargetBlock(nnx.Module):

    def __init__(self, rngs):
        self.mlp = {"up_proj": TargetLinear(rngs)}


class TargetModel(nnx.Module):

    def __init__(self, rngs):
        self.tgt_lm_head = nnx.Param(jnp.zeros((2, 4)))
        self.model = {"layers": {0: TargetBlock(rngs)}}


# ----- Test -----
class WeightTransfer(jtu.JaxTestCase):

    def test_transfer_state(self):
        rng = nnx.Rngs(0)
        src_model = SourceModel(rng)
        tgt_model = TargetModel(rng)

        # Get split states
        _, src_state = nnx.split(src_model)
        _, tgt_state = nnx.split(tgt_model)

        # Overwrite known values
        src_state["layers"][0]["kernel"].value = jnp.ones((4, 4)) * 42.0
        src_state["layers"][0]["bias"].value = jnp.ones((4, )) * 7.0
        src_state["src_lm_head"].value = jnp.ones((2, 4)) * 6.0
        # Mapping for both kernel and bias
        mappings = {
            "layers.*.kernel": ("model.layers.*.mlp.up_proj.kernel", (None, )),
            "layers.*.bias": ("model.layers.*.mlp.up_proj.bias", (None, )),
            "src_lm_head": ("tgt_lm_head", (None, None)),
        }

        # Transfer
        new_tgt_state = transfer_state_with_mappings(src_state, tgt_state,
                                                     mappings)

        # Assert correctness
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["kernel"].value, 42.0)
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["bias"].value, 7.0)
        assert jnp.allclose(new_tgt_state["tgt_lm_head"].value, 6.0)


# ----- Test -----
# Dummy classes to mock configurations
class MockVLLMConfig:

    def __init__(self, model_config):
        self.model_config = model_config


class MockModelConfig:

    def __init__(self,
                 hf_config,
                 hidden_size,
                 head_size,
                 is_multimodal_model=False):
        self.hf_config = hf_config
        self._hidden_size = hidden_size
        self._head_size = head_size
        self.is_multimodal_model = is_multimodal_model
        self.model = "mock_model_path"

    def get_hidden_size(self):
        return self._hidden_size

    def get_head_size(self):
        return self._head_size


class MockHFConfig:

    def __init__(self, num_attention_heads, num_key_value_heads):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


# Dummy model for testing load_hf_weights
class SimpleModel(nnx.Module):

    def __init__(self, rngs):
        self.param = nnx.Param(jnp.zeros(1), rngs=rngs)


@pytest.fixture
def mock_config():
    hf_config = MockHFConfig(num_attention_heads=4, num_key_value_heads=2)
    model_config = MockModelConfig(hf_config, hidden_size=16, head_size=4)
    return MockVLLMConfig(model_config)


@pytest.fixture
def mesh():
    # Simulate a mesh with 2 devices for model parallelism
    return Mesh(np.array(jax.devices()[:2]), ('model', ))


@pytest.fixture
def mappings():
    return {
        "model.layers.*.self_attn.q_proj": "transformer.h.*.attn.c_attn_q",
        "model.layers.*.self_attn.k_proj": "transformer.h.*.attn.c_attn_k",
        "model.layers.*.self_attn.v_proj": "transformer.h.*.attn.c_attn_v",
        "model.layers.*.self_attn.o_proj": "transformer.h.*.attn.c_proj",
        "model.embed_tokens": "transformer.wte",
    }


# Fixture to mock dependencies for load_hf_weights_on_thread
@pytest.fixture
def mock_thread_deps(monkeypatch):
    base_path = "tpu_commons.models.jax.utils.weight_utils"

    mocks = {
        "get_named_sharding":
        mock.MagicMock(return_value=mock.MagicMock()),
        "model_weights_generator":
        mock.MagicMock(),
        "get_param_and_sharding":
        mock.MagicMock(),
        "shard_put":
        mock.MagicMock(side_effect=lambda tensor, sharding, mesh: tensor),
        "get_padded_head_dim":
        mock.MagicMock(side_effect=lambda x: x),  # No padding by default
        "logger":
        mock.MagicMock(),
    }

    monkeypatch.setattr(f"{base_path}.nnx.get_named_sharding",
                        mocks["get_named_sharding"])
    monkeypatch.setattr(f"{base_path}.model_weights_generator",
                        mocks["model_weights_generator"])
    monkeypatch.setattr(f"{base_path}.get_param_and_sharding",
                        mocks["get_param_and_sharding"])
    monkeypatch.setattr(f"{base_path}.shard_put", mocks["shard_put"])
    monkeypatch.setattr(f"{base_path}.utils.get_padded_head_dim",
                        mocks["get_padded_head_dim"])
    monkeypatch.setattr(f"{base_path}.logger", mocks["logger"])

    # Setup mock return for get_param_and_sharding
    mock_model_weight = nnx.Variable(jnp.zeros(1))
    mocks["get_param_and_sharding"].return_value = (mock_model_weight,
                                                    mock.MagicMock())
    mocks["mock_model_weight"] = mock_model_weight

    return mocks


def test_load_q_proj(mock_config, mesh, mappings, mock_thread_deps):
    config = mock_config.model_config
    hf_key = "model.layers.0.self_attn.q_proj.weight"
    hf_weight = jnp.ones((config.hf_config.num_attention_heads *
                          config.get_head_size() * config.get_hidden_size(), ))
    mock_thread_deps["model_weights_generator"].return_value = [(hf_key,
                                                                 hf_weight)]

    # Expected shape: (hidden_size, num_heads, head_dim)
    # No repeat since sharding_size // num_heads < 1
    expected_shape = (config.get_hidden_size(),
                      config.hf_config.num_attention_heads,
                      config.get_head_size())
    mock_thread_deps["mock_model_weight"].value = jnp.zeros(expected_shape)

    load_hf_weights_on_thread(mock_config, mock.MagicMock(), mappings, mesh,
                              "dummy.bin")

    mock_thread_deps["get_param_and_sharding"].assert_called_once_with(
        mock.ANY, mock_thread_deps["get_named_sharding"].return_value,
        "transformer.h.*.attn.c_attn_q")

    transformed_weight = mock_thread_deps["shard_put"].call_args[0][0]
    assert transformed_weight.shape == expected_shape


def test_load_k_proj_gqa(mock_config, mappings, mock_thread_deps):
    mesh_sharding_4 = Mesh(np.array(jax.devices()[:4]), ('model', ))
    sharding_size = 4
    config = mock_config.model_config
    hf_key = "model.layers.0.self_attn.k_proj.weight"
    hf_weight = jnp.ones((config.hf_config.num_key_value_heads *
                          config.get_head_size() * config.get_hidden_size(), ))
    mock_thread_deps["model_weights_generator"].return_value = [(hf_key,
                                                                 hf_weight)]

    # Expected shape: (hidden_size, num_kv_heads * repeat_factor, head_dim)
    num_kv_heads = config.hf_config.num_key_value_heads
    repeat_factor = sharding_size // num_kv_heads
    expected_heads = num_kv_heads * repeat_factor
    expected_shape = (config.get_hidden_size(), expected_heads,
                      config.get_head_size())
    mock_thread_deps["mock_model_weight"].value = jnp.zeros(expected_shape)

    load_hf_weights_on_thread(mock_config, mock.MagicMock(), mappings,
                              mesh_sharding_4, "dummy.bin")

    transformed_weight = mock_thread_deps["shard_put"].call_args[0][0]
    assert transformed_weight.shape == expected_shape


def test_load_embedding(mock_config, mesh, mappings, mock_thread_deps):
    config = mock_config.model_config
    hf_key = "model.embed_tokens.weight"
    expected_shape = (100, config.get_hidden_size())
    hf_weight = jnp.ones(expected_shape)
    mock_thread_deps["model_weights_generator"].return_value = [(hf_key,
                                                                 hf_weight)]

    mock_thread_deps["mock_model_weight"].value = jnp.zeros(expected_shape)

    load_hf_weights_on_thread(mock_config, mock.MagicMock(), mappings, mesh,
                              "dummy.bin")

    mock_thread_deps["get_param_and_sharding"].assert_called_once_with(
        mock.ANY, mock_thread_deps["get_named_sharding"].return_value,
        "transformer.wte")
    transformed_weight = mock_thread_deps["shard_put"].call_args[0][0]
    assert transformed_weight.shape == expected_shape
    assert np.array_equal(transformed_weight, hf_weight)


def test_load_with_head_padding(mock_config, mesh, mappings, mock_thread_deps):
    mock_thread_deps["get_padded_head_dim"].side_effect = lambda x: x + 4
    config = mock_config.model_config
    head_dim_original = config.get_head_size()
    head_dim_padded = head_dim_original + 4
    # sharding_size = mesh.shape["model"]
    # num_heads = config.hf_config.num_attention_heads

    hf_key = "model.layers.0.self_attn.q_proj.weight"
    hf_weight = jnp.ones((config.hf_config.num_attention_heads *
                          head_dim_original * config.get_hidden_size(), ))
    mock_thread_deps["model_weights_generator"].return_value = [(hf_key,
                                                                 hf_weight)]

    # Assertion is skipped due to head_dim_pad > 0, so no need to mock model_weight shape

    load_hf_weights_on_thread(mock_config, mock.MagicMock(), mappings, mesh,
                              "dummy.bin")

    transformed_weight = mock_thread_deps["shard_put"].call_args[0][0]
    # reshape: (4, 4, 16) -> pad: (4, 8, 16) -> transpose: (16, 4, 8)
    # No repeat: sharding_size // num_heads < 1
    expected_shape = (config.get_hidden_size(),
                      config.hf_config.num_attention_heads, head_dim_padded)
    assert transformed_weight.shape == expected_shape


# Fixture to mock dependencies for load_hf_weights
@pytest.fixture
def mock_load_deps(monkeypatch):
    base_path = "tpu_commons.models.jax.utils.weight_utils"
    mocks = {
        "get_model_weights_files": mock.MagicMock(),
        "load_hf_weights_on_thread": mock.MagicMock(),
        "nnx_update": mock.MagicMock(),
        "ThreadPoolExecutor": mock.MagicMock(),
    }
    monkeypatch.setattr(f"{base_path}.get_model_weights_files",
                        mocks["get_model_weights_files"])
    monkeypatch.setattr(f"{base_path}.load_hf_weights_on_thread",
                        mocks["load_hf_weights_on_thread"])
    monkeypatch.setattr(f"{base_path}.nnx.update", mocks["nnx_update"])
    monkeypatch.setattr(f"{base_path}.ThreadPoolExecutor",
                        mocks["ThreadPoolExecutor"])
    return mocks


def test_load_weights_multiple_files(mock_config, mesh, mappings,
                                     mock_load_deps):
    model = SimpleModel(rngs=nnx.Rngs(0))
    file_list = ["file1.bin", "file2.bin"]
    mock_load_deps["get_model_weights_files"].return_value = file_list

    load_hf_weights(mock_config, model, mappings, mesh)

    mock_load_deps["get_model_weights_files"].assert_called_once_with(
        mock_config.model_config.model)
    assert mock_load_deps["load_hf_weights_on_thread"].call_count == 2
    mock_load_deps["nnx_update"].assert_called_once()
    update_args = mock_load_deps["nnx_update"].call_args[0]
    assert update_args[0] is model
    assert isinstance(update_args[1], nnx.State)


def test_load_weights_no_files(mock_config, mesh, mappings, mock_load_deps):
    model = SimpleModel(rngs=nnx.Rngs(0))
    mock_load_deps["get_model_weights_files"].return_value = []

    load_hf_weights(mock_config, model, mappings, mesh)

    mock_load_deps["load_hf_weights_on_thread"].assert_not_called()
    mock_load_deps["nnx_update"].assert_not_called()
