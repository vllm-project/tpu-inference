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

from dataclasses import field
from types import SimpleNamespace
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.models.jax.llama_guard_4 import (LlamaGuard4ForCausalLM,
                                                    LlamaGuard4WeightLoader)


class MockParamLlamaGuard4:
    """A mock for a parameter used in the LlamaGuard4 model."""
    shape: Tuple[int, ...]
    dtype: jnp.dtype = jnp.bfloat16
    sharding_spec: Tuple[str | None, ...] | None = None
    value: Any = field(init=False)
    out_sharding: Any = field(init=False)

    def __init__(self, shape=(32, 128)):
        self.shape = shape
        self.value = jnp.zeros(self.shape, dtype=self.dtype)
        # The sharding spec is accessed during weight loading
        self.out_sharding = SimpleNamespace(spec=self.sharding_spec)

    # Allow the mock parameter's value to be updated
    def __setattr__(self, name, value):
        if name in [
                'value', 'shape', 'dtype', 'out_sharding', 'sharding_spec'
        ]:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the LlamaGuard4 model."""

    def __init__(self,
                 model_name: str,
                 random_weights: bool = False,
                 tensor_parallelism: int = 1):
        self.model_config = MagicMock(spec=ModelConfig)
        self.load_config = MagicMock()
        self.load_config.download_dir = None

        # Downsizing the following to avoid OOM
        self.model_config.get_vocab_size.return_value = 1024
        self.model_config.get_hidden_size.return_value = 128
        self.model_config.model = model_name

        self.additional_config = {
            "random_weights": random_weights,
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": tensor_parallelism
                }
            }
        }

        self.cache_config = MagicMock(cache_dtype="auto")

        # Mock the underlying HF config values for parameter detection
        # Downsized to avoid OOM
        text_config_mock = MagicMock()
        text_config_mock.num_attention_heads = 4
        text_config_mock.num_key_value_heads = 2
        text_config_mock.head_dim = 32
        # Explicit numeric values: MagicMock auto-attrs are Mock objects and
        # blow up only at forward time inside jitted math.
        text_config_mock.num_layers = 2
        text_config_mock.rms_norm_eps = 1e-5
        text_config_mock.intermediate_size = 256
        text_config_mock.hidden_act = "silu"
        # transformers >= 5.6 layout: rope_theta lives in rope_parameters,
        # the flat rope_theta attribute no longer exists, and rope_scaling
        # is a property alias of rope_parameters. Llama-Guard-4 ships
        # llama3-style scaling keys, so mirror that faithfully.
        llama3_rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        text_config_mock.rope_parameters = llama3_rope_parameters
        text_config_mock.rope_scaling = llama3_rope_parameters
        del text_config_mock.rope_theta

        hf_config_mock = MagicMock()
        hf_config_mock.text_config = text_config_mock

        vision_config_mock = MagicMock()
        vision_config_mock.image_size = 336
        vision_config_mock.patch_size = 14
        vision_config_mock.hidden_size = 1408
        vision_config_mock.num_attention_heads = 16
        vision_config_mock.rope_parameters = {
            "rope_theta": 10000.0,
            "rope_type": "default"
        }
        del vision_config_mock.rope_theta
        vision_config_mock.intermediate_size = 5632
        vision_config_mock.projector_input_dim = 4096
        vision_config_mock.projector_output_dim = 4096
        vision_config_mock.projector_dropout = 0.0
        hf_config_mock.vision_config = vision_config_mock
        hf_config_mock.image_token_index = 200092

        self.model_config.hf_config = hf_config_mock


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with all required axes for testing.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices())

    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'model', 'expert')) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config_llama_guard_4() -> MockVllmConfig:
    return MockVllmConfig(model_name="meta-llama/Llama-Guard-4-12B")


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.models.jax.llama_guard_4.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


class TestLlamaGuard4ForCausalLM:
    """Tests for the main LlamaGuard4ForCausalLM model class."""

    def test_init_llama_guard_4(self, mock_vllm_config_llama_guard_4, rng,
                                mesh):
        """Tests correct initialization and parameter detection."""
        with jax.set_mesh(mesh):
            model = LlamaGuard4ForCausalLM(mock_vllm_config_llama_guard_4, rng,
                                           mesh)

        # Check model name is correctly set in the config
        assert "llama-guard-4" in model.vllm_config.model_config.model.lower()

        assert model.hidden_size == 128

    def test_create_model_with_random_weights(self,
                                              mock_vllm_config_llama_guard_4,
                                              rng, mesh):
        """
        Tests that random weight initialization creates concrete, non-zero-variance arrays.
        """
        with jax.set_mesh(mesh):
            model = LlamaGuard4ForCausalLM(
                vllm_config=mock_vllm_config_llama_guard_4,
                rng=rng,
                mesh=mesh,
                force_random_weights=True)

            embedding_weight = model.embedder.input_embedding_table_VD.value
            attention_q_kernel = model.layers[0].attn.kernel_q_proj_DNH.value
            final_norm_scale = model.final_norm.scale.value

            assert isinstance(embedding_weight, jax.Array)
            assert isinstance(attention_q_kernel, jax.Array)
            assert isinstance(final_norm_scale, jax.Array)

            assert jnp.std(embedding_weight) > 0
            assert jnp.std(attention_q_kernel) > 0

            assert jnp.all(final_norm_scale == 1.0)

    def test_load_weights_called_correctly(self, rng, mesh):
        """Tests that the weight loader is called correctly for checkpoint loading."""
        vllm_config = MockVllmConfig(model_name="llama-guard-4-test",
                                     random_weights=False)
        with jax.set_mesh(mesh):
            model = LlamaGuard4ForCausalLM(vllm_config, rng, mesh)

        with patch(
                'tpu_inference.models.jax.llama_guard_4.LlamaGuard4WeightLoader'
        ) as mock_loader_cls:
            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            model.load_weights(rng)

            mock_loader_cls.assert_called_once_with(vllm_config=vllm_config,
                                                    hidden_size=128,
                                                    attn_heads=4,
                                                    num_key_value_heads=2,
                                                    attn_head_dim=32)
            mock_loader_instance.load_weights.assert_called_once_with(model)


class TestLlamaGuard4WeightLoader:
    """Tests for the LlamaGuard4WeightLoader class."""

    @pytest.fixture
    def weight_loader(self):
        return LlamaGuard4WeightLoader(
            vllm_config=MockVllmConfig("test-model"),
            hidden_size=5120,
            attn_heads=40,
            num_key_value_heads=8,
            attn_head_dim=128)

    @pytest.mark.parametrize("hf_key, expected", [
        ("language_model.model.layers.15.self_attn.q_proj.weight",
         "layers.15.attn.kernel_q_proj_DNH"),
        ("language_model.model.layers.0.feed_forward.gate_proj.weight",
         "layers.0.custom_module.kernel_gating_DF"),
        ("language_model.model.embed_tokens.weight",
         "embedder.input_embedding_table_VD"),
        ("language_model.model.norm.weight", "final_norm.scale"),
        ("language_model.lm_head.weight", "lm_head.input_embedding_table_DV"),
        ("unmapped.key.name", "unmapped.key.name"),
    ])
    def test_map_loaded_to_standardized_name(self, weight_loader, hf_key,
                                             expected):
        """Tests the mapping from HuggingFace key names to internal names."""
        assert weight_loader.map_loaded_to_standardized_name(
            hf_key) == expected

    def test_load_weights_transformation(self, weight_loader, rng, mesh):
        """Tests that weights are correctly reshaped, transposed, and loaded."""
        vllm_config = MockVllmConfig(model_name="llama-guard-4-small-test",
                                     random_weights=False)

        with jax.set_mesh(mesh):
            model = LlamaGuard4ForCausalLM(vllm_config, rng, mesh)

        hidden_size = 5120
        vocab_size = 202048

        original_weight = jnp.ones((vocab_size, hidden_size))
        dummy_weights = [
            ("language_model.model.embed_tokens.weight", original_weight),
        ]
        weight_loader.names_and_weights_generator = dummy_weights

        # Mock get_param to return a mock param with the target shape
        mock_param = MockParamLlamaGuard4(shape=(vocab_size, hidden_size))

        with patch("tpu_inference.models.jax.llama_guard_4.get_param", return_value=mock_param), \
            patch("tpu_inference.models.jax.llama_guard_4.shard_put", return_value=jnp.ones(mock_param.value.shape)) as mock_shard_put:

            weight_loader.load_weights(model)

            # Assert that shard_put was called with the correctly transposed weight
            mock_shard_put.assert_called_once()

            # Get the actual array passed to shard_put
            called_with_weight = mock_shard_put.call_args[0][0]

            # Check if the shape of the array passed to shard_put matches the model's expected shape.
            assert called_with_weight.shape == mock_param.value.shape


class TestLlamaGuard4ForwardContract:
    """Pins the forward-return contract expected by run_model's jit
    out_shardings (kv_caches, hidden, aux_hidden_states, expert_ids)."""

    def test_forward_returns_4_tuple(self, mock_vllm_config_llama_guard_4,
                                     rng):
        from tpu_inference.layers.common.attention_metadata import \
            AttentionMetadata
        from tpu_inference.runner.kv_cache import create_kv_caches

        # Single-device mesh: the ragged-attention shard_map requires the
        # tiny request metadata to divide the data axis evenly.
        # (jax.devices() raises rather than returning [] when no backend
        # exists, so check the platform explicitly.)
        if jax.devices()[0].platform != "tpu":
            pytest.skip("Forward pass requires a TPU device.")
        device_mesh = np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1))
        mesh = Mesh(device_mesh,
                    axis_names=('data', 'attn_dp', 'model', 'expert'))

        with jax.set_mesh(mesh):
            model = LlamaGuard4ForCausalLM(
                vllm_config=mock_vllm_config_llama_guard_4,
                rng=rng,
                mesh=mesh,
                force_random_weights=True)

            # num_blocks must be divisible by the mesh's data-axis size
            # (this file's mesh fixture uses all local devices).
            kv_caches = create_kv_caches(
                num_blocks=16,
                block_size=32,
                num_kv_heads=model.num_key_value_heads,
                head_size=model.head_dim,
                mesh=mesh,
                layer_names=["layer"] * len(model.layers),
                cache_dtype=jnp.bfloat16)

            num_tokens = 8
            num_reqs = 1
            input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
            attention_metadata = AttentionMetadata(
                input_positions=jnp.ones((num_tokens, ), dtype=jnp.int32),
                block_tables=jnp.zeros((num_reqs, 4),
                                       dtype=jnp.int32).reshape(-1),
                seq_lens=jnp.ones((num_reqs, ), dtype=jnp.int32),
                query_start_loc=jnp.array([0, num_tokens], dtype=jnp.int32),
                request_distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
            )

            out = model(kv_caches, input_ids, attention_metadata)

        assert len(out) == 4
        new_kv_caches, hidden, aux_hidden, expert_ids = out
        assert len(new_kv_caches) == len(kv_caches)
        assert hidden.shape == (num_tokens, model.hidden_size)
        assert aux_hidden == []
        assert expert_ids is None

    def test_vision_rope_call_returns_jax_array(self):
        from tpu_inference.layers.jax.rope import Llama4VisionRotaryEmbedding
        rope = Llama4VisionRotaryEmbedding(image_size=336,
                                           patch_size=14,
                                           hidden_size=1408,
                                           num_attention_heads=16,
                                           rope_theta=10000.0,
                                           dtype=jnp.bfloat16)
        out = rope()
        # Must be a concrete array, not an nnx.Param: newer JAX no longer
        # implicitly converts nnx.Param via __jax_array__ inside jit.
        assert isinstance(out, jax.Array)
