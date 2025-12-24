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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.llama_eagle3 import (Eagle3LlamaDecoderLayer,
                                                   EagleLlama3ForCausalLM)
from tpu_inference.runner.kv_cache import create_kv_caches


class MockSpeculativeConfig:

    def __init__(self):
        self.num_speculative_tokens = 3
        self.method = "eagle3"
        self.draft_model_config = None


class MockVllmConfig:

    def __init__(self, model: str, draft_model: str, kv_cache_dtype):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.speculative_config = MockSpeculativeConfig()
        self.speculative_config.draft_model_config = ModelConfig(
            draft_model,
            dtype="bfloat16",
            max_model_len=2048,
            skip_tokenizer_init=True,
            trust_remote_code=True)
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)


@pytest.fixture
def mock_vllm_config() -> MockVllmConfig:
    return MockVllmConfig(model="meta-llama/Meta-Llama-3-8B-Instruct",
                          draft_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                          kv_cache_dtype="auto")


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, -1))

    with Mesh(device_mesh, axis_names=('data', 'attn_dp', 'model')) as m:
        yield m


@pytest.fixture
def mock_model_inputs(mock_vllm_config: MockVllmConfig):
    """Provides mock inputs for the EagleLlama3 model."""
    batch_size = 2
    seq_len = 16
    target_hidden_size = mock_vllm_config.model_config.get_hidden_size()

    input_ids = jnp.ones((batch_size * seq_len, ), dtype=jnp.int32)
    hidden_states = jnp.ones((batch_size * seq_len, target_hidden_size),
                             dtype=jnp.bfloat16)
    attention_metadata = AttentionMetadata(
        input_positions=jnp.arange(batch_size * seq_len, dtype=jnp.int32),
        block_tables=jnp.zeros((batch_size, 1), dtype=jnp.int32).reshape(-1),
        seq_lens=jnp.full((batch_size, ), seq_len, dtype=jnp.int32),
        query_start_loc=jnp.arange(0, (batch_size + 1) * seq_len,
                                   seq_len,
                                   dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, batch_size], dtype=jnp.int32),
    )
    return input_ids, hidden_states, attention_metadata


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestEagleLlama3ForCausalLM:
    """Tests for the EagleLlama3ForCausalLM model."""

    def test_eagle3_decoder_layer_init(self, mock_vllm_config: MockVllmConfig,
                                       rng: PRNGKey, mesh: Mesh):
        """Tests the initialization of the Eagle3LlamaDecoderLayer."""
        hf_config = mock_vllm_config.speculative_config.draft_model_config.hf_config
        dtype = jnp.bfloat16
        rngs = nnx.Rngs(rng)

        layer = Eagle3LlamaDecoderLayer(
            hf_config,
            dtype,
            rngs,
            mesh,
            kv_cache_dtype=mock_vllm_config.cache_config.cache_dtype)

        # Check if projection layers are overridden with correct input size
        original_hidden_size = hf_config.hidden_size
        expected_input_size = 2 * original_hidden_size

        assert layer.self_attn.q_proj.kernel.value.shape[
            0] == expected_input_size
        assert layer.self_attn.k_proj.kernel.value.shape[
            0] == expected_input_size
        assert layer.self_attn.v_proj.kernel.value.shape[
            0] == expected_input_size
        assert isinstance(layer.hidden_norm, nnx.RMSNorm)

    @pytest.mark.parametrize("mock_vllm_config", [
        MockVllmConfig("meta-llama/Meta-Llama-3-8B-Instruct",
                       "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "auto"),
        MockVllmConfig("meta-llama/Meta-Llama-3-8B-Instruct",
                       "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "fp8"),
    ])
    def test_forward_pass(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey,
                          mesh: Mesh, mock_model_inputs):
        """Tests the forward pass of the EagleLlama3ForCausalLM model."""

        draft_model_config = mock_vllm_config.speculative_config.draft_model_config
        hf_config = draft_model_config.hf_config
        model = EagleLlama3ForCausalLM(mock_vllm_config, rng, mesh)

        input_ids, hidden_states, attention_metadata = mock_model_inputs

        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=16,
            num_kv_heads=hf_config.num_key_value_heads,
            head_size=hf_config.hidden_size // hf_config.num_attention_heads,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if mock_vllm_config.cache_config.cache_dtype == "fp8" else
            jnp.bfloat16)

        _, output_hidden_states, aux_hidden_states = model(
            kv_caches, input_ids, hidden_states, attention_metadata)

        logits = model.compute_logits(output_hidden_states)

        target_model_config = mock_vllm_config.model_config

        assert output_hidden_states.shape == (
            input_ids.shape[0], draft_model_config.get_hidden_size())
        assert logits.shape == (input_ids.shape[0],
                                target_model_config.get_vocab_size())
        assert len(aux_hidden_states) == 1
        assert aux_hidden_states[0].shape == output_hidden_states.shape

    @patch("tpu_inference.models.jax.llama_eagle3.load_hf_weights")
    def test_load_weights(self, mock_load_hf_weights: MagicMock,
                          mock_vllm_config: MockVllmConfig, rng: PRNGKey,
                          mesh: Mesh):
        """Tests that the load_weights function is called correctly."""
        model = EagleLlama3ForCausalLM(mock_vllm_config, rng, mesh)
        model.load_weights(rng)

        mock_load_hf_weights.assert_called_once()
        call_args = mock_load_hf_weights.call_args.kwargs

        assert call_args["vllm_config"] is mock_vllm_config
        assert call_args["model"] is model
        assert call_args["mesh"] is mesh
        assert call_args["is_draft_model"] is True

        metadata_map = call_args["metadata_map"]
        assert "midlayer.hidden_norm" in metadata_map.name_map
        assert "lm_head" in metadata_map.name_map
        assert "d2t" in metadata_map.name_map
        assert "q_proj" in metadata_map.reshape_map
        assert metadata_map.reshape_map["q_proj"][-1] == (
            2 * mock_vllm_config.model_config.get_hidden_size())
