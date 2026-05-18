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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.gemma4_mtp import Gemma4MTPForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches


class DummyTextConfig:

    def __init__(self):
        self.hidden_size = 256
        self.vocab_size = 1024
        self.num_hidden_layers = 2
        self.rms_norm_eps = 1e-6
        self.layer_types = ["full_attention", "full_attention"]
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.head_dim = 32
        self.global_head_dim = 32
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.attention_bias = False
        self.intermediate_size = 512
        self.final_logit_softcapping = 30.0


class DummyDraftConfig:

    def __init__(self):
        self.text_config = DummyTextConfig()
        self.backbone_hidden_size = 256
        self.tie_word_embeddings = False
        self.use_ordered_embeddings = True
        self.num_centroids = 32
        self.centroid_intermediate_top_k = 4


class MockVllmConfig:

    def __init__(self, use_ordered_embeddings=True, kv_cache_dtype="auto"):
        self.model_config = MagicMock()
        self.model_config.dtype = jnp.bfloat16
        self.model_config.get_vocab_size = lambda: 1024
        self.model_config.get_hidden_size = lambda: 256

        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None

        # Setup draft config
        self.speculative_config = MagicMock()
        draft_model_config = MagicMock()

        draft_hf_config = DummyDraftConfig()
        draft_hf_config.use_ordered_embeddings = use_ordered_embeddings

        draft_model_config.hf_config = draft_hf_config
        draft_model_config.get_hidden_size = lambda: 256

        self.speculative_config.draft_model_config = draft_model_config


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
    """Provides mock inputs for the Gemma4MTP model."""
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


class TestGemma4MTPForCausalLM:
    """Tests for the Gemma4MTPForCausalLM model."""

    @pytest.mark.parametrize("use_ordered_embeddings", [True, False])
    def test_gemma4_mtp_init(self, use_ordered_embeddings: bool, rng: PRNGKey,
                             mesh: Mesh):
        """Tests initialization of the Gemma4MTPForCausalLM model."""
        vllm_config = MockVllmConfig(
            use_ordered_embeddings=use_ordered_embeddings)
        with jax.set_mesh(mesh):
            model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)

        assert model.model is not None
        assert len(
            model.model.layers
        ) == vllm_config.speculative_config.draft_model_config.hf_config.text_config.num_hidden_layers

        if use_ordered_embeddings:
            assert model.masked_embedding is not None
            assert model.masked_embedding.num_centroids == 32
            assert model.masked_embedding.centroid_intermediate_top_k == 4
        else:
            assert model.masked_embedding is None

    @pytest.mark.parametrize("use_ordered_embeddings", [True, False])
    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
    def test_forward_pass(self, use_ordered_embeddings: bool,
                          kv_cache_dtype: str, rng: PRNGKey, mesh: Mesh,
                          mock_model_inputs):
        """Tests the forward pass of the Gemma4MTPForCausalLM model."""
        vllm_config = MockVllmConfig(
            use_ordered_embeddings=use_ordered_embeddings,
            kv_cache_dtype=kv_cache_dtype)
        with jax.set_mesh(mesh):
            model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)

        input_ids, hidden_states, attention_metadata = mock_model_inputs

        draft_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = draft_hf_config.text_config

        head_size = utils.get_padded_head_dim(text_config.global_head_dim)

        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=16,
            num_kv_heads=text_config.num_key_value_heads,
            head_size=head_size,
            mesh=mesh,
            layer_names=["layer"] * text_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if kv_cache_dtype == "fp8" else jnp.bfloat16)

        # We mock the centroids' token ordering embedding with valid values (ranges 0-vocab_size)
        if use_ordered_embeddings:
            model.masked_embedding.token_ordering.value = jnp.arange(
                text_config.vocab_size, dtype=jnp.int32)

        with jax.set_mesh(mesh):
            kv_caches, draft_hidden_states, backbone_hidden_states_list, _ = model(
                kv_caches, input_ids, hidden_states, attention_metadata)

        assert len(kv_caches) == text_config.num_hidden_layers
        assert draft_hidden_states.shape == (input_ids.shape[0],
                                             text_config.hidden_size)
        assert len(backbone_hidden_states_list) == 1

        backbone_hidden_states = backbone_hidden_states_list[0]
        assert backbone_hidden_states.shape == (
            input_ids.shape[0], draft_hf_config.backbone_hidden_size)

        # Compute logits
        logits = model.compute_logits(draft_hidden_states)
        assert logits.shape == (input_ids.shape[0], text_config.vocab_size)

        # Get top tokens
        top_tokens = model.get_top_tokens(draft_hidden_states)
        assert top_tokens.shape == (input_ids.shape[0], )
