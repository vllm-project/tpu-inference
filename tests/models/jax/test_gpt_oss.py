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
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.models.jax.gpt_oss import GptOss
from tpu_inference.models.jax.jax_intermediate_tensor import JaxIntermediateTensors
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:
    def __init__(self, model: str, num_hidden_layers: int = 4):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.speculative_config = None
        self.cache_config = MagicMock(cache_dtype="auto")
        self.additional_config = MagicMock()
        self.additional_config.get.return_value = False
        
        # Override HF config for testing
        hf_config = self.model_config.hf_config
        hf_config.num_hidden_layers = num_hidden_layers
        hf_config.num_local_experts = 2
        hf_config.vocab_size = 1000
        hf_config.num_attention_heads = 4
        hf_config.num_key_value_heads = 2
        hf_config.head_dim = 32
        hf_config.hidden_size = 128
        hf_config.intermediate_size = 256
        hf_config.num_experts_per_tok = 1
        hf_config.rms_norm_eps = 1e-6
        hf_config.swiglu_limit = 1.0
        hf_config.rope_theta = 10000.0
        hf_config.rope_scaling = {"factor": 1.0, "beta_slow": 1.0, "beta_fast": 1.0, "original_max_position_embeddings": 2048}
        hf_config.sliding_window = 1024
        hf_config.tie_word_embeddings = False


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    with Mesh(device_mesh, axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_model_inputs():
    num_tokens = 4
    num_reqs = 1
    input_ids = jnp.ones((num_tokens,), dtype=jnp.int32)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    
    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=jnp.zeros((num_reqs, 4), dtype=jnp.int32).reshape(-1),
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        query_start_loc=jnp.array([0, num_tokens], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    return input_ids, attention_metadata


class TestGptOss:
    def test_init_single_rank(self, rng, mesh):
        mock_pp = MagicMock(is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1)
        with patch("tpu_inference.models.jax.gpt_oss.get_pp_group", return_value=mock_pp), 
             patch("tpu_inference.layers.jax.pp_utils.get_pp_group", return_value=mock_pp):
            config = MockVllmConfig("test-model", num_hidden_layers=4)
            model = GptOss(config, rng, mesh)
            
            assert not isinstance(model.embedder, PPMissingLayer)
            assert not isinstance(model.final_norm, PPMissingLayer)
            assert not isinstance(model.lm_head, PPMissingLayer)
            assert len(model.layers) == 4
            for layer in model.layers:
                assert not isinstance(layer, PPMissingLayer)

    def test_init_pipeline_parallel_rank0(self, rng, mesh):
        # 2 stages, rank 0 (layers 0, 1)
        mock_pp = MagicMock(is_first_rank=True, is_last_rank=False, rank_in_group=0, world_size=2)
        with patch("tpu_inference.models.jax.gpt_oss.get_pp_group", return_value=mock_pp), 
             patch("tpu_inference.layers.jax.pp_utils.get_pp_group", return_value=mock_pp):
            config = MockVllmConfig("test-model", num_hidden_layers=4)
            model = GptOss(config, rng, mesh)
            
            assert not isinstance(model.embedder, PPMissingLayer)
            assert isinstance(model.final_norm, PPMissingLayer)
            assert isinstance(model.lm_head, PPMissingLayer)
            
            assert len(model.layers) == 4
            assert not isinstance(model.layers[0], PPMissingLayer)
            assert not isinstance(model.layers[1], PPMissingLayer)
            assert isinstance(model.layers[2], PPMissingLayer)
            assert isinstance(model.layers[3], PPMissingLayer)

    def test_init_pipeline_parallel_rank1(self, rng, mesh):
        # 2 stages, rank 1 (layers 2, 3)
        mock_pp = MagicMock(is_first_rank=False, is_last_rank=True, rank_in_group=1, world_size=2)
        with patch("tpu_inference.models.jax.gpt_oss.get_pp_group", return_value=mock_pp), 
             patch("tpu_inference.layers.jax.pp_utils.get_pp_group", return_value=mock_pp):
            config = MockVllmConfig("test-model", num_hidden_layers=4)
            model = GptOss(config, rng, mesh)
            
            assert isinstance(model.embedder, PPMissingLayer)
            assert not isinstance(model.final_norm, PPMissingLayer)
            assert not isinstance(model.lm_head, PPMissingLayer)
            
            assert len(model.layers) == 4
            assert isinstance(model.layers[0], PPMissingLayer)
            assert isinstance(model.layers[1], PPMissingLayer)
            assert not isinstance(model.layers[2], PPMissingLayer)
            assert not isinstance(model.layers[3], PPMissingLayer)

    def test_forward_rank0(self, rng, mesh, mock_model_inputs):
        mock_pp = MagicMock(is_first_rank=True, is_last_rank=False, rank_in_group=0, world_size=2)
        with patch("tpu_inference.models.jax.gpt_oss.get_pp_group", return_value=mock_pp), 
             patch("tpu_inference.layers.jax.pp_utils.get_pp_group", return_value=mock_pp):
            config = MockVllmConfig("test-model", num_hidden_layers=4)
            model = GptOss(config, rng, mesh)
            
            input_ids, attention_metadata = mock_model_inputs
            kv_caches = create_kv_caches(
                num_blocks=4, block_size=32, num_kv_heads=2, head_size=32,
                mesh=mesh, layer_names=["layer"] * 2, cache_dtype=jnp.bfloat16
            )
            
            # For rank 0, call should return JaxIntermediateTensors
            kv_caches, output, aux = model(kv_caches, input_ids, attention_metadata)
            
            assert isinstance(output, JaxIntermediateTensors)
            assert "hidden_states" in output.tensors
            assert output["hidden_states"].shape == (4, 128)

    def test_forward_rank1(self, rng, mesh, mock_model_inputs):
        mock_pp = MagicMock(is_first_rank=False, is_last_rank=True, rank_in_group=1, world_size=2)
        with patch("tpu_inference.models.jax.gpt_oss.get_pp_group", return_value=mock_pp), 
             patch("tpu_inference.layers.jax.pp_utils.get_pp_group", return_value=mock_pp):
            config = MockVllmConfig("test-model", num_hidden_layers=4)
            model = GptOss(config, rng, mesh)
            
            input_ids, attention_metadata = mock_model_inputs
            kv_caches = create_kv_caches(
                num_blocks=4, block_size=32, num_kv_heads=2, head_size=32,
                mesh=mesh, layer_names=["layer"] * 2, cache_dtype=jnp.bfloat16
            )
            
            # For rank 1, we need to pass intermediate_tensors
            hidden_states = jnp.ones((4, 128), dtype=jnp.bfloat16)
            intermediate_tensors = JaxIntermediateTensors({"hidden_states": hidden_states})
            
            kv_caches, output, aux = model(kv_caches, input_ids, attention_metadata, 
                                           intermediate_tensors=intermediate_tensors)
            
            # For last rank, output should be hidden states after final norm
            assert isinstance(output, jax.Array)
            assert output.shape == (4, 128)
